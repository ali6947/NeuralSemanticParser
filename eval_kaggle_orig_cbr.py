import argparse
import torch

from allennlp.models.archival import Archive, load_archive, archive_model
from allennlp.data.vocabulary import Vocabulary
from smbop.modules.relation_transformer import *
import json
from allennlp.common import Params

# from smbop.models.smbop import SmbopParser
# from smbop.models.cbr_smbop import CBRSmbopParser
# from smbop.models.cbr_smbop_w_leafs import CBRSmbopParser
# from smbop.models.cbr_same_schema_leafs import CBRSameSchemaLeafs
# from smbop.models.smbop_NL_enr_schema_aft_rat import SmbopParser
# from smbop.models.enr_schema_plus_cbr import EnrSchemaPlusCBR
# from smbop.models.smbop_enr_NL_align_loss import SmbopParser
# from smbop.models.smbop_enr_pbase_proper_MHA import EnrSchemaSmbopParserMasked
# from smbop.models.cbr_smbop_w_leafs import CBRSmbopParser
from smbop.models.smbop_with_concat_cases import SmbopParser

from smbop.modules.lxmert import LxmertCrossAttentionLayer
# from smbop.dataset_readers.spider_basic_pkl_singledb import SmbopSpiderDatasetReader
from smbop.data_loaders.cbr_with_same_schema import CBRSameSchemaDataLoader
from smbop.dataset_readers.cbr_with_same_schema import CBRSameSchemaDatasetReader

import itertools
import smbop.utils.node_util as node_util
import numpy as np
import numpy as np
import json
import tqdm
from allennlp.models import Model
from allennlp.common.params import *
from allennlp.data import DatasetReader, Instance
import tqdm
from allennlp.predictors import Predictor
import json
import pickle
import os
import shutil
import tarfile
from pathlib import Path
from allennlp.data.fields import ListField

def create_archive(archive_dir):
    filename = os.path.join(archive_dir,'model.tar.gz')
    tmp_dir = os.path.join(archive_dir,'model')
    os.makedirs(tmp_dir)
    best_weights = os.path.join(archive_dir,'best.th')
    shutil.copyfile(best_weights, os.path.join(tmp_dir,'weights.th'))
    config_file = os.path.join(archive_dir, 'config.json')
    shutil.copyfile(config_file, os.path.join(tmp_dir,'config.json'))
    vocab_file = os.path.join(archive_dir, 'vocabulary/non_padded_namespaces.txt')
    os.makedirs(os.path.join(tmp_dir,'vocabulary'))
    shutil.copyfile(vocab_file, os.path.join(tmp_dir,'vocabulary/non_padded_namespaces.txt'))
    with tarfile.open(filename, "w:gz") as tar:
        tar.add(tmp_dir, arcname=os.path.basename(tmp_dir))
    shutil.rmtree(tmp_dir)
    return filename

def update_instance_with_cases(ex, cases, idx):
    ins_fields={}
    field_names = ex.fields.keys()    
    #case_indices = np.random.choice(len(cases),3)
    #cases = [cases[i] for i in case_indices]
    #cases = [ex]
    for field_type in field_names:
        ex_item = ex[field_type]
        case_items = [item[field_type] for item in cases]
        #print('\n=============\n', len(cases),'\n=============\n')
        #case_items = [cases[idx][field_type]]
        if field_type == 'gold_sql':
            pass
            #print('case: ',case_items[0].metadata)
        all_items = [ex_item] + case_items
        list_field = ListField(all_items)
        ins_fields[field_type] = list_field
    instance = Instance(ins_fields)
    return instance

def case_apply_token_indexers(instance: Instance,reader: DatasetReader) -> None:
    if isinstance(instance.fields["cases"], ListField):
        for text_field in instance.fields["cases"].field_list:
            text_field.token_indexers = reader._utterance_token_indexers
    else:
        instance.fields["cases"].token_indexers = reader._utterance_token_indexers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive_dir",type=str)
    parser.add_argument("--dev_path", type=str, default="dataset/dev.json")
    parser.add_argument("--table_path", type=str, default="dataset/tables.json")
    parser.add_argument("--dataset_path", type=str, default="dataset/database")
    parser.add_argument("--output", type=str, default="predictions_with_vals_fixed4.txt")
    parser.add_argument(
        "--output_gold", type=str, default="gold_predictions_with_vals_fixed4.txt"
    )
    parser.add_argument("--cases", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    overrides = {
        "dataset_reader": {
            "tables_file": args.table_path,
            "dataset_path": args.dataset_path,
        }
    }
    overrides["validation_dataset_reader"] = {
        "tables_file": args.table_path,
        "dataset_path": args.dataset_path,
    }

    if args.cases is not None:
        cases = pickle.load(open(args.cases,"rb"))
        #cases = cases[0:10]
    else:
        cases = None

    archive_dir = args.archive_dir
    predictor = Predictor.from_path(
        archive_dir, cuda_device=args.gpu, overrides=overrides
    )
    print("after pred")

    final_beam_acc_stats = []
    leaf_acc_stats = []
    with open(args.output_gold,'w') as gg:
        with open(args.output, "w") as g:    
            with open(args.dev_path,'rb') as f:
                dev_pkl = pickle.load(f)
                leaf_log=[]
                avg_prec=[]
                inv_rank=[]
                gold_tree_rec=[]
                gold_leaf_rec=[]
                for i, el in enumerate(tqdm.tqdm(dev_pkl)):
                    if i == 0:
                        el_0 = el
                    #print('example: ',el['gold_sql'].metadata)

                    '''
                    el['gold_sql'].metadata = el_0['gold_sql'].metadata
                    el['is_gold_leaf'].tensor = torch.zeros_like(el['is_gold_leaf'].tensor)
                    el['hash_gold_levelorder'].tensor = torch.zeros_like(el['hash_gold_levelorder'].tensor)
                    el['hash_gold_tree'].tensor = torch.zeros_like(el['hash_gold_tree'].tensor)
                    el['is_gold_span'].tensor = torch.zeros_like(el['is_gold_span'].tensor)
                    '''

                    instance = el
                    if i == 0:
                        instance_0 = instance
                    if instance is not None and cases is not None:
                        instance = update_instance_with_cases(instance, cases, i)

                    if instance is not None:
                        predictor._dataset_reader.apply_token_indexers(instance)
                        case_apply_token_indexers(instance,predictor._dataset_reader)
                        if cases is not None:
                            #print('cases is not None')
                            with torch.cuda.amp.autocast(enabled=True):
                                out = predictor._model.forward_on_instances(
                                    [instance]
                                )
                                pred = out[0]["sql_list"]
                                db_id = el['db_id'].metadata
                        elif isinstance(instance['db_id'],ListField):
                            with torch.cuda.amp.autocast(enabled=True):
                                out = predictor._model.forward_on_instances(
                                    [instance]
                                )
                                pred = out[0]["sql_list"]
                                db_id = list(el['db_id'])[0].metadata
                        else:
                            with torch.cuda.amp.autocast(enabled=True):
                                out = predictor._model.forward_on_instances(
                                    [instance, instance_0]
                                )
                                pred = out[0]["sql_list"]
                                db_id = el['db_id'].metadata
                        avg_prec.append(float(out[0]['avg_prec']))
                        leaf_log.append(float(out[0]['leaf_log']))
                        inv_rank.append(float(out[0]['inv_rank']))
                        gold_tree_rec.append(float(out[0]['final_beam_acc']))
                        gold_leaf_rec.append(float(out[0]['leaf_acc']))
                        g.write(f"{pred}\t{db_id}\n")
                        # print(f"{instance['gold_sql'].metadata}\t{db_id}")
                        gg.write(f"{instance['gold_sql'].metadata}\t{db_id}\n")
                    else:
                        pred = "NO PREDICTION"
                    
                print(f'leaf_log: {np.mean(leaf_log):.3f}')
                print(f'avg_prec: {np.mean(avg_prec):.3f}')
                print(f'inv_rank: {np.mean(inv_rank):.3f}')
                print(f'gold_tree_recall: {np.mean(gold_tree_rec):.3f}')
                print(f'gold_leaf_recall: {np.mean(gold_leaf_rec):.3f}')                
# print(f'Mean final_beam_acc: {np.mean(final_beam_acc_stats)}')
#         #print(f'final_beam_acc_stats: {final_beam_acc_stats}')
#         print(f'Mean leaf_acc: {np.mean(leaf_acc_stats)}')


if __name__ == "__main__":
    main()
