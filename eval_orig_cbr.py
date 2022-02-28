import argparse
import torch

from allennlp.models.archival import Archive, load_archive, archive_model
from allennlp.data.vocabulary import Vocabulary
from smbop.modules.relation_transformer import *
import json
from allennlp.common import Params
# from smbop.models.smbop_with_concat_cases import SmbopParser
# from smbop.models.smbop_with_concat_cases_utt_aug import SmbopParser
from smbop.models.smbop_with_concat_cases_vanilla_mix_new_mets_tmp import SmbopParser
# from smbop.models.cbr_smbop import CBRSmbopParser
from smbop.modules.lxmert import LxmertCrossAttentionLayer
# from smbop.dataset_readers.spider_basic_pkl_singledb import SmbopSpiderDatasetReader
# from smbop.dataset_readers.spider_round_robin_cases import SmbopSpiderDatasetReader
from smbop.dataset_readers.cbr_with_same_schema import CBRSameSchemaDatasetReader # not the actual data loader for orig_cbr but doesnt matter since we dont use it, actual is spider_round_robin_cases
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
import os
import shutil
import tarfile
import pickle
from pathlib import Path
from allennlp.data.fields import ListField,TextField,TensorField
from itertools import cycle
from copy import deepcopy

def create_archive(archive_dir):
    filename = os.path.join(archive_dir,'copy_model.tar.gz')
    tmp_dir = os.path.join(archive_dir,'copy_model')
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive_dir",type=str)
    parser.add_argument("--dev_path", type=str, default="/mnt/infonas/data/alirehan/smbop/pickle_objs/complete_spider_val_with_cases.pkl")
    parser.add_argument("--table_path", type=str, default="dataset/tables.json")
    parser.add_argument("--dataset_path", type=str, default="dataset/database")
    parser.add_argument(
        "--output", type=str, default="predictions_with_vals_fixed4.txt"
    )
    parser.add_argument(
        "--output_gold", type=str, default="gold_predictions_with_vals_fixed4.txt"
    )
    parser.add_argument(
        "--db_id", type=str, default="world_1"
        )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    '''
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
    '''


    archive_dir = args.archive_dir
    '''
    if os.path.isfile(os.path.join(archive_dir,'model.tar.gz')):
        print('Found: ',os.path.join(archive_dir,'model.tar.gz'))
        archive_file = os.path.join(archive_dir, 'model.tar.gz')
    elif os.path.isfile(os.path.join(archive_dir, 'copy_model.tar.gz')):
        print('Found: ', os.path.join(archive_dir,'copy_model.tar.gz'))
        archive_file = os.path.join(archive_dir, 'copy_model.tar.gz')
    else:
        print('Creating (Not Found): ',os.path.join(archive_dir,'copy_model.tar.gz'))
        assert os.path.exists(archive_dir)
        archive_file = create_archive(archive_dir)

    print(archive_file)
    '''
    predictor = Predictor.from_path(
        archive_dir, cuda_device=args.gpu) #overrides=overrides

    print("after pred")

    with open(args.output, "w") as g:
        with open(args.output_gold,'w') as gg:
            with open(args.dev_path, "rb") as f:
                pkl_obj = pickle.load(f)
                set0=False
                leaf_log=[]
                avg_prec=[]
                inv_rank=[]
                gold_tree_rec=[]
                gold_leaf_rec=[]
                leaf_rec=[]
                for i, ex in enumerate(tqdm.tqdm(pkl_obj)):
                    gsql=ex['gold_sql'].metadata
                    ex_db_id = ex['db_id'].metadata
                    # if ex_db_id!=args.db_id:
                    #     continue
                    tokens_with_caseSQL=ex['enc'].tokens.copy()
                    pool = cycle(ex['cases'].metadata)
                    for item in pool:
                        ######### some pickles have cases as tuples some dont so we take care of that here
                        if type(item)==type((2,3)):
                            inst,NL_ques=item
                        else:
                            inst=item
                        #############

                        if len(tokens_with_caseSQL)>=1024:
                            break
                        for token in inst['enc'].tokens[1:]:
                            tokens_with_caseSQL.append(token)
                            if token.text=='[SEP]':
                                    break
                        gold_sql_tokens=predictor._dataset_reader._tokenizer.tokenize(inst['gold_sql'].metadata+' [SEP] [SEP]') #for bigbird
                        tokens_with_caseSQL.extend(gold_sql_tokens[1:])
                    tokens_with_caseSQL=TextField(tokens_with_caseSQL)
                    # print(tokens_with_caseSQL)
                    ex.add_field('cases',tokens_with_caseSQL) #NOTE: add_field overwrise the first argument ke name wali field, if already present
                    ex.fields["cases"].token_indexers = predictor._dataset_reader._utterance_token_indexers
                    instance = ex
                    if not set0:
                        instance_0 = instance
                        set0=True
                    else:
                        pass #below code was for ensuring there is no gold leakage
                        # instance.add_field('gold_sql',(deepcopy(instance_0['gold_sql'])))
                        # instance.add_field('is_gold_leaf',TensorField(torch.zeros_like(instance_0['is_gold_leaf'].tensor)))
                        # instance.add_field('hash_gold_levelorder',TensorField(torch.zeros_like(instance_0['hash_gold_levelorder'].tensor)))
                        # instance.add_field('hash_gold_tree',deepcopy(instance_0['hash_gold_tree']))
                        # instance.add_field('is_gold_span',TensorField(torch.zeros_like(instance_0['is_gold_span'].tensor)))
                    if instance is not None:
                        predictor._dataset_reader.apply_token_indexers(instance)
                        with torch.cuda.amp.autocast(enabled=True):
                            out = predictor._model.forward_on_instances(
                                [instance, instance_0]
                            )
                            pred = out[0]["sql_list"]
                            avg_prec.append(float(out[0]['avg_prec']))
                            leaf_log.append(float(out[0]['leaf_log']))
                            inv_rank.append(float(out[0]['inv_rank']))
                            gold_tree_rec.append(float(out[0]['final_beam_acc']))
                            gold_leaf_rec.append(float(out[0]['leaf_acc']))
                    else:
                        pred = "NO PREDICTION"
                    g.write(f"{pred}\t{ex_db_id}\n")
                    gg.write(f"{gsql}\t{ex_db_id}\n")
                print(f'leaf_log: {np.mean(leaf_log):.3f}')
                print(f'avg_prec: {np.mean(avg_prec):.3f}')
                print(f'inv_rank: {np.mean(inv_rank):.3f}')
                print(f'gold_tree_recall: {np.mean(gold_tree_rec):.3f}')
                print(f'gold_leaf_recall: {np.mean(gold_leaf_rec):.3f}')


if __name__ == "__main__":
    main()
