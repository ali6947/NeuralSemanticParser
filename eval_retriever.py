import argparse
import torch

from allennlp.models.archival import Archive, load_archive, archive_model
from allennlp.data.vocabulary import Vocabulary
from smbop.modules.relation_transformer import *
import json
from allennlp.common import Params
from smbop.dataset_readers.spider_retriever_nn import SmbopSpiderRetrieverDatasetReader
from smbop.dataset_readers.spider_retriever_nn_val import SmbopSpiderRetrieverValDatasetReader
# from smbop.vocabulary.vocab_retriever import RetriverSpiderVocabulary
# from smbop.models.smbop_vanilla import SmbopParser
# from smbop.models.cbr_smbop import CBRSmbopParser
from smbop.models.smbop_pretrain import SmbopSimPretrained
from smbop.modules.relation_transformer import RelationTransformer
from smbop.modules.lxmert import LxmertCrossAttentionLayer
from smbop.data_loaders.data_loader_retriever import RetriverMultiProcessDataLoader
import itertools
import smbop.utils.node_util as node_util
import numpy as np
import json
from tqdm import tqdm
from allennlp.models import Model
from allennlp.common.params import *
from allennlp.data import DatasetReader, Instance
from allennlp.predictors import Predictor
import json
from allennlp.data.fields import  ListField, MetadataField
from copy import deepcopy
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive_path",type=str)
    parser.add_argument("--dev_inst", type=str, default="pickle_objs/all_InstanceObj_spider_val_grappa.pkl")
    parser.add_argument("--nn_masks_file", type=str, default="pickle_objs/spider_val_classes_5_20_25_25_25_only_eval.pkl")
    # parser.add_argument(
    #     "--output", type=str, default="predictions_with_vals_fixed4.txt"
    # )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    # overrides = {
    #     "dataset_reader": {
    #         "tables_file": args.table_path,
    #         "dataset_path": args.dataset_path,
    #     }
    # }
    # overrides["validation_dataset_reader"] = {
    #     "tables_file": args.table_path,
    #     "dataset_path": args.dataset_path,
    # }
    predictor = Predictor.from_path(
        args.archive_path, cuda_device=args.gpu)
    print("after pred")

    with open(args.dev_inst,'rb') as f:
        allinst=pickle.load(f)
    with open(args.nn_masks_file,'rb') as f:
        nn_masks=pickle.load(f)

    [instance.add_field('inst_id',MetadataField(i)) for i,instance in enumerate(allinst)]
    accvec=[]
    no_classa=0
    np.random.seed(45000)
    for x in tqdm(allinst[600:800]):
        inst=deepcopy(x)
        nn_indices=[]
        iid=inst['inst_id'].metadata
        for nn_mask in nn_masks:
            nn_indices.append(np.where(nn_mask[iid]==1)[0])
        if nn_indices[0].shape[0]==0:
            no_classa+=1
            continue
        chosen_inst=[np.random.choice(x) for x in nn_indices if x.shape[0]!=0]
        if len(chosen_inst)<5:
            dr_idxs=np.concatenate(nn_indices[1:])
            dr_idxs=np.setdiff1d(dr_idxs,chosen_inst)
            more_idxs=np.random.choice(dr_idxs,5-len(chosen_inst),replace=False)
            chosen_inst=chosen_inst+list(more_idxs)
        assert iid not in chosen_inst and np.unique(chosen_inst).shape[0]==5
        nn_inst=[allinst[nn] for nn in chosen_inst]
        ins_fields = {}
        # field_names = inst.fields.keys()
        
        field_names = ['enc','db_id','is_gold_leaf','lengths','offsets','relation','span_hash','is_gold_span','inst_id']
        for field_type in field_names:
            ex_item = inst[field_type]
            nn_items = [item[field_type] for item in nn_inst]
            all_items = [ex_item] + nn_items
            list_field = ListField(all_items)
            ins_fields[field_type] = list_field
        fin_inst=Instance(ins_fields)
        predictor._dataset_reader.apply_token_indexers(fin_inst)
        with torch.cuda.amp.autocast(enabled=True):
            out = predictor._model.forward_on_instances(
                [fin_inst, deepcopy(fin_inst)]
            )
            accvec.append(out[0]['nn_acc'])
        
    print(accvec[:10])
    print(sum(accvec))
    print(np.mean(accvec))
    print(no_classa)



            


if __name__ == "__main__":
    main()
