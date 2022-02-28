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
# from smbop.models.retriever_soft import SmbopSimPretrained
from smbop.models.retriever_get_cls import SmbopSimPretrained
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
    parser.add_argument("--archive_path",type=str,default='pretrained_ckpt/retriever_v1/')
    
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

    dirs1=['split_0', 'split_1', 'split_2']
    dirs2=['flight_2', 'car_1', 'cre_Doc_Template_Mgt', 'dog_kennels', 'world_1']
    BASE_PATH= "/mnt/infonas/data/awasthi/semantic_parsing/smbop/pickles/spider_val_cbr_splits_30/"
    BASE_PATH_BB="/mnt/infonas/data/alirehan/semantic_parsing/pickle_objs/bigbird_base/spider_val_cbr_splits_30"
    # files=['val', 'train', 'test']

    for f1 in dirs1:
        for f2 in dirs2:

            with open(os.path.join(BASE_PATH,f1,f2,'train.pkl'),'rb') as f: # train gives cases
                allinst_train=pickle.load(f)

            with open(os.path.join(BASE_PATH_BB,f1,f2,'train.pkl'),'rb') as f:
                allinst_train_bigbird=pickle.load(f)

            with open(os.path.join(BASE_PATH,f1,f2,'test.pkl'),'rb') as f:
                allinst=pickle.load(f)

            with open(os.path.join(BASE_PATH_BB,f1,f2,'test.pkl'),'rb') as f:
                allinst_bigbird=pickle.load(f)

            for xy in [allinst,allinst_bigbird,allinst_train,allinst_train_bigbird]:
                [instance.add_field('inst_id',MetadataField(i)) for i,instance in enumerate(xy)]

            inst_w_cases=[]
            num_eg=len(allinst)
            num_eg_train=len(allinst_train)
            cls_embs=torch.zeros(num_eg,256)
            cls_embs_train=torch.zeros(num_eg_train,256)

            for eg_idx in (range(num_eg)):
                inst=deepcopy(allinst[eg_idx])
                iid=inst['inst_id'].metadata
                
                nn_inst=[deepcopy(inst)]
                ins_fields = {}
                
                field_names = ['enc','db_id','is_gold_leaf','lengths','offsets','relation','span_hash','is_gold_span','inst_id']        
                for field_type in field_names:
                    ex_item = inst[field_type]
                    nn_items = [item[field_type] for item in nn_inst]
                    all_items = [ex_item] + nn_items
                    list_field = ListField(all_items)
                    ins_fields[field_type] = list_field
                fin_inst=Instance(ins_fields)
                predictor._dataset_reader.apply_token_indexers(fin_inst)
                try:
                    with torch.cuda.amp.autocast(enabled=True):
                        out = predictor._model.forward_on_instance(
                            fin_inst
                        )
                        cls_embs[eg_idx]=torch.tensor(out['cls_toks'])
                except:
                    pass

            for eg_idx in (range(num_eg_train)):
                inst=deepcopy(allinst_train[eg_idx])
                iid=inst['inst_id'].metadata
                
                nn_inst=[deepcopy(inst)]
                ins_fields = {}
                
                field_names = ['enc','db_id','is_gold_leaf','lengths','offsets','relation','span_hash','is_gold_span','inst_id']        
                for field_type in field_names:
                    ex_item = inst[field_type]
                    nn_items = [item[field_type] for item in nn_inst]
                    all_items = [ex_item] + nn_items
                    list_field = ListField(all_items)
                    ins_fields[field_type] = list_field
                fin_inst=Instance(ins_fields)
                predictor._dataset_reader.apply_token_indexers(fin_inst)
                try:
                    with torch.cuda.amp.autocast(enabled=True):
                        out = predictor._model.forward_on_instance(
                            fin_inst
                        )
                        cls_embs_train[eg_idx]=torch.tensor(out['cls_toks'])
                except:
                    pass

            all_sims=torch.nn.functional.cosine_similarity(cls_embs.unsqueeze(1),cls_embs_train.unsqueeze(0),dim=-1)
            # with open('pickle_objs/bigbird_base/grappa_val_train_cossim.pkl','wb') as fcos:
            #     pickle.dump(all_sims,fcos)
            print(torch.max(all_sims,dim=-1)[0])
            for eg_idx in (range(num_eg)):
                accvec=all_sims[eg_idx]  
                sidx=np.argsort(accvec)[-5:]
                sidx=[x for x in sidx if accvec[x]>0.7]
                instbb=deepcopy(allinst_bigbird[eg_idx])
                nn_cases=[ deepcopy(allinst_train_bigbird[sii]) for sii in reversed(sidx)]
                instbb.add_field('cases',MetadataField(nn_cases))
                inst_w_cases.append(instbb)


            with open(os.path.join(BASE_PATH_BB,f1,f2,'test_w_top5_cases_from_train_no_leaf_thresh_gt_0.7.pkl'),'wb') as fw:
                pickle.dump(inst_w_cases,fw)

if __name__ == "__main__":
    main()
