import os
import json
import torch
from allennlp.data.vocabulary import Vocabulary
# from smbop.dataset_readers.kaggledb_bigbird import KaggleDBQAReader
from smbop.dataset_readers.spider_bigbird import SmbopSpiderDatasetReader
from allennlp.data import DatasetReader, Instance
import tqdm
from allennlp.data.token_indexers import PretrainedTransformerIndexer 
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.fields import TensorField, MetadataField, TextField
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.data_loaders import SimpleDataLoader


# PRETRAINED_MODEL_PATH="/mnt/infonas/data/awasthi/semantic_parsing/roberta-base"
PRETRAINED_MODEL_PATH="google/bigbird-roberta-base"
DATABASE_DIR="/mnt/infonas/data/awasthi/semantic_parsing/smbop/dataset/database"
BASE_PATH_SRC = "/mnt/infonas/data/awasthi/semantic_parsing/smbop/pickles/spider_val_cbr_splits_30/"
BASE_PATH_DST="/mnt/infonas/data/alirehan/semantic_parsing/pickle_objs/bigbird_base/spider_val_cbr_splits_30_tmp"
TABLES_PATH = "/mnt/infonas/data/awasthi/semantic_parsing/smbop/dataset/tables.json"

q_token_indexer = PretrainedTransformerIndexer(model_name=PRETRAINED_MODEL_PATH)
q_tokenizer = PretrainedTransformerTokenizer(model_name=PRETRAINED_MODEL_PATH)
vocab = Vocabulary.from_pretrained_transformer(model_name=PRETRAINED_MODEL_PATH)


kaggledb_reader = SmbopSpiderDatasetReader(
        lazy = False,
        question_token_indexers = {"tokens": q_token_indexer},
        keep_if_unparsable = False,
        tables_file = os.path.join(TABLES_PATH),
        dataset_path = DATABASE_DIR,
        cache_directory = "scratch/cache/train",
        include_table_name_in_column=True,
        fix_issue_16_primary_keys=False,
        qq_max_dist=2,
        cc_max_dist=2,
        tt_max_dist=2,
        max_instances=10,
        decoder_timesteps=9,
        limit_instances=-1,
        value_pred=True,
        use_longdb=True,

)
# print(kaggledb_reader.cls_token)
# print(kaggledb_reader.eos_token)
print('constructed dataset reader')

dirs1=['split_0', 'split_1', 'split_2']
dirs2=['flight_2', 'car_1', 'cre_Doc_Template_Mgt', 'dog_kennels', 'world_1']
files=['val', 'train', 'test']

for f1 in dirs1:
  for f2 in dirs2:
    for f3 in files:

      if not os.path.isdir(os.path.join(BASE_PATH_DST,f1)):
        os.mkdir(os.path.join(BASE_PATH_DST,f1))

      if not os.path.isdir(os.path.join(BASE_PATH_DST,f1,f2)):
        os.mkdir(os.path.join(BASE_PATH_DST,f1,f2))

      print(f1,f2,f3)
      kaggledb_reader.process_and_dump_pickle(os.path.join(BASE_PATH_SRC,f1,f2,f3+'.json'), os.path.join(BASE_PATH_DST,f1,f2,f3+'.pkl'))
