from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, TokenIndexer, Field, Instance
from allennlp.data.fields import TextField, ListField, IndexField, MetadataField, TensorField
from allennlp.data.fields import (
    TextField,
    ListField,
    IndexField,
    MetadataField,
    ArrayField,
)

import torch
import anytree
from anytree.search import *
from collections import defaultdict
from overrides import overrides
from time import time
from typing import Dict
from smbop.utils import moz_sql_parser as msp
import pickle
import smbop.utils.node_util as node_util
import smbop.utils.hashing as hashing
import smbop.utils.ra_preproc as ra_preproc
from anytree import Node, LevelOrderGroupIter
import dill
import itertools
from collections import defaultdict, OrderedDict
import json
import logging
import numpy as np
import os
from smbop.utils.replacer import Replacer
import time
from smbop.dataset_readers.enc_preproc import *
import smbop.dataset_readers.disamb_sql as disamb_sql
from smbop.utils.cache import TensorCache
# from smbop.dataset_readers.spider_basic_pkl_bigbird import SmbopSpiderDatasetReader
from smbop.dataset_readers.spider import SmbopSpiderDatasetReader
logger = logging.getLogger(__name__)


@DatasetReader.register("smbop_retriver_val")
class SmbopSpiderRetrieverValDatasetReader(SmbopSpiderDatasetReader):
    def __init__(
        self,
        lazy: bool = True,
        question_token_indexers: Dict[str, TokenIndexer] = None,
        keep_if_unparsable: bool = True,
        tables_file: str = None,
        dataset_path: str = "dataset/database",
        cache_directory: str = "cache/train",
        include_table_name_in_column=True,
        fix_issue_16_primary_keys=False,
        qq_max_dist=2,
        cc_max_dist=2,
        tt_max_dist=2,
        max_instances=10000000,
        decoder_timesteps=9,
        limit_instances=-1,
        value_pred=True,
        use_longdb=True,
        nn_idx_file: str = None,  # list of pairs having (idx, list of nn idx)
        all_TED: str=None
    ):
        
        with open(nn_idx_file,'rb') as f:
            self.nn_idx=pickle.load(f)
        
        with open(all_TED,'rb') as f:
            self.all_TED=pickle.load(f)

        super().__init__(
            lazy,
            question_token_indexers,
            keep_if_unparsable,
            tables_file,
            dataset_path,
            cache_directory,
            include_table_name_in_column,
            fix_issue_16_primary_keys,
            qq_max_dist,
            cc_max_dist,
            tt_max_dist,
            max_instances,
            decoder_timesteps,
            limit_instances,
            value_pred,
            use_longdb,
        )

    @overrides
    def _read(self, file_path: str):
        if file_path.endswith(".pkl"):
            yield from self._read_examples_file(file_path)
        else:
            raise ConfigurationError(f"Don't know how to read filetype of {file_path}")



    def _read_examples_file(self, file_path: str):
        # cache_dir = os.path.join("cache", file_path.split("/")[-1])
        print('Reading NN for val yay: ',file_path)    
        with open(file_path, "rb") as data_file:
            inst_list = pickle.load(data_file)
            [inst.add_field('inst_id',MetadataField(i)) for i,inst in enumerate(inst_list)]
            # [self.apply_token_indexers(x) for x in inst_list]
            for idx,nns in self.nn_idx:
                nn_inst=[inst_list[x] for x in nns]
                ins_fields = {}
                # field_names = inst.fields.keys()
                inst=inst_list[idx]
                field_names = ['enc','db_id','is_gold_leaf','lengths','offsets','relation','span_hash','is_gold_span','inst_id']
                for field_type in field_names:
                    ex_item = inst[field_type]
                    nn_items = [item[field_type] for item in nn_inst]
                    all_items = [ex_item] + nn_items
                    list_field = ListField(all_items)
                    ins_fields[field_type] = list_field

                ### NOTE: added new by ALi 30/12/2021
                TEDlist=[0]
                for nnitem in nn_inst:
                    first_idx=min(nnitem['inst_id'].metadata,inst['inst_id'].metadata)
                    second_idx=max(nnitem['inst_id'].metadata,inst['inst_id'].metadata)
                    TEDlist.append(self.all_TED[first_idx,second_idx])

                ins_fields['teds']=TensorField(torch.tensor(TEDlist))
                ##########
                
                yield Instance(ins_fields)



    def apply_token_indexers(self, instance: Instance) -> None:
        for text_field in instance.fields["enc"].field_list:
            text_field.token_indexers = self._utterance_token_indexers

