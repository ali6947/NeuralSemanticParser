from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, TokenIndexer, Field, Instance
from allennlp.data.fields import TextField, ListField, IndexField, MetadataField
from allennlp.data.fields import (
    TextField,
    ListField,
    IndexField,
    MetadataField,
    ArrayField,
)

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
# from smbop.dataset_readers.spider_basic_pkl_bigbird import SmbopSpiderDatasetReaderBigBird
from smbop.dataset_readers.spider import SmbopSpiderDatasetReader
logger = logging.getLogger(__name__)


@DatasetReader.register("smbop_retriever")
class SmbopSpiderRetrieverDatasetReader(SmbopSpiderDatasetReader):
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
        nn_mask_file: str = None, # used only when val is false
        
    ):
        
        with open(nn_mask_file,'rb') as f:
            self.nn_masks=pickle.load(f)



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

        self.max_instances=max_instances
        print(max_instances)

    @overrides
    def _read(self, file_path: str):
        if file_path.endswith(".pkl"):
            yield from self._read_examples_file(file_path)
        else:
            raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

    def _read_examples_file(self, file_path: str):
        # cache_dir = os.path.join("cache", file_path.split("/")[-1])
        print('Reading NN yay: ',file_path)    
        with open(file_path, "rb") as data_file:
            inst_list = pickle.load(data_file)
            [inst.add_field('inst_id',MetadataField(i)) for i,inst in enumerate(inst_list)]
            # [self.apply_token_indexers(x) for x in inst_list]
            unused=0
            for total_cnt, ex in enumerate(inst_list):
                if total_cnt>self.max_instances:
                    break
                
                all_class_exist=True
                for mask_idx,nn_mask in enumerate(self.nn_masks):
                    sum_bits=np.sum(nn_mask[total_cnt]) # if all are 0 then sum bits is 0 and we have no nn in this class
                    if sum_bits==0 and mask_idx==0:
                        all_class_exist=False
                        break
                if not all_class_exist:
                    unused+=1
                    continue
                yield ex
            print(f'unused instances: {unused}')



    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["enc"].token_indexers = self._utterance_token_indexers

    # def apply_token_indexers(self, instance: Instance) -> None:
    #     for text_field in instance.fields["enc"].field_list:
    #         text_field.token_indexers = self._utterance_token_indexers

