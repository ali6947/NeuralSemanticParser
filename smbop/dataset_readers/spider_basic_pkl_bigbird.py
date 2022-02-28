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
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token_class import Token
import anytree
from anytree.search import *
from collections import defaultdict
from overrides import overrides
from time import time
from typing import Dict
from smbop.utils import moz_sql_parser as msp
import sys
import smbop.utils.node_util as node_util
import smbop.utils.hashing as hashing
import smbop.utils.ra_preproc as ra_preproc #relational algebra preprocessing
from anytree import Node, LevelOrderGroupIter
import dill # not used
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
from smbop.dataset_readers.spider import SmbopSpiderDatasetReader
from smbop.utils.cache import TensorCache
from itertools import cycle
logger = logging.getLogger(__name__)


# @DatasetReader.register("smbop")
class SmbopSpiderDatasetReaderBigBird(SmbopSpiderDatasetReader):
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
    ):
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
        self.cache_directory = cache_directory
        self.cache = TensorCache(cache_directory)
        self.value_pred = value_pred
        self._decoder_timesteps = decoder_timesteps
        self._max_instances = max_instances
        self.limit_instances = limit_instances
        self.load_less = limit_instances!=-1

        self._utterance_token_indexers = question_token_indexers

        self._tokenizer = self._utterance_token_indexers["tokens"]._allennlp_tokenizer

        # self.cls_token = self._tokenizer.tokenize("a")[0] #NOTE: for grappa
        # self.eos_token = self._tokenizer.tokenize("a")[-1] #NOTE: for grappa

        self.cls_token = self._tokenizer.tokenize("[SEP] [SEP]")[0]
        self.eos_token = self._tokenizer.tokenize("[SEP] [SEP]")[-1] #NOTE: another option1 for big bird

        # self.cls_token = Token(text='[CLS]',text_id=65,type_id=0,idx=None,idx_end=None)
        # self.eos_token = Token(text='[SEP]',text_id=66,type_id=0,idx=None,idx_end=None)  #NOTE: for big bird, option 2 resembles idx and idx_end for grappa

        self._keep_if_unparsable = keep_if_unparsable

        self._tables_file = tables_file
        self._dataset_path = dataset_path

        # ratsql
        self.enc_preproc = EncPreproc(
            tables_file,
            dataset_path,
            include_table_name_in_column,
            fix_issue_16_primary_keys,
            qq_max_dist,
            cc_max_dist,
            tt_max_dist,
            use_longdb,
        )
        self._create_action_dicts()
        self.replacer = Replacer(tables_file)

    
    def _read(self, file_path: str):
        if file_path.endswith(".json"):
            yield from self._read_examples_file(file_path)
        elif file_path.endswith(".pkl"):
            print('reading a pkl!')
            import pickle
            with open(file_path,'rb') as data_file:
                all_data=pickle.load(data_file)
                cnt=0                
                for total_cnt,ex in enumerate(all_data):
                    if cnt >= self._max_instances:
                        break
                    else:
                        #################### Basic reader, as if using the json reader, to be used wiht allInstobj type pickles
                        self.apply_token_indexers(ex)
                        ex.add_field('inst_id',MetadataField(total_cnt))
                        yield ex
                        cnt+=1
                        ####################

                        ##################### Here the case is cls, NLques, sep, schema related items, sep, NLques_case1, sep, schema_related_items_case1 ....
                        # tokens_with_cases=ex['enc'].tokens.copy()
                        # for inst,NLques in ex['cases'].metadata[:4]:
                        #     tokens_with_cases.extend(inst['enc'].tokens[1:])
                        # # print(len(tokens_with_cases))
                        # tokens_with_cases=TextField(tokens_with_cases)
                        # ex.add_field('cases',tokens_with_cases) #NOTE: add_field overwrise the first argument ke name wali field, if already present
                        # ex.fields["cases"].token_indexers = self._utterance_token_indexers
                            
                        # yield ex
                        # cnt+=1
                        #####################

                        ##################### Here the case is cls, NLques, sep, schema related items, sep, NLques_case1, sep, SQL1 .... (number of cases limited by max seq len)
                        # tokens_with_caseSQL=ex['enc'].tokens.copy()
                        # for inst,NLques in ex['cases'].metadata:
                        #     if len(tokens_with_caseSQL)>=1024:
                        #         break
                        #     for token in inst['enc'].tokens[1:]:
                        #         tokens_with_caseSQL.append(token)
                        #         if token.text=='[SEP]':
                        #                 break
                        #     gold_sql_tokens=self._tokenizer.tokenize(inst['gold_sql'].metadata+' [SEP] [SEP]')
                        #     tokens_with_caseSQL.extend(gold_sql_tokens[1:])
                            

                        # tokens_with_caseSQL=TextField(tokens_with_caseSQL)
                        # ex.add_field('cases',tokens_with_caseSQL) #NOTE: add_field overwrise the first argument ke name wali field, if already present
                        # ex.fields["cases"].token_indexers = self._utterance_token_indexers

                        # yield ex
                        # cnt+=1
                        #####################

                        ##################### Here the case is cls, NLques, sep, schema related items, sep, NLques_case1, sep, SQL1 .... (keep adding cases until len doesnt reach 1024)
                        # tokens_with_caseSQL=ex['enc'].tokens.copy()
                        # pool = cycle(ex['cases'].metadata)
                        # for inst,NLques in pool:
                        #     if len(tokens_with_caseSQL)>=1024:
                        #         break
                        #     for token in inst['enc'].tokens[1:]:
                        #         tokens_with_caseSQL.append(token)
                        #         if token.text=='[SEP]':
                        #                 break
                        #     gold_sql_tokens=self._tokenizer.tokenize(inst['gold_sql'].metadata+' [SEP] [SEP]')
                        #     tokens_with_caseSQL.extend(gold_sql_tokens[1:])

                        # tokens_with_caseSQL=TextField(tokens_with_caseSQL)
                        # ex.add_field('cases',tokens_with_caseSQL) #NOTE: add_field overwrise the first argument ke name wali field, if already present
                        # ex.fields["cases"].token_indexers = self._utterance_token_indexers
                        # yield ex
                        # cnt+=1
                        #####################


        else:
            raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

    def _read_examples_file(self, file_path: str):
        cache_dir = os.path.join("cache", file_path.split("/")[-1])
        print('Reading: ',file_path)
        cnt = 0
        cache_buffer = []
        cont_flag = True
        sent_set = set()
        # for total_cnt,ins in self.cache:
        #     if cnt >= self._max_instances:
        #         break
        #     if ins is not None:
        #         yield ins
        #         cnt += 1
        #     sent_set.add(total_cnt)
        #     if self.load_less and len(sent_set) > self.limit_instances:
        #         cont_flag = False
        #         break

        if cont_flag:
            with open(file_path, "r") as data_file:
                json_obj = json.load(data_file)
                for total_cnt, ex in enumerate(json_obj):
                    # if total_cnt<1000:
                    #     yield 'tmp'
                    #     continue
                    if cnt >= self._max_instances:
                        break
                    if len(cache_buffer)>50:
                        self.cache.write(cache_buffer)
                        cache_buffer = []
                    if total_cnt in sent_set:
                        continue
                    else:    
                        ins = self.create_instance(ex)
                        cache_buffer.append([total_cnt, ins]) 
                    if ins is not None:
                        yield ins,ex['question'] #NOTE: item after comma extra added  !!!
                        cnt +=1
            self.cache.write(cache_buffer)



    def text_to_instance(
        self, utterance: str, db_id: str, sql=None, sql_with_values=None
    ):
        fields: Dict[str, Field] = {
            "db_id": MetadataField(db_id),
        }

        tokenized_utterance = self._tokenizer.tokenize(utterance+' [SEP] [SEP]')


        has_gold = sql is not None

        if has_gold:
            try:
                tree_dict = msp.parse(sql)
                tree_dict_values = msp.parse(sql_with_values)
            except msp.ParseException as e:
                print(f"could'nt create AST for:  {sql}")
                return None
            tree_obj = ra_preproc.ast_to_ra(tree_dict["query"])
            tree_obj_values = ra_preproc.ast_to_ra(tree_dict_values["query"])

            arit_list = anytree.search.findall(
                tree_obj, filter_=lambda x: x.name in ["sub", "add"]
            )  # TODO: fixme
            haslist_list = anytree.search.findall(
                tree_obj,
                filter_=lambda x: hasattr(x, "val") and isinstance(x.val, list),
            )
            if arit_list or haslist_list:
                print(f"could'nt create RA for:  {sql}")
                
                return None
            if self.value_pred:
                for a, b in zip(tree_obj_values.leaves, tree_obj.leaves):
                    if b.name == "Table" or ("." in str(b.val)):
                        continue
                    b.val = a.val
                    if (
                        isinstance(a.val, int) or isinstance(a.val, float)
                    ) and b.parent.name == "literal":
                        parent_node = b.parent
                        parent_node.children = []
                        parent_node.name = "Value"
                        parent_node.val = b.val

            for leaf in tree_obj.leaves:
                leaf.val = self.replacer.pre(leaf.val, db_id)
                if not self.value_pred and node_util.is_number(leaf.val):
                    leaf.val = "value"

            leafs = list(set(node_util.get_leafs(tree_obj)))
            hash_gold_levelorder, hash_gold_tree = self._init_fields(tree_obj)

            fields.update(
                {
                    "hash_gold_levelorder": ArrayField(
                        hash_gold_levelorder, padding_value=-1, dtype=np.int64
                    ),
                    "hash_gold_tree": ArrayField(
                        np.array(hash_gold_tree), padding_value=-1, dtype=np.int64
                    ),
                    "gold_sql": MetadataField(sql_with_values),
                    "tree_obj": MetadataField(tree_obj),
                }
            )
        
        desc = self.enc_preproc.get_desc(tokenized_utterance, db_id) # output of preprocess_item in enc_preproc.py
        
        entities, added_values, relation = self.extract_relation(desc)
        # print(entities)

        question_concated = [[x] for x in tokenized_utterance[1:-1]]
        # question_concated = [[x] for x in tokenized_utterance[1:]] #NOTE: changed because bigbird doesnt add any token at the end, use above line for grappa
        schema_tokens_pre, schema_tokens_pre_mask = table_text_encoding(
            entities[len(added_values) + 1 :]
        )

        schema_size = len(entities)
        schema_tokens_pre = added_values + ["*"] + schema_tokens_pre

        schema_tokens = [
            [y for y in x if y.text not in ["_"]]
            # for x in [self._tokenizer.tokenize(x)[1:-1] for x in schema_tokens_pre]
            for x in [self._tokenizer.tokenize(x+' [SEP] [SEP]')[1:-1] for x in schema_tokens_pre] #NOTE:  changed because bigbird doesnt add any token at the end(this was fixed by 2 sep token, actually it adds but Token type id doesnt allow take last 2 tokens to be returned), use above line for grappa
            # for x in [self._tokenizer.tokenize(x)[1:] for x in schema_tokens_pre] #NOTE: changed because bigbird doesnt add any token at the end(this was fixed by 2 sep token, actually it adds but Token type id doesnt allow take last 2 tokens to be returned), this is also for bigbird when we say use a single sep
        ]

        entities_as_leafs = [x.split(":")[0] for x in entities[len(added_values) + 1 :]]
        entities_as_leafs = added_values + ["*"] + entities_as_leafs
        orig_entities = [self.replacer.post(x, db_id) for x in entities_as_leafs]
        entities_as_leafs_hash, entities_as_leafs_types = self.hash_schema(
            entities_as_leafs, added_values
        )

        fields.update(
            {
                "relation": ArrayField(relation, padding_value=-1, dtype=np.int32),
                "entities": MetadataField(entities_as_leafs),
                 "orig_entities": MetadataField(orig_entities),
                 "leaf_hash": ArrayField(
                    entities_as_leafs_hash, padding_value=-1, dtype=np.int64
                ),
                "leaf_types": ArrayField(
                    entities_as_leafs_types,
                    padding_value=self._type_dict["nan"],
                    dtype=np.int32,
                )
            })

        if has_gold:
            leaf_indices, is_gold_leaf, depth = self.is_gold_leafs(
                tree_obj, leafs, schema_size, entities_as_leafs
            )
            fields.update(
                {
                    "is_gold_leaf": ArrayField(
                        is_gold_leaf, padding_value=0, dtype=np.int32
                    ),
                    "leaf_indices": ArrayField(
                        leaf_indices, padding_value=-1, dtype=np.int32
                    ),
                    "depth": ArrayField(depth, padding_value=0, dtype=np.int32),
                }
            )

        utt_len = len(tokenized_utterance[1:-1])
        # utt_len = len(tokenized_utterance[1:]) #NOTE: changed because bigbird doesnt add any token at the end, use above line for grappa. Now above line works with big bird also because we add sep
        if self.value_pred:
            span_hash_array = self.hash_spans(tokenized_utterance)
            fields["span_hash"] = ArrayField(
                span_hash_array, padding_value=-1, dtype=np.int64
            )

        if has_gold and self.value_pred:
            value_list = np.array(
                [self.hash_text(x) for x in node_util.get_literals(tree_obj)],
                dtype=np.int64,
            )
            is_gold_span = np.isin(span_hash_array.reshape([-1]), value_list).reshape(
                [utt_len, utt_len]
            )
            fields["is_gold_span"] = ArrayField(
                is_gold_span, padding_value=False, dtype=np.bool
            )

        enc_field_list = []
        offsets = []
        mask_list = (
            [False]
            + ([True] * len(question_concated))
            + [False]
            + ([True] * len(added_values))
            + [True]
            + schema_tokens_pre_mask
            + [False]
        )
        for mask, x in zip(
            mask_list,
            [[self.cls_token]]
            + question_concated
            + [[self.eos_token]]
            + schema_tokens
            + [[self.eos_token]],
        ):
            start_offset = len(enc_field_list)
            enc_field_list.extend(x)
            if mask:
                offsets.append([start_offset, len(enc_field_list) - 1])

        fields["lengths"] = ArrayField(
            np.array(
                [
                    [0, len(question_concated) - 1],
                    [len(question_concated), len(question_concated) + schema_size - 1],
                ]
            ),
            dtype=np.int32,
        )
        fields["offsets"] = ArrayField(
            np.array(offsets), padding_value=0, dtype=np.int32
        )
        fields["enc"] = TextField(enc_field_list)
        # print('######')
        # print(fields['enc'])
        # print('######')

        ins = Instance(fields)
        return ins

    

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["enc"].token_indexers = self._utterance_token_indexers


