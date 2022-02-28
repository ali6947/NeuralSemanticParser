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
from smbop.utils.cache import TensorCache

from smbop.dataset_readers.spider import SmbopSpiderDatasetReader, table_text_encoding
import pickle
from tqdm import tqdm

logger = logging.getLogger(__name__)


@DatasetReader.register("kaggle_dbqa")
class KaggleDBQAReader(SmbopSpiderDatasetReader):
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

        self.cls_token = self._tokenizer.tokenize("[SEP] [SEP]")[0]
        self.eos_token = self._tokenizer.tokenize("[SEP] [SEP]")[-1]

        self._keep_if_unparsable = keep_if_unparsable

        self._tables_file = tables_file
        self._dataset_path = dataset_path

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

        self.all_keywords=["select",
    "from",
    "where",
    "group",'by',
    "order",
    "limit",
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists","none", "-", "+", "*", "/","none", "max", "min", "count", "sum", "avg","and", "or","intersect", "union", "except","desc", "asc",
    "distinct",'ilike','having','as','join','on']


    def process_instance(self, instance: Instance, index: int):
        return instance

    @overrides
    def _read(self, file_path: str):
        if file_path.endswith(".pkl"):
            yield from self._read_examples_file(file_path)
        else:
            raise ConfigurationError(f"Filetype of {file_path} must be of type .pkl")

    def _read_examples_file(self, file_path: str):
        # cache_dir = os.path.join("cache", file_path.split("/")[-1])
        print('Reading: ',file_path)
        cnt = 0
        with open(file_path, "rb") as data_file:
            pkl_obj = pickle.load(data_file)
            for total_cnt, ex in enumerate(pkl_obj):
                if cnt >= self._max_instances:
                    break
                yield ex
                cnt +=1

    def process_and_dump_pickle(self, input_file_path, output_file_path):
        if not input_file_path.endswith(".json"):
            raise ValueError(f"Don't know how to process filetype of {input_file_path}")
        if not output_file_path.endswith(".pkl"):
            raise ValueError(f"Output file {output_file_path} must be of type .pkl")
        instances = []
        print('Reading: ',input_file_path)
        with open(input_file_path, "r") as data_file:
            json_obj = json.load(data_file)
            for total_cnt, ex in tqdm(enumerate(json_obj)):   
                ins = self.create_instance(ex)
                if ins is None:
                    print(f"Unable to process instance with id {total_cnt}")
                else:
                    # print(anytree.RenderTree(ins['tree_obj'].metadata))
                    # print('*****')
                    instances.append(ins)
        print('Reading and processing complete.')
        print(f'Dumping instances into {output_file_path}')
        disamb_sql.reset_cache()
        with open(output_file_path,"wb") as output_file:
            pickle.dump(instances,output_file)

    def isfloat(self,value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def split_non_alpha_numeric(self,s): #treating '_' as alpha numeric
        spli=[]
        st=0
        cw=''
        while st<len(s):
            if s[st].isalnum() or s[st]=='_':
                cw+=s[st]
                st+=1
            else:
                if cw!='':
                    spli.append(cw)
                spli.append(s[st])
                cw=''
                st+=1
        if cw!='':
            spli.append(cw)
        return spli

    def break_query_into_spans(self,quer):
        spans=[]
        cw=''
        st=0
        inside_literal=False
        lit_char=None
        while st<len(quer):
            if quer[st]==' ':
                if inside_literal:
                    cw+=quer[st]
                else:
                    if cw!='':
                        spans.append(cw)
                    cw=''
            elif inside_literal and lit_char==quer[st] and (st==0 or (st!=0 and quer[st-1]!='\\')): # that is we encounter end of literal span and the enclosing quotes are not escaped
                inside_literal=False
                lit_char=None
                cw+=quer[st]
                spans.append(cw)
                cw=''
            elif not inside_literal and quer[st] in ['\'','"']: #literal begins
                inside_literal=True
                lit_char=quer[st]
                if cw!='':
                    print(quer)
                    assert False
                else:
                    cw+=quer[st]
                    pass
            else:
                cw+=quer[st]
            st+=1
        if cw!='':
            spans.append(cw)
        return spans


    def create_instance(self,ex):
        sql = None
        sql_with_values = None

        if "query_toks" in ex:
            #ex = disamb_sql.fix_number_value(ex) #standardizes ex["query_toks.*"]
            pre_query_as_list=self.break_query_into_spans(ex['query'])
            query_as_list=[]
            for qtok in pre_query_as_list:
                if self.isfloat(qtok):
                    query_as_list.append(qtok)
                elif qtok[0] in ['\'','"'] and qtok[-1] in ['\'','"']:
                    query_as_list.append(qtok)
                elif qtok.lower() in self.all_keywords:
                    query_as_list.append(qtok.lower())
                else:
                    spli=self.split_non_alpha_numeric(qtok)
                    spli=[x.lower() for x in spli]
                    query_as_list.extend(spli)
            sql = disamb_sql.disambiguate_items(
                ex["db_id"],
                query_as_list,
                self._tables_file,
                allow_aliases=False,
            ) # modified query tokens (applied on sql without values)
            
            sql = disamb_sql.sanitize(sql)  # the tree_obj eventually has all names in lower case so we did lower casing here only 
            sql_with_values = disamb_sql.sanitize(ex["query"]) #this can have upper case also does not matter since it aint used and we dont copy col/table names from it into tree_obj
        # print(f'sql: {sql}')
        # print(f'sql_with_values: {sql_with_values}')
        ins = self.text_to_instance(
            utterance=ex["question"],
            db_id=ex["db_id"],
            sql=sql,
            sql_with_values=sql_with_values,
        )

        return ins

    def text_to_instance(
        self, utterance: str, db_id: str, sql=None, sql_with_values=None
    ):
        fields: Dict[str, Field] = {
            "db_id": MetadataField(db_id),
        }

        tokenized_utterance = self._tokenizer.tokenize(utterance+' [SEP] [SEP]')
        # print(tokenized_utterance)
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
                leaf.val = self.replacer.pre(leaf.val.lower() if isinstance(leaf.val,str) else leaf.val, db_id)
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


        question_concated = [[x] for x in tokenized_utterance[1:-1]]
        schema_tokens_pre, schema_tokens_pre_mask = table_text_encoding(
            entities[len(added_values) + 1 :] # why +1 ?
        )

        schema_size = len(entities)
        schema_tokens_pre = added_values + ["*"] + schema_tokens_pre

        schema_tokens = [
            [y for y in x if y.text not in ["_"]]
            for x in [self._tokenizer.tokenize(x+' [SEP] [SEP]')[1:-1] for x in schema_tokens_pre]
        ]
        # print(schema_tokens)

        entities_as_leafs = [x.split(":")[0] for x in entities[len(added_values) + 1 :]]
        entities_as_leafs = added_values + ["*"] + entities_as_leafs
        orig_entities = [self.replacer.post(x.lower() if isinstance(x, str) else x, db_id) for x in entities_as_leafs]
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
        # print([[self.cls_token]]
        #     , question_concated
        #     , [[self.eos_token]]
        #     , schema_tokens
        #     , [[self.eos_token]])
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
        # print(enc_field_list)

        ins = Instance(fields)
        return ins