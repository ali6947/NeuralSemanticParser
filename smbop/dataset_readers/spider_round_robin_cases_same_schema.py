import pickle

from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, TokenIndexer, Field, Instance
from allennlp.data.fields import TextField, ListField, IndexField, MetadataField, ArrayField
from typing import Dict
from overrides import overrides

from smbop.dataset_readers import spider
from smbop.dataset_readers.spider import SmbopSpiderDatasetReader
import numpy as np
from itertools import cycle
@DatasetReader.register("cbr_with_same_schema")
class CBRSameSchemaDatasetReader(SmbopSpiderDatasetReader):
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
        is_training=True,
        neighbour_file: str = None,
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
        self.is_training = is_training
        self.neighbours = pickle.load(open(neighbour_file,'rb'))
        print(f'\n**** loaded neighbours from {neighbour_file} ***\n')

    @overrides
    def _read(self, file_path: str):
        if file_path.endswith(".pkl"):
            yield from self._read_examples_file(file_path)
        else:
            raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

    def _read_examples_file(self, file_path: str):
        print('Reading: ',file_path)
        cnt = 0
        cache_buffer = []
        cont_flag = True
        if cont_flag:
            with open(file_path, "rb") as data_file:
                pkl_obj = pickle.load(data_file)
                # [item.add_field('inst_id',MetadataField(i)) for i,item in enumerate(pkl_obj)]
                self.all_instances = pkl_obj
                for total_cnt, ex in enumerate(pkl_obj):
                    if cnt >= self._max_instances:
                        break
                    tokens_with_caseSQL=ex['enc'].tokens.copy()
                    if self.is_training:
                        same_db_ids = np.random.choice(self.neighbours[total_cnt],7) # hardcoding
                        #same_db_ids = np.random.choice(len(self.all_instances), 5)
                    else:
                        same_db_ids = np.random.choice(self.neighbours[total_cnt],7) # hardcoding
                        #same_db_ids = np.random.choice(len(self.all_instances), 10)
                    nbr_instances = [self.all_instances[nbr] for nbr in same_db_ids]
                    pool = cycle(nbr_instances)
                    for inst in pool:
                        if len(tokens_with_caseSQL)>=1024:
                            break
                        for token in inst['enc'].tokens[1:]:
                            tokens_with_caseSQL.append(token)
                            if token.text=='[SEP]':
                                    break
                        gold_sql_tokens=self._tokenizer.tokenize(inst['gold_sql'].metadata+' [SEP] [SEP]') #for bigbird
                        tokens_with_caseSQL.extend(gold_sql_tokens[1:])

                    tokens_with_caseSQL=TextField(tokens_with_caseSQL)
                    ex.add_field('cases',tokens_with_caseSQL) #NOTE: add_field overwrise the first argument ke name wali field, if already present
                    ex.fields["cases"].token_indexers = self._utterance_token_indexers
                    # del ex.fields['inst_id']
                    yield ex
                    cnt+=1
                    

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["enc"].token_indexers = self._utterance_token_indexers





