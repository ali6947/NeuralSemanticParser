from collections import deque
import logging
from multiprocessing.process import BaseProcess
import random
import traceback
from typing import List, Iterator, Optional, Iterable, Union, TypeVar

from overrides import overrides
import torch
import torch.multiprocessing as mp

from allennlp.common.util import lazy_groups_of, shuffle_iterable
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict, allennlp_collate
from allennlp.data.dataset_readers import DatasetReader, WorkerInfo, DatasetReaderInput
from allennlp.data.fields import TextField
from allennlp.data.samplers import BatchSampler
from allennlp.data.vocabulary import Vocabulary
import allennlp.nn.util as nn_util
from allennlp.data.data_loaders import MultiProcessDataLoader

import pickle
import numpy as np
from allennlp.data import  Instance
from allennlp.data.fields import  ListField, MetadataField, ArrayField
from itertools import cycle
from copy import deepcopy

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

@DataLoader.register("cbr_with_same_schema")
class CBRSameSchemaDataLoader(MultiProcessDataLoader):
    def __init__(
        self,
        reader: DatasetReader,
        data_path: DatasetReaderInput,
        *,
        batch_size: int = None,
        drop_last: bool = False,
        shuffle: bool = False,
        batch_sampler: BatchSampler = None,
        batches_per_epoch: int = None,
        num_workers: int = 0,
        max_instances_in_memory: int = None,
        start_method: str = "fork",
        cuda_device: Optional[Union[int, str, torch.device]] = None,
        quiet: bool = False,
    ) -> None:
        super().__init__(
            reader,
            data_path,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            batch_sampler=batch_sampler,
            batches_per_epoch=batches_per_epoch,
            num_workers=num_workers,
            max_instances_in_memory=max_instances_in_memory,
            start_method=start_method,
            cuda_device=cuda_device,
            quiet=quiet)



        ########################### added by ali/aa
        self.same_db_nbrs = self.reader.neighbours
        self.is_training = self.reader.is_training
        self.all_instances=self.reader.all_instances
        self.tokenised_gold_SQL=[self.reader._tokenizer.tokenize(inst['gold_sql'].metadata+' [SEP] [SEP]') for inst in self.all_instances]
        [x.add_field('inst_id',MetadataField(idx)) for idx,x in enumerate(self.all_instances)]
        ########################### added by ali/aa

    @overrides
    def index_with(self, vocab: Vocabulary) -> None:
        self._vocab = vocab
        if self._instances:
            for instance in self._maybe_tqdm(self._instances):
                instance.index_fields(vocab)

    def _instances_to_batches(
        self, instance_iterator: Iterable[Instance], move_to_device
    ) -> Iterator[TensorDict]:
        instance_iterator = (self._index_instance(instance) for instance in instance_iterator)
        if move_to_device and self.cuda_device is not None:
            tensorize = lambda batch: nn_util.move_to_device(  # noqa: E731
                self.collate_fn(batch), self.cuda_device
            )
        else:
            tensorize = self.collate_fn

        if self.batch_sampler is not None:
            instance_chunks: Iterable[List[Instance]]

            if self.max_instances_in_memory is not None:
                instance_chunks = lazy_groups_of(instance_iterator, self.max_instances_in_memory)
            else:
                instance_chunks = [list(instance_iterator)]

            for instances in instance_chunks:
                batches = (
                    [instances[i] for i in batch_indices]
                    for batch_indices in self.batch_sampler.get_batch_indices(instances)
                )
                ########################### added by ali/aa
                for batch in batches:
                    new_batch = self._modify_batch(batch)
                    yield tensorize(new_batch)
                ###########################
                    
        else:
            # Safe to assume this is not `None` when `self.batch_sampler` is `None`.
            assert self.batch_size is not None

            if self.shuffle:
                if self.max_instances_in_memory is not None:
                    instance_iterator = shuffle_iterable(
                        instance_iterator,
                        self.max_instances_in_memory,
                    )
                else:
                    # At this point we've already loaded the instances in memory and indexed them,
                    # so this won't take long.
                    instance_iterator = list(instance_iterator)
                    random.shuffle(instance_iterator)

            for batch in lazy_groups_of(instance_iterator, self.batch_size):
                if self.drop_last and len(batch) < self.batch_size:
                    break
                new_batch = self._modify_batch(batch)
                yield tensorize(new_batch)

    def _modify_batch(self, batch):
        new_batch=[]
        for ex in batch:
            iid=ex['inst_id'].metadata
            if len(self.same_db_nbrs[iid]) == 0:
                print('\n=============\n')
                print(f'WARNING: Instance {iid} has no same db nbr')
                print('\n=============\n')
                continue
            if self.is_training:
                if len(self.same_db_nbrs[iid])>8:
                    same_db_ids = np.random.choice(self.same_db_nbrs[iid],8,replace=False) # hardcoding    
                else:
                    same_db_ids = np.random.choice(self.same_db_nbrs[iid],8) # hardcoding
                #same_db_ids = np.random.choice(len(self.all_instances), 5)
            else:
                if len(self.same_db_nbrs[iid])>8:
                    same_db_ids = np.random.choice(self.same_db_nbrs[iid],8,replace=False) # hardcoding    
                else:
                    same_db_ids = np.random.choice(self.same_db_nbrs[iid],8) # hardcoding
                #same_db_ids = np.random.choice(len(self.all_instances), 10)
                
            
            # nbr_instances = [self._index_instance(self.all_instances[nbr]) for nbr in same_db_ids]
            nbr_instances = [self.all_instances[nbr] for nbr in same_db_ids]
            nbr_SQL=[self.tokenised_gold_SQL[nbr] for nbr in same_db_ids]
            # tokens_with_caseSQL=ex['enc'].tokens.copy()
            # pool = cycle(nbr_instances)
            # for inst in pool:
            #     if len(tokens_with_caseSQL)>=1024:
            #         break
            #     for token in inst['enc'].tokens[1:]:
            #         tokens_with_caseSQL.append(token)
            #         if token.text=='[SEP]':
            #             break
            #     gold_sql_tokens=self.reader._tokenizer.tokenize(inst['gold_sql'].metadata+' [SEP] [SEP]') #for bigbird
            #     tokens_with_caseSQL.extend(gold_sql_tokens[1:])

            # tokens_with_caseSQL=TextField(tokens_with_caseSQL)
            # ex.add_field('cases',tokens_with_caseSQL) #NOTE: add_field overwrise the first argument ke name wali field, if already present
            # ex.fields["cases"].token_indexers = self.reader._utterance_token_indexers

            tokens_with_caseSQL=ex['enc'].tokens.copy()
            pool = cycle(zip(nbr_instances,nbr_SQL))
            for inst,tokSQL in pool:
                if len(tokens_with_caseSQL)>=1024:
                    break
                for token in inst['enc'].tokens[1:]:
                    tokens_with_caseSQL.append(token)
                    if token.text=='[SEP]':
                        break
                # gold_sql_tokens=self.reader._tokenizer.tokenize(inst['gold_sql'].metadata+' [SEP] [SEP]') #for bigbird
                tokens_with_caseSQL.extend(tokSQL[1:])

            tokens_with_caseSQL=TextField(tokens_with_caseSQL)
            ex.add_field('cases',tokens_with_caseSQL) #NOTE: add_field overwrise the first argument ke name wali field, if already present
            ex.fields["cases"].token_indexers = self.reader._utterance_token_indexers
            

            new_inst=ex
            # del new_inst.fields['inst_id']
            new_inst['cases'].index(self._vocab)
            new_batch.append(new_inst)
        return new_batch
