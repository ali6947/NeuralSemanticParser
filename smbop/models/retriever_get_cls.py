import itertools
import json
import logging
import os
import time
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import Dict

import allennlp
import torch
from allennlp.common.util import *
from allennlp.data import TokenIndexer, Vocabulary
from allennlp.models import Model
from allennlp.modules import (
    Seq2SeqEncoder,
    TextFieldEmbedder,
)

from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util
from allennlp.nn.util import masked_mean
from allennlp.training.metrics import Average
from anytree import PostOrderIter
from overrides import overrides

import smbop.utils.node_util as node_util
from smbop.eval_final.evaluation import evaluate_single #returns exact match
from smbop.utils import ra_postproc
from smbop.utils import vec_utils
from smbop.utils import hashing

from smbop.models.smbop import SmbopParser, get_failed_set

logger = logging.getLogger(__name__)


@Model.register("smbop_sim_pretrained")
class SmbopSimPretrained(SmbopParser):
    '''
    All the init arguments are probably loaded from the json config file
    '''
    def __init__(
        self,
        experiment_name: str,
        vocab: Vocabulary,
        question_embedder: TextFieldEmbedder, #grappa etc. (type: pytorch_transformer)
        schema_encoder: Seq2SeqEncoder, # (type: relation transformer)
        beam_encoder: Seq2SeqEncoder, # (type: pytorch_transformer)
        tree_rep_transformer: Seq2SeqEncoder, # (type: pytorch_transformer)
        utterance_augmenter: Seq2SeqEncoder, # (type: cross_attention)
        beam_summarizer: Seq2SeqEncoder, # (type: pytorch_transformer)
        decoder_timesteps=9,
        beam_size=30,
        misc_params=None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(experiment_name,
        vocab,
        question_embedder, #grappa etc. (type: pytorch_transformer)
        schema_encoder, # (type: relation transformer)
        beam_encoder, # (type: pytorch_transformer)
        tree_rep_transformer, # (type: pytorch_transformer)
        utterance_augmenter, # (type: cross_attention)
        beam_summarizer, # (type: pytorch_transformer)
        decoder_timesteps,
        beam_size,
        misc_params,
        dropout)

        # self._pre_sim_layer = torch.nn.Sequential(
        #             torch.nn.Linear(self.d_frontier, self.d_frontier),
        #             torch.nn.Dropout(p=dropout),
        #             torch.nn.LayerNorm(self.d_frontier),
        #             self.activation_func(),
        #             torch.nn.Linear(self.d_frontier, self.d_frontier),
        #         )
        self._action_dim = beam_encoder.get_output_dim()
        self._nn_acc = Average()
        self.sim_space=128
        self.activation_func = torch.nn.ReLU

        self.sim_projection=torch.nn.Sequential(
                    torch.nn.Linear(2*self._action_dim, self.sim_space),
                    torch.nn.Dropout(p=dropout),
                    torch.nn.LayerNorm(self.sim_space),
                    self.activation_func(),
                )

        self.kl_loss=torch.nn.KLDivLoss(reduction='batchmean')

    def _flatten_cases_tensor(self,tensor):
        original_shape = list(tensor.shape)
        if len(original_shape) > 2:
            new_shape = [-1] + original_shape[2:]
        elif len(original_shape) == 2:
            new_shape = [-1]
        else:
            raise ValueError("tensor should have atleast two dimensions")
        new_tensor = tensor.reshape(new_shape)
        return new_tensor

    def _flatten_cases_list(self,ex_list):
        flattened_list = [item for sublist in ex_list for item in sublist]
        return flattened_list

    def forward(
        self,
        enc,
        db_id,
        is_gold_leaf=None,
        lengths=None,
        offsets=None,
        relation=None,
        span_hash=None,
        is_gold_span=None,
        inst_id=None,
        teds=None,
    ):
        case_size = is_gold_span.shape[1]
        
        for key in enc["tokens"]:
            enc["tokens"][key] = self._flatten_cases_tensor(enc["tokens"][key])

        db_id = self._flatten_cases_list(db_id)
        is_gold_leaf = self._flatten_cases_tensor(is_gold_leaf)
        lengths = self._flatten_cases_tensor(lengths)
        offsets = self._flatten_cases_tensor(offsets)
        relation = self._flatten_cases_tensor(relation)
        span_hash = self._flatten_cases_tensor(span_hash)
        is_gold_span = self._flatten_cases_tensor(is_gold_span)

        batch_size = len(db_id)
        actual_batch_size = batch_size // case_size
        actual_batch_idx = torch.arange(actual_batch_size) * case_size
        boolean_batch_idx = torch.zeros(batch_size)
        boolean_batch_idx[actual_batch_idx]=1.0 
        list_actual_batch_idx = list(actual_batch_idx.numpy())
        actual_enc = {}
        actual_enc["tokens"] = {}
        for key in enc["tokens"]:
            actual_enc["tokens"][key] = enc["tokens"][key][actual_batch_idx]

        total_start = time.time()
        outputs = {}
        beam_list = []
        item_list = []
        self._device = enc["tokens"]["token_ids"].device
        self.move_to_gpu(self._device)
        batch_size = len(db_id)
        self.hasher = hashing.Hasher(self._device)
        (
            embedded_schema, # B x E x D ?
            schema_mask,
            embedded_utterance,
            utterance_mask,
        ) = self._encode_utt_schema(enc, offsets, relation, lengths)


        batch_size, utterance_length, _ = embedded_utterance.shape # B x T x D
        start = time.time()
        loss = torch.tensor([0], dtype=torch.float32, device=self._device)
        pre_loss = torch.tensor([0], dtype=torch.float32, device=self._device)
        vector_loss = torch.tensor(
            [0] * batch_size, dtype=torch.float32, device=self._device
        )
        # tree_sizes_vector = torch.tensor(
        #     [0] * batch_size, dtype=torch.float32, device=self._device
        # )

        cls_toks=embedded_utterance[:,0,:] # B x D
        main_cls=cls_toks[boolean_batch_idx==1]
        outputs['cls_toks']=main_cls.to('cpu')
        return outputs

    def _compute_validation_outputs(
        self,
        outputs,
        batch_size,
        **kwargs,
    ):
        batch_size = batch_size
        leaf_acc_list = []
        nn_acc_list=[]
        if not self.training:

            if (
                kwargs["is_gold_leaf"] is not None
                and kwargs["top_beam_indices"] is not None
            ):
                for top_beam_indices_el, is_gold_leaf_el in zip(
                    kwargs["top_beam_indices"], kwargs["is_gold_leaf"]
                ):
                    is_gold_leaf_idx = is_gold_leaf_el.nonzero().squeeze().tolist()
                    leaf_acc = int(
                        all([x in top_beam_indices_el for x in is_gold_leaf_idx])
                    )
                    leaf_acc_list.append(leaf_acc)
                    self._leafs_acc(leaf_acc)

            ############# added by ali
            if kwargs['exp_sim_score'] is not None:
                closest_nn=torch.argmax(kwargs['exp_sim_score'],axis=-1)
                if len(closest_nn.shape)==0:
                    closest_nn=closest_nn.unsqueeze(0)
                    
                for nni in closest_nn:
                   acc_val=int(nni==0)
                   nn_acc_list.append(acc_val)
                   self._nn_acc(acc_val)
                
            #############
            # TODO: change this!! this causes bugs!
            for b in range(batch_size):
                outputs["inf_time"] = [kwargs["inf_time"]]+([None]*(batch_size-1))
                outputs["total_time"] = [kwargs["total_time"]] + \
                    ([None]*(batch_size-1))

        outputs["leaf_acc"] = leaf_acc_list or ([None]*batch_size)
        outputs['nn_acc'] = nn_acc_list 

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        out = {}
        if not self.training:
            lacc = self._leafs_acc.get_metric(reset)
            nnacc =self._nn_acc.get_metric(reset)
            out["leafs_acc"]=lacc
            out['nns_acc']=nnacc
            out['combined_acc']=(lacc+nnacc)/2
            # out['self._spider_acc._count'] = self._spider_acc._count
        return out

