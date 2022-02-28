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

        tree_sizes_vector = torch.tensor(
            [1] * batch_size, dtype=torch.float32, device=self._device
        ) # total tree nodes in loss
        if self.value_pred:
            span_scores, start_logits, end_logits = self.score_spans(
                embedded_utterance, utterance_mask
            ) # (B x T x T), (B x T), (B x T)
            span_mask = torch.isfinite(span_scores).bool()
            final_span_scores = span_scores.clone() #(B x T x T)
            delta = final_span_scores.shape[-1] - span_hash.shape[-1]
            span_hash = torch.nn.functional.pad(
                span_hash,
                pad=(0, delta, 0, delta),
                mode="constant",
                value=-1,
            )
            if self.training:
                is_gold_span = torch.nn.functional.pad(
                    is_gold_span,
                    pad=(0, delta, 0, delta),
                    mode="constant",
                    value=0,
                )
                batch_idx, start_idx, end_idx = is_gold_span.nonzero().t()
                final_span_scores[
                    batch_idx, start_idx, end_idx
                ] = allennlp.nn.util.max_value_of_dtype(final_span_scores.dtype) # to ensure gold spans in top-k while training

                is_span_end = is_gold_span.sum(-2).float() # B x T
                is_span_start = is_gold_span.sum(-1).float() # B x T

                span_start_probs = allennlp.nn.util.masked_log_softmax(
                    start_logits, utterance_mask.bool(), dim=1
                )
                span_end_probs = allennlp.nn.util.masked_log_softmax(
                    end_logits, utterance_mask.bool(), dim=1
                )

                vector_loss += (-span_start_probs * is_span_start.squeeze()).sum(-1) - (
                    span_end_probs * is_span_end.squeeze()
                ).sum(-1) # B 
                tree_sizes_vector += 2 * is_span_start.squeeze().sum(-1) #Total nodels to finally normalize loss term (2 for both start and end)

            else:
                final_span_scores = span_scores #(B x T x T)
            

            _, leaf_span_mask, best_spans = allennlp.nn.util.masked_topk(
                final_span_scores.view([batch_size, -1]),
                span_mask.view([batch_size, -1]),
                self._num_values,
            ) # _ , B x K, B x K

        leaf_schema_scores = self._rank_schema(embedded_schema) # B x E x 1?
        leaf_schema_scores = leaf_schema_scores / self.temperature # no temperature used for values?
        if is_gold_leaf is not None:
            is_gold_leaf = torch.nn.functional.pad(
                is_gold_leaf,
                pad=(0, leaf_schema_scores.size(-2) - is_gold_leaf.size(-1)),
                mode="constant",
                value=0,
            )

        if self.training:
            final_leaf_schema_scores = leaf_schema_scores.clone() # B x E x 1
            if not self.is_oracle:
                avg_leaf_schema_scores = allennlp.nn.util.masked_log_softmax( # B x E x 1
                    final_leaf_schema_scores,
                    schema_mask.unsqueeze(-1).bool(),
                    dim=1,
                )
                loss_tensor = (
                    -avg_leaf_schema_scores * is_gold_leaf.unsqueeze(-1).float() # B x E x 1
                )
                vector_loss += loss_tensor.squeeze().sum(-1) # B
                tree_sizes_vector += is_gold_leaf.squeeze().sum(-1).float() # B

            final_leaf_schema_scores = final_leaf_schema_scores.masked_fill( # B x E x 1 -- to keep gold schema values in top-k
                is_gold_leaf.bool().unsqueeze(-1),
                allennlp.nn.util.max_value_of_dtype(final_leaf_schema_scores.dtype),
            )
        else:
            final_leaf_schema_scores = leaf_schema_scores

        final_leaf_schema_scores = final_leaf_schema_scores.masked_fill( # B x E x 1
            ~schema_mask.bool().unsqueeze(-1),
            allennlp.nn.util.min_value_of_dtype(final_leaf_schema_scores.dtype),
        )

        min_k = torch.clamp(schema_mask.sum(-1), 0, self._n_schema_leafs) # B x 1
        _, leaf_schema_mask, top_beam_indices = allennlp.nn.util.masked_topk(
            final_leaf_schema_scores.squeeze(-1), mask=schema_mask.bool(), k=min_k
        ) # _, B x min_k.max(), B x min_k.max()

        cls_toks=embedded_utterance[:,0,:] # B x D
        main_cls=cls_toks[boolean_batch_idx==1] # b x D
        case_cls=cls_toks[boolean_batch_idx==0] # bC x D

        b,D=main_cls.shape # bs is b
        main_cls=main_cls.unsqueeze(1) # b x 1 x D
        
        case_cls=case_cls.reshape((b,-1,D)) # b x C x D
        cos_sims=torch.nn.functional.cosine_similarity(main_cls,case_cls,dim=-1) # b x C
        exp_sims=torch.exp(cos_sims)
        normalised_sims=exp_sims/torch.sum(exp_sims,dim=-1,keepdim=True)
        normalised_sims=normalised_sims.squeeze() # b x C

        if self.training:

            exp_teds=torch.exp(-teds)[:,1:] # b x C, remove the main example
            normalised_teds=exp_teds/torch.sum(exp_teds,dim=-1,keepdim=True) # b x C
            normalised_teds=normalised_teds.squeeze() # b x C
            similarity_loss=-(normalised_teds*torch.log(normalised_sims)).sum(dim=-1).mean(dim=-1)

            pre_loss = (vector_loss / tree_sizes_vector).mean()
            loss = pre_loss.squeeze() + similarity_loss.squeeze()
            
            assert not bool(torch.isnan(loss))
            outputs["loss"] = loss
            self._compute_validation_outputs(
                outputs,
                batch_size,
            )
            return outputs
        else:
            end = time.time()
            self._compute_validation_outputs(
                outputs,
                batch_size,
                is_gold_leaf=is_gold_leaf,
                top_beam_indices=top_beam_indices,
                exp_sim_score=normalised_sims,
                inf_time=end - start,
                total_time=end - total_start,
            )
            outputs['exp_sim_score']=normalised_sims
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

