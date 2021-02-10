# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from .file_utils import cached_path

from pytorch_pretrained_bert.modeling import BertModel, PreTrainedBertModel

from allennlp.nn import util

logger = logging.getLogger(__name__)

class BertForChineseZero(PreTrainedBertModel):
    """BERT model for Chinese Zero Anaphora Resolution.
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: either
            - a BertConfig class instance with the configuration to build a new model, or
            - a str with the name of a pre-trained model to load selected in the list of:
                . `bert-base-uncased`
                . `bert-large-uncased`
                . `bert-base-cased`
                . `bert-base-multilingual`
                . `bert-base-chinese`
                The pre-trained model will be downloaded and cached if needed.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = BertForChineseZero(config)
    g_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForChineseZero, self).__init__(config)
        self.bert = BertModel(config)

        self.W_z = nn.Linear(config.hidden_size * 2, config.hidden_size)        
        self.W_c = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.v_a = nn.Linear(config.hidden_size, 2, bias=False)
        
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, zps, candidates_labels_set, is_training=True):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batchsize = sequence_output.size(0)
        hidden_size = sequence_output.size(-1)
        
        n_zp = zps.size(1)
        zps_starts, zps_ends = [index.squeeze(-1) for index in zps.split(1, dim=-1)]
        
        zps_starts = F.relu(zps_starts.float()).long()
        zps_ends = F.relu(zps_ends.float()).long()        
        
        # [batchsize, n_zp, hidden_size]
        zp_starts_embeddings = util.batched_index_select(sequence_output, zps_starts)
        zp_ends_embeddings = util.batched_index_select(sequence_output, zps_ends)        

        candidates_starts, candidates_ends, labels = [index.squeeze(-1) for index in candidates_labels_set.split(1, dim=-1)]
        # [batchsize, n_zp, n_candidates]
        candidates_starts = F.relu(candidates_starts.float()).long()
        # [batchsize, n_zp * n_candidates]
        candidates_starts = candidates_starts.view(batchsize, -1)

        candidates_ends = F.relu(candidates_ends.float()).long()
        candidates_ends = candidates_ends.view(batchsize, -1)
        
        # [batchsize, n_zp * n_candidates, hidden_size]        
        candidates_starts_embeddings = util.batched_index_select(sequence_output, candidates_starts)
        # [batchsize, n_zp, n_candidates, hidden_size]        
        candidates_starts_embeddings = candidates_starts_embeddings.view(batchsize, n_zp, -1, hidden_size)

        # [batchsize, n_zp * n_candidates, hidden_size]                
        candidates_ends_embeddings = util.batched_index_select(sequence_output, candidates_ends)
        # [batchsize, n_zp, n_candidates, hidden_size]                
        candidates_ends_embeddings = candidates_ends_embeddings.view(batchsize, n_zp, -1, hidden_size)        

        # [batchsize, n_zp, n_candidates, hidden_size]
        expand_zp_starts_embeddings = zp_starts_embeddings.unsqueeze(2).expand_as(candidates_starts_embeddings)
        expand_zp_ends_embeddings = zp_ends_embeddings.unsqueeze(2).expand_as(candidates_ends_embeddings)
        
        # feed_foward
        h_z = self.W_z(torch.cat([expand_zp_starts_embeddings, expand_zp_ends_embeddings], -1))
        h_c = self.W_c(torch.cat([candidates_starts_embeddings, candidates_ends_embeddings], -1))
        # [batchsize, n_zp, n_candidates, 2]
        logits = self.v_a(torch.tanh(h_z + h_c))
        
        if is_training is True:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss
        else:
            ret_dict = {}
            ret_dict["antecedent_labels_set"] = F.softmax(logits, dim=-1)

            return ret_dict
