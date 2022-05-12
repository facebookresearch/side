#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BiEncoder component + loss function for 'all-in-batch' training
"""
import collections
import logging
from typing import Tuple

import torch
from torch import Tensor as T
from torch import nn

from dpr.models.ranker_loss import PassageLevelRankerNllLoss, RankerLoss, DocumentLevelRankerNllLoss
from dpr.utils.model_utils import CheckpointState

logger = logging.getLogger(__name__)


QueryContextsBatch = collections.namedtuple(
    "QueryContextsBatch",
    [
        "question_ids",
        "question_segments",
        "context_ids",
        "ctx_segments",
        "is_positive",
        "hard_negatives",
        "encoder_type",
    ],
)


class BiEncoder(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        cfg,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
    ):
        super(BiEncoder, self).__init__()
        self.cfg = cfg
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder
        logger.info("!!! fix_ctx_encoder=%s", fix_ctx_encoder)

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        fix_encoder: bool = False,
        representation_token_pos=0,
    ) -> (T, T, T):
        sequence_output = None
        pooled_output = None
        hidden_states = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, pooled_output, hidden_states = sub_model(
                        ids,
                        segments,
                        attn_mask,
                        representation_token_pos=representation_token_pos,
                    )

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, pooled_output, hidden_states = sub_model(
                    ids,
                    segments,
                    attn_mask,
                    representation_token_pos=representation_token_pos,
                )

        return sequence_output, pooled_output, hidden_states

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        representation_token_pos=0,
    ) -> Tuple[T, T]:

        q_encoder = self.question_model if encoder_type is None or encoder_type == "question" else self.ctx_model
        _q_seq, q_pooled_out, _q_hidden = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
        )

        ctx_encoder = self.ctx_model if encoder_type is None or encoder_type == "ctx" else self.question_model
        _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
            ctx_encoder, context_ids, ctx_segments, ctx_attn_mask, self.fix_ctx_encoder
        )

        return q_pooled_out, ctx_pooled_out

    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        self.load_state_dict(saved_state.model_dict, strict=strict)

    def get_state_dict(self):
        return self.state_dict()

    def get_loss_function(self) -> RankerLoss:
        return PassageLevelRankerNllLoss(self.cfg)

    def prepare_model_inputs(self, batch: dict, dataset):
        return {
            "question_ids": batch["question_ids"],
            "question_segments": batch["question_segments"],
            "question_attn_mask": dataset.tensorizer.get_attn_mask(batch["question_ids"]),
            "context_ids": batch["context_ids"],
            "ctx_segments": batch["ctx_segments"],
            "ctx_attn_mask": dataset.tensorizer.get_attn_mask(batch["context_ids"]),
        }


class DocumentBiEncoder(BiEncoder):
    def __init__(
        self,
        cfg,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
    ):
        super().__init__(
            cfg=cfg,
            question_model=question_model,
            ctx_model=ctx_model,
            fix_q_encoder=fix_q_encoder,
            fix_ctx_encoder=fix_ctx_encoder,
        )

    def get_loss_function(self) -> RankerLoss:
        return DocumentLevelRankerNllLoss(self.cfg)
