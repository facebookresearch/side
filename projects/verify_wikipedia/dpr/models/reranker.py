#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BiEncoder component + loss function for 'all-in-batch' training
"""

import logging
from typing import Tuple

import torch
from torch import Tensor as T
from torch import nn

from dpr.models.biencoder import RankerLoss, DocumentBiEncoder
from dpr.models.ranker_loss import RerankerNllLoss

logger = logging.getLogger(__name__)


class ColbertRerankerBiEncoder(DocumentBiEncoder):

    def __init__(self,
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

    def get_interactions(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        representation_token_pos = 0,
    ):
        q_encoder = self.question_model if encoder_type is None or encoder_type == "question" else self.ctx_model
        _q_seq, q_pooled_out, _q_hidden = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            self.fix_q_encoder,
            representation_token_pos = representation_token_pos,
        )

        ctx_encoder = self.ctx_model if encoder_type is None or encoder_type == "ctx" else self.question_model
        _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
        ctx_encoder, context_ids, ctx_segments, ctx_attn_mask, self.fix_ctx_encoder
        )

        return torch.einsum("qih,cjh->qcij", _q_seq, _ctx_seq)

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

        q_b_size, q_len = question_ids.size()
        ctx_b_size, c_len = context_ids.size()
        one_ctx_size = ctx_b_size // q_b_size

        r = self.get_interactions(
            question_ids,
            question_segments,
            question_attn_mask,
            context_ids,
            ctx_segments,
            ctx_attn_mask,
            encoder_type,
            representation_token_pos,
        )
        r = r.max(-1).values
        r = r.mean(-1)
        r = r.view(q_b_size, -1)

        one_ctx_size = ctx_b_size // q_b_size
        rows = torch.arange(q_b_size).view(-1, 1).repeat(1, one_ctx_size).view(-1)
        cols = torch.arange(q_b_size * one_ctx_size).view(-1)

        return torch.zeros(q_b_size, device=r.device), r[rows, cols].view(q_b_size, -1)

    def get_loss_function(self) -> RankerLoss:
        return RerankerNllLoss(self.cfg)


class BERTRerankerCrossEncoder(DocumentBiEncoder):

    def __init__(self,
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
        if question_model.encode_proj is None:
            raise Exception("Set \"encoder.projection_dim: 1\" for BERTRerankerCrossEncoder")

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
        q_b_size, q_len = question_ids.size()
        ctx_b_size, c_len = context_ids.size()

        _, all_cross_pooled_out, _ = self.get_representation(
            q_encoder,
            context_ids,
            ctx_segments,
            ctx_attn_mask,
            self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
        )

        return torch.zeros(q_b_size, device=all_cross_pooled_out.device), all_cross_pooled_out.view(q_b_size, -1)

    def get_loss_function(self) -> RankerLoss:
        return RerankerNllLoss(self.cfg)

    def prepare_model_inputs(self, batch: dict, dataset):

        question_ids = batch["question_ids"]
        question_segments = batch["question_segments"]
        question_attn_mask = dataset.tensorizer.get_attn_mask(batch["question_ids"])
        context_ids = batch["context_ids"]
        ctx_attn_mask = dataset.tensorizer.get_attn_mask(batch["context_ids"])

        q_b_size, q_len = question_ids.size()
        ctx_b_size, c_len = context_ids.size()

        all_cross_ids = question_ids.repeat(1, ctx_b_size // q_b_size).view(- 1, q_len).repeat(1, 2)
        all_cross_ids[:, q_len:] = dataset.tensorizer.get_pad_id()

        cols = torch.arange(1, c_len, device=question_ids.device).view(1, -1).repeat(ctx_b_size, 1) + ((all_cross_ids != dataset.tensorizer.get_pad_id()).sum(-1).view(-1, 1) - 1)
        rows = torch.arange(ctx_b_size, device=question_ids.device).view(-1, 1).repeat(1, c_len - 1)

        all_cross_ids[rows, cols] = context_ids[:, 1:]

        all_cross_segments = question_segments.repeat(1, ctx_b_size // q_b_size).view(- 1, q_len).repeat(1, 2)
        all_cross_segments[:, q_len:] = 0
        all_cross_segments[rows, cols] = 1

        all_cross_attn_mask = question_attn_mask.repeat(1, ctx_b_size // q_b_size).view(- 1, q_len).repeat(1, 2)
        all_cross_attn_mask[:, q_len:] = 0
        all_cross_attn_mask[rows, cols] = ctx_attn_mask[:, 1:]

        return {
            "question_ids": question_ids,
            "question_segments": question_segments,
            "question_attn_mask": question_attn_mask,
            "context_ids": all_cross_ids,
            "ctx_segments": all_cross_segments,
            "ctx_attn_mask": all_cross_attn_mask,
        }