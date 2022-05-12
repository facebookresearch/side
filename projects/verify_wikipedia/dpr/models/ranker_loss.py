#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor as T

logger = logging.getLogger(__name__)


def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)


class RankerLoss(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        global_step: int,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        raise NotImplementedError

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        raise NotImplementedError

    @staticmethod
    def get_similarity_function():
        raise NotImplementedError


class PassageLevelRankerNllLoss(RankerLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        global_step: int,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors)
        if isinstance(positive_idx_per_question, list):
            positive_idx_per_question = torch.tensor(positive_idx_per_question)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            positive_idx_per_question.to(softmax_scores.device),
            reduction="mean",
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == positive_idx_per_question.to(max_idxs.device)).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = PassageLevelRankerNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores


class DocumentLevelRankerNllLoss(PassageLevelRankerNllLoss):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_mask: list,
        global_step: int,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors. Difference to the PassageLevelRankerNllLoss
        is that there are multiple positive passages per training instance and we do not have supervision on which
        of the positive passage is the correct one that contains the relevant text. This class contains a few different
        strategies to handle this kind of case. They mostly revolve around using the score of the question and passage
        encoder to weight the softmax loss according to the model's current belief, i.e. the expectation maximisation
        algorithm (EM).

            "hard_em": pick the loss with the positive passage that receives the highest scores amongst all positive passages
            "first_passage": pick the first positive passage (in document order)
            "random_passage": pick a random passage
            "soft_em": weight the losses of the positive passage according to their scores amongst all positive passages
            "random": randomly weight all the passages
            "uniform": uniformly weight all the passages

        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        loss, correct_predictions_count = None, None

        if self.cfg.loss_strategy in {"hard_em", "first_passage", "random_passage"}:

            positive_idx_per_question = None

            if self.cfg.loss_strategy == "hard_em":
                e_step_scores = scores.clone().detach()
                e_step_scores[~positive_mask] = -1.0e6
                positive_idx_per_question = e_step_scores.max(-1).indices

            elif self.cfg.loss_strategy == "first_passage":
                true_coords = positive_mask.nonzero()
                first_pass_idx = torch.ones_like(positive_mask, dtype=torch.long) * 10_000
                first_pass_idx[true_coords[:, 0], true_coords[:, 1]] = true_coords[:, 1]
                positive_idx_per_question = first_pass_idx.min(1).indices

            elif self.cfg.loss_strategy == "random_passage":
                e_step_scores = torch.randn(scores.size())
                e_step_scores[~positive_mask] = -1.0e6
                positive_idx_per_question = e_step_scores.max(-1).indices

            positive_mask[torch.arange(len(positive_idx_per_question)), positive_idx_per_question] = False
            scores_mask = torch.ones_like(scores, requires_grad=False)
            scores_mask[positive_mask] = -1.0e6
            scores_with_mask = scores + scores_mask
            softmax_scores = torch.log_softmax(scores_with_mask, dim=1)

            loss = -(softmax_scores[torch.arange(len(positive_idx_per_question)), positive_idx_per_question]).sum()

        elif self.cfg.loss_strategy in {
            "soft_em",
            "random",
            "uniform",
        }:

            # positive_mask = torch.tensor([
            #     [True, True, False, False, False, False, False, ],
            #     [False, False, False, True, True, True, False, ],
            # ])

            true_coords = positive_mask.nonzero()
            # true_coords =
            # tensor([[0, 0],
            #         [0, 1],
            #         [1, 3],
            #         [1, 4],
            #         [1, 5]])

            positive_mask_scores = positive_mask[true_coords[:, 0]]
            positive_mask_scores[torch.arange(positive_mask_scores.size(0)), true_coords[:, 1]] = False
            # Mask out the positive scores for each instance
            # positive_mask_scores =
            # tensor([[False, True, False, False, False, False, False],
            #         [True, False, False, False, False, False, False],
            #         [False, False, False, False, True, True, False],
            #         [False, False, False, True, False, True, False],
            #         [False, False, False, True, True, False, False]])

            scores_softmax_mask = torch.ones_like(positive_mask_scores, dtype=torch.float, device=scores.device)
            scores_softmax_mask[positive_mask_scores] = 0

            scores_with_mask = scores[true_coords[:, 0]] * scores_softmax_mask
            log_softmax_scores = torch.log_softmax(scores_with_mask, dim=-1)
            # log_softmax_scores = torch.log(
            # tensor([[0.2626, 0.0000, 0.0705, 0.2840, 0.0632, 0.0606, 0.2590],
            #         [0.0000, 0.2288, 0.0738, 0.2971, 0.0661, 0.0634, 0.2709],
            #         [0.0506, 0.3419, 0.5314, 0.0520, 0.0000, 0.0000, 0.0241],
            #         [0.0469, 0.3170, 0.4926, 0.0000, 0.1211, 0.0000, 0.0224],
            #         [0.0438, 0.2960, 0.4600, 0.0000, 0.0000, 0.1793, 0.0209]]))
            log_softmax_scores = log_softmax_scores[
                torch.arange(positive_mask_scores.size(0), device=positive_mask.device), true_coords[:, 1]
            ]

            if self.cfg.loss_strategy == "soft_em":
                e_step_scores = scores.detach().clone()
                # apply temperature
                if global_step and self.cfg.loss_strategy == "soft_em" and self.cfg.soft_em.warmup_steps > 0:
                    global_step_clamped = min(global_step, self.cfg.soft_em.warmup_steps)
                    temperature = (self.cfg.soft_em.end_temperature - self.cfg.soft_em.start_temperature) * (
                        global_step_clamped / self.cfg.soft_em.warmup_steps
                    ) + self.cfg.soft_em.start_temperature
                    e_step_scores = e_step_scores * temperature
                softmax_mask = torch.zeros_like(positive_mask, dtype=torch.float, device=e_step_scores.device)
                softmax_mask[~positive_mask] = -1.0e6
                e_step_probs = torch.softmax(e_step_scores + softmax_mask, dim=-1)
                # e_step_probs =
                # tensor([[0.6502, 0.3498, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                #         [0.0000, 0.0000, 0.0000, 0.1621, 0.7200, 0.1179, 0.0000]])
                weighted_logsoftmax = log_softmax_scores * e_step_probs[true_coords[:, 0], true_coords[:, 1]]
                loss = -(weighted_logsoftmax).sum()
            elif self.cfg.loss_strategy == "random":
                random_weights = torch.randn_like(scores[true_coords[:, 0], true_coords[:, 1]].view(-1, 1))
                random_weights = torch.softmax(random_weights, dim=-1)
                weighted_logsoftmax = log_softmax_scores * random_weights
                loss = -(weighted_logsoftmax).sum()
            elif self.cfg.loss_strategy == "uniform":
                uniform = torch.ones_like(scores)
                softmax_mask = torch.zeros_like(positive_mask, dtype=torch.float, device=uniform.device)
                softmax_mask[~positive_mask] = -1.0e6
                uniform = torch.softmax(uniform + softmax_mask, dim=-1)
                # e_step_probs =
                # tensor([[0.6502, 0.3498, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                #         [0.0000, 0.0000, 0.0000, 0.1621, 0.7200, 0.1179, 0.0000]])
                weighted_logsoftmax = log_softmax_scores * uniform[true_coords[:, 0], true_coords[:, 1]]
                loss = -(weighted_logsoftmax).sum()

            # weighted_logsoftmax[torch.arange(positive_mask_all_scores.size(0)), true_coords[:, 1]] =
            # tensor([-65024.1797, -34975.6328, -16210.2422, -71997.6328, -11791.9492])

        else:
            raise Exception(f"Unkownn loss_strategy {self.cfg.loss_strategy}")

        # if hasattr(self.cfg, "regularize") and self.cfg.regularize.passages == "spread":
        #     ctx_vectors * ctx_vectors.t()

        # is the highest scored true passage also the highest scored overall passage?
        max_score, max_idxs = torch.max(scores, 1)
        e_step_scores = scores.clone().detach()
        e_step_scores[~positive_mask] = -1.0e6
        positive_idx_per_question = e_step_scores.max(-1).indices
        correct_predictions_count = (max_idxs == positive_idx_per_question.to(max_idxs.device)).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count


class RerankerNllLoss(DocumentLevelRankerNllLoss):
    """
    The RerankerNllLoss expects that the forward function returns a fake output for the q_vector s and a tensor with
    scores for each query x contexts. The q_vector is mainly used to keep track of how many questions the batch had.
    TODO: It would be better to create a RankerOutput class that keeps track of this
    """

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = RerankerNllLoss.get_similarity_function()
        return f(ctx_vectors)

    @staticmethod
    def get_similarity_function():
        def similarity_function(ctx_vectors):
            """
            calculates q->ctx scores for every row in ctx_vector
            :param q_vector:
            :param ctx_vector:
            :return:
            """
            # q_pooled_out, None

            return ctx_vectors

        return similarity_function
