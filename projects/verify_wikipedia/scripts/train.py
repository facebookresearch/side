#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Pipeline to train DPR Biencoder
"""

import logging
import math
import os
import random
import subprocess
import sys
import time
from typing import Tuple, List, Union

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T

from dpr.models import init_biencoder_components
from dpr.models.biencoder import BiEncoder, QueryContextsBatch
from dpr.models.hf_models import get_optimizer
from dpr.options import (
    set_seed,
    get_encoder_params_state_from_cfg,
    set_cfg_params_from_state,
    setup_logger,
)
from dpr.dataset.input_transform import SchemaPreprocessor
from dpr.utils.conf_utils import DatasetsCfg
# get the token to be used for representation selection
from dpr.utils.data_utils import DEFAULT_SELECTOR
from dpr.utils.data_utils import (
    ShardedDataIterator,
    MultiSetDataIterator,
    LocalShardedDataIterator, Dataset,
)
from dpr.utils.dist_utils import all_gather_list, setup_cfg_gpu
from dpr.utils.model_utils import (
    setup_fp16_and_distributed_mode,
    move_to_device,
    get_schedule_linear,
    CheckpointState,
    get_model_file,
    get_model_obj,
    load_states_from_checkpoint,
)
from misc.utils import load_options_from_argv_yaml

logger = logging.getLogger()
setup_logger(logger)


class Trainer(object):

    def __init__(self, cfg: DictConfig):

        self.shard_id = cfg.local_rank if cfg.local_rank != -1 else 0
        self.distributed_factor = cfg.distributed_world_size or 1
        self.start_epoch = 0
        self.start_batch = 0
        self.scheduler_state = None
        self.best_validation_loss_result = None
        self.best_validation_acc_result = None
        self.best_cp_name = None
        self.dev_iterator = None

        logger.info("***** Initializing components for training *****")

        # if model file is specified, encoder parameters from saved state should be used for initialization
        model_file = get_model_file(cfg, cfg.checkpoint_file_name)
        saved_state = None
        if model_file:
            saved_state = load_states_from_checkpoint(model_file)
            set_cfg_params_from_state(saved_state.encoder_params, cfg)

        self.tensorizer, self.model, _ = init_biencoder_components(cfg.base_model.encoder_model_type, cfg)

        self.loss = self.model.get_loss_function()
        self.prepare_model_inputs_fn = self.model.prepare_model_inputs

        self.model, _ = setup_fp16_and_distributed_mode(
            self.model,
            _,
            cfg.device,
            cfg.n_gpu,
            cfg.local_rank,
            cfg.fp16,
            cfg.fp16_opt_level,
        )

        self.optimizer = get_optimizer(
            self.model,
            learning_rate=cfg.trainer.learning_rate,
            adam_eps=cfg.trainer.adam_eps,
            weight_decay=cfg.trainer.weight_decay,
        )
        if saved_state:
            self._load_saved_state(saved_state, cfg)

        self.cfg = cfg
        self.ds_cfg = DatasetsCfg(cfg, tensorizer=self.tensorizer)


    def get_data_iterator(
        self,
        batch_size: int,
        is_train_set: bool,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
        rank: int = 0,
    ):

        hydra_datasets = self.ds_cfg.train_datasets if is_train_set else self.ds_cfg.dev_datasets
        sampling_rates = self.ds_cfg.sampling_rates

        logger.info(
            "Initializing task/set data %s",
            self.ds_cfg.train_datasets_names if is_train_set else self.ds_cfg.dev_datasets_names,
        )

        single_ds_iterator_cls = LocalShardedDataIterator if self.cfg.local_shards_dataloader else ShardedDataIterator

        sharded_iterators = [
            single_ds_iterator_cls(
                ds,
                shard_id=self.shard_id,
                num_shards=self.distributed_factor,
                # shard_id=self.shard_id if is_train_set else 0,
                # num_shards=self.distributed_factor if is_train_set else 1,
                batch_size=batch_size,
                shuffle=shuffle,
                shuffle_seed=shuffle_seed,
                offset=offset,
                strict_batch_size=self.distributed_factor > 1,
            )
            for ds in hydra_datasets
        ]

        return MultiSetDataIterator(
            sharded_iterators,
            shuffle_seed,
            shuffle,
            sampling_rates=sampling_rates if is_train_set else [1],
            # rank=rank if is_train_set else 0,
            rank=rank,
        )

    def run_train(self):
        cfg = self.cfg

        train_iterator = self.get_data_iterator(
            cfg.train_iterator.batch_size,
            True,
            shuffle=True,
            shuffle_seed=cfg.seed,
            offset=self.start_batch,
            rank=cfg.local_rank,
        )
        max_iterations = train_iterator.get_max_iterations()
        logger.info("  Total iterations per epoch=%d", max_iterations)
        if max_iterations == 0:
            logger.warning("No data found for training.")
            return

        updates_per_epoch = train_iterator.max_iterations // cfg.trainer.gradient_accumulation_steps

        total_updates = updates_per_epoch * cfg.trainer.num_train_epochs
        logger.info(" Total updates=%d", total_updates)
        warmup_steps = cfg.trainer.warmup_steps

        if self.scheduler_state:
            # TODO: ideally we'd want to just call
            # scheduler.load_state_dict(self.scheduler_state)
            # but it doesn't work properly as of now

            logger.info("Loading scheduler state %s", self.scheduler_state)
            shift = int(self.scheduler_state["last_epoch"])
            logger.info("Steps shift %d", shift)
            scheduler = get_schedule_linear(
                self.optimizer,
                warmup_steps,
                total_updates,
                steps_shift=shift,
            )
        else:
            scheduler = get_schedule_linear(self.optimizer, warmup_steps, total_updates)

        if cfg.trainer.eval_per_epoch > 1:
            eval_step = math.ceil(train_iterator.max_iterations / cfg.trainer.eval_per_epoch)
        else:
            eval_step = 10 ** 20

        if cfg.local_rank in [-1, 0]:
            logger.info("  Eval step = %d", eval_step)
            logger.info("***** Training *****")

        for epoch in range(self.start_epoch, int(cfg.trainer.num_train_epochs)):
            if cfg.local_rank in [-1, 0]:
                logger.info("***** Epoch %d *****", epoch)
            self._train_epoch(scheduler, epoch, eval_step, train_iterator)

        if cfg.local_rank in [-1, 0]:
            logger.info("Training finished. Best validation checkpoint %s", self.best_cp_name)

    def validate_and_save(self, epoch: int, iteration: int, scheduler):
        cfg = self.cfg
        # for distributed mode, save checkpoint for only one process
        save_cp = cfg.local_rank in [-1, 0]

        if not cfg.dev_datasets:
            validation_loss, validation_acc = 0.0, 0.0
        else:
            validation_loss, validation_acc = self.validate_nll()

        if save_cp:
            cp_name = self._save_checkpoint(scheduler, epoch, iteration, [str(epoch)])
            logger.info("Saved checkpoint to %s", cp_name)

            if validation_loss < (self.best_validation_loss_result or validation_loss + 1):
                best_cp_name = self._save_checkpoint(scheduler, epoch, iteration, ["best_validation_loss"])
                self.best_validation_loss_result = validation_loss
                self.best_cp_name = best_cp_name
                logger.info("New Best validation checkpoint %s", best_cp_name)

            if validation_acc > (self.best_validation_acc_result or 0.0):
                best_cp_name = self._save_checkpoint(scheduler, epoch, iteration, ["best_validation_acc"])
                self.best_validation_acc_result = validation_acc
                self.best_cp_name = best_cp_name
                logger.info("New Best validation checkpoint %s", best_cp_name)

    def validate_nll(self) -> Tuple[float, float]:
        logger.info("NLL validation ...")
        cfg = self.cfg
        self.model.eval()

        if not self.dev_iterator:
            self.dev_iterator = self.get_data_iterator(
                cfg.valid_iterator.batch_size,
                False,
                shuffle=False,
                rank=cfg.local_rank,
            )
        data_iterator = self.dev_iterator

        global_total_loss = 0.0
        start_time = time.time()
        global_total_correct_predictions = 0
        log_result_step = cfg.trainer.log_batch_step
        batches = 0
        ds_id = 0
        global_total_positives = 0

        for i, samples_batch in enumerate(data_iterator.iterate_ds_data()):
            if isinstance(samples_batch, Tuple):
                samples_batch, ds_id = samples_batch

            dataset = self.ds_cfg.train_datasets[ds_id]

            if i % cfg.trainer.log_batch_step == 0:
                logger.info("Eval step: %d ,rnk=%s", i, cfg.local_rank)

            biencoder_batch = dataset.create_model_input(
                samples=samples_batch,
                dataset=dataset,
                cfg=cfg,
                is_training=False,
            )

            # get the token to be used for representation selection
            dataset = self.ds_cfg.dev_datasets[ds_id]
            rep_positions = dataset.selector.get_positions(biencoder_batch.question_ids, self.tensorizer)
            encoder_type = dataset.encoder_type

            loss, nr_correct, global_batch_size = _do_biencoder_fwd_pass(
                self.model,
                self.loss,
                biencoder_batch,
                dataset,
                self.prepare_model_inputs_fn,
                cfg,
                encoder_type=encoder_type,
                rep_positions=rep_positions,
            )
            global_total_loss += loss.item()
            global_total_correct_predictions += nr_correct
            global_total_positives += global_batch_size
            batches += 1
            if (i + 1) % log_result_step == 0:
                logger.info(
                    "Eval step: %d , used_time=%f sec., loss=%f ",
                    i,
                    time.time() - start_time,
                    loss.item(),
                )

        global_total_loss = global_total_loss / global_total_positives
        # total_samples = batches * cfg.trainer.dev_batch_size * self.distributed_factor
        correct_ratio = float(global_total_correct_predictions / global_total_positives)
        logger.info(
            "NLL Validation: loss = %f correct prediction ratio  %d/%d ~  %f",
            global_total_loss,
            global_total_correct_predictions,
            global_total_positives,
            correct_ratio,
        )
        return global_total_loss, correct_ratio

    def _train_epoch(
        self,
        scheduler,
        epoch: int,
        eval_step: int,
        train_data_iterator: MultiSetDataIterator,
    ):

        scaler = None
        if self.cfg.fp16:
            scaler = torch.cuda.amp.GradScaler()

        cfg = self.cfg
        rolling_train_loss = 0.0
        epoch_loss = 0
        epoch_correct_predictions = 0
        epoch_correct_predictions_step = 0

        log_result_step = cfg.trainer.log_batch_step
        rolling_loss_step = cfg.trainer.train_rolling_loss_step
        seed = cfg.seed
        self.model.train()
        epoch_batches = train_data_iterator.max_iterations
        data_iteration = 0

        dataset_idx = 0
        for i, samples_batch in enumerate(train_data_iterator.iterate_ds_data(epoch=epoch)):
            if isinstance(samples_batch, Tuple):
                samples_batch, dataset_idx = samples_batch

            dataset = self.ds_cfg.train_datasets[dataset_idx]
            # special_token = ds_cfg.special_token
            encoder_type = dataset.encoder_type

            # to be able to resume shuffled ctx- pools
            data_iteration = train_data_iterator.get_iteration()
            random.seed(seed + epoch + data_iteration)

            biencoder_batch = dataset.create_model_input(
                samples=samples_batch,
                cfg=cfg,
                dataset=dataset,
                is_training=True,
            )

            selector = dataset.selector if dataset else DEFAULT_SELECTOR

            rep_positions = selector.get_positions(biencoder_batch.question_ids, self.tensorizer)

            loss_scale = cfg.loss_scale_factors[dataset_idx] if cfg.loss_scale_factors else None
            global_step = epoch * epoch_batches + data_iteration

            # prepared_batch = self.prepare_model_inputs_fn(biencoder_batch._asdict(), dataset)
            # for inst in prepared_batch["context_ids"]:
            #     print(inst.tolist())
            #     print(self.tensorizer.tokenizer.convert_ids_to_tokens(inst))
            # exit()

            try:
                # logger.warning(f"Cuda memory BEFORE fwd in data_iteration {data_iteration} batch {i}. {query('memory.used').split()}")
                global_loss, global_nr_is_correct, global_batch_size = _do_biencoder_fwd_pass(
                    self.model,
                    self.loss,
                    biencoder_batch,
                    dataset,
                    self.prepare_model_inputs_fn,
                    cfg,
                    global_step=global_step,
                    encoder_type=encoder_type,
                    rep_positions=rep_positions,
                    loss_scale=loss_scale,
                )
                # logger.warning(f"Cuda memory AFTER fwd in data_iteration {data_iteration} batch {i}. {query('memory.used').split()}")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"Cuda OOM in data_iteration {data_iteration} batch {i}. {query('memory.used').split()}")
                    self.model.zero_grad()
                    continue
                else:
                    raise e

            epoch_correct_predictions += global_nr_is_correct
            epoch_correct_predictions_step += global_batch_size
            epoch_loss += global_loss.item()
            rolling_train_loss += global_loss.item()

            if cfg.fp16:

                scaler.scale(global_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if cfg.trainer.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.trainer.max_grad_norm)
            else:
                # logger.info("!! loss %s %s", loss, loss.device)
                global_loss.backward()

                # if cfg.local_rank in [-1, 0]:
                    # TODO: tmp debug code
                    # """
                    # m = get_model_obj(self.biencoder)
                    # cm = m.ctx_model
                    # qm = m.question_model
                    # if i % log_result_step == 0:
                    #     logger.info("!! Grad total Norm %s", _print_norms(m))
                    #     logger.info("!! Grad ctx Norm %s", _print_norms(cm))
                    #     logger.info("!! Grad q Norm %s", _print_norms(qm))
                    # """

                if cfg.trainer.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.trainer.max_grad_norm)

            if (i + 1) % cfg.trainer.gradient_accumulation_steps == 0:
                self.optimizer.step()
                scheduler.step()
                self.model.zero_grad()

            if i % log_result_step == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                if cfg.local_rank in [-1, 0]:
                    logger.warning(f"Cuda memory in data_iteration {data_iteration} batch {i}. {query('memory.used').split()}")
                    logger.info(
                        "Epoch: %d: Step: %d/%d, loss=%f, lr=%f",
                        epoch,
                        data_iteration,
                        epoch_batches,
                        global_loss.item(),
                        lr,
                    )

            if cfg.local_rank in [-1, 0]:
                if (i + 1) % rolling_loss_step == 0:
                    logger.info("Train batch %d", data_iteration)
                    latest_rolling_train_av_loss = rolling_train_loss / rolling_loss_step
                    latest_rolling_train_epoch_correct_predictions = epoch_correct_predictions / epoch_correct_predictions_step
                    logger.warning(f"Cuda memory in data_iteration {data_iteration} batch {i}. {query('memory.used').split()}")
                    logger.info(
                        "Avg. loss/acc per last %d batches: %f / %f",
                        rolling_loss_step,
                        latest_rolling_train_av_loss,
                        latest_rolling_train_epoch_correct_predictions,
                    )
                    rolling_train_loss = 0.0

            if data_iteration % eval_step == 0:
                if cfg.local_rank in [-1, 0]:
                    logger.info(
                        "rank=%d, Validation: Epoch: %d Step: %d/%d",
                        cfg.local_rank,
                        epoch,
                        data_iteration,
                        epoch_batches,
                    )
                self.validate_and_save(epoch, train_data_iterator.get_iteration(), scheduler)
                self.model.train()

        logger.info("Epoch finished on %d", cfg.local_rank)
        self.validate_and_save(epoch, data_iteration, scheduler)

        epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        logger.info("Av Loss per epoch=%f", epoch_loss)
        logger.info("epoch total correct predictions=%d", epoch_correct_predictions)

    def _save_checkpoint(self, scheduler, epoch: int, offset: int, tag: List[str]= None) -> str:
        cfg = self.cfg
        model_to_save = get_model_obj(self.model)
        tag_str = ""
        if len(tag) > 0:
            tag_str = "." + ".".join(tag)
        cp = os.path.join(cfg.output_dir, cfg.checkpoint_file_name + tag_str)
        meta_params = get_encoder_params_state_from_cfg(cfg)
        state = CheckpointState(
            model_dict=model_to_save.get_state_dict(),
            optimizer_dict=self.optimizer.state_dict(),
            scheduler_dict=scheduler.state_dict(),
            offset=offset,
            epoch=epoch,
            best_validation_loss_result=self.best_validation_loss_result,
            best_validation_acc_result=self.best_validation_acc_result,
            encoder_params=meta_params,
        )
        torch.save(state._asdict(), cp)
        logger.info("Saved checkpoint at %s", cp)
        return cp

    def _load_saved_state(self, saved_state: CheckpointState, cfg):
        epoch = saved_state.epoch
        # offset is currently ignored since all checkpoints are made after full epochs
        offset = saved_state.offset
        if offset == 0:  # epoch has been completed
            epoch += 1

        if cfg.resume.restore_best_metric_from_checkpoint:
            self.best_validation_loss_result = saved_state.best_validation_loss_result
            self.best_validation_acc_result = saved_state.best_validation_acc_result

        logger.info("Loading checkpoint @ batch=%s and epoch=%s", offset, epoch)

        if cfg.resume.ignore_checkpoint_offset:
            self.start_epoch = 0
            self.start_batch = 0
        else:
            self.start_epoch = epoch
            # TODO: offset doesn't work for multiset currently
            self.start_batch = 0  # offset

        model_to_load = get_model_obj(self.model)
        logger.info("Loading saved model state ...")

        # TODO: tmp - load only ctx encoder state

        # logger.info("!!! loading only ctx encoder state")
        """
        prefix_len = len("ctx_model.")
        ctx_state = {
            key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if key.startswith("ctx_model.")
        }
        model_to_load.ctx_model.load_state_dict(ctx_state, strict=False)
        return
        """

        model_to_load.load_state(saved_state, strict=True)
        logger.info("Saved state loaded")
        if not cfg.resume.ignore_checkpoint_optimizer:
            if saved_state.optimizer_dict:
                logger.info("Loading saved optimizer state ...")
                self.optimizer.load_state_dict(saved_state.optimizer_dict)

            if saved_state.scheduler_dict:
                self.scheduler_state = saved_state.scheduler_dict

def gather_to_main_device(
        cfg,
        local_q_vector,
        local_ctx_vectors,
        local_positive_idxs = None,
        local_hard_negatives_idxs: list = None,
):
    distributed_world_size = cfg.distributed_world_size or 1

    if distributed_world_size > 1:

        q_vector_to_send = torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach_()
        ctx_vector_to_send = torch.empty_like(local_ctx_vectors).cpu().copy_(local_ctx_vectors).detach_()

        global_question_ctx_vectors = all_gather_list(
            [
                q_vector_to_send,
                ctx_vector_to_send,
                local_positive_idxs,
                local_hard_negatives_idxs,
            ],
            max_size=cfg.global_loss_buf_sz,
        )

        global_q_vector = []
        global_ctxs_vector = []

        # ctxs_per_question = local_ctx_vectors.size(0)
        _positive_idx_per_question = []
        hard_negatives_per_question = []

        total_ctxs = 0

        for i, item in enumerate(global_question_ctx_vectors):
            q_vector, ctx_vectors, positive_idx, hard_negatives_idxs = item

            if i != cfg.local_rank:
                global_q_vector.append(q_vector.to(local_q_vector.device))
                global_ctxs_vector.append(ctx_vectors.to(local_q_vector.device))
                _positive_idx_per_question.append(positive_idx.to(local_q_vector.device))
                # positive_idx_per_question.extend([v + total_ctxs for v in positive_idx.to(local_q_vector.device)])
                hard_negatives_per_question.extend([[v + total_ctxs for v in l] for l in hard_negatives_idxs])
            else:
                global_q_vector.append(local_q_vector)
                global_ctxs_vector.append(local_ctx_vectors)
                _positive_idx_per_question.append(positive_idx)
                # positive_idx_per_question.extend([v + total_ctxs for v in local_positive_idxs])
                hard_negatives_per_question.extend([[v + total_ctxs for v in l] for l in local_hard_negatives_idxs])
            total_ctxs += ctx_vectors.size(0)

        _positive_idx_per_question = torch.cat(_positive_idx_per_question, dim=0)
        global_q_vector = torch.cat(global_q_vector, dim=0)
        global_ctxs_vector = torch.cat(global_ctxs_vector, dim=0)
        if cfg.model_class not in {
            "BERTRerankerCrossEncoder",
            "ColbertRerankerBiEncoder",
        }:
            nz1 = _positive_idx_per_question.nonzero()
            nz2 = _positive_idx_per_question.view(-1).nonzero()
            positive_idx_per_question = torch.zeros(
                (
                    _positive_idx_per_question.size(0),
                    _positive_idx_per_question.size(0) * _positive_idx_per_question.size(1)
                ),
                device=_positive_idx_per_question.device,
                dtype=torch.bool,
            )
            positive_idx_per_question[nz1[:, 0], nz2.view(-1)] = 1
        else:
            positive_idx_per_question = _positive_idx_per_question

    else:
        global_q_vector = local_q_vector
        global_ctxs_vector = local_ctx_vectors
        positive_idx_per_question = local_positive_idxs
        hard_negatives_per_question = local_hard_negatives_idxs

    return global_q_vector, \
            global_ctxs_vector, \
            positive_idx_per_question, \
            hard_negatives_per_question


def _calc_loss(
    cfg,
    loss_function,
    local_q_vector,
    local_ctx_vectors,
    local_positive_idxs,
    global_step,
    local_hard_negatives_idxs: list = None,
    loss_scale: float = None,
) -> Tuple[T, int, int]:
    """
    Calculates In-batch negatives schema loss and supports to run it in DDP mode by exchanging the representations
    across all the nodes.
    """
    global_q_vector, \
    global_ctxs_vector, \
    global_positive_idx, \
    hard_negatives_per_question = gather_to_main_device(
        cfg,
        local_q_vector,
        local_ctx_vectors,
        local_positive_idxs,
        local_hard_negatives_idxs,
    )

    # logger.info("!!! global_q_vector %s rank=%s", global_q_vector.device, cfg.local_rank)
    # logger.info("!!! global_q_vector %s rank=%s", global_q_vector.device, cfg.local_rank)
    global_loss, global_nr_is_correct = loss_function.calc(
        global_q_vector,
        global_ctxs_vector,
        global_positive_idx,
        global_step,
        hard_negatives_per_question,
        loss_scale=loss_scale,
    )

    global_batch_size = global_positive_idx.size(0)
    global_loss = global_loss / global_batch_size
    global_nr_is_correct = global_nr_is_correct.sum()

    return global_loss, global_nr_is_correct, global_batch_size


def _print_norms(model):
    total_norm = 0
    for n, p in model.named_parameters():
        if p.grad is None:
            # logger.info("!!! no grad for p=%s", n)
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def show_gpu(msg=""):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """
    import subprocess
    def query(field):
        return (subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
             '--format=csv,nounits,noheader'],
            encoding='utf-8'))

    def to_int(result):
        return int(result.strip().split('\n')[0])

    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used / total
    return ('\n' + msg, f'{100 * pct:2.1f}% ({used} out of {total})')


def _do_biencoder_fwd_pass(
    biencoder: BiEncoder,
    loss_function,
    input: QueryContextsBatch,
    dataset: Union[Dataset, SchemaPreprocessor],
    prepare_model_inputs_fn,
    cfg,
    encoder_type: str,
    global_step=None,
    rep_positions=0,
    loss_scale: float = None,
) -> Tuple[torch.Tensor, int, int]:

    input = move_to_device(input._asdict(), cfg.device)

    def get_output():
        if biencoder.training:
            model_out = biencoder(
                **prepare_model_inputs_fn(input, dataset),
                encoder_type=encoder_type,
                representation_token_pos=rep_positions,
            )
        else:
            with torch.no_grad():
                model_out = biencoder(
                    **prepare_model_inputs_fn(input, dataset),
                    encoder_type=encoder_type,
                    representation_token_pos=rep_positions,
                )
        return model_out

    if cfg.fp16:
        with torch.cuda.amp.autocast():
            local_q_vector, local_ctx_vectors = get_output()
            global_loss, global_nr_is_correct, global_batch_size = _calc_loss(
                cfg,
                loss_function,
                local_q_vector,
                local_ctx_vectors,
                input["is_positive"],
                global_step,
                input["hard_negatives"],
                loss_scale=loss_scale,
            )
    else:
        local_q_vector, local_ctx_vectors = get_output()
        global_loss, global_nr_is_correct, global_batch_size = _calc_loss(
            cfg,
            loss_function,
            local_q_vector,
            local_ctx_vectors,
            input["is_positive"],
            global_step,
            input["hard_negatives"],
            loss_scale=loss_scale,
        )

    if cfg.trainer.gradient_accumulation_steps > 1:
        global_loss = global_loss / cfg.trainer.gradient_accumulation_steps

    return global_loss, global_nr_is_correct, global_batch_size


def query(field):
    return (subprocess.check_output(
        ['nvidia-smi', f'--query-gpu={field}',
         '--format=csv,nounits,noheader'],
        encoding='utf-8'))


@hydra.main(config_path="../conf", config_name="training_config")
def main(cfg: DictConfig):

    if cfg.trainer.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                cfg.trainer.gradient_accumulation_steps
            )
        )

    if cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)

    cfg = setup_cfg_gpu(cfg)
    set_seed(cfg)

    if cfg.local_rank in [-1, 0]:
        logger.info("CFG (after gpu  configuration):")
        logger.info("%s", OmegaConf.to_yaml(cfg))

    if cfg.local_rank == -1 and torch.cuda.device_count() > 0:
        assert cfg.train_iterator.batch_size % torch.cuda.device_count() == 0, f"cfg.train_iterator.batch_size={cfg.train_iterator.batch_size} % torch.cuda.device_count()={torch.cuda.device_count()} == 0"

    trainer = Trainer(cfg)

    with open(f"./eval_config_template.yaml", "w") as f:
        OmegaConf.save(cfg.eval_config_template, f)

    if cfg.train_datasets and len(cfg.train_datasets) > 0:
        trainer.run_train()
    elif cfg.model_file and cfg.dev_datasets:
        logger.info("No train files are specified. Run 2 types of validation for specified model file")
        trainer.validate_nll()
    else:
        logger.warning("Neither train_file or (model_file & dev_file) parameters are specified. Nothing to do.")


if __name__ == "__main__":
    logger.info("Sys.argv: %s", sys.argv)
    load_options_from_argv_yaml()
    main()


