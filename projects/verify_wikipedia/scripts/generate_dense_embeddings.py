#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import glob
import logging
import math
import os
import pathlib
import pickle
import shutil
import sys
import time

import filelock
from typing import List, Tuple, Union

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch import nn

from dpr.models import init_biencoder_components
from dpr.options import set_cfg_params_from_state,  setup_logger
from dpr.dataset.input_transform import Passage, SchemaPreprocessor
from dpr.dataset.retrieval import ContextSource

from dpr.utils.data_utils import Tensorizer
from dpr.utils.dist_utils import setup_cfg_gpu
from dpr.utils.model_utils import (
    setup_fp16_and_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
)
from misc.utils import load_options_from_argv_yaml

logger = logging.getLogger()
setup_logger(logger)


def gen_ctx_vectors(
    cfg: DictConfig,
    checkpoint_cfg: DictConfig,
    ctx_rows: List[Tuple[object, Passage]],
    ctx_src: Union[ContextSource, SchemaPreprocessor],
    model: nn.Module,
) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = cfg.batch_size
    total = 0
    results = []
    max_phi = 0
    for j, batch_start in enumerate(range(0, n, bsz)):
        batch = ctx_rows[batch_start: batch_start + bsz]
        batch_token_tensors = [
            ctx_src.preprocess_passage(ctx[1]) for ctx in batch
        ]

        ctx_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0), checkpoint_cfg.device)
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), checkpoint_cfg.device)
        ctx_attn_mask = move_to_device(ctx_src.tensorizer.get_attn_mask(ctx_ids_batch), checkpoint_cfg.device)
        with torch.no_grad():
            _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
        out = out.cpu()

        ctx_ids = [r[0] for r in batch]

        assert len(ctx_ids) == out.size(0)
        total += len(ctx_ids)

        results.extend([(ctx_ids[i], out[i].view(-1).numpy(), *ctx_src.get_meta(batch[i][1])) for i in range(out.size(0))])

        phi = (out.numpy() ** 2).sum(1).max()
        if phi > max_phi:
            max_phi = phi

        if total % 10 == 0:
            logger.info("Encoded passages %d", total)
    return results, max_phi


@hydra.main(config_path="../conf", config_name="generate_embeddings")
def main(cfg: DictConfig):

    # delete the output from a previous run
    if hasattr(cfg, "rerun") and cfg.rerun:
        logger.info("Rerunning this configuration!")
        lock = filelock.FileLock(f"{cfg.model_file}.rerun_lock")
        with lock:
            if not os.path.exists(f"{cfg.model_file}.rerun_refresh_counter"):
                logger.info("Cleaning up previous output!")
                # this process is the first process to enter this
                if os.path.exists(f"{cfg.model_file}.generate_embeddings"):
                    os.rename(f"{cfg.model_file}.generate_embeddings", f"{cfg.model_file}.generate_embeddings_bak")
                    os.rename(os.path.dirname(cfg.out_file), f"{os.path.dirname(cfg.out_file)}_bak")
                with open(f"{cfg.model_file}.rerun_refresh_counter", "w") as f:
                    f.writelines([f"{cfg.shard_id}\n"])

            else:
                logger.info("Previous output already refreshed!")
                # this process is *not* the first process to enter so we count how many have visited to clean up after
                with open(f"{cfg.model_file}.rerun_refresh_counter", "a+") as f:
                    f.writelines([f"{cfg.shard_id}\n"])
                with open(f"{cfg.model_file}.rerun_refresh_counter") as f:
                    rerun_refresh_processes = len(list(f.readlines()))

                # this is the last process to enter this, so we clean up
                if rerun_refresh_processes == cfg.num_shards:
                    os.remove(f"{cfg.model_file}.rerun_refresh_counter")

    if os.path.exists(cfg.out_file + "_" + str(cfg.shard_id)):
        logger.info(f"Shard {cfg.shard_id} already processed. File '{cfg.out_file}_{cfg.shard_id}' already exists. Exiting.")
        return True

    cfg.checkpoint_dir = str(pathlib.Path(hydra.utils.to_absolute_path(cfg.checkpoint_dir)))

    checkpoint_cfg = OmegaConf.load(f"{cfg.checkpoint_dir}/.hydra/config.yaml")

    assert cfg.model_file, "Please specify encoder checkpoint as model_file param"
    assert cfg.ctx_src, "Please specify passages source as ctx_src param"

    checkpoint_cfg = setup_cfg_gpu(checkpoint_cfg)

    time.sleep((cfg.shard_id))

    cfg.model_file = str(pathlib.Path(hydra.utils.to_absolute_path(cfg.model_file)))

    if not os.path.exists(f"{cfg.model_file}.generate_embeddings"):
        lock = filelock.FileLock(f"{cfg.model_file}.lock")
        with lock:
            if not os.path.exists(f"{cfg.model_file}.generate_embeddings"):
                shutil.copy(cfg.model_file, f"{cfg.model_file}.generate_embeddings")
            saved_state = load_states_from_checkpoint(f"{cfg.model_file}.generate_embeddings")
    else:
        lock = filelock.FileLock(f"{cfg.model_file}.lock")
        with lock:
            saved_state = load_states_from_checkpoint(f"{cfg.model_file}.generate_embeddings")

    set_cfg_params_from_state(saved_state.encoder_params, checkpoint_cfg)

    logger.info("CFG:")
    logger.info("%s", OmegaConf.to_yaml(checkpoint_cfg))
    logger.info("%s", OmegaConf.to_yaml(cfg))

    tensorizer, encoder, _ = init_biencoder_components(checkpoint_cfg.base_model.encoder_model_type, checkpoint_cfg, inference_only=True)

    encoder = encoder.ctx_model if cfg.encoder_type == "ctx" else encoder.biencoder_model

    encoder, _ = setup_fp16_and_distributed_mode(
        encoder,
        None,
        checkpoint_cfg.device,
        checkpoint_cfg.n_gpu,
        checkpoint_cfg.local_rank,
        checkpoint_cfg.fp16,
        checkpoint_cfg.fp16_opt_level,
    )
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info("Loading saved model state ...")
    logger.debug("saved model keys =%s", saved_state.model_dict.keys())

    prefix_len = len("ctx_model.")
    ctx_state = {
        key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if key.startswith("ctx_model.")
    }
    model_to_load.load_state_dict(ctx_state, strict=False)

    logger.info("reading data source: %s", cfg.ctx_src)

    if os.path.exists(cfg.ctx_src):
        cfg.temp_dataset.file = cfg.ctx_src
        ctx_src = hydra.utils.instantiate(cfg.temp_dataset, cfg, tensorizer)
    else:
        ctx_src = hydra.utils.instantiate(cfg.datasets[cfg.ctx_src], cfg, tensorizer)

    all_passages_dict = {}

    len_all_passages = len(ctx_src)
    shard_size = math.ceil(len_all_passages / cfg.num_shards)
    start_idx = cfg.shard_id * shard_size
    end_idx = start_idx + shard_size

    ctx_src.load_data_to(all_passages_dict, start_idx, end_idx)
    shard_passages = [(k, v) for k, v in all_passages_dict.items()]

    logger.info(
        "Producing encodings for passages range: %d to %d (out of total %d)",
        start_idx,
        end_idx,
        len_all_passages,
    )

    data, phi = gen_ctx_vectors(
        cfg=cfg,
        checkpoint_cfg=checkpoint_cfg,
        ctx_rows=shard_passages,
        ctx_src=ctx_src,
        model=encoder,
    )

    file = cfg.out_file + "_" + str(cfg.shard_id)
    meta = cfg.out_file + "_meta_" + str(cfg.shard_id)
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info("Writing results to %s" % file)
    with open(file, mode="wb") as f:
        pickle.dump(data, f)
    with open(meta, mode="wb") as f:
        pickle.dump({"phi": phi}, f)

    logger.info("Total passages processed %d. Written to %s. Phi %f", len(data), file, phi)

    return True


if __name__ == "__main__":

    logger.info("Sys.argv: %s", sys.argv)

    with open(str(pathlib.Path(__file__).parent.parent.absolute()) + "/conf/generate_embeddings.yaml") as f:
        gen_embs_conf = yaml.safe_load(f)
    load_options_from_argv_yaml(constrain_to=gen_embs_conf)

    if main():
        sys.exit(0)
    else:
        sys.exit(1)
