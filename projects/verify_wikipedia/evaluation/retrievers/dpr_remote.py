# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import pickle
import zlib
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

from dpr.dense_retriever import DenseRPCRetriever
from dpr.models import init_biencoder_components
from dpr.options import set_cfg_params_from_state
from dpr.utils.dist_utils import setup_cfg_gpu
from dpr.utils.model_utils import (
    setup_fp16_and_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)
from evaluation.retrievers.base_retriever import Retriever

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class DPR(Retriever):
    def __init__(self, name, cfg):
        super().__init__(name)

        self.cfg = cfg
        _ckpt_cfg = OmegaConf.load(f"{cfg.retriever.checkpoint_dir}/.hydra/config.yaml")
        self.checkpoint_cfg = setup_cfg_gpu(_ckpt_cfg)

        logger.info("CFG (after gpu  configuration):")
        logger.info("%s", OmegaConf.to_yaml(cfg))

        if os.path.exists(f"{cfg.retriever.model_file}.generate_embeddings"):
            saved_state = load_states_from_checkpoint(f"{cfg.retriever.model_file}.generate_embeddings")
        elif os.path.exists(f"{cfg.retriever.model_file}"):
            saved_state = load_states_from_checkpoint(f"{cfg.retriever.model_file}")
        else:
            raise Exception(
                f"Checkpoint {cfg.retriever.model_file}.generate_embeddings or {cfg.retriever.model_file} doesn't exist."
            )

        set_cfg_params_from_state(saved_state.encoder_params, self.checkpoint_cfg)

        tensorizer, encoder, _ = init_biencoder_components(
            self.checkpoint_cfg.base_model.encoder_model_type, self.checkpoint_cfg, inference_only=True
        )

        encoder = encoder.question_model
        encoder, _ = setup_fp16_and_distributed_mode(
            encoder,
            None,
            self.checkpoint_cfg.device,
            self.checkpoint_cfg.n_gpu,
            self.checkpoint_cfg.local_rank,
            self.checkpoint_cfg.fp16,
        )
        encoder.eval()

        # load weights from the model file
        model_to_load = get_model_obj(encoder)
        logger.info("Loading saved model state ...")

        encoder_prefix = "question_model."
        prefix_len = len(encoder_prefix)

        logger.info("Encoder state prefix %s", encoder_prefix)
        question_encoder_state = {
            key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if key.startswith(encoder_prefix)
        }
        model_to_load.load_state_dict(question_encoder_state)
        vector_size = model_to_load.get_out_size()
        logger.info("Encoder vector_size=%d", vector_size)

        ctx_src = hydra.utils.instantiate(
            cfg.retriever.datasets[cfg.retriever.ctx_src], self.checkpoint_cfg, tensorizer=tensorizer
        )

        self.retriever = DenseRPCRetriever(
            question_encoder=encoder,
            qa_src=ctx_src,
            batch_size=cfg.retriever.batch_size,
            index_cfg_path=cfg.retriever.rpc_retriever_cfg_file,
            dim=vector_size,
            use_l2_conversion=True,
            # nprobe=128,
        )
        self.retriever.load_index(cfg.retriever.rpc_index_id)

    def set_queries_data(self, queries_data):
        # get questions & answers
        self.questions = [x["query"].strip() for x in queries_data]
        self.query_ids = [x["id"] for x in queries_data]

    def run(self):

        questions_tensor = self.retriever.generate_question_vectors(self.questions)
        top_ids_and_scores = self.retriever.get_top_docs(
            questions_tensor.numpy(),
            self.cfg.n_docs,
            filter_pos=4,
            filter_value="1",
        )

        provenance = {}

        for record, query_id in tqdm(zip(top_ids_and_scores, self.query_ids)):
            docs_meta, scores = record
            element = []

            for score, meta in zip(scores, docs_meta):
                chunk_id, text, title, url = meta[:4]
                element.append(
                    {
                        "score": str(score),
                        "chunk_id": str(chunk_id),
                        "text": str(zlib.decompress(text).decode()),
                        "wikipedia_title": str(zlib.decompress(title).decode()),
                        "url": str(zlib.decompress(url).decode()),
                    }
                )

            assert query_id not in provenance
            provenance[query_id] = element

        return provenance
