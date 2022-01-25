# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import glob
import logging

import hydra
from omegaconf import OmegaConf

from dpr.dense_retriever import LocalFaissRetriever, get_all_passages
from dpr.indexer.faiss_indexers import DenseFlatIndexer, DenseHNSWFlatIndexer
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
        print(OmegaConf.to_yaml(cfg))

        _ckpt_cfg = OmegaConf.load(f"{cfg.retriever.checkpoint_dir}/.hydra/config.yaml")
        self.checkpoint_cfg = setup_cfg_gpu(_ckpt_cfg)
        self.cfg = cfg

        assert int(self.cfg.n_docs) > 0

        logger.info("CFG (after gpu  configuration):")
        logger.info("%s", OmegaConf.to_yaml(cfg))

        saved_state = load_states_from_checkpoint(f"{cfg.retriever.model_file}.generate_embeddings")
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
            key[prefix_len:]: value
            for (key, value) in saved_state.model_dict.items()
            if key.startswith(encoder_prefix)
        }
        model_to_load.load_state_dict(question_encoder_state)
        vector_size = model_to_load.get_out_size()
        logger.info("Encoder vector_size=%d", vector_size)

        ctx_sources = []
        ctx_src = hydra.utils.instantiate(cfg.retriever.datasets[cfg.retriever.ctx_src], self.checkpoint_cfg, tensorizer=tensorizer)
        ctx_sources.append(ctx_src)
        self.all_passages = get_all_passages(ctx_sources)

        if self.cfg.retriever.hnsw_index:
            index = DenseHNSWFlatIndexer(vector_size)
            index.deserialize(cfg.retriever.hnsw_index_path)
            self.retriever = LocalFaissRetriever(
                encoder, cfg.retriever.batch_size, ctx_src, index
            )
        else:
            logger.info("Reading from encoded vectors=%s", f"{cfg.retriever.out_file}_[0-9]*")
            # index all passages
            index = DenseFlatIndexer()
            index.init_index(vector_sz=vector_size)
            self.retriever = LocalFaissRetriever(
                encoder, cfg.retriever.batch_size, ctx_src, index
            )
            self.retriever.index_encoded_data(
                vector_files=glob.glob(f"{cfg.retriever.out_file}_[0-9]*"),
                buffer_size=index.buffer_size,
            )

    def set_queries_data(self, queries_data):
        # get questions & answers
        self.questions = [
            x["query"].strip()
            for x in queries_data
        ]
        self.query_ids = [x["id"] for x in queries_data]

    def run(self):

        questions_tensor = self.retriever.generate_question_vectors(self.questions)

        top_ids_and_scores = self.retriever.get_top_docs(
            questions_tensor.numpy(), int(self.cfg.n_docs)
        )

        provenance = {}

        for record, query_id in zip(top_ids_and_scores, self.query_ids):
            top_ids, scores = record
            element = []

            # sort by score in descending order
            for score, id in sorted(zip(scores, top_ids), reverse=True):

                text = self.all_passages[id][0]
                index = self.all_passages[id][1]
                url = self.all_passages[id][2]

                wikipedia_id = None

                element.append(
                    {
                        "score": str(score),
                        "text": str(text),
                        "wikipedia_title": str(index),
                        "wikipedia_id": str(wikipedia_id),
                        "chunk_id": id,
                        "url": url,
                    }
                )

            assert query_id not in provenance
            provenance[query_id] = element

        return provenance