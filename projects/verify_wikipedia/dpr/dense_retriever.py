#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import logging
import pickle
import time
from typing import List, Tuple, Union, Iterator

import numpy as np
import torch
import tqdm
from torch import Tensor as T
from torch import nn

from dpr.indexer.faiss_indexers import (
    DenseIndexer,
)
from dpr.models.biencoder import (
    BiEncoder,
)
from dpr.options import setup_logger
from dpr.dataset.input_transform import SchemaPreprocessor
from dpr.dataset.retrieval import ContextSource

logger = logging.getLogger()
setup_logger(logger)


def generate_question_vectors(
    question_encoder: torch.nn.Module,
    qa_src: Union[ContextSource, SchemaPreprocessor],
    questions: List[str],
    bsz: int,
) -> T:
    n = len(questions)
    query_vectors = []
    selector = qa_src.selector
    with torch.no_grad():
        for j, batch_start in enumerate(range(0, n, bsz)):
            batch_questions = questions[batch_start : batch_start + bsz]
            batch_tensors = [qa_src.preprocess_query(q) for q in batch_questions]

            q_ids_batch = torch.stack(batch_tensors, dim=0).cuda()
            q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
            q_attn_mask = qa_src.tensorizer.get_attn_mask(q_ids_batch)

            if selector:
                rep_positions = selector.get_positions(q_ids_batch, qa_src.tensorizer)

                _, out, _ = BiEncoder.get_representation(
                    question_encoder,
                    q_ids_batch,
                    q_seg_batch,
                    q_attn_mask,
                    representation_token_pos=rep_positions,
                )
            else:
                _, out, _ = question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

            query_vectors.extend(out.cpu().split(1, dim=0))

            if len(query_vectors) % 100 == 0:
                logger.info("Encoded queries %d", len(query_vectors))

    query_tensor = torch.cat(query_vectors, dim=0)
    logger.info("Total encoded queries tensor %s", query_tensor.size())
    assert query_tensor.size(0) == len(questions)
    return query_tensor


class DenseRetriever(object):
    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        qa_src: Union[ContextSource, SchemaPreprocessor],
    ):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.qa_src = qa_src

    def generate_question_vectors(self, questions: List[str]) -> T:

        bsz = self.batch_size
        self.question_encoder.eval()
        return generate_question_vectors(
            self.question_encoder,
            self.qa_src,
            questions,
            bsz,
        )


class LocalFaissRetriever(DenseRetriever):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        qa_src: SchemaPreprocessor,
        index: DenseIndexer,
    ):
        super().__init__(question_encoder, batch_size, qa_src)
        self.index = index

    def index_encoded_data(
        self,
        vector_files: List[str],
        buffer_size: int,
        path_id_prefixes: List = None,
    ):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        buffer = []
        for i, item in enumerate(iterate_encoded_files(vector_files, path_id_prefixes=path_id_prefixes)):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        logger.info("Data indexing completed.")

    def get_top_docs(self, query_vectors: np.array, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info("index search time: %f sec.", time.time() - time0)
        # self.index = None
        return results


class DenseRPCRetriever(DenseRetriever):
    def __init__(
        self,
        question_encoder: nn.Module,
        qa_src: SchemaPreprocessor,
        batch_size: int,
        index_cfg_path: str,
        dim: int,
        use_l2_conversion: bool = False,
        nprobe: int = 256,
    ):
        # from distributed_faiss.client import IndexClient // using a copy because of a small bugfix TODO make PR to df
        from dpr.indexer.client import IndexClient
        from distributed_faiss.index_cfg import IndexCfg

        # super().__init__(question_encoder, batch_size, tensorizer)
        super().__init__(question_encoder, batch_size, qa_src)
        self.dim = dim
        self.index_id = "dr"
        self.nprobe = nprobe
        logger.info("Connecting to index server ...")
        self.index_client = IndexClient(index_cfg_path)
        self.use_l2_conversion = use_l2_conversion
        logger.info("Connected")

    def load_index(self, index_id):
        from distributed_faiss.index_cfg import IndexCfg

        self.index_id = index_id

        logger.info("Loading remote index %s", index_id)

        idx_cfg = IndexCfg()
        idx_cfg.nprobe = self.nprobe
        if self.use_l2_conversion:
            idx_cfg.metric = "l2"

        self.index_client.load_index(self.index_id, cfg=idx_cfg, force_reload=False)
        logger.info("Index loaded")
        self._wait_index_ready(index_id)

    def index_encoded_data(
        self,
        vector_files: List[str],
        buffer_size: int = 1000,
        path_id_prefixes: List = None,
    ):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        from distributed_faiss.index_cfg import IndexCfg

        buffer = []
        idx_cfg = IndexCfg()

        idx_cfg.dim = self.dim
        logger.info("Index train num=%d", idx_cfg.train_num)
        idx_cfg.faiss_factory = "flat"
        index_id = self.index_id
        self.index_client.create_index(index_id, idx_cfg)

        def send_buf_data(buffer, index_client):
            buffer_vectors = [np.reshape(encoded_item[1], (1, -1)) for encoded_item in buffer]
            buffer_vectors = np.concatenate(buffer_vectors, axis=0)
            meta = [encoded_item[0] for encoded_item in buffer]
            index_client.add_index_data(index_id, buffer_vectors, meta)

        for i, item in enumerate(iterate_encoded_files(vector_files, path_id_prefixes=path_id_prefixes)):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                send_buf_data(buffer, self.index_client)
                buffer = []
        if buffer:
            send_buf_data(buffer, self.index_client)
        logger.info("Embeddings sent.")
        self._wait_index_ready(index_id)

    def get_top_docs(
        self, query_vectors: np.array, top_docs: int = 100, search_batch: int = 512, filter_pos=3, filter_value=True
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        if self.use_l2_conversion:
            aux_dim = np.zeros(len(query_vectors), dtype="float32")
            query_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
            logger.info("query_hnsw_vectors %s", query_vectors.shape)
            self.index_client.cfg.metric = "l2"

        results = []
        for i in tqdm.tqdm(range(0, query_vectors.shape[0], search_batch)):
            time0 = time.time()
            query_batch = query_vectors[i : i + search_batch]
            logger.info("query_batch: %s", query_batch.shape)
            # scores, meta = self.index_client.search(query_batch, top_docs, self.index_id)

            scores, meta = self.index_client.search_with_filter(
                query_batch, top_docs, self.index_id, filter_pos=filter_pos, filter_value=filter_value
            )

            logger.info("index search time: %f sec.", time.time() - time0)
            results.extend([(meta[q], scores[q]) for q in range(len(scores))])
        return results

    def _wait_index_ready(self, index_id: str):
        from distributed_faiss.index_state import IndexState

        # TODO: move this method into IndexClient class
        while self.index_client.get_state(index_id) != IndexState.TRAINED:
            time.sleep(10)
        logger.info(
            "Remote Index is ready. Index data size %d",
            self.index_client.get_ntotal(index_id),
        )


def get_all_passages(ctx_sources):
    try:

        all_passages = {}
        for ctx_src in ctx_sources:
            ctx_src.load_data_to(all_passages)
            logger.info("Loaded ctx data: %d", len(all_passages))

        if len(all_passages) == 0:
            raise RuntimeError("No passages data found. Please specify ctx_file param properly.")

        return all_passages

    except NotImplementedError:

        for ctx_src in ctx_sources:
            ctx_src.load_data()
            logger.info(f"Loaded ctx data: {ctx_src}")


def iterate_encoded_files(vector_files: list, path_id_prefixes: List = None) -> Iterator[Tuple]:
    for i, file in enumerate(vector_files):
        logger.info("Reading file %s", file)
        id_prefix = None
        if path_id_prefixes:
            id_prefix = path_id_prefixes[i]
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                doc = list(doc)
                if id_prefix and not str(doc[0]).startswith(id_prefix):
                    doc[0] = id_prefix + str(doc[0])
                yield doc
