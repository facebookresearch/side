import logging
import os
import random
import socket
import tempfile
from typing import List, Union

import filelock
import torch
from omegaconf import DictConfig

from dpr.models.biencoder import QueryContextsBatch
from dpr.dataset.input_transform import SchemaPreprocessor, WaferQueryContextsRawText, WaferPreprocessor
from dpr.dataset.utils import find_or_download_files
from dpr.utils.data_utils import read_data_from_json_files, Dataset, Tensorizer
from dpr.utils.dist_utils import infer_slurm_init

logger = logging.getLogger(__name__)


class JsonDataset(Dataset):

    def __init__(
        self,
        file: str,
    ):
        super().__init__()
        self.file = file
        self.data_files = []

    def load_data(self, start_pos: int = -1, end_pos: int = -1, sharding_fn=None):
        # lock and wait for each process to load its shard to avoid swapping in DDP training on single nodes
        hostname = socket.gethostbyaddr(socket.gethostname())[0]
        max_concurrency = 4
        _, local_rank, world_size, _ = infer_slurm_init() 
        lock = filelock.FileLock(os.path.join(f"{tempfile.gettempdir()}", f".{hostname}.lock.{int(local_rank % max_concurrency)}"))
        with lock:
            if not self.data:
                self._load_all_data()
            start_pos, end_pos = sharding_fn(len(self.data))
            if start_pos >= 0 and end_pos >= 0:
                logger.info("Selecting subset range from %d to %d", start_pos, end_pos)
                self.data = self.data[start_pos:end_pos]

    def _load_all_data(self):
        self.data_files = find_or_download_files(self.file)
        logger.info("Data files: %s", self.data_files)
        data = read_data_from_json_files(self.data_files)
        # filter those without positive ctx
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info("Total cleaned data size: %d", len(self.data))

    def create_model_input(
            self,
            samples: List[WaferQueryContextsRawText],
            dataset: Union[Dataset, SchemaPreprocessor],
            cfg,
            is_training: bool,
    ):
        raise NotImplementedError

    def prepare_model_inputs(self, batch: dict):
        raise NotImplementedError


class JsonWaferDataset(JsonDataset, WaferPreprocessor):

    def __init__(
        self,
        cfg: DictConfig,
        tensorizer: Tensorizer,
        file: str,
        shuffle_negatives: bool=False,
    ):
        JsonDataset.__init__(
            self,
            file=file,
        )
        WaferPreprocessor.__init__(
            self,
            cfg=cfg,
            tensorizer=tensorizer,
        )
        self.shuffle_negatives = shuffle_negatives

    def __getitem__(self, index) -> WaferQueryContextsRawText:
        json_sample: dict = self.data[index]
        r = WaferQueryContextsRawText()
        r.query = json_sample["question"]

        positive_ctxs = json_sample["positive_ctxs"]

        negative_ctxs = json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
        negative_doc_ctxs = json_sample["negative_doc_ctxs"] if "negative_doc_ctxs" in json_sample else []
        hard_negative_ctxs = json_sample["hard_negative_ctxs"] if "hard_negative_ctxs" in json_sample else []
        hard_negative_doc_ctxs = json_sample["hard_negative_doc_ctxs"] if "hard_negative_doc_ctxs" in json_sample else []

        for ctx in positive_ctxs + negative_ctxs + negative_doc_ctxs + hard_negative_ctxs + hard_negative_doc_ctxs:
            if "title" not in ctx:
                ctx["title"] = None

        r.positive_passages = [self.passage_struct_from_dict(ctx) for ctx in positive_ctxs]
        r.negative_passages = [self.passage_struct_from_dict(ctx) for ctx in negative_ctxs]
        r.negative_doc_passages = [self.passage_struct_from_dict(ctx) for ctx in negative_doc_ctxs]
        r.hard_negative_passages = [self.passage_struct_from_dict(ctx) for ctx in hard_negative_ctxs]
        r.hard_negative_doc_passages = [self.passage_struct_from_dict(ctx) for ctx in hard_negative_doc_ctxs]
        return r

    def create_model_input(
            self,
            samples: List[WaferQueryContextsRawText],
            dataset: Union[Dataset, SchemaPreprocessor],
            cfg,
            is_training: bool,
    ) -> QueryContextsBatch:

        question_tensors = []
        ctx_tensors = []
        positive_and_negative_ctx_lens = []
        neg_ctx_indices = []

        for sample in samples:

            question = sample.query

            positive_ctx = sample.positive_passages
            neg_ctxs = sample.negative_passages
            neg_doc_ctxs = sample.negative_doc_passages
            hard_neg_ctxs = sample.hard_negative_passages
            hard_neg_doc_ctxs = sample.hard_negative_doc_passages

            #
            # Set the number of positive and negative contexts
            #

            if is_training:

                num_positives = cfg.train_iterator.num_positives
                other_negatives = cfg.train_iterator.other_negatives
                other_doc_negatives = cfg.train_iterator.other_doc_negatives
                hard_negatives = cfg.train_iterator.hard_negatives

            else:

                # if validation then reuse the train config if nothing is defined

                if cfg.valid_iterator.num_positives is not None:
                    num_positives = cfg.valid_iterator.num_positives
                else:
                    num_positives = cfg.train_iterator.num_positives

                if cfg.valid_iterator.other_negatives is not None:
                    other_negatives = cfg.valid_iterator.other_negatives
                else:
                    other_negatives = cfg.train_iterator.other_negatives

                if cfg.valid_iterator.other_doc_negatives is not None:
                    other_doc_negatives = cfg.valid_iterator.other_doc_negatives
                else:
                    other_doc_negatives = cfg.train_iterator.other_doc_negatives

                if cfg.valid_iterator.hard_negatives is not None:
                    hard_negatives = cfg.valid_iterator.hard_negatives
                else:
                    hard_negatives = cfg.train_iterator.hard_negatives


            #
            # Trim the available contexts down to the defined number of contexts
            #

            # positive contexts

            positive_ctx = positive_ctx[0:num_positives]

            # randomly sampled negative single contexts (one contexts from multiple documents)

            if is_training and cfg.train_iterator.shuffle_negatives and len(neg_ctxs) > 0:
                random.shuffle(neg_ctxs)
            if cfg.model_class in {
                "BERTRerankerCrossEncoder",
                "ColbertRerankerBiEncoder",
            }:
                neg_ctxs = neg_ctxs * 5  # TODO: hack for WAFER reranker dev to make up for instances with too few negatives
                neg_ctxs = neg_ctxs[0:other_negatives]
            else:
                neg_ctxs = neg_ctxs[0:other_negatives]

            # randomly sampled negative single contexts (multiple contexts from one document)

            if is_training and cfg.train_iterator.shuffle_negatives and len(neg_doc_ctxs) > 0:
                random.shuffle(neg_doc_ctxs)
            neg_doc_ctxs = neg_doc_ctxs[0:other_doc_negatives]

            # hard sampled negative single contexts (one contexts from multiple documents)

            if is_training and cfg.train_iterator.shuffle_negatives and len(hard_neg_ctxs) > 0:
                random.shuffle(hard_neg_ctxs)
            hard_neg_ctxs = hard_neg_ctxs[0:hard_negatives]

            # hard sampled negative document contexts (multiple contexts from one document)

            if is_training and cfg.train_iterator.shuffle_negatives and len(hard_neg_doc_ctxs) > 0:
                random.shuffle(hard_neg_doc_ctxs)
            hard_neg_doc_ctxs = hard_neg_doc_ctxs[0:cfg.train_iterator.hard_doc_negatives]

            #
            # Trim all negatives (other, hard, ...) by the number of positives
            # Thus the total number of contexts per instance is defined by the total number of negative contexts
            #
            ns = [neg_ctxs, neg_doc_ctxs, hard_neg_ctxs, hard_neg_doc_ctxs]
            p = len(positive_ctx)
            while True:
                for i in range(len(ns)):
                    if p > 0 and len(ns[i]) > 0:
                        ns[i] = ns[i][:-1]
                        p -= 1
                    if p == 0 or sum([len(n) for n in ns]) == 0:
                        break
                if p == 0 or sum([len(n) for n in ns]) == 0:
                    break

            #
            # Combine all contexts
            #

            neg_ctxs, neg_doc_ctxs, hard_neg_ctxs, hard_neg_doc_ctxs = ns
            all_ctxs = positive_ctx + neg_ctxs + neg_doc_ctxs + hard_neg_ctxs + hard_neg_doc_ctxs

            #
            # Compute offsets
            #

            current_ctxs_len = len(ctx_tensors)
            positive_ctx_len = len(positive_ctx)
            neg_ctxs_len = len(neg_ctxs) + len(neg_doc_ctxs) + len(hard_neg_ctxs) + len(hard_neg_doc_ctxs)

            negatives_start_idx = positive_ctx_len
            negatives_end_idx = positive_ctx_len + neg_ctxs_len

            #
            # Preprocess contexts
            #

            sample_ctxs_tensors = [
                dataset.preprocess_passage(ctx, is_training)
                for ctx in all_ctxs
            ]

            #
            # Finalize
            #

            ctx_tensors.extend(sample_ctxs_tensors)

            positive_and_negative_ctx_lens.append([positive_ctx_len, neg_ctxs_len])
            neg_ctx_indices.append(
                [
                    i
                    for i in range(
                    current_ctxs_len + negatives_start_idx,
                    current_ctxs_len + negatives_end_idx,
                )
                ]
            )

            question_tensors.append(dataset.preprocess_query(question, is_training))

        #
        # Create batch tensor
        #

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        #
        # Create batch tensor
        #

        if cfg.n_gpu > 0:
            divider = cfg.n_gpu
        else:
            divider = 1
        truncate_by = ctxs_tensor.size(0) % divider
        if truncate_by > 0 and cfg.model_class in {
            "BERTRerankerCrossEncoder",
            "ColbertRerankerBiEncoder",
        }:
            ctxs_tensor = ctxs_tensor[:-truncate_by]

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        ctxt_total_len = sum([sum(l) for l in positive_and_negative_ctx_lens])
        positive_ctx_indices = []
        offset = 0

        for positive_and_negative_ctx_len in positive_and_negative_ctx_lens:
            pos_len, neg_len = positive_and_negative_ctx_len
            next_offset = offset + sum(positive_and_negative_ctx_len)
            if cfg.model_class in {
                "BERTRerankerCrossEncoder",
                "ColbertRerankerBiEncoder",
            } or cfg.model_class == "DocumentBiEncoder" and self.cfg.distributed_world_size > 1:
                positive_ctx_indices.append([True] * pos_len + [False] * neg_len)
            else:
                positive_ctx_indices.append(
                    [False] * offset + [True] * pos_len + [False] * neg_len + [False] * (ctxt_total_len - next_offset))
            offset = next_offset

        positive_ctx_indices = torch.tensor(positive_ctx_indices, dtype=bool)

        if truncate_by > 0 and cfg.model_class in {
            "BERTRerankerCrossEncoder",
            "ColbertRerankerBiEncoder",
        }:
            positive_ctx_indices = positive_ctx_indices[:, -truncate_by]

        return QueryContextsBatch(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            neg_ctx_indices,
            "question",
        )
