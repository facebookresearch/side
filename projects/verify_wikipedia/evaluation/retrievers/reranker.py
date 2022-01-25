# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os.path
from collections import OrderedDict
from pathlib import Path
import torch
import json

import hydra
from omegaconf import OmegaConf
import tqdm

from dpr.models import init_biencoder_components
from dpr.models.biencoder import RankerLoss
from dpr.options import set_cfg_params_from_state
from dpr.dataset.input_transform import SchemaPreprocessor
from dpr.utils.dist_utils import setup_cfg_gpu
from dpr.utils.model_utils import (
    setup_fp16_and_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint, move_to_device,
)
from evaluation.retrievers.base_retriever import Retriever
from scripts.train import gather_to_main_device

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_ctxt_encoder_input_in_dpr_schema(VALID_SPLIT_DATA_FILES__RETRIEVERS, gold_queries_data, nr_ctxs, strict=False):

    out = OrderedDict()

    for i, instance in enumerate(gold_queries_data):
        if instance["id"] not in out:
            out[instance["id"]] = {
                "id": instance["id"],
                "question": instance["query"],
                "ctxs": list(),
                "provenances": list(),
                "evaluation": set()
            }

    for retriever_sys, valid_split_data_file in VALID_SPLIT_DATA_FILES__RETRIEVERS:
        with open(valid_split_data_file) as f:

            for i, line in enumerate(tqdm.tqdm(f)):

                jline_sys = json.loads(line)

                if jline_sys["id"] not in out:
                    # logger.info(f"Skipping instance '{jline_sys['id']}' which is not in gold")
                    continue

                for output in jline_sys["output"]:
                    if "provenance" not in output:
                        continue
                    if len(output["provenance"]) == 0:
                        continue
                    if len(output["provenance"]) == 0 or "text" not in output["provenance"][0]:
                        continue

                    # rand_nr = random.randint(5, nr_ctxs)

                    for i, provenance in enumerate(output["provenance"][:nr_ctxs]):
                        text, url = provenance["text"], provenance["url"],

                        if "wikipedia_title" in provenance:
                            title = provenance["wikipedia_title"]
                        else:
                            title = provenance["title"]

                        if "chunk_id" in provenance:
                            chunk_id = provenance["chunk_id"]
                        elif "sha" in provenance:
                            chunk_id = provenance["sha"]
                        else:
                            chunk_id = None

                        out[jline_sys["id"]]["ctxs"].append(
                            {
                                "id": chunk_id,
                                "doc_id": 0,
                                "passage_id": 0,
                                "url": url,
                                "title": title,
                                "text": text,
                                "system_hit": retriever_sys,
                                "rank": f"[RANK:{i}]",
                            }
                        )
                        provenance["retriever_sys"] = retriever_sys
                        out[jline_sys["id"]]["provenances"].append(provenance)

                out[jline_sys["id"]]["evaluation"].add(retriever_sys)

    before_filter = len(out)
    logger.info(f"query_ctx size before filter: {before_filter}")
    out = list(
        filter(
            lambda x: len(x["evaluation"]) == len(VALID_SPLIT_DATA_FILES__RETRIEVERS),
            list(out.values())
        )
    )
    after_filter = len(out)
    logger.info(f"query_ctxy size after filter: {after_filter}")

    if strict:
        assert after_filter == before_filter

    return out


class Reranker(Retriever):

    def __init__(self, name, cfg):
        super().__init__(name)

        self.cfg = cfg
        _ckpt_cfg = OmegaConf.load(f"{cfg.retriever.checkpoint_dir}/.hydra/config.yaml")
        self.checkpoint_cfg = setup_cfg_gpu(_ckpt_cfg)

        logger.info("CFG (after gpu  configuration):")
        logger.info("%s", OmegaConf.to_yaml(cfg))

        saved_state = load_states_from_checkpoint(f"{cfg.retriever.model_file}")
        set_cfg_params_from_state(saved_state.encoder_params, self.checkpoint_cfg)

        tensorizer, reranker, _ = init_biencoder_components(
            self.checkpoint_cfg.base_model.encoder_model_type, self.checkpoint_cfg, inference_only=True
        )

        self.tensorizer = tensorizer
        self.ctx_src : SchemaPreprocessor = hydra.utils.instantiate(cfg.retriever.datasets[cfg.retriever.ctx_src], self.checkpoint_cfg, tensorizer=tensorizer)
        self.loss_func : RankerLoss = reranker.get_loss_function()
        self.biencoder_prepare_model_inputs = reranker.prepare_model_inputs

        self.reranker, _ = setup_fp16_and_distributed_mode(
            reranker,
            None,
            self.checkpoint_cfg.device,
            self.checkpoint_cfg.n_gpu,
            self.checkpoint_cfg.local_rank,
            self.checkpoint_cfg.fp16,
        )
        self.reranker.eval()

        # load weights from the model file
        logger.info("Loading saved model state ...")
        reranker = get_model_obj(self.reranker)
        reranker.load_state(saved_state, strict=True)

        self.query_ctxts = None
        self.annotate_interactions = self.cfg.retriever.retrieved_data_nr_ctxs == 1

        self.cfg.retriever.retrieved_data_files = [s.strip().split(":") for s in self.cfg.retriever.retrieved_data_files]

        # self.annotate_interactions = False

    def get_queries_data(self, gold_queries_data):
        return get_ctxt_encoder_input_in_dpr_schema(
            self.cfg.retriever.retrieved_data_files,
            gold_queries_data=gold_queries_data,
            nr_ctxs=self.cfg.retriever.retrieved_data_nr_ctxs,
        )

    def set_queries_data(self, queries_data):
        self.query_ctxts = queries_data

    def run(self):

        result = {}

        def collate(batch):
            """
            Collates a list of batch items into processed tensors for a bienencoder model to consume.
            :param batch:
            :return:
            """
            batch_inputs = list()
            batch_query_ctxs = list()
            orig_lens = list()

            for query_ctxs in batch:
                question_ids = self.ctx_src.preprocess_query(query_ctxs["question"], training=False)
                ctx_ids = torch.stack(
                    [self.ctx_src.preprocess_passage(self.ctx_src.passage_struct_from_dict(p), training=False) for p in
                     query_ctxs["ctxs"]], dim=0)
                ctx_ids_size = ctx_ids.size(0)
                orig_lens.append(ctx_ids_size)
                should_ctx_ids_size = len(self.cfg.retriever.retrieved_data_files) * self.cfg.retriever.retrieved_data_nr_ctxs
                if should_ctx_ids_size > ctx_ids_size:
                    padding = torch.zeros_like(ctx_ids[0:(should_ctx_ids_size - ctx_ids_size), :])
                    ctx_ids = torch.cat([ctx_ids, padding], dim=0)
                inputs = {
                    "question_ids": question_ids,
                    "question_segments": torch.zeros_like(question_ids),
                    "context_ids": ctx_ids,
                    "ctx_segments": torch.zeros_like(ctx_ids),
                }
                batch_inputs.append(inputs)
                batch_query_ctxs.append(query_ctxs)

            # pad the the last batch to play well with gather() from multi-GPU
            while len(batch_inputs) < self.checkpoint_cfg.n_gpu*multiplier:
                batch_inputs.append(
                    {
                        "question_ids": torch.zeros_like(batch_inputs[-1]["question_ids"]),
                        "question_segments": torch.zeros_like(batch_inputs[-1]["question_segments"]),
                        "context_ids": torch.zeros_like(batch_inputs[-1]["context_ids"]),
                        "ctx_segments": torch.zeros_like(batch_inputs[-1]["ctx_segments"]),
                    }
                )

            splits = [len(bi["context_ids"]) for bi in batch_inputs]
            batch_inputs_merged = {k: torch.cat([bi[k].view(-1, bi[k].size(-1)) for bi in batch_inputs], dim=0) for k in batch_inputs[0].keys()}
            batch_inputs_merged = self.biencoder_prepare_model_inputs(batch_inputs_merged, self.ctx_src)

            return orig_lens, splits, batch_inputs_merged, batch_query_ctxs

        multiplier = 1 #TODO: hardcoded nr of batch items per GPU (with nr_of_ctxs passages, atm hardcoded 100)
        loader = torch.utils.data.DataLoader(self.query_ctxts, batch_size=max(1, self.checkpoint_cfg.n_gpu*multiplier), collate_fn=collate)

        for orig_lens, splits, batch_inputs_merged, batch_query_ctxs in tqdm.tqdm(loader):

            batch_inputs_merged = move_to_device(batch_inputs_merged, self.checkpoint_cfg.device)

            interactions = None

            with torch.no_grad():
                q_out, ctx_out = self.reranker(
                        **batch_inputs_merged,
                        encoder_type="question",
                        representation_token_pos=0,
                    )
                if self.annotate_interactions and hasattr(get_model_obj(self.reranker), "get_interactions"):
                    interactions = get_model_obj(self.reranker).get_interactions(
                        **batch_inputs_merged,
                        encoder_type="question",
                        representation_token_pos=0,
                    )
                    interactions = interactions.detach().cpu().tolist()

            q_out, ctx_out, _, _ = gather_to_main_device(
                self.checkpoint_cfg,
                q_out,
                ctx_out,
            )

            if len(q_out.size()) > 1:
                q_out_splits = torch.split(q_out, 1)
                ctx_out_splits = torch.split(ctx_out, splits)
            else:
                # crossencoder sends fake output with size len(q_out.size()) == 1
                q_out_splits = torch.split(q_out, 1)
                if ctx_out.size(0) * ctx_out.size(1) == sum(splits):
                    ctx_out_splits = torch.split(ctx_out.view(ctx_out.size(0) * ctx_out.size(1), -1), splits) if len(ctx_out.size()) > 1 else [None] * len(splits)
                else:
                    logger.warning(f"{ctx_out.size(0)}*{ctx_out.size(1)} = {ctx_out.size(0) * ctx_out.size(1)} != {sum(splits)} {splits}")
                    continue

            assert len(q_out_splits) == len(ctx_out_splits)
            scores_chunks = list()

            for q_out, ctx_out in zip(q_out_splits, ctx_out_splits):
                loss_func_get_scores = self.loss_func.get_scores(q_out, ctx_out).view(-1)
                scores_chunks.append(loss_func_get_scores)

            # loss_func_get_scores = self.loss_func.get_scores(q_out, ctx_out).view(-1).cpu()
            # print(loss_func_get_scores.size())
            # scores_chunks = [sc.tolist() for sc in torch.split(loss_func_get_scores, 1)]

            for scores_chunk, query_ctxs, orig_len in zip(scores_chunks, batch_query_ctxs, orig_lens):
                outputs = list()
                assert orig_len == len(query_ctxs["provenances"]), f"scores_chunk={scores_chunk.size()} orig_len={orig_len} len(provenances)={len(query_ctxs['provenances'])}"
                scores_chunk = scores_chunk[:orig_len]
                # sort by score in descending order
                for score, provenance_id in sorted(zip(scores_chunk, list(range(len(scores_chunk)))), reverse=True):
                    provenance = query_ctxs["provenances"][provenance_id]
                    provenance["score"] = score.item()
                    if interactions:
                        provenance["question_ids"] = batch_inputs_merged["question_ids"].detach().cpu().tolist()
                        provenance["context_ids"] = batch_inputs_merged["context_ids"].detach().cpu().tolist()
                        provenance["interactions"] = interactions
                    outputs.append(provenance)
                result[query_ctxs["id"]] = outputs

        return result