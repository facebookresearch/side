# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import csv
import json
import logging
import multiprocessing
import os
import pickle
from multiprocessing.pool import ThreadPool
from pathlib import Path

import hydra
import jnius_config
from hydra import compose
from omegaconf import OmegaConf
from tqdm import tqdm

from evaluation.retrievers.base_retriever import Retriever

logging.basicConfig(level=logging.INFO)

def _run_thread(arguments):
    thread_id = arguments["id"]
    index = arguments["index_dir"]
    n_docs = arguments["n_docs"]
    queries = arguments["data"]

    # BM25 parameters #TODO
    # bm25_a = arguments["bm25_a"]
    # bm25_b = arguments["bm25_b"]
    # searcher.set_bm25(bm25_a, bm25_b)

    from pyserini.search import SimpleSearcher

    searcher = SimpleSearcher(index)

    if thread_id == 0:
        queries = tqdm(queries)

    provenance = {}
    for query in queries:
        query_id = query["id"]
        pyserini_query = (
            query["query"].strip()
        )

        hits = searcher.search(pyserini_query, n_docs)

        query_results = []
        for hit in hits:
            meta_json = json.loads(hit.docid)
            query_results.append(
                {
                    "score": hit.score,
                    "text": str(hit.raw).strip(),
                    "title": meta_json["title"],
                    "chunk_id": meta_json["id"],
                    "url": meta_json["url"],
                }
            )
        provenance[query_id] = query_results

    return provenance


# split a list in num parts evenly
def _chunk_it(seq, num):
    assert num > 0
    chunk_len = len(seq) // num
    chunks = [seq[i * chunk_len : i * chunk_len + chunk_len] for i in range(num)]

    diff = len(seq) - chunk_len * num  # 0 <= diff < num
    for i in range(diff):
        chunks[i].append(seq[chunk_len * num + i])

    return chunks


def _get_records(filename, id2url_map):
    anserini_records = []
    num_passages_stored = 0

    with open(filename, "r") as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t") 
        for row in tqdm(read_tsv):
            idx = str(row[0]).strip()
            passage = row[1]
            title = row[2]
            url = row[3]
            element = {
                "id": str(idx),
                "url": id2url_map[str(idx)] if id2url_map else str(url),
                "title": str(title),
                "sha": "",
            }
            anserini_records.append(
                {
                    "id": json.dumps(element),
                    "contents": "{} {}".format(title.strip(), passage.strip()),
                }
            )
            num_passages_stored += 1

    logging.info(
        "filename: {}\nnum_passages_stored: {}\n".format(
            filename,
            num_passages_stored,
        )
    )
    return anserini_records


def _append_records_on_file(cfg, NUM_TREADS, anserini_records):
    idx = 0
    for chunk in tqdm(_chunk_it(anserini_records, NUM_TREADS)):
        filename = cfg.retriever.tmp_dir + "data{}.jsonl".format(idx)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "a+") as outfile:
            for element in chunk:
                json.dump(element, outfile)
                outfile.write("\n")
        idx += 1


def create_index(cfg):
    logging.info(f"Create index in {cfg.retriever.index_dir}.")

    NUM_TREADS = cfg.retriever.indexing_threads
    if not NUM_TREADS:
        NUM_TREADS = os.cpu_count()

    id2url_map = None
    if cfg.retriever.id2url_map_pickle:
        logging.info("loading id2url")
        id2url_map = pickle.load(open(cfg.retriever.id2url_map_pickle, "rb"))

    print(cfg)
    logging.info("Reading {}".format(cfg.retriever.ctx_src))

    ctx_src = hydra.utils.instantiate(cfg.retriever.datasets[cfg.retriever.ctx_src], cfg, tensorizer=None)

    ctx_srcs = list()
    if hasattr(ctx_src, "file") and ctx_src.file is not None:
        ctx_srcs.append(ctx_src.file)
    elif hasattr(ctx_src, "files") and ctx_src.files is not None and len(ctx_src.files) > 0:
        ctx_srcs.extend(ctx_src.files)

    # 1. create data for anserini
    if len(ctx_srcs) > 0:
        # filenames = glob.glob(cfg.ctx_src + "/*.tsv")
        for i, filename in enumerate(ctx_srcs):
            logging.info("filename: {} [{}/{}]\nReading...".format(filename, i, len(ctx_srcs)))
            anserini_records = _get_records(filename, id2url_map)
            # 2. split in NUM_TREADS files
            logging.info(f"filename: {filename} [{i}/{len(ctx_srcs)}]\nAppending to {NUM_TREADS} temp files in {cfg.retriever.tmp_dir}")
            _append_records_on_file(cfg, NUM_TREADS, anserini_records)
    # elif os.path.isfile(cfg.ctx_src):
    #     anserini_records = _get_records(cfg.ctx_src, id2url_map)
    #     logging.info(f"Writing {NUM_TREADS} temp files in {cfg.tmp_dir}")
    #     _append_records_on_file(cfg, NUM_TREADS, anserini_records)
    else:
        raise ValueError("Unsupported input - It is a special file (socket, FIFO, device file)")

    # 2. create the BM25 index
    logging.info("Starting ingestion")
    os.system(
        f"python3 -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads {NUM_TREADS} -input {cfg.retriever.tmp_dir} -index {cfg.retriever.index_dir} -storePositions -storeDocvectors -storeRaw"
    )


class BM25(Retriever):

    def __init__(self, name, cfg):
        super().__init__(name)
        self.cfg = cfg
        print(OmegaConf.to_yaml(cfg))

        if self.cfg.retriever.Xms and self.cfg.retriever.Xmx:
            # to solve Insufficient memory for the Java Runtime Environment
            jnius_config.add_options(
                "-Xms{}".format(self.cfg.retriever.Xms), "-Xmx{}".format(self.cfg.retriever.Xmx), "-XX:-UseGCOverheadLimit"
            )
            logging.info("Configured options:", jnius_config.get_options())

        if not os.path.exists(self.cfg.retriever.index_dir):
            logging.info(f"No index found in {self.cfg.retriever.index_dir}.")
            create_index(self.cfg)

        self.num_threads = min(self.cfg.retriever.threads, int(multiprocessing.cpu_count()))

        # initialize a ranker per thread
        self.arguments = []
        for id in tqdm(range(self.num_threads)):
            self.arguments.append(
                {
                    "id": id,
                    "index_dir": self.cfg.retriever.index_dir,
                    "n_docs": self.cfg.n_docs,
                }
            )

    def preprocess_instance(self, datapoint):
        if self.cfg.retriever.question_transform_type == "sentence_1":
            datapoint["input"] = datapoint["meta"]["sentences"][-1]
        elif self.cfg.retriever.question_transform_type == "title+sentence_1":
            datapoint["input"] = datapoint["meta"]["wikipedia_title"] + " " + datapoint["meta"]["sentences"][-1]
        return datapoint

    def set_queries_data(self, queries_data, logger=None):
        chunked_queries = _chunk_it(queries_data, self.num_threads)
        for i, arg in enumerate(self.arguments):
            self.arguments[i]["data"] = chunked_queries[i]

    def run(self):
        pool = ThreadPool(self.num_threads)
        results = pool.map(_run_thread, self.arguments)

        provenance = {}
        for x in results:
            provenance.update(x)
        pool.terminate()
        pool.join()

        return provenance
