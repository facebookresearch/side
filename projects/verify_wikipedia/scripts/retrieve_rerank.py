# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import logging
import math
import os
import sys

import filelock
import hydra
from omegaconf import OmegaConf

from evaluation.retrievers.base_retriever import Retriever
from misc import utils
from misc.utils import load_options_from_argv_yaml


def setup_logger(logger):
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_formatter = logging.Formatter("[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(log_formatter)
    logger.addHandler(console)

logger = logging.getLogger()
setup_logger(logger)


def output_file_name(output_folder, dataset_file, output_suffix=""):
    if not output_suffix:
        output_suffix = ""
    basename = os.path.basename(dataset_file)
    output_file = os.path.join(output_folder, basename) + output_suffix
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    return output_file

def apply_ranker(
    test_config,
    ranker: Retriever,
    logger,
    debug=False,
    output_folder="",
    num_shards=1,
    shard_id=0,
):

    for dataset in test_config.evaluation_datasets:
        dataset = hydra.utils.instantiate(test_config.datasets[dataset])

        logger.info("TASK: {}".format(dataset.task_family))
        logger.info("DATASET: {}".format(dataset.name))

        output_file = output_file_name(output_folder, dataset.file, test_config.output_suffix)
        if os.path.exists(output_file):
            logger.info(
                "Skip output file {} that already exists.".format(output_file)
            )
            continue

        raw_data = utils.load_data(dataset.file)
        validated_data = {}
        queries_data = list()
        for instance in raw_data:
            if dataset.validate_datapoint(instance, logger=logger):
                instance = ranker.preprocess_instance(instance)
                if instance["id"] in validated_data:
                    raise ValueError("ids are not unique in input data!")
                validated_data[instance["id"]] = instance
                queries_data.append(
                    {"query": instance["input"], "id": instance["id"]}
                )

        queries_data = ranker.get_queries_data(queries_data)

        if debug:
            # just consider the top10 datapoints
            queries_data = queries_data[:10]
            print("query_data: {}", format(queries_data))

        if num_shards > 1:
            len_all_query_ctxts = len(queries_data)
            shard_size = math.ceil(len_all_query_ctxts / num_shards)
            start_idx = shard_id * shard_size
            end_idx = start_idx + shard_size
            queries_data = queries_data[start_idx:end_idx]
            logger.info(f"sharded query_ctxy size: {len(queries_data)}")
        else:
            logger.info(f"query_ctxy size: {len(queries_data)}")

        ranker.set_queries_data(queries_data)

        # get predictions
        provenance = ranker.run()

        if len(provenance) != len(queries_data):
            logger.warning(
                "different numbers of queries: {} and predictions: {}".format(
                    len(queries_data), len(provenance)
                )
            )

        # write prediction files
        if provenance:
            logger.info("writing prediction file to {}".format(output_file))

            predictions = []
            for query_id in provenance.keys():
                if query_id in validated_data:
                    instance = validated_data[query_id]
                    new_output = [{"provenance": provenance[query_id]}]
                    # append the answers
                    if "output" in instance:
                        for o in instance["output"]:
                            if "answer" in o:
                                new_output.append({"answer": o["answer"]})
                    instance["output"] = new_output
                    predictions.append(instance)

            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)

            if num_shards > 1:

                lock = filelock.FileLock(output_file_name(output_folder, dataset.file, ".lock"))
                with lock:
                    with open(output_file, "w+") as outfile:
                        for p in predictions:
                            json.dump(p, outfile)
                            outfile.write("\n")
                    output_files = glob.glob(output_file_name(output_folder, dataset.file, ".[0-9]*-[0-9]*"))
                    if len(output_files) == num_shards:
                        sorted(output_files)
                        with open(output_file_name(output_folder, dataset.file, ""), "w") as cat:
                            for output_file in output_files:
                                cat.writelines(open(output_file))

            else:

                with open(output_file, "w+") as outfile:
                    for p in predictions:
                        json.dump(p, outfile)
                        outfile.write("\n")


@hydra.main(config_path="../conf", config_name="retrieval")
def main(cfg):

    # logger.info("loading {} ...".format(OmegaConf.to_yaml(cfg)))

    shard_id = int(os.environ.get("SLURM_ARRAY_TASK_ID")) if os.environ.get("SLURM_ARRAY_TASK_ID") else 1
    num_shards = int(os.environ.get("SLURM_ARRAY_TASK_COUNT")) if os.environ.get("SLURM_ARRAY_TASK_COUNT") else 1
    task_id = int(os.environ.get("SLURM_ARRAY_JOB_ID")) if os.environ.get("SLURM_ARRAY_JOB_ID") else cfg.task_id

    if num_shards > 1:
        cfg.output_suffix = f".{shard_id:02}-{num_shards}"

    module_name = '.'.join(cfg.retriever._target_.split('.')[:-1])
    clazz_name = cfg.retriever._target_.split('.')[-1]
    retriever_module = __import__(name=module_name, fromlist=clazz_name)
    retriever_clazz = getattr(retriever_module, clazz_name)
    retriever = retriever_clazz(clazz_name, cfg)

    if cfg.output_folder is None or len(cfg.output_folder) == 0:
        cfg.output_folder = "./"

    if cfg.output_folder_in_checkpoint_dir:
        output_folder = f"{cfg.output_folder}__{task_id}"
    else:
        output_folder = cfg.output_folder

    apply_ranker(
        test_config=cfg,
        ranker=retriever,
        logger=logger,
        output_folder=output_folder,
        num_shards=num_shards,
        shard_id=shard_id,
    )

    OmegaConf.save(cfg, f"{output_folder}/retriever_cfg.yaml")
    if hasattr(retriever, "checkpoint_cfg"):
        OmegaConf.save(retriever.checkpoint_cfg, f"{output_folder}/checkpoint_cfg.yaml")


if __name__ == "__main__":
    load_options_from_argv_yaml()
    logger.info("Sys.argv: %s", sys.argv)
    main()
