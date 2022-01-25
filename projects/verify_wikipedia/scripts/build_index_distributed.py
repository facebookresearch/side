"""
Script allows to build a faiss index (either remotely on ditributed-faiss or locally)
from a set of embedding files.

Adapted from https://github.com/fairinternal/Sphere/blob/main/scripts/build_index_distributed.py

"""

import argparse
import faiss
import glob
import logging
import numpy as np
import pickle
import time

# from distributed_faiss.client import IndexClient
from dpr.indexer.client import IndexClient
from distributed_faiss.index import faiss_special_index_factories
from distributed_faiss.index_cfg import IndexCfg
from distributed_faiss.index_state import IndexState


logging.basicConfig(level=logging.INFO)


class RemoteIndex(object):
    def __init__(
        self,
        index_factory: str,
        discovery_config,
        index_id,
        l2_space_conversion: bool = True,
    ):
        self.index_client = IndexClient(discovery_config)
        self.index_id = index_factory.replace(",", "_") + "_" + index_id
        idx_cfg = IndexCfg()

        if index_factory in faiss_special_index_factories:
            idx_cfg.index_builder_type = index_factory
        else:
            idx_cfg.faiss_factory = index_factory

        # TODO: read vector size from embeddings
        idx_cfg.dim = 769 if l2_space_conversion else 768
        idx_cfg.metric = "l2" if "IVF" in index_factory or l2_space_conversion else "dot"

        # TODO: parametrize instead of hardcoding
        idx_cfg.train_num = 50000
        idx_cfg.centroids = 0
        idx_cfg.nprobe = 128

        idx_cfg.save_interval_sec = 2 * 60 * 60

        # Dropping index before creation might be useful at early stage of experimentation.
        if args.drop_index:
            logging.info(f"Drop index")
            self.index_client.drop_index(self.index_id)

        if not self.index_client.load_index(self.index_id, cfg=idx_cfg, force_reload=False):
            logging.info(f"Creating new index")
            self.index_client.create_index(self.index_id, idx_cfg)
        else:
            total = self.index_client.get_ntotal(self.index_id)
            logging.info(f"Total index ready data: {total}")

    def add(self, hnsw_vectors, metadata):
        self.index_client.add_index_data(
            self.index_id, hnsw_vectors, metadata, train_async_if_triggered=True
        )

    def save(self, dest=None):
        logging.info("Saving the index")
        self.index_client.save_index(self.index_id)

    def wait_for_index(self) -> int:
        while self.index_client.get_state(self.index_id) != IndexState.TRAINED:
            logging.info(
                "Remote Index is not ready. State: {}. Size: {}".format(
                    self.index_client.get_state(self.index_id),
                    self.index_client.get_ntotal(self.index_id),
                )
            )
            time.sleep(60)
            self.index_client.add_buffer_to_index(self.index_id)
            total = self.index_client.get_ntotal(self.index_id)
            logging.info(f"Total index ready data: {total}")
        total = self.index_client.get_ntotal(self.index_id)
        logging.info(f"Total index ready data: {total}")
        return total


class LocalIndex(object):
    def __init__(self, index_factory):
        """dim = 769 if "IVF" in index_factory else 768
        metric = (
            faiss.METRIC_L2 if "IVF" in index_factory else faiss.METRIC_INNER_PRODUCT
        )
        """
        dim = 768
        metric = faiss.METRIC_INNER_PRODUCT
        self.index = faiss.index_factory(dim, index_factory, metric)

    def add(self, hnsw_vectors, metadata=None):
        self.index.add(hnsw_vectors)

    def save(self, dest):
        faiss.write_index(self.index, dest)

    def wait_for_index(self):
        pass


def get_phi_for_hnsw(metadata_path):
    phi = 0
    metas = glob.glob(metadata_path)
    for meta in metas:
        with open(meta, "rb") as reader:
            metadata = pickle.load(reader)
            if metadata["phi"] > phi:
                phi = metadata["phi"]
    return phi


def get_hnsw_vectors(vector_chunk, phi):
    norms = [(doc_vector ** 2).sum() for doc_vector in vector_chunk]
    for norm in norms:
        if phi < norm:
            raise ValueError("Vector norm larger than phi")
    aux_dims = [np.sqrt(phi - norm) for norm in norms]
    hnsw_vectors = [
        np.hstack((doc_vector.reshape(1, -1), aux_dims[i].reshape(-1, 1)))
        for i, doc_vector in enumerate(vector_chunk)
    ]
    hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)
    return hnsw_vectors


def main(args):

    index = (
        LocalIndex(args.index_factory)
        if args.type == "local"
        else RemoteIndex(
            args.index_factory,
            args.discovery_config,
            args.index_id,
            l2_space_conversion=args.hnsw,
        )
    )

    # Ids already added to the index
    indexed_ids = index.index_client.get_ids(index.index_id)
    logging.info("Total ids size %d", len(indexed_ids))

    buffer_size = 1000

    if args.hnsw:
        phi = get_phi_for_hnsw(args.embeddings + "_meta_*")

    files = glob.glob(args.embeddings + "_[0-9]*")
    time0 = time.time()

    for file in files:
        time_file = time.time()
        with open(file, "rb") as reader:
            logging.info("Reading {}".format(file))
            try:
                doc_vectors_with_meta = pickle.load(reader)
            except Exception:
                logging.info("Couldn't unpickle {}".format(file))
                continue
            logging.info("Adding {} vectors".format(len(doc_vectors_with_meta)))
            vectors = []
            ids = []
            metas = []

            # We assume  text, title, is_wiki are pickled along with vectors, might require
            # refactoring otherwise
            logging.info("Read vectors from {}".format(file))
            for id__vector__meta in doc_vectors_with_meta:
                id = id__vector__meta[0]
                vector = id__vector__meta[1]
                meta = id__vector__meta[2:]
                if id in indexed_ids:
                    continue
                vectors.append(vector)
                metas.append((id, *meta))

            meta_chunks = [
                metas[i: i + buffer_size] for i in range(0, len(metas), buffer_size)
            ]
            vector_chunks = [
                vectors[i : i + buffer_size] for i in range(0, len(vectors), buffer_size)
            ]

            time_chunk = time.time()
            assert len(vector_chunks) == len(meta_chunks)
            for i, (meta_chunk, vector_chunk) in enumerate(zip(meta_chunks, vector_chunks)):
                assert all([not np.isnan(v).any() and not np.isinf(v).any() for v in vector_chunk])
                vector_chunk = (
                    get_hnsw_vectors(vector_chunk, phi) if args.hnsw else np.array(vector_chunk)
                )
                index.add(vector_chunk, meta_chunk)

                if i % 100 == 0:
                    logging.info("Added chunk {} in {} sec".format(i, time.time() - time_chunk))
                    time_chunk = time.time()

            logging.info(
                "Done adding vectors from {}. Took {} sec".format(file, time.time() - time_file)
            )

    index.wait_for_index()
    logging.info("index build time: %f sec.", time.time() - time0)
    index.save(args.index_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        default="",
        type=str,
        help="prefix to embeddings files, same as out_file in https://github.com/fairinternal/DPR/blob/cc_news_indexing/generate_dense_embeddings.py#L153-L154",
    )
    parser.add_argument("--index_factory", default="flat", type=str)
    parser.add_argument(
        "--index_id",
        default="default_index",
        type=str,
        help="identifier which will be appended to index_facotry",
    )
    parser.add_argument(
        "--discovery_config",
        type=str,
    )
    parser.add_argument("--type", choices=["local", "remote"], type=str)

    parser.add_argument("--hnsw", action="store_true")
    parser.add_argument("--drop_index", action="store_true")
    parser.add_argument("--include_wiki", action="store_true")

    parser.add_argument("--index_path", type=str)

    args = parser.parse_args()
    main(args)
