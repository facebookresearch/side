import collections
import csv
import logging
import os
import pickle
import sys
import zlib
from pathlib import Path
from typing import Dict, List

import filelock
import hydra
import torch
from omegaconf import DictConfig

from dpr.dataset.input_transform import Passage, WaferPassage, WaferPreprocessor
from dpr.dataset.utils import get_file_len, find_or_download_files
from dpr.utils.data_utils import Tensorizer, DEFAULT_SELECTOR

logger = logging.getLogger(__name__)


class ContextSource(torch.utils.data.Dataset):

    def __init__(
            self,
            file: str,
            selector: DictConfig = None,
            encoder_type: str = None,
    ):
        if selector:
            self.selector = hydra.utils.instantiate(selector)
        else:
            self.selector = DEFAULT_SELECTOR
        self.file = file
        self.data_files = []

    def get_data_paths(self):
        self.data_files = find_or_download_files(self.file)
        assert len(self.data_files) > 0, f"self.file={self.file} ||| self.data_files={self.data_files}"
        self.file = self.data_files[0]

    def get_meta(self, ctxt):
        raise NotImplementedError

    def load_data_to(self, ctxs: Dict[object, Passage]):
        raise NotImplementedError


class WaferCsvCtxSrc(ContextSource, WaferPreprocessor):
    def __init__(
            self,
            cfg: DictConfig,
            tensorizer: Tensorizer,
            file: str,
            id_prefix: str = None,
            selector: DictConfig = None,
    ):
        ContextSource.__init__(
            self,
            file=file,
            selector=selector,
        )
        WaferPreprocessor.__init__(
            self,
            cfg=cfg,
            tensorizer=tensorizer,
        )
        self.id_prefix = id_prefix
 
    def __len__(self):
        os.makedirs(f"{Path.home()}/.cache/files/", exist_ok=True)
        lock = filelock.FileLock(f"{Path.home()}/.cache/files/{self.file.replace('/', '_')}.lock")
        with lock:
            if not os.path.exists(f"{Path.home()}/.cache/files/{self.file.replace('/', '_')}.size"):
                size = get_file_len(self.file)
                with open(f"{Path.home()}/.cache/files/{self.file.replace('/', '_')}.size", "w") as f:
                    f.writelines(f"{size}")
            else:
                with open(f"{Path.home()}/.cache/files/{self.file.replace('/', '_')}.size") as f:
                    size = int(next(f))
        return size

    def load_data_to(self, ctxs: Dict[object, WaferPassage], start=0, end=sys.maxsize):
        super().get_data_paths()
        logger.info("Reading file %s", self.file)
        with open(self.file) as fd:
            if start > 0:
                for i, line in enumerate(fd):
                    if i + 1 == start:
                        break
            reader = csv.reader(fd, delimiter="\t")
            for i, row in enumerate(reader):
                if start + i == end:
                    break
                if row[0] == "id":
                    continue
                if self.id_prefix:
                    sample_id = self.id_prefix + str(row[0])
                else:
                    sample_id = row[0]

                ctxs[sample_id] = WaferPassage(
                    text=row[1].strip('"'),
                    title=row[2],
                    url=row[3],
                    passage_id=row[4],
                    bm25_hit=row[5],
                    rank=None,
                )

    def get_meta(self, ctxt: WaferPassage): 
        return (
            zlib.compress(ctxt.text.encode()),
            zlib.compress(ctxt.title.encode()),
            zlib.compress(ctxt.url.encode()),
            ctxt.passage_id,
            ctxt.bm25_hit,
        )

CCNETCsvCtxSrcPassage = collections.namedtuple("CCNETCsvCtxSrcPassage", ["text", "title", "url", "is_wiki"])


class CCNETCsvCtxSrc(WaferCsvCtxSrc):

    def __init__(self,
                 chunk_id__to__url__map,
                 cfg,
                 tensorizer,
                 file,
                 id_prefix,
                 selector,
                 ):
        WaferCsvCtxSrc.__init__(
            self,
            cfg=cfg,
            tensorizer=tensorizer,
            file=file,
            id_prefix=id_prefix,
            selector=selector,
        )
        self.chunk_id__to__url__map = chunk_id__to__url__map

    def load_data_to(self, ctxs: Dict[object, CCNETCsvCtxSrcPassage], start=0, end=sys.maxsize):
        super().get_data_paths()
        logger.info("Reading file %s", self.file)
        with open(self.file) as fd:
            if start > 0:
                for i, line in enumerate(fd):
                    if i + 1 == start:
                        break
            reader = csv.reader(fd, delimiter="\t")
            for i, row in enumerate(reader):
                if start + i == end:
                    break
                if row[0] == "id":
                    continue
                if self.id_prefix:
                    sample_id = self.id_prefix + str(row[0])
                else:
                    sample_id = row[0]
                ctxs[sample_id] = CCNETCsvCtxSrcPassage(
                    text=row[1].strip('"'),
                    title=row[2],
                    url=self.chunk_id__to__url__map(sample_id),
                    is_wiki=row[4],
                )

    def get_meta(self, ctxt: CCNETCsvCtxSrcPassage):
        return (
            zlib.compress(ctxt.text.encode()),
            zlib.compress(ctxt.title.encode()),
            zlib.compress(ctxt.url.encode()),
            ctxt.is_wiki,
        )

#TODO: create multiple files retriever data?
class CCNETMultiCsvCtxSrc(ContextSource, WaferPreprocessor):
    def __init__(
            self,
            cfg: DictConfig,
            tensorizer: Tensorizer,
            files: str,
            id_prefix: str = None,
            selector: DictConfig = None,
    ):
        WaferPreprocessor.__init__(
            self,
            cfg=cfg,
            tensorizer=tensorizer,
        )
        if selector:
            self.selector = hydra.utils.instantiate(selector)
        else:
            self.selector = DEFAULT_SELECTOR

        # TODO remove or make configurable
        self.chunk_id__to__url__map = None
        def lazy_pickle_dict(key):
            if self.chunk_id__to__url__map is None:
                with open("/checkpoint/piktus/sphere/chunk_id_url.pkl", "rb") as f:
                    self.chunk_id__to__url__map = pickle.load(f)
            return self.chunk_id__to__url__map[key]

        self.ctx_srcs = list()
        for file in files:
            self.ctx_srcs.append(
                CCNETCsvCtxSrc(
                    chunk_id__to__url__map=lazy_pickle_dict,
                    cfg=cfg,
                    tensorizer=tensorizer,
                    file=file,
                    id_prefix=id_prefix,
                    selector=selector,
                )
            )

    def __len__(self):
        size = 0
        for ctx_src in self.ctx_srcs:
            size += len(ctx_src)
        return size

    def load_data_to(self, ctxs: Dict[object, WaferPassage], start=0, end=sys.maxsize):
        dataset_start = 0
        dataset_end = 0
        for ctx_src in self.ctx_srcs:
            dataset_end += len(ctx_src)
            if start < dataset_end:
                if end <= dataset_end:
                    ctx_src.load_data_to(ctxs, start - dataset_start, end - dataset_start)
                    break
                else:
                    ctx_src.load_data_to(ctxs, start - dataset_start, dataset_end - dataset_start)
                    start = dataset_end
            dataset_start = dataset_end

    def get_meta(self, ctxt: CCNETCsvCtxSrcPassage):
        return self.ctx_srcs[0].get_meta(ctxt)
