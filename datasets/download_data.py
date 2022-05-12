# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os.path
import requests

from tqdm import tqdm
from pathlib import Path
import tarfile

SIDE_URL = "http://dl.fbaipublicfiles.com/side/"

# wafer constants
WAFER_FILENAMES = [
    "wafer-train.jsonl.tar.gz",
    "wafer-dev.jsonl.tar.gz",
    "wafer-test.jsonl.tar.gz",
    "wafer-fail-dev.jsonl.tar.gz",
    "wafer-fail-test.jsonl.tar.gz",
]


def download_file(url, file_path, overwrite):
    file_name = url.split("/")[-1]
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get("content-length", 0))

    if not overwrite and os.path.isfile(file_path):
        current_size = os.path.getsize(file_path)
        if total_size == current_size:
            print(" - Skipping " + file_name + " - already exists.")
            return

    block_size = 1024  # 1 Kibibyte
    t = tqdm(
        total=total_size,
        unit="iB",
        unit_scale=True,
        desc=" - Downloading " + file_name + ": ",
    )
    with open(file_path, "wb") as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()


def download_wafer(dest_dir, overwrite):
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    print("Downloading compressed sparse index:")

    for filename in WAFER_FILENAMES:
        print("Downloading", filename)
        download_file(
            SIDE_URL + filename,
            dest_dir + "/" + filename,
            overwrite,
        )

        print("Extracting", filename)
        my_tar = tarfile.open(dest_dir + "/" + filename)
        my_tar.extractall(dest_dir + "/")
        my_tar.close()

        print("Removing", filename)
        os.remove(dest_dir + filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest_dir",
        required=True,
        type=str,
        help="The path to a directory where index files should be stored",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["wafer"],
        type=str,
        help="The dataset to download.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If flag set, existing files will be overwritter, otherwise skipping download.",
    )
    args = parser.parse_args()

    if args.dataset == "wafer":
        download_wafer(args.dest_dir, args.overwrite)
    else:
        raise ValueError
