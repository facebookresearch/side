# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import json
from tqdm.auto import tqdm
import pickle
import json
import argparse

from genre.fairseq_model import GENRE


def load_data(filename):
    data = []
    with open(filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data


def store_data(filename, data):
    with open(filename, "w+") as outfile:
        for idx, element in enumerate(data):
            # print(round(idx * 100 / len(data), 2), "%", end="\r")
            # sys.stdout.flush()
            json.dump(element, outfile)
            outfile.write("\n")


def main(args, test_data):
    print(
        "loading model {} from {}".format(
            args.checkpoint_file,
            args.model_path,
        )
    )
    model = (
        GENRE.from_pretrained(
            args.model_path, checkpoint_file=args.checkpoint_file, arg_overrides=True
        )
        .eval()
        .to("cuda:1")
    )

    genre_output = []
    buffer = []
    batch_size = 100
    beam_size = 15
    for record in tqdm(test_data):
        s = record["input"]

        buffer.append(s)

        if len(buffer) == batch_size:
            a = model.sample(
                buffer,
                beam=beam_size,
                max_len_b=15,
                # prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
            genre_output.extend(a)
            buffer = []

    if len(buffer) > 0:
        a = model.sample(
            buffer,
            beam=beam_size,
            max_len_b=15,
            # prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )
        genre_output.extend(a)
        buffer = []

    new_data = []
    print("storing output file in {}".format(args.out_filename))
    for record, output in zip(test_data, genre_output):
        for x in output:
            if type(x["score"]) != float:
                x["score"] = x["score"].item()

        # append title prediction to input
        record["input"] = (
            record["meta"]["sentences"][-1]
            + " "
            + output[0]["text"].strip()
            + " "
            + record["meta"]["wikipedia_title"]
        )
        record["output"] = output
        new_data.append(record)

    store_data(args.out_filename, new_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        dest="model_path",
        type=str,
        default="models/sparse/title_generator/",
        help="model path.",
    )

    parser.add_argument(
        "--checkpoint_file",
        dest="checkpoint_file",
        type=str,
        default="checkpoint_best.pt",
        help="checkpoint file.",
    )

    parser.add_argument(
        "--test_filename",
        dest="test_filename",
        type=str,
        required=True,
        help="test filename.",
    )

    parser.add_argument(
        "--out_filename",
        dest="out_filename",
        type=str,
        required=True,
        help="output filename.",
    )

    args = parser.parse_args()

    print("lading test data from {}".format(args.test_filename))
    test_data = load_data(args.test_filename)

    main(args, test_data)
