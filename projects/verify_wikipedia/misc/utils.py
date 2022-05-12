# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import sys

from omegaconf import OmegaConf, DictConfig


def load_data(filename):
    data = []
    with open(filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data


def load_options_from_argv_yaml(constrain_to=None):
    # convert the cli params into Hydra format
    config_file = None
    hydra_formatted_args = list()
    if constrain_to:
        constrain_to = flatten_yaml_dict(constrain_to, return_dict=True)
    for arg in sys.argv:
        if len(arg.split("=")) == 1 and arg.endswith(".yaml"):
            config_file = OmegaConf.load(arg)
            for k, v in flatten_yaml_dict(yd=config_file, return_dict=True).items():
                if constrain_to:
                    if k in constrain_to:
                        hydra_formatted_args.append(k + "=" + str(v).replace("'", "").replace('"', ""))
                else:
                    hydra_formatted_args.append(k + "=" + str(v))
        else:
            hydra_formatted_args.append(arg)
    sys.argv = hydra_formatted_args
    return config_file


def flatten_yaml_dict(yd=None, prefix: str = "", level: int = 0, return_dict: bool = True):
    result_args = list()
    if len(prefix) == 0:
        for k, v in yd.items():
            if isinstance(v, dict) or isinstance(v, DictConfig):
                result_args.extend(flatten_yaml_dict(v, k, level=0, return_dict=False))
            # elif isinstance(v, str):
            else:
                result_args.append(k + "=" + str(v).replace("'", "").replace('"', ""))
            # else:
            #     print(type(v))
    else:
        for k, v in yd.items():
            if isinstance(v, dict) or isinstance(v, DictConfig):
                result_args.extend(flatten_yaml_dict(v, prefix + "." + k, level=level + 1))
            # elif isinstance(v, str):
            else:
                result_args.append(prefix + "." + k + "=" + str(v).replace("'", "").replace('"', ""))
            # else:
            #     print(type(v))
    if level > 0:
        return result_args
    elif return_dict:
        return dict([(kv.split("=")[0], kv.split("=")[1]) for kv in result_args])
    else:
        return result_args
