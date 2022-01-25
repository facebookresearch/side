#!/usr/bin/env python3

import glob
import os
import sys
from pathlib import Path

if __name__ == "__main__":

    if len(sys.argv) < 2 or not sys.argv[1].endswith(".yaml"):

        print("Provide experiment config yaml.")
        exit()

    else:

        experiment_config = sys.argv[1]

        print(f"Select slurm config to run {experiment_config}:")
        choices = dict()
        default = None
        for i, script in enumerate(glob.glob(f"{Path(__file__).parent}/slurm/*.sh")):
            if "sbatch" in script:
                if script == "internal/train/slurm/sbatch_singlenode_devpartition.sh":
                    print(f"[_{i}_]", script)
                    default = i
                else:
                    print(f"[ {i} ]", script)
                choices[i] = script

        choice = input()
        if choice is None or len(choice) == 0:
            choice = default
        else:
            choice = int(choice)
        session = os.system(
            f'sbatch {choices[choice]} {Path(__file__).parent.parent.parent.absolute()} {Path(experiment_config).absolute()} &',
        )

        if session != 0:
            print(session)