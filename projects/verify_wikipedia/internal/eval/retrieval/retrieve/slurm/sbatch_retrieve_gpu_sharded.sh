#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Usage: sbatch launch_distributed.sh EXPERIMENT.yaml
#SBATCH --job-name=kilt
#SBATCH --partition=learnlab
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --constraint=volta32gb
#SBATCH --cpus-per-task=4
#SBATCH --mem=240G
#SBATCH --time=72:00:00
#SBATCH --open-mode=append
#SBATCH --comment="kilt job"
#SBATCH --output=logs/retrieve_%A_%a.log

BASEDIR=$1
echo "BASEDIR = $BASEDIR"
WRAPPED_SCRIPT=$BASEDIR/internal/eval/retrieval/retrieve/slurm/wrapped_retrieve.sh
echo "WRAPPED_SCRIPT = $WRAPPED_SCRIPT"
shift;
echo "Calling command $WRAPPER"
srun --label $WRAPPED_SCRIPT "$@"
