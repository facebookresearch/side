#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

. /usr/share/modules/init/sh
eval "$(conda shell.bash hook)"
conda activate side

BASEDIR=$( pwd )
echo "BASEDIR=$BASEDIR"

module purge
module unload cuda
module load cuda
module load NCCL
module load cudnn
module unload java
export JAVA_HOME=/private/home/fabiopetroni/jdk-11.0.9
PATH=${PATH}:${JAVA_HOME}/bin

export PYTHONPATH=$BASEDIR:$BASEDIR/../distributed-faiss:$BASEDIR/../KILT

# execute retrieval
python $BASEDIR/scripts/retrieve_rerank.py $@ task_id=$SLURM_ARRAY_JOB_ID
