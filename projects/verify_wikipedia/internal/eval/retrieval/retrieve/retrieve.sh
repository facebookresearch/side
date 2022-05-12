# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

BASEDIR=$( pwd )

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
CUDA_VISIBLE_DEVICES=0 python $BASEDIR/scripts/retrieve_rerank.py $@
