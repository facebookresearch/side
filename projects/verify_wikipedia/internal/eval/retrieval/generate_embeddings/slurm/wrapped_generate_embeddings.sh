#!/bin/bash

. /usr/share/modules/init/sh
eval "$(conda shell.bash hook)"
conda activate side

module purge
module unload cuda
module load cuda
module load NCCL
module load cudnn

export WORLD_SIZE=${SLURM_NTASKS}
export RANK=${SLURM_PROCID}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASEDIR=$SCRIPT_DIR/../../../../../
cd $BASEDIR
export PYTHONPATH=$BASEDIR

python $BASEDIR/scripts/generate_dense_embeddings.py \
  shard_id=$SLURM_ARRAY_TASK_ID \
  num_shards=$SLURM_ARRAY_TASK_COUNT \
  $@
