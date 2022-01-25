#!/bin/bash
. /usr/share/modules/init/sh
eval "$(conda shell.bash hook)"
conda activate side

BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../../../../../

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
