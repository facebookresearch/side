#!/bin/bash

. /usr/share/modules/init/sh
eval "$(conda shell.bash hook)"
conda activate side

module purge
module unload cuda
module load cuda
module load NCCL
module load cudnn
module unload java

PATH=${PATH}:${JAVA_HOME}/bin
EXPERIMENT_yaml=$1
TASK_ID=$2
ADDITIONAL_ARG=$3

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASEDIR=$SCRIPT_DIR/../../../
cd $BASEDIR

export JAVA_HOME=/private/home/fabiopetroni/jdk-11.0.9
export WORLD_SIZE=${SLURM_NTASKS}
export RANK=${SLURM_PROCID}
export PYTHONPATH=$BASEDIR

echo "Running job $SLURM_JOB_ID on $SLURM_NNODES nodes: $SLURM_NODELIST"
echo "World Size: $WORLD_SIZE"
echo "Rank: $RANK"

python $BASEDIR/scripts/train.py $EXPERIMENT_yaml $TASK_ID $ADDITIONAL_ARG
