#!/bin/bash
# Usage: sbatch launch_distributed.sh EXPERIMENT.yaml
#SBATCH --job-name=kilt
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --constraint=volta32gb
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/train_%j.log

# echo "Starting distributed job $SLURM_JOB_ID on $SLURM_NNODES nodes: $SLURM_NODELIST"

SCRIPT_DIR=$1
EXPERIMENT_yaml=$2
ADDITIONAL_ARG=$3

WRAPPED_SCRIPT=$SCRIPT_DIR/internal/train/slurm/wrapped_train.sh

echo "Calling command $WRAPPER"
srun --label $WRAPPED_SCRIPT $EXPERIMENT_yaml task_id=$SLURM_JOB_ID $ADDITIONAL_ARG