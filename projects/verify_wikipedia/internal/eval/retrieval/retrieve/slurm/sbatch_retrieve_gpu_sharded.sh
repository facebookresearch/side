#!/bin/bash
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
WRAPPED_SCRIPT=$BASEDIR/internal/eval/retrieval/retrieve/slurm/wrapped_retrieve.sh
shift;
echo "Calling command $WRAPPER"
srun --label $WRAPPED_SCRIPT "$@"
