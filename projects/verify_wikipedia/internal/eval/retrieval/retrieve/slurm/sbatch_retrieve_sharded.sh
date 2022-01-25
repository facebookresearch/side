#!/bin/bash
# Usage: sbatch launch_distributed.sh EXPERIMENT.yaml
#SBATCH --job-name=kilt
#SBATCH --partition=learnlab
#SBATCH --array=0-39
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=72:00:00
#SBATCH --open-mode=append
#SBATCH --comment="kilt job"
#SBATCH --output=logs/retrieve_%A_%a.log

BASEDIR=$1
WRAPPED_SCRIPT=$BASEDIR/internal/eval/retrieval/retrieve/slurm/wrapped_retrieve.sh
shift;
echo "Calling command $WRAPPER"
srun --label $WRAPPED_SCRIPT "$@"

