#!/bin/bash
# Usage: sbatch launch_distributed.sh TRAINING_OUTPUT_DIR EXPERIMENT_YAML"
#SBATCH --job-name=kilt
#SBATCH --partition=learnlab
#SBATCH --array=0-512%64
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --constraint=volta16gb
#SBATCH --cpus-per-task=2
#SBATCH --mem=256GB
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --comment="kilt job"
#SBATCH --output=logs/generate_embeddings_%A_%a.log

echo "Starting distributed job $SLURM_JOB_ID on $SLURM_NNODES nodes: $SLURM_NODELIST"

BASEDIR=$1

WRAPPED_SCRIPT=$BASEDIR/internal/eval/retrieval/generate_embeddings/slurm/wrapped_generate_embeddings.sh
shift;
echo "Calling command $WRAPPER"
srun --label $WRAPPED_SCRIPT "$@"