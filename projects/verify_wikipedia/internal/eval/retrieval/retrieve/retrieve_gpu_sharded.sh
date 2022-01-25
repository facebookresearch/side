EXPERIMENT_YAML=$(realpath $1)
CHECKPOINT_DIR=$(dirname $EXPERIMENT_YAML)
BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../../../../

# generate embeddings
sbatch $BASEDIR/internal/eval/retrieval/retrieve/slurm/sbatch_retrieve_gpu_sharded.sh \
  $BASEDIR \
  $EXPERIMENT_YAML
