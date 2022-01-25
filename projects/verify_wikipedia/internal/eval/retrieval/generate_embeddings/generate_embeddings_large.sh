BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../../../../

# generate embeddings
sbatch $BASEDIR/internal/eval/retrieval/generate_embeddings/slurm/sbatch_sharded_large.sh $BASEDIR $@
