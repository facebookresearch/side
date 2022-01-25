EMBEDDINGS_SHARDS_DIR=$2/shard_
DIST_FAISS_DATA_DIR=$3

BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../../../../

echo "This will drop the current index and recreate/retrain the index based on the embeddings in $EMBEDDINGS_SHARDS_DIR"

read

python /scripts/build_index_distributed.py \
    --index_factory hnswsq \
    --type remote \
    --embeddings $EMBEDDINGS_SHARDS_DIR \
    --discovery_config $DIST_FAISS_DATA_DIR/discovery-config.txt \
    --hnsw \
    --drop_index
