TASK_ID=45680605
DATA_DIR=/checkpoint/fabiopetroni/WAI/Samuel

DIST_FAISS_DATA_DIR=$DATA_DIR/distributed-faiss-data/$TASK_ID/
EMBEDDING_SHARD_DIR=$DATA_DIR/ctxt_embeddings/$TASK_ID/best_validation_acc__wafer_ccnet/shard_

BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

bash $BASEDIR/index_distributed_faiss.sh $EMBEDDING_SHARD_DIR $DIST_FAISS_DATA_DIR
