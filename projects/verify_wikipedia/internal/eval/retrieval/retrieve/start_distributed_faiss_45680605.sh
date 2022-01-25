TASK_ID=45680605
DIST_FAISS_DATA_DIR=/checkpoint/fabiopetroni/WAI/Samuel/distributed-faiss-data/$TASK_ID/
BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../../../../../

mkdir -p $DIST_FAISS_DATA_DIR

bash $BASEDIR/start_distributed_faiss.sh $TASK_ID $DIST_FAISS_DATA_DIR
