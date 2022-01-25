DIST_FAISS_DATA_DIR=$1

mkdir -p $DIST_FAISS_DATA_DIR

BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../../../../../

export PYTHONPATH=$BASEDIR:$BASEDIR/distributed-faiss

python $BASEDIR/distributed-faiss/scripts/server_launcher.py \
    --discovery-config $DIST_FAISS_DATA_DIR/discovery-config.txt \
    --log-dir $DIST_FAISS_DATA_DIR/logs \
    --save-dir $DIST_FAISS_DATA_DIR/index \
    --partition learnlab \
    --num-servers 32 \
    --num-servers-per-node 2 \
    --timeout-min 4320 \
    --mem-gb 500 \
    --base-port 12034