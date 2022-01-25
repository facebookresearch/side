BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../../../../

module purge
module unload cuda
module load cuda
module load NCCL
module load cudnn
module unload java
export JAVA_HOME=/private/home/fabiopetroni/jdk-11.0.9
PATH=${PATH}:${JAVA_HOME}/bin

export PYTHONPATH=$BASEDIR:$BASEDIR/../distributed-faiss:$BASEDIR/../KILT

# execute retrieval
CUDA_VISIBLE_DEVICES=0 python $BASEDIR/scripts/retrieve_rerank.py $@
