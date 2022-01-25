# side-internal

## About

This repository contains the code for training and evaluating **DPR**, **BM25** and **Reranker** models on the [WAFER](https://fb.workplace.com/notes/612716063022116) [(paper draft)](https://fb.workplace.com/work/file_viewer/1131061870964095/?surface=POST_ATTACHMENT) dataset. This repository is adapted from [DPR](https://github.com/fairinternal/dpr), [KILT-internal](https://github.com/fairinternal/kilt-internal), [KILT](https://github.com/facebookresearch/kilt) and [Sphere](https://github.com/fairinternal/sphere).


## Table of contents

1. [Features and differences of DPR](#features-and-extensions-of-dpr)
2. [Setup](#setup)
3. [Training and Evaluate DPR](#training-and-evaluate-dpr)
4. [Training and Evaluate Reranker](#training-and-evaluate-reranker)
5. [Overview preprocessed datasets](#overview-preprocessed-datasets)

## Features and extensions of DPR

#### Modularized and centralized schmema for datasets:
  - Original DPR manipulates the text in many different places which makes adaption to other schema difficult
  - Original DPR expects that the query and passage text are already transformed in the data, which makes experimentation with different preprocessing variants or combining them extremly difficult and leads to many different copies of the same dataset. In contrast, in this code the [input transformation](https://github.com/fairinternal/side-internal/blob/107f21503278d73de1462539807c5504c09d407c/dpr/dataset/input_transform.py#L49) happens on the fly, and the transformation is therefore a hyperparameter and also enables things like input dropout.


#### Training **DPR** and **Reranker** models:
  - Ranker models: BiEncoder, Crossencoder, ColBERT
  - Base models: HF-BERT, HF-RoBERTa, HF-DeBERTA
  - Additional [loss functions](https://github.com/fairinternal/side-internal/blob/107f21503278d73de1462539807c5504c09d407c/dpr/models/ranker_loss.py#L113) for datasets with supervision on the document and not the passage-level: 
      - hard-em
      - soft-em with temperature annealing
      - uniform
      - random 

#### Creating WAFER training data for DPR and the reranker:
  - BiEncoder:
      - random negatives
      - random document negatives (multiple consecutive passages from a random document)
      - hard negatives (for example from BM25 or a prior DPR model)
      - hard negative documents (multiple consecutive passages from a random document around a BM25/DPR hit)
  - Crossencoder:
      - Creates a mix of DPR and BM25 retrieveable passages 

#### Bugfixes and small usability improvements for DPR:
  - Resuming with validation metrics
  - All scripts have a slurm wrapper for single-node+multi-gpu (DP) and multi-node+multi-task+single-gpu (DDP) setting.
  - Slight improvements for the configurability and naming schemes
  - Fixing the loss scale
  - Checkpoint saving and loading
  - ...

#### Creating a **DPR** or **BM25** index:
  - Create a DPR index, i.e. generate embeddings (sharded and resumable) and then generate a local or a remote index
  - Create a BM25 index on the fly

#### Running evaluation with **DPR**, **BM25** and the **Reranker**:
  - Running retrieval and reranking locally or distributed (shards) to speed up inference
  - Resumable when run in shards (e.g. to compensate for node failures)
  
## Setup

The project depencies are defined in [requirements.txt](https://github.com/fairinternal/side-internal/blob/main/requirements.txt). Install the project with 

**NOTE: The `side` conda environment is essential for some slurm scripts!**

    conda create -n side -y python=3.8 && conda activate side
    cd YOUR_WORKDIR
    git glone git@github.com:fairinternal/side-internal.git
    git clone git@github.com:fairinternal/distributed-faiss.git
    cd distributed-faiss
    pip install -e .
    cd ..
    cd side-internal
    pip install -r requirements.txt

## Training and Evaluate DPR

### Run training

In `internal/train/configs/` are example configurations for training a model with the best presets that we found so far. 

    internal/train/configs/
    ├── biencoder-prototype.yaml
    ├── biencoder.yaml
    ├── crossencoder-prototype.yaml
    └── crossencoder.yaml

To start training run

    python internal/train/launch.py PATH_TO_CONFIG    


For example using a prototyping config with a small dataset:

    python internal/train/launch.py internal/train/configs/biencoder-prototype.yaml

shows you the choice to enter a number from 0-3  

    Select slurm config to run internal/train/configs/biencoder-prototype.yaml:

    [ 0 ] internal/train/slurm/sbatch_multinode.sh
    [ 1 ] internal/train/slurm/sbatch_multinode_devpartition.sh
    [ 2 ] internal/train/slurm/sbatch_singlenode.sh
    [_3_] internal/train/slurm/sbatch_singlenode_devpartition.sh

which will run the configuration with the respective configuration (defaults to [ 3 ] by just pressing enter).

The output file is located in `CWD/outputs/HYDRA_JOB_NAME/SLURM_JOB_ID/` (can be configured) and the slurm log files are located in `CWD/logs`.

See here for all [training config options](https://github.com/fairinternal/side-internal/blob/main/conf/training_config.yaml).

### Output files

The file `.../.hydra/config.yaml` contains all configurations for the model.

    PATH_TO/side-internal/outputs/biencoder/52237054/
    ├── .hydra
    │   ├── config.yaml
    │   ├── hydra.yaml
    │   └── overrides.yaml
    ├── crossencoder.log
    ├── eval_config_template.yaml
    └── outputs
        ├── checkpoint.0
        ├── checkpoint.1
        ├── checkpoint.best_validation_acc
        └── checkpoint.best_validation_loss


### Create training data for DPR

    python scripts/create_datasets/create_dpr_train_data.py

See here for all [create data config options](https://github.com/fairinternal/side-internal/blob/main/conf/create_data/reproduce_dpr_training.yaml).

### Create big index

Creating an index consists of two steps: generating embeddings and then creating the index. The script to generate embeddings expects three arguments: 1. the checkpoint directory, 2. the context source (i.e. the passage collection) and 3. the pattern to which the shards of embeddings are written to. In the following example, the config value for the context source `wafer_ccnet` points to the dataset file in [conf/datasets/retrieval.yaml](https://github.com/fairinternal/side-internal/blob/main/conf/datasets/retrieval.yaml) 

First generate the embeddings. The following call launches 64 parallel jobs for 512 shards and this can be resumed in case that nodes fail. The implementation of the 'resume' functionality is lazy and just starts each job but checks if the output file already exists. This takes roughly a day to finish. 

    TASK_ID=45680605
    CHECKPOINT_DIR=/checkpoint/fabiopetroni/WAI/Samuel/checkpoints/biencoder/$TASK_ID/
    EMBEDDING_SHARD_FILE_PATTERN=/checkpoint/fabiopetroni/WAI/Samuel/ctxt_embeddings/$TASK_ID/best_validation_acc__wafer_ccnet/shard_
    
    bash internal/eval/retrieval/generate_embeddings/generate_embeddings_large.sh \
      checkpoint_dir=$CHECKPOINT_DIR \
      ctx_src=wafer_ccnet \
      out_file=$EMBEDDING_SHARD_FILE_PATTERN


When all shards have been computed they can be fed to the distributed-faiss server to build the index. To do this, start now an instance of an distributed-faiss server in a seperate screen/tmux session (interupting the job does not cancel the job, this has to be done manually with scancel):

    TASK_ID=45680605
    DIST_FAISS_DATA_DIR=/checkpoint/fabiopetroni/WAI/Samuel/distributed-faiss-data/$TASK_ID/

    bash internal/eval/retrieval/build_big_index/start_distributed_faiss.sh $DIST_FAISS_DATA_DIR

When the server is running (i.e. squeue shows 16 nodes allocated), call

    TASK_ID=45680605
    DIST_FAISS_DATA_DIR=/checkpoint/fabiopetroni/WAI/Samuel/distributed-faiss-data/$TASK_ID/
    EMBEDDING_SHARD_FILE_PATTERN=/checkpoint/fabiopetroni/WAI/Samuel/ctxt_embeddings/$TASK_ID/best_validation_acc__wafer_ccnet/shard_

    bash internal/eval/retrieval/build_big_index/index_distributed_faiss.sh $EMBEDDING_SHARD_FILE_PATTER $DATA_DIR

This call will now loop all the shards and send the embeddings to the server, in total this takes around 2-3 days. 


### Run DPR and BM25 evaluation on large-scale index

In `internal/eval/retrieval/retrieve/configs/` are ready made configurations that contain the paths to the **best checkpoints** and **precomputed indexes** for running DPR or BM25 retrieval 

    internal/eval/retrieval/retrieve/configs/
    ├── best_dpr_remote.yaml                      Run DPR retrieval on big index for evaluation data
    ├── best_dpr_remote_training_data.yaml        Run DPR retrieval on big index for training data (e.g. to create reranker training data)
    └── bm25_sphere.yaml                          Run BM25 retrieval on big index for evaluation data 

#### Running retrieval with DPR  

  1. First start the distributed server in a seperate screen/tmux session (interupting the job does not cancel the job, this has to be done manually with scancel):

    bash internal/eval/retrieval/retrieve/start_distributed_faiss_45680605.sh

  2. As soon as the distributed server runs (16 nodes are shown in squeue), then retrieval for *evaluation data* can be done locally (on a devfair machine) with `retrieve.sh` but some devfairs might not have free ressources. So the safest way is to do the inference with the sharded mode which is also faster: 

    bash internal/eval/retrieval/retrieve/retrieve_gpu_sharded.sh internal/eval/retrieval/retrieve/configs/best_dpr_remote.yaml

#### Running retrieval with BM25

BM25 retrieval is too slow and has always to be done sharded. To do so, call for example 

    bash internal/eval/retrieval/retrieve/retrieve_sharded.sh internal/eval/retrieval/retrieve/configs/bm25_sphere.yaml

#### Output files 

The output files are located in `outputs/predictions/RETRIEVER_NAME/TASK_ID/`

    outputs/predictions/
    ├── evaluation.retrievers.bm25.BM25
    │   └── 52240796
    │       ├── retrieve_rerank.log
    │       ├── retriever_cfg.yaml
    │       └── wafer-dev-kiltweb.jsonl
    ├── evaluation.retrievers.dpr_remote.DPR
    │   └── 52240797
    │       ├── retrieve_rerank.log
    │       ├── retriever_cfg.yaml
    │       └── wafer-dev-kiltweb.jsonl
    └── evaluation.retrievers.reranker.Reranker

You can configure the location of the output files with the key `output_folder`. For example to save the prediction files in the checkpoint directory (to associate the predictions with a model's checkpoint), you can configure the path to:

    output_folder: ${retriever.checkpoint_dir}/predictions/${retriever.checkpoint_load_suffix}__${retriever.ctx_src}

Now you can use the [Evaluate predictions](https://github.com/fairinternal/side-internal/blob/main/notebooks/Evaluate%20predictions.ipynb) notebook to compute the scores (adapt the wildcard path there to match your output directory). 

#### Default configurations 

All default configurations for all scenarios are under `conf/retriever`

    conf/retriever
    ├── bm25.yaml             Run retrieval on a small index for prototyping
    ├── bm25_sphere.yaml      Run retrieval on a large index for prototyping
    ├── default.yaml
    ├── dpr.yaml              Run retrieval on a small index for prototyping
    └── dpr_remote.yaml       Run retrieval on a large index for prototyping

### Create local prototyping index and run retrieval evaluation during model development

To evaluate a model on a small prototyping index you can generate embeddings for a chunk of the training data and then run retrieval in this prototyping index. Lets say the task id of a newly trained model is 9999999 in your development dir. Then generate embeddings and run retrieval with:

    TASK_ID=9999999
    CHECKPOINT_DIR=PATH_TO/side-internal/outputs/biencoder/$TASK_ID/
    
    bash internal/eval/retrieval/generate_embeddings/generate_embeddings_wait.sh \
      checkpoint_dir=$CHECKPOINT_DIR \
      ctx_src=wafer_200k

    bash internal/eval/retrieval/retrieve/retrieve_gpu_sharded.sh \
      ctx_src=wafer_200k \
      checkpoint_dir=$CHECKPOINT_DIR 

The output files will be written into your checkpoint directory. 

    crossencoder/9999999/
    ├── .hydra
    │   ├── config.yaml
    │   ├── hydra.yaml
    │   └── overrides.yaml
    ├── crossencoder.log
    ├── eval_config_template.yaml
    ├── outputs
    │   ├── checkpoint.best_validation_acc
    │   └── checkpoint.best_validation_loss
    └── predictions
        └── best_validation_acc__wafer_ccnet
            ├── checkpoint_cfg.yaml
            ├── retriever_cfg.yaml
            ├── wafer-dev-kiltweb.jsonl
            ├── wafer-dev-kiltweb.jsonl.00-4
            ├── wafer-dev-kiltweb.jsonl.01-4
            ├── wafer-dev-kiltweb.jsonl.02-4
            ├── wafer-dev-kiltweb.jsonl.03-4
            └── wafer-dev-kiltweb.jsonl.lock
        
Now you can use the [Evaluate predictions](https://github.com/fairinternal/side-internal/blob/main/notebooks/Evaluate%20predictions.ipynb) notebook to compute the scores (adapt the wildcard path there to match your output directory). 


## Training and Evaluate Reranker

### Training of the reranker

Same as training DPR (see Training DPR above).

### Create training data for the Reranker

Creating the training data for the reranker involves the following steps:

  1. Retrieve K passages for the training data from the big CCNET index with BM25 and DPR
  2. Create a mixed set of passages to create an intermediate index context source
  3. Create an intermediate index
  4. Retrieve again K passages *for* the training data from the intermediate index with BM25 and DPR
  5. Use the K passages for reranking (i.e. only learn to rerank passages that can be retrieved)

For the first step run retrieval with BM25 and DPR for chunks of the training data.

    bash internal/eval/retrieval/retrieve/retrieve_gpu_sharded.sh internal/eval/retrieval/retrieve/configs/best_dpr_remote_training_data.yaml

and 

    bash internal/eval/retrieval/retrieve/retrieve_sharded.sh internal/eval/retrieval/retrieve/configs/bm25_sphere_training_data.yaml


Then, for each chunks of the training data, create a mix of all passages from both retrievers for the training data. See here for all [create data config options](https://github.com/fairinternal/side-internal/blob/main/conf/create_data/reproduce_dpr_training.yaml).

    python scripts/create_datasets/create_reranker_index_from_retrieved.py

Then, for each chunk's index context source, create an index and perform retrieval for each index. Using the retrieved files create training data chunks with 

    python scripts/create_datasets/create_reranker_train_data.py



### Evaluate Reranker

Running the reranker is too slow and is better done sharded. To do so, call for example 

    bash internal/eval/retrieval/retrieve/retrieve_gpu_sharded.sh internal/eval/retrieval/retrieve/configs/best_reranker.yaml

#### Output files 

The output files are located in `outputs/predictions/RETRIEVER_NAME/TASK_ID/`

    outputs/predictions/
    ├── evaluation.retrievers.reranker.Reranker
    │   └── 52240797
    │       ├── retrieve_rerank.log
    │       ├── retriever_cfg.yaml
    │       └── wafer-dev-kiltweb.jsonl
    └── evaluation.retrievers.reranker.Reranker

#### Visualize Colbert interactions

To visualize Colbert interactions to explain another reranker use the notebook [Reranker Colbert highlights](https://github.com/fairinternal/side-internal/blob/main/notebooks/Reranker%20Colbert%20highlights.ipynb). To create such files set the config key `retriever.retrieved_data_nr_ctxs = 1` in the retrieval config `internal/eval/retrieval/retrieve/configs/best_colbert_reranker.yaml`. 

![Colbert interactions](https://github.com/fairinternal/side-internal/blob/main/docs/colbert_highlights.png)


## Overview preprocessed datasets

    /checkpoint/fabiopetroni/WAI/Samuel/data/
        ├── training_split
        ├── retrieved_bm25
        ├── retrieved_dpr
        ├── training_dpr
        ├── training_reranker
        └── training_reranker_source

 - `training_split`: training data split into chunks of 200k per chunk
 - `retrieved_bm25`: retrieved passages for chunks 11-21 with BM25 from the big Sphere index 
 - `retrieved_dpr`: retrieved passages for chunks 11-21 with the best trained DPR from our big Sphere index
 - `training_dpr`: training data for DPR with random and hard (BM25) negatives
 - `training_reranker`: training data for the reranker created with the procedure described above
 - `training_reranker_source`: the source files for creating the training data in `training_reranker` (created from `training_split/wafer-train-kiltweb_03.jsonl`, `retrieved_bm25/wafer-train-kiltweb_03.jsonl`, `retrieved_dpr/wafer-train-kiltweb_03.jsonl`)

## TODOs

 - Create setup.py




