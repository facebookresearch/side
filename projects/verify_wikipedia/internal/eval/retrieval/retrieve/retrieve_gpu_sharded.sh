# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

EXPERIMENT_YAML=$(realpath $1)
BASEDIR=$( pwd )

# generate embeddings
sbatch $BASEDIR/internal/eval/retrieval/retrieve/slurm/sbatch_retrieve_gpu_sharded.sh \
  $BASEDIR \
  $EXPERIMENT_YAML
