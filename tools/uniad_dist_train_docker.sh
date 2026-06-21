#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
GPUS=$2                                              #
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
NNODES=`expr $GPUS / $GPUS_PER_NODE`

MASTER_PORT=${MASTER_PORT:-28596}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}  
RANK=${RANK:-0}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

# -------------------------------------------------- #
# Ensure the dataset is available at ./data          #
# When launched via tools/uniad_docker_run.sh the    #
# repo (including ./data) is bind-mounted, so the     #
# directory normally already exists.                 #
# Otherwise a symlink to DATA_SOURCE is created.      #
# DATA_SOURCE defaults to the shared Lustre dataset.  #
# -------------------------------------------------- #
DATA_SOURCE="${DATA_SOURCE:-/adlustre/nuscenes/data}"
DATA_TARGET="$(dirname "$0")/../data"

if [ -e ${DATA_TARGET} ]; then
    echo "Using existing data directory: ${DATA_TARGET} -> $(readlink -f ${DATA_TARGET})"
elif [ -n "${DATA_SOURCE}" ]; then
    echo "Creating symlink: ${DATA_TARGET} -> ${DATA_SOURCE}"
    ln -s ${DATA_SOURCE} ${DATA_TARGET}
else
    echo "Error: ${DATA_TARGET} does not exist and DATA_SOURCE is not set"
    exit 1
fi

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

export TORCH_CUDA_ARCH_LIST=9.0
export LD_LIBRARY_PATH=/root/anaconda3/envs/uniad_new/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export PATH=/root/cuda-12.2/bin/:$PATH
export LD_LIBRARY_PATH=/root/cuda-12.2/lib64/:$LD_LIBRARY_PATH
export CUDA_HOME=/root/cuda-12.2
source /root/anaconda3/etc/profile.d/conda.sh
conda activate uniad_new
export NCCL_IB_GID_INDEX="3"
sed -i '1s|#!/ssd2/wenshengzhao/anaconda3/envs/uniad_new/bin/python|#!/root/anaconda3/envs/uniad_new/bin/python|' /root/anaconda3/envs/uniad_new/bin/torchrun


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=${NNODES} \
    --node_rank=${RANK} \
    $(dirname "$0")/train.py \
    $CFG \
    --launcher pytorch ${@:3} \
    --deterministic \
    --work-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/train.$T