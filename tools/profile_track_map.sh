#!/usr/bin/env bash

# Profiler script for UniAD training with torch.profiler
# Usage: ./tools/profile_track_map.sh <config> <gpus> [optional args]
# Example: ./tools/profile_track_map.sh ./projects/configs/stage1_track_map/base_track_map.py 1

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
GPUS=$2                                              #
# -------------------------------------------------- #

if [ -z "$CFG" ] || [ -z "$GPUS" ]; then
    echo "Usage: $0 <config> <gpus> [optional args]"
    echo "Example: $0 ./projects/configs/stage1_track_map/base_track_map.py 1"
    exit 1
fi

GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
NNODES=`expr $GPUS / $GPUS_PER_NODE`

MASTER_PORT=${MASTER_PORT:-28596}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}  
RANK=${RANK:-0}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")_profiler/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

# -------------------------------------------------- #
# Create data symlink if not exists                  #
# -------------------------------------------------- #
DATA_SOURCE="/home/xueshu/work/data/nuscenes/data"
DATA_TARGET="$(dirname "$0")/../data"

if [ ! -e ${DATA_TARGET} ]; then
    echo "Creating symlink: ${DATA_TARGET} -> ${DATA_SOURCE}"
    ln -s ${DATA_SOURCE} ${DATA_TARGET}
elif [ ! -L ${DATA_TARGET} ]; then
    echo "Warning: ${DATA_TARGET} exists but is not a symlink"
    echo "Please check your data directory setup"
else
    echo "Data symlink already exists: ${DATA_TARGET} -> $(readlink ${DATA_TARGET})"
fi

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

# -------------------------------------------------- #
# Profiler Configuration                             #
# -------------------------------------------------- #
# You can customize these via environment variables:
PROFILER_WAIT=${PROFILER_WAIT:-10}        # Wait iterations before profiling
PROFILER_WARMUP=${PROFILER_WARMUP:-1}    # Warmup iterations
PROFILER_ACTIVE=${PROFILER_ACTIVE:-3}    # Active profiling iterations
PROFILER_REPEAT=${PROFILER_REPEAT:-1}    # Repeat profiling cycles

echo "=================================================="
echo "Running Profiler Training"
echo "Config: ${CFG}"
echo "GPUs: ${GPUS}"
echo "Work Dir: ${WORK_DIR}"
echo "Profiler Settings:"
echo "  - Wait: ${PROFILER_WAIT} iters"
echo "  - Warmup: ${PROFILER_WARMUP} iters"
echo "  - Active: ${PROFILER_ACTIVE} iters (recording)"
echo "  - Repeat: ${PROFILER_REPEAT} cycle(s)"
echo "=================================================="


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
    $(dirname "$0")/profiler_train.py \
    $CFG \
    --profiler-wait ${PROFILER_WAIT} \
    --profiler-warmup ${PROFILER_WARMUP} \
    --profiler-active ${PROFILER_ACTIVE} \
    --profiler-repeat ${PROFILER_REPEAT} \
    --profiler-export tensorboard \
    --profiler-profile-memory \
    --profiler-no-shapes \
    --profiler-no-stack \
    --launcher pytorch ${@:3} \
    --deterministic \
    --work-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/profiler.$T

echo ""
echo "=================================================="
echo "Profiler training completed!"
echo "=================================================="
echo "Profiler logs saved to: ${WORK_DIR}profiler_logs/"
echo "Training logs: ${WORK_DIR}logs/profiler.$T"
echo ""
echo "To view the profiling results with TensorBoard:"
echo "  tensorboard --logdir=${WORK_DIR}profiler_logs/"
echo ""
echo "Then open your browser and navigate to:"
echo "  http://localhost:6006/#pytorch_profiler"
echo "=================================================="
