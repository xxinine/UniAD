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

echo "==================== BF16 Training Configuration ===================="
echo "Config: $CFG"
echo "GPUs: $GPUS (${GPUS_PER_NODE} per node, ${NNODES} nodes)"
echo "Work Dir: $WORK_DIR"
echo "====================================================================="

export TORCH_CUDA_ARCH_LIST=9.0
export LD_LIBRARY_PATH=/root/anaconda3/envs/uniad_new/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export PATH=/root/cuda-12.2/bin/:$PATH
export LD_LIBRARY_PATH=/root/cuda-12.2/lib64/:$LD_LIBRARY_PATH
export CUDA_HOME=/root/cuda-12.2
source /root/anaconda3/etc/profile.d/conda.sh
conda activate uniad_new
export NCCL_IB_GID_INDEX="3"
sed -i '1s|#!/ssd2/wenshengzhao/anaconda3/envs/uniad_new/bin/python|#!/root/anaconda3/envs/uniad_new/bin/python|' /root/anaconda3/envs/uniad_new/bin/torchrun

# Check BF16 support
echo ""
echo "Checking BF16 support..."
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('BF16 supported:', torch.cuda.is_bf16_supported())"
echo ""

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
    2>&1 | tee ${WORK_DIR}logs/train_bf16.$T
