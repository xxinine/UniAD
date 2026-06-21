#!/usr/bin/env bash
# =============================================================================
# Multi-node training launcher. Runs INSIDE the uniad2.0 container at
# /workspace/UniAD (started with --network host). Topology comes from env vars
# so the same image/command works on every node, only NODE_RANK differs.
#
#   NNODES        total number of nodes        (default 4)
#   GPUS_PER_NODE GPUs per node                (default 4)
#   NODE_RANK     this node's rank, 0..NNODES-1 (default 0)
#   MASTER_ADDR   rank-0 node private IP       (default 10.1.0.4)
#   MASTER_PORT   rendezvous port              (default 28596)
#
# Usage (inside container):
#   NODE_RANK=0 ./tools/uniad_dist_train_multinode.sh <CFG> [extra train.py args]
#
# This file intentionally mirrors tools/uniad_dist_train_docker.sh (single-node)
# but parameterizes the distributed topology and uses TCP NCCL transport, since
# the A100 PCIe nodes have no InfiniBand.
# =============================================================================

T=`date +%m%d%H%M`

# -------------------------------------------------- #
CFG=$1
# -------------------------------------------------- #
NNODES=${NNODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"10.1.0.4"}
MASTER_PORT=${MASTER_PORT:-28596}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

# -------------------------------------------------- #
# Ensure dataset is available at ./data (bind-mounted #
# from the shared lustre when the container starts).  #
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

# ---- runtime env (identical to single-node docker script) ---------------- #
export TORCH_CUDA_ARCH_LIST=9.0
export LD_LIBRARY_PATH=/root/anaconda3/envs/uniad_new/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export PATH=/root/cuda-12.2/bin/:$PATH
export LD_LIBRARY_PATH=/root/cuda-12.2/lib64/:$LD_LIBRARY_PATH
export CUDA_HOME=/root/cuda-12.2
source /root/anaconda3/etc/profile.d/conda.sh
conda activate uniad_new
sed -i '1s|#!/ssd2/wenshengzhao/anaconda3/envs/uniad_new/bin/python|#!/root/anaconda3/envs/uniad_new/bin/python|' /root/anaconda3/envs/uniad_new/bin/torchrun

# ---- NCCL over TCP (no InfiniBand on A100 PCIe nodes) --------------------- #
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
# Optional verbosity for the first bring-up; comment out for normal runs.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

echo "[multinode] NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE} NODE_RANK=${NODE_RANK} MASTER=${MASTER_ADDR}:${MASTER_PORT} IF=${NCCL_SOCKET_IFNAME}"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    $(dirname "$0")/train.py \
    $CFG \
    --launcher pytorch ${@:2} \
    --deterministic \
    --work-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/train.node${NODE_RANK}.$T
