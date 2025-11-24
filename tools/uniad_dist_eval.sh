#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
CKPT=$2                                              #
GPUS=$3                                              #    
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

MASTER_PORT=${MASTER_PORT:-28596}
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

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    $(dirname "$0")/test.py \
    $CFG \
    $CKPT \
    --launcher pytorch ${@:4} \
    --eval bbox \
    --show-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/eval.$T