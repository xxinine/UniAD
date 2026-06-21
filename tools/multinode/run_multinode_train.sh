#!/usr/bin/env bash
# =============================================================================
# Master-side orchestrator for 4-node x 4-GPU (16 GPU) UniAD training.
# Run this ON the master (vm-gpu-italy-1, 10.1.0.4).
#
# It launches the uniad2.0 container on every node with --network host and runs
# tools/uniad_dist_train_multinode.sh inside, passing the correct NODE_RANK.
#
# Subcommands:
#   connectivity        check master-port reachability from every worker
#   run <CFG>           launch 16-GPU training (rank0 local, rank1-3 via SSH)
#   logs                follow rank-0 (master) training log
#   stop                stop/remove the training container on all nodes
#
# Example:
#   ./tools/multinode/run_multinode_train.sh run \
#       ./projects/configs/stage1_track_map/base_track_map.py
# =============================================================================
set -euo pipefail

# rank -> node private IP (rank 0 is the master / this host)
NODE_IPS=(10.1.0.4 10.1.0.5 10.1.0.6 10.1.0.7)
MASTER_ADDR="${MASTER_ADDR:-10.1.0.4}"
MASTER_PORT="${MASTER_PORT:-28596}"
NNODES="${NNODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

SSH_USER="azureuser"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/vm-gpu-spot_key.pem}"
SSH_OPTS=(-i "${SSH_KEY}" -o StrictHostKeyChecking=no -o ConnectTimeout=10)

IMG="xxining/uniad2.0:20250703191817"
CONTAINER="uniad_mn"
REPO_DIR="/data/work/UniAD"
DATA_SRC="/adlustre/nuscenes/data"
TRAIN_SCRIPT="./tools/uniad_dist_train_multinode.sh"

# docker run flags shared by all nodes (host network for cross-node NCCL).
docker_flags() {
    local rank="$1"
    echo "--rm -d --name ${CONTAINER} --network host --runtime=nvidia \
-e NVIDIA_VISIBLE_DEVICES=all --ipc=host --shm-size=128g \
--ulimit memlock=-1 --ulimit stack=67108864 \
-e NNODES=${NNODES} -e GPUS_PER_NODE=${GPUS_PER_NODE} -e NODE_RANK=${rank} \
-e MASTER_ADDR=${MASTER_ADDR} -e MASTER_PORT=${MASTER_PORT} \
-v ${REPO_DIR}:/workspace/UniAD -v ${DATA_SRC}:/workspace/UniAD/data:ro \
-w /workspace/UniAD"
}

is_master() { [ "$1" = "0" ]; }

cmd_connectivity() {
    echo "checking master port ${MASTER_ADDR}:${MASTER_PORT} reachability from each worker..."
    for rank in 1 2 3; do
        local ip="${NODE_IPS[$rank]}"
        printf 'rank%s %-12s ' "$rank" "$ip"
        ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" \
            "timeout 5 bash -c 'echo > /dev/tcp/${MASTER_ADDR}/${MASTER_PORT}' 2>/dev/null && echo PORT-OPEN || echo 'port closed (ok if master not started yet)'; \
             ping -c1 -W2 ${MASTER_ADDR} >/dev/null 2>&1 && echo '  ping OK' || echo '  ping FAIL'"
    done
}

launch_node() {
    local rank="$1" cfg="$2"
    local ip="${NODE_IPS[$rank]}"
    local flags; flags="$(docker_flags "${rank}")"
    local cmd="docker rm -f ${CONTAINER} >/dev/null 2>&1; \
docker run ${flags} ${IMG} bash -lc '${TRAIN_SCRIPT} ${cfg}'"
    if is_master "${rank}"; then
        echo "[rank0/master ${ip}] launching container..."
        bash -lc "${cmd}"
    else
        echo "[rank${rank} ${ip}] launching container via ssh..."
        ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" "${cmd}"
    fi
}

cmd_run() {
    local cfg="${1:?usage: run <CFG>}"
    # Start workers (rank 1..3) first so they are waiting, then the master.
    for rank in 1 2 3; do launch_node "${rank}" "${cfg}"; done
    launch_node 0 "${cfg}"
    echo
    echo "All 4 containers launched (16 GPUs). Following rank-0 log..."
    echo "Use '$0 logs' to re-attach, '$0 stop' to stop all."
    sleep 3
    docker logs -f "${CONTAINER}"
}

cmd_logs() { docker logs -f "${CONTAINER}"; }

# Cross-node NCCL all-reduce sanity check before real training.
nccl_incontainer_cmd() {
    local rank="$1"
    # timeout makes the test self-terminate if NCCL hangs, so the --rm
    # container always cleans up (avoids wedged containerd state).
    echo "source /root/anaconda3/etc/profile.d/conda.sh; conda activate uniad_new; \
export LD_LIBRARY_PATH=/root/anaconda3/envs/uniad_new/lib/python3.8/site-packages/torch/lib:\$LD_LIBRARY_PATH; \
export NCCL_SOCKET_IFNAME=eth0; export NCCL_IB_DISABLE=1; export NCCL_NET_PLUGIN=none; export NCCL_COLLNET_ENABLE=0; export NCCL_PROTO=Simple; export NCCL_P2P_DISABLE=1; export NCCL_DEBUG=INFO; \
timeout 150 python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --nnodes=${NNODES} --node_rank=${rank} \
--master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} tools/multinode/nccl_test.py \
2>&1 | tee /workspace/UniAD/tools/multinode/nccl_node${rank}.log"
}

launch_nccl_node() {
    local rank="$1" detached="$2"
    local ip="${NODE_IPS[$rank]}"
    local incmd; incmd="$(nccl_incontainer_cmd "${rank}")"
    local flags="--rm ${detached} --name ${CONTAINER} --network host --runtime=nvidia \
-e NVIDIA_VISIBLE_DEVICES=all --ipc=host --shm-size=128g --ulimit memlock=-1 --ulimit stack=67108864 \
-v ${REPO_DIR}:/workspace/UniAD -v ${DATA_SRC}:/workspace/UniAD/data:ro -w /workspace/UniAD"
    local cmd="docker rm -f ${CONTAINER} >/dev/null 2>&1; docker run ${flags} ${IMG} bash -lc '${incmd}'"
    if is_master "${rank}"; then
        bash -lc "${cmd}"
    else
        echo "[rank${rank} ${ip}] starting nccl test container..."
        ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" "${cmd}"
    fi
}

cmd_nccltest() {
    for rank in $(seq 1 $((NNODES-1))); do launch_nccl_node "${rank}" "-d"; done
    echo "[rank0/master] running nccl test in foreground (${NNODES} nodes x ${GPUS_PER_NODE} GPUs)..."
    sleep 3
    launch_nccl_node 0 ""   # foreground, prints result
}

cmd_stop() {
    for rank in 0 1 2 3; do
        local ip="${NODE_IPS[$rank]}"
        if is_master "${rank}"; then
            docker rm -f "${CONTAINER}" >/dev/null 2>&1 && echo "stopped rank0 ${ip}" || true
        else
            ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" \
                "docker rm -f ${CONTAINER} >/dev/null 2>&1 && echo stopped rank${rank} ${ip}" || true
        fi
    done
}

case "${1:-}" in
    connectivity) cmd_connectivity ;;
    nccltest)     cmd_nccltest ;;
    run)          cmd_run "${2:-}" ;;
    logs)         cmd_logs ;;
    stop)         cmd_stop ;;
    *) echo "usage: $0 {connectivity|nccltest|run <CFG>|logs|stop}"; exit 1 ;;
esac
