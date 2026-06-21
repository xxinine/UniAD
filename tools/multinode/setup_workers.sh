#!/usr/bin/env bash
# =============================================================================
# Master-side dispatcher: provision UniAD training WORKER nodes over SSH.
# Run this ON the master (vm-gpu-italy-1, 10.1.0.4).
#
# Subcommands (run in order, confirm between stages):
#   ping              test SSH to all workers
#   copy              copy provision_worker.sh to all workers
#   phase1            run phase1 on all workers (disk/docker/repo/image/kernel)
#   reboot            reboot all workers and wait until SSH is back
#   phase2            run phase2 on all workers (lustre kmod + mount)
#   verify            print readiness report for all workers
#
# Worker private IPs are fixed (same VNet 10.1.0.0/16, NIC eth0).
# =============================================================================
set -euo pipefail

WORKERS=(10.1.0.5 10.1.0.6 10.1.0.7)
SSH_USER="azureuser"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/vm-gpu-spot_key.pem}"
REMOTE_SCRIPT="/tmp/provision_worker.sh"
LOCAL_SCRIPT="$(cd "$(dirname "$0")" && pwd)/provision_worker.sh"

SSH_OPTS=(-i "${SSH_KEY}" -o StrictHostKeyChecking=no -o ConnectTimeout=10)

ssh_w()  { ssh "${SSH_OPTS[@]}" "${SSH_USER}@$1" "${@:2}"; }
scp_w()  { scp "${SSH_OPTS[@]}" "$1" "${SSH_USER}@$2:$3"; }

cmd_ping() {
    for w in "${WORKERS[@]}"; do
        printf '%-12s ' "$w"
        ssh_w "$w" 'echo OK $(hostname)' 2>&1 || echo "UNREACHABLE"
    done
}

cmd_copy() {
    for w in "${WORKERS[@]}"; do
        echo "=== copy -> $w ==="
        scp_w "${LOCAL_SCRIPT}" "$w" "${REMOTE_SCRIPT}"
        ssh_w "$w" "chmod +x ${REMOTE_SCRIPT}"
    done
}

# Run a phase on all workers in parallel, streaming each log to a temp file.
run_phase_parallel() {
    local phase="$1"
    local pids=() logs=()
    for w in "${WORKERS[@]}"; do
        local lf="/tmp/provision_${w}_${phase}.log"
        logs+=("$lf")
        ( ssh_w "$w" "bash ${REMOTE_SCRIPT} ${phase}" >"$lf" 2>&1 ) &
        pids+=($!)
        echo "started ${phase} on ${w} (pid $!) -> ${lf}"
    done
    local rc=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "[OK]   ${WORKERS[$i]} ${phase}"
        else
            echo "[FAIL] ${WORKERS[$i]} ${phase} (see ${logs[$i]})"
            rc=1
        fi
    done
    echo "----- tail of each log -----"
    for lf in "${logs[@]}"; do
        echo "### ${lf}"
        tail -n 6 "${lf}"
    done
    return $rc
}

cmd_phase1() { run_phase_parallel phase1; }
cmd_phase2() { run_phase_parallel phase2; }

cmd_reboot() {
    for w in "${WORKERS[@]}"; do
        echo "rebooting ${w}"
        ssh_w "$w" 'sudo systemctl reboot' || true
    done
    echo "waiting for workers to come back (kernel switch)..."
    for w in "${WORKERS[@]}"; do
        printf 'waiting %s ' "$w"
        for _ in $(seq 1 60); do
            if ssh "${SSH_OPTS[@]}" -o ConnectTimeout=5 "${SSH_USER}@$w" \
                   'uname -r' >/tmp/k_$w 2>/dev/null; then
                echo "UP kernel=$(cat /tmp/k_$w)"
                break
            fi
            printf '.'
            sleep 10
        done
    done
}

cmd_verify() {
    for w in "${WORKERS[@]}"; do
        echo "========================================"
        ssh_w "$w" "bash ${REMOTE_SCRIPT} verify" 2>&1 || echo "verify failed on $w"
    done
}

case "${1:-}" in
    ping)    cmd_ping ;;
    copy)    cmd_copy ;;
    phase1)  cmd_phase1 ;;
    reboot)  cmd_reboot ;;
    phase2)  cmd_phase2 ;;
    verify)  cmd_verify ;;
    *) echo "usage: $0 {ping|copy|phase1|reboot|phase2|verify}"; exit 1 ;;
esac
