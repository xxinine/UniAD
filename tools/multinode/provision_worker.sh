#!/usr/bin/env bash
# =============================================================================
# Idempotent provisioning for a UniAD multi-node training WORKER node.
# Runs locally ON a worker (dispatched via SSH by setup_workers.sh).
#
# Usage:
#   provision_worker.sh phase1   # data disk + nvidia-docker + repo + image + kernel switch
#   provision_worker.sh phase2   # (after reboot) lustre kmod + mount /adlustre
#   provision_worker.sh verify   # report readiness
#
# Safe to re-run: every step checks current state and skips if already done.
# =============================================================================
set -euo pipefail

PHASE="${1:-}"

# ---- configuration (matches master vm-gpu-italy-1) --------------------------
DATA_MNT="/data"
LUSTRE_MGS="10.1.1.4@tcp0:/lustrefs"
LUSTRE_MNT="/adlustre"
REPO_URL="https://github.com/xxinine/UniAD.git"
REPO_DIR="/data/work/UniAD"
REPO_BRANCH="feat/multinode-train"
DOCKER_IMG="xxining/uniad2.0:20250703191817"
DOCKER_DATA_ROOT="/data/docker"

LTS_META="linux-image-azure-lts-22.04"
LTS_KERNEL="5.15.0-1114-azure"
AMLFS_KMOD="amlfs-lustre-client-2.15.8-34-gc0f2040"
NVIDIA_KMOD="linux-modules-nvidia-535-5.15.0-1114-azure"

log() { echo "[$(hostname) $(date +%H:%M:%S)] $*"; }
die() { echo "[$(hostname)] ERROR: $*" >&2; exit 1; }

# Wait until apt/dpkg locks are free (Ubuntu unattended-upgrades may hold them).
wait_for_apt() {
    for _ in $(seq 1 90); do
        if ! sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 \
           && ! sudo fuser /var/lib/apt/lists/lock >/dev/null 2>&1; then
            return 0
        fi
        log "waiting for apt lock (held by another process)..."
        sleep 10
    done
    die "timed out waiting for apt lock"
}

# -----------------------------------------------------------------------------
# Step A: format + mount the 1 TB data disk at /data
# -----------------------------------------------------------------------------
setup_data_disk() {
    if mountpoint -q "${DATA_MNT}"; then
        log "data disk already mounted at ${DATA_MNT} -> skip"
        return
    fi

    # Identify the raw 1 TB data disk (no partitions, no mountpoint, type=disk).
    # Device letter differs per node, so match by size, never by name.
    local dev=""
    while read -r name size type mnt; do
        [ "${type}" = "disk" ] || continue
        [ "${size}" = "1T" ] || continue
        # skip if it (or any child) is already mounted
        if lsblk -nr "/dev/${name}" -o MOUNTPOINT | grep -q '[^[:space:]]'; then
            continue
        fi
        dev="/dev/${name}"
        break
    done < <(lsblk -dn -o NAME,SIZE,TYPE,MOUNTPOINT)

    [ -n "${dev}" ] || die "could not find an unused 1T data disk"
    log "selected data disk: ${dev}"

    local uuid
    uuid="$(sudo blkid -s UUID -o value "${dev}" 2>/dev/null || true)"
    if [ -z "${uuid}" ]; then
        log "formatting ${dev} as ext4 (whole-disk, matches master)"
        sudo mkfs.ext4 -F "${dev}"
        uuid="$(sudo blkid -s UUID -o value "${dev}" 2>/dev/null || true)"
    else
        log "${dev} already has filesystem (UUID=${uuid}) -> not reformatting"
    fi
    [ -n "${uuid}" ] || die "failed to read UUID of ${dev}"

    sudo mkdir -p "${DATA_MNT}"
    if ! grep -q "UUID=${uuid}" /etc/fstab; then
        log "adding ${DATA_MNT} to /etc/fstab"
        echo "UUID=${uuid}  ${DATA_MNT}  ext4  defaults,nofail  0  2" | sudo tee -a /etc/fstab >/dev/null
    fi
    sudo mount "${DATA_MNT}"
    sudo mkdir -p "${REPO_DIR%/UniAD}"
    sudo chown -R "$(id -u):$(id -g)" "${DATA_MNT}/work" 2>/dev/null || true
    log "data disk mounted: $(findmnt -n "${DATA_MNT}")"
}

# -----------------------------------------------------------------------------
# Step C: nvidia-container-toolkit + docker daemon (nvidia runtime, data-root)
# -----------------------------------------------------------------------------
setup_docker_nvidia() {
    # Azure's unattended-upgrade may activate a conflicting snap docker. Match
    # master by disabling snap docker so the standard docker.service owns the
    # socket and reads /etc/docker/daemon.json (data-root /data/docker).
    if snap services docker >/dev/null 2>&1; then
        if systemctl is-active --quiet snap.docker.dockerd 2>/dev/null \
           || snap services docker 2>/dev/null | grep -q active; then
            log "disabling conflicting snap docker"
            sudo snap disable docker || true
            # snap disable removes /var/run/docker.sock; restart apt docker to
            # recreate the socket and reclaim it.
            sudo systemctl restart docker
            for _ in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 2; done
        fi
    fi
    sudo systemctl enable --now docker 2>/dev/null || true

    if ! dpkg -l nvidia-container-toolkit >/dev/null 2>&1; then
        log "installing nvidia-container-toolkit"
        wait_for_apt
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
            | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
        sudo apt-get update -qq
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
    else
        log "nvidia-container-toolkit already installed -> skip"
    fi

    # Write daemon.json (nvidia runtime + data-root on the 1T disk), like master.
    local want='{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "data-root": "/data/docker"
}'
    sudo mkdir -p /etc/docker
    if [ "$(cat /etc/docker/daemon.json 2>/dev/null || true)" != "${want}" ]; then
        log "writing /etc/docker/daemon.json"
        echo "${want}" | sudo tee /etc/docker/daemon.json >/dev/null
        sudo mkdir -p "${DOCKER_DATA_ROOT}"
        sudo systemctl restart docker
    else
        log "daemon.json already correct -> skip"
    fi

    # Wait for the docker daemon to be ready after a (re)start before checking.
    for _ in $(seq 1 30); do
        docker info >/dev/null 2>&1 && break
        sleep 2
    done

    docker info 2>/dev/null | grep -q "Docker Root Dir: ${DOCKER_DATA_ROOT}" \
        || die "docker data-root not ${DOCKER_DATA_ROOT}"
    docker info 2>/dev/null | grep -qi "Runtimes:.*nvidia" \
        || die "docker nvidia runtime missing"
    log "docker ok (nvidia runtime, data-root=${DOCKER_DATA_ROOT})"
}

# -----------------------------------------------------------------------------
# Step D: clone repo + checkout branch
# -----------------------------------------------------------------------------
setup_repo() {
    if [ -d "${REPO_DIR}/.git" ]; then
        log "repo exists -> fetch + checkout ${REPO_BRANCH}"
        git -C "${REPO_DIR}" fetch --all --prune
    else
        log "cloning repo into ${REPO_DIR}"
        sudo mkdir -p "$(dirname "${REPO_DIR}")"
        sudo chown -R "$(id -u):$(id -g)" "$(dirname "${REPO_DIR}")"
        git clone "${REPO_URL}" "${REPO_DIR}"
    fi
    git -C "${REPO_DIR}" checkout "${REPO_BRANCH}"
    git -C "${REPO_DIR}" pull --ff-only origin "${REPO_BRANCH}" || true
    log "repo on branch: $(git -C "${REPO_DIR}" branch --show-current)"
}

# -----------------------------------------------------------------------------
# Step E: pull docker image
# -----------------------------------------------------------------------------
pull_image() {
    if docker image inspect "${DOCKER_IMG}" >/dev/null 2>&1; then
        log "docker image present -> skip"
    else
        log "pulling ${DOCKER_IMG} (may take a while)"
        docker pull "${DOCKER_IMG}"
    fi
}

# -----------------------------------------------------------------------------
# Step B (part 1): add AMLFS apt repo + switch to LTS 5.15 kernel
# -----------------------------------------------------------------------------
add_amlfs_repo() {
    if [ ! -f /etc/apt/sources.list.d/amlfs.list ]; then
        log "adding AMLFS apt repo"
        wait_for_apt
        sudo apt-get update -qq || true
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl apt-transport-https lsb-release gnupg
        # shellcheck disable=SC1091
        source /etc/lsb-release
        echo "deb [arch=amd64] https://packages.microsoft.com/repos/amlfs-${DISTRIB_CODENAME}/ ${DISTRIB_CODENAME} main" \
            | sudo tee /etc/apt/sources.list.d/amlfs.list >/dev/null
        curl -sL https://packages.microsoft.com/keys/microsoft.asc \
            | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/microsoft.gpg >/dev/null
        sudo apt-get update -qq
    else
        log "AMLFS repo already present -> skip"
    fi
}

switch_to_lts_kernel() {
    if [ "$(uname -r)" = "${LTS_KERNEL}" ]; then
        log "already running LTS kernel ${LTS_KERNEL} -> skip kernel switch"
        return
    fi

    if ! dpkg -l "${LTS_META}" >/dev/null 2>&1; then
        log "installing LTS kernel meta ${LTS_META} (-> ${LTS_KERNEL})"
        wait_for_apt
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${LTS_META}"
    fi

    # Remove HWE metapackages so the system does not roll back to 6.8.
    for meta in linux-image-azure linux-azure; do
        if dpkg -l "${meta}" >/dev/null 2>&1; then
            log "removing HWE meta ${meta}"
            sudo DEBIAN_FRONTEND=noninteractive apt-get remove -y "${meta}" || true
        fi
    done
    # Remove installed 6.8 kernels so 5.15 becomes default on reboot.
    for k in $(dpkg -l 'linux-image-6.8.*-azure' 2>/dev/null | awk '/^ii/{print $2}'); do
        log "removing kernel ${k}"
        sudo DEBIAN_FRONTEND=noninteractive apt-get remove -y "${k}" || true
    done
    sudo DEBIAN_FRONTEND=noninteractive apt-get autoremove -y || true
    sudo update-grub

    # The running 6.8 kernel image cannot be removed yet, and GRUB orders by
    # version (6.8 > 5.15), so explicitly pin the LTS entry as the default.
    pin_grub_to_lts
    log "kernel switch staged; REBOOT required to boot ${LTS_KERNEL}"
}

# Pin GRUB default to the LTS kernel so reboot does not land back on 6.8.
pin_grub_to_lts() {
    local sub entry
    # The LTS kernel lives inside the "Advanced options for Ubuntu" submenu, so
    # the saved entry must be the full "submenu>entry" path, not just the leaf.
    sub="$(grep "submenu 'Advanced options for Ubuntu'" /boot/grub/grub.cfg 2>/dev/null \
           | sed -E "s/.*menuentry_id_option '([^']+)'.*/\1/" | head -1)"
    entry="$(grep "menuentry 'Ubuntu, with Linux ${LTS_KERNEL}'" /boot/grub/grub.cfg 2>/dev/null \
             | grep -v recovery | head -1 \
             | sed -E "s/.*menuentry_id_option '([^']+)'.*/\1/")"
    [ -n "${sub}" ] && [ -n "${entry}" ] || die "could not find GRUB submenu/entry for ${LTS_KERNEL}"
    sudo sed -i 's/^GRUB_DEFAULT=.*/GRUB_DEFAULT=saved/' /etc/default/grub
    sudo update-grub
    sudo grub-set-default "${sub}>${entry}"
    log "GRUB default pinned to: ${sub}>${entry}"
}

# Install the NVIDIA kernel module matching the LTS kernel and load it.
# Azure's CUDA apt repo + unattended-upgrade may have left the nvidia 535 stack
# at an inconsistent "1ubuntu1" version with no prebuilt module for 5.15. We
# converge the userspace to the Ubuntu-archive 0ubuntu0.22.04.1 (what master
# runs) and install the prebuilt kernel module, matching master exactly.
install_nvidia_module() {
    if [ "$(uname -r)" != "${LTS_KERNEL}" ]; then
        log "not on ${LTS_KERNEL}; skipping nvidia module install"
        return
    fi
    if nvidia-smi >/dev/null 2>&1 && dpkg -l "${NVIDIA_KMOD}" >/dev/null 2>&1; then
        log "nvidia already working ($(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) GPUs) -> skip"
        return
    fi

    # Disable the NVIDIA CUDA repo so apt no longer offers the conflicting
    # 1ubuntu1 packages.
    local f
    for f in $(grep -rl "developer.download.nvidia.com/compute/cuda" /etc/apt/sources.list.d/ 2>/dev/null); do
        log "disabling CUDA repo ${f}"
        sudo mv "${f}" "${f}.disabled"
    done
    wait_for_apt
    sudo apt-get update -qq || true

    # Downgrade only the userspace nvidia 535 packages that came from the CUDA
    # repo, drop nvidia-dkms-535 (master does not use it), and install the
    # prebuilt kernel module/objects/signatures for the LTS kernel.
    local ver pkgs dg
    ver="535.309.01-0ubuntu0.22.04.1"
    pkgs=$(dpkg -l | awk '/535.309.01-1ubuntu1/ {print $2}' | sed 's/:amd64//' | grep -v '^nvidia-dkms-535$' || true)
    dg=""
    for p in ${pkgs}; do dg="${dg} ${p}=${ver}"; done
    log "converging nvidia userspace to ${ver} and installing ${NVIDIA_KMOD}"
    wait_for_apt
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades --no-install-recommends \
        ${dg} nvidia-dkms-535- \
        "${NVIDIA_KMOD}" \
        "linux-objects-nvidia-535-${LTS_KERNEL}" \
        "linux-signatures-nvidia-${LTS_KERNEL}"

    sudo modprobe nvidia 2>/dev/null || true
    sleep 3
    if nvidia-smi >/dev/null 2>&1; then
        log "nvidia-smi OK ($(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) GPUs)"
    else
        log "WARNING: nvidia-smi still failing after module install"
    fi
}

# -----------------------------------------------------------------------------
# Step B (part 2, after reboot): install kmod + mount lustre
# -----------------------------------------------------------------------------
setup_lustre() {
    [ "$(uname -r)" = "${LTS_KERNEL}" ] \
        || die "running kernel $(uname -r) != ${LTS_KERNEL}; reboot into LTS kernel first"

    add_amlfs_repo
    if ! dpkg -l "${AMLFS_KMOD}" >/dev/null 2>&1; then
        log "installing lustre kmod ${AMLFS_KMOD}=$(uname -r)"
        wait_for_apt
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${AMLFS_KMOD}=$(uname -r)"
    else
        log "lustre kmod already installed -> skip"
    fi
    sudo modprobe lustre

    sudo mkdir -p "${LUSTRE_MNT}"
    if ! grep -q "${LUSTRE_MNT}" /etc/fstab; then
        log "adding ${LUSTRE_MNT} to /etc/fstab"
        echo "${LUSTRE_MGS} ${LUSTRE_MNT} lustre noatime,user_xattr,_netdev,nofail 0 0" \
            | sudo tee -a /etc/fstab >/dev/null
    fi
    if ! mountpoint -q "${LUSTRE_MNT}"; then
        log "mounting lustre ${LUSTRE_MGS} -> ${LUSTRE_MNT}"
        sudo mount -t lustre -o noatime,user_xattr "${LUSTRE_MGS}" "${LUSTRE_MNT}"
    fi
    ls "${LUSTRE_MNT}/nuscenes/data" >/dev/null 2>&1 \
        && log "lustre data visible: ${LUSTRE_MNT}/nuscenes/data" \
        || log "WARNING: ${LUSTRE_MNT}/nuscenes/data not found"
}

# -----------------------------------------------------------------------------
verify() {
    echo "==== $(hostname) readiness ===="
    echo "kernel        : $(uname -r)  (expect ${LTS_KERNEL})"
    echo "data /data    : $(mountpoint -q ${DATA_MNT} && echo MOUNTED || echo NO)"
    echo "lustre        : $(mountpoint -q ${LUSTRE_MNT} && echo MOUNTED || echo NO)"
    echo "lustre data   : $(ls ${LUSTRE_MNT}/nuscenes/data >/dev/null 2>&1 && echo OK || echo MISSING)"
    echo "docker nvidia : $(docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia' && echo OK || echo MISSING)"
    echo "docker root   : $(docker info 2>/dev/null | grep 'Docker Root Dir' | awk '{print $NF}')"
    echo "docker image  : $(docker image inspect ${DOCKER_IMG} >/dev/null 2>&1 && echo OK || echo MISSING)"
    echo "repo branch   : $(git -C ${REPO_DIR} branch --show-current 2>/dev/null || echo MISSING)"
    echo "gpus          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
}

case "${PHASE}" in
    phase1)
        log "=== PHASE 1 (pre-reboot) ==="
        setup_data_disk
        setup_docker_nvidia
        setup_repo
        pull_image
        add_amlfs_repo
        switch_to_lts_kernel
        log "=== PHASE 1 done. Reboot required. ==="
        ;;
    phase2)
        log "=== PHASE 2 (post-reboot) ==="
        setup_docker_nvidia
        install_nvidia_module
        setup_lustre
        verify
        log "=== PHASE 2 done. ==="
        ;;
    verify)
        verify
        ;;
    *)
        die "usage: $0 {phase1|phase2|verify}"
        ;;
esac
