#!/usr/bin/env bash
# Shared launch helpers for gwp-mot Hydra training scripts.
#
# Usage from a thin wrapper (in the repo root):
#   export GWP_DEFAULT_NPROC=8            # default GPUs/proc when not on platform
#   export MASTER_PORT="${MASTER_PORT:-29500}"
#   source "$(dirname "${BASH_SOURCE[0]}")/scripts/launch_lib.sh"
#   gwp_launch <task_name> [extra hydra overrides...]
#
# Tunables honored via env (all optional):
#   CUDA_VISIBLE_DEVICES, NUM_NODES/NODE_RANK/NPROC_PER_NODE (or MLP_* on platform),
#   MASTER_ADDR/MASTER_PORT, GWP_MOT_OUTPUT_ROOT, GWP_MOT_TMPDIR, WANDB_MODE,
#   ACCEL_CONFIG, CONDA_ENV.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

gwp_setup_rdma() {
    local rdma_ifs rdma_net
    rdma_ifs=$(ls /sys/class/infiniband/ 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    if [ -n "$rdma_ifs" ]; then
        export NCCL_IB_DISABLE=0
        export NCCL_IB_HCA="$rdma_ifs"
        export NCCL_NET_GDR_LEVEL=2
        echo "  NCCL: RDMA enabled, IB HCA=$rdma_ifs"
        return
    fi
    rdma_net=$(ip link show 2>/dev/null | grep -E 'rdma|roce|ib' | awk -F: '{print $2}' | tr -d ' ' | head -1)
    if [ -n "$rdma_net" ]; then
        export NCCL_IB_DISABLE=0
        export NCCL_SOCKET_IFNAME="$rdma_net"
        echo "  NCCL: RDMA net interface=$rdma_net"
    fi
}

gwp_setup_env() {
    export date="${date:-$(date +%m%d_%H%M)}"
    export GWP_MOT_OUTPUT_ROOT="${GWP_MOT_OUTPUT_ROOT:-/shared_disk/users/hengtao.li/codex/gwp-mot}"

    NUM_NODES="${MLP_WORKER_NUM:-${NUM_NODES:-1}}"
    NODE_RANK="${MLP_ROLE_INDEX:-${NODE_RANK:-0}}"
    NPROC_PER_NODE="${MLP_WORKER_GPU:-${NPROC_PER_NODE:-${GWP_DEFAULT_NPROC:-8}}}"
    TOTAL_PROCS=$((NUM_NODES * NPROC_PER_NODE))

    export MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}"
    export MASTER_PORT="${MLP_WORKER_0_PORT:-${MASTER_PORT:-29500}}"

    export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
    export TORCH_DISTRIBUTED_TIMEOUT_SEC="${TORCH_DISTRIBUTED_TIMEOUT_SEC:-3600}"
    export WANDB_MODE="${WANDB_MODE:-offline}"

    [ "${GWP_SETUP_RDMA:-1}" = "1" ] && gwp_setup_rdma || true

    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV:-/mnt/pfs/users/hengtao.li/conda_envs/gwpmot}" 2>/dev/null || true
}

gwp_launch() {
    local task="$1"; shift
    gwp_setup_env
    cd "$REPO_ROOT"

    local log_dir="$GWP_MOT_OUTPUT_ROOT/logs/$task"
    local log_file="$log_dir/${date}_node${NODE_RANK}.log"
    export TMPDIR="${GWP_MOT_TMPDIR:-/tmp/gwp-mot/${task}_${date}_node${NODE_RANK}_$$}"
    mkdir -p "$log_dir" "$TMPDIR"

    local accel_config="${ACCEL_CONFIG:-scripts/accelerate_configs/config_deepspeed_zero2.json}"

    echo "=== gwp-mot launch: task=$task ==="
    echo "  REPO_ROOT=$REPO_ROOT"
    echo "  NPROC_PER_NODE=$NPROC_PER_NODE TOTAL_PROCS=$TOTAL_PROCS NODE_RANK=$NODE_RANK/$NUM_NODES"
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
    echo "  MASTER=$MASTER_ADDR:$MASTER_PORT  WANDB_MODE=$WANDB_MODE"
    echo "  ACCEL_CONFIG=$accel_config"
    echo "  hydra overrides: $*"
    echo "  LOG=$log_file"
    echo "================================"

    accelerate launch \
        --config_file "$accel_config" \
        ${CUDA_VISIBLE_DEVICES:+--gpu_ids "$CUDA_VISIBLE_DEVICES"} \
        --num_processes "$TOTAL_PROCS" \
        --num_machines "$NUM_NODES" \
        --machine_rank "$NODE_RANK" \
        --main_process_ip "$MASTER_ADDR" \
        --main_process_port "$MASTER_PORT" \
        scripts/train_hydra.py task="$task" "$@" \
        2>&1 | tee "$log_file"
}
