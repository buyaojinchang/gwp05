#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

gwp_python_nvidia_libs() {
    "$CONDA_ENV/bin/python" - <<'PY'
from pathlib import Path
import site

libs = []
for sp in site.getsitepackages():
    root = Path(sp) / "nvidia"
    if root.exists():
        libs += [str(p) for p in root.glob("*/lib") if p.is_dir()]

print(":".join(libs))
PY
}

gwp_setup_env() {
    export date="${date:-$(date +%m%d_%H%M)}"
    export CONDA_ENV="${CONDA_ENV:-/inspire/hdd/project/robot-dna/sunmingyang-240108120101/wam_locomanip/0_conda_env/gwp05}"
    export GWP_MOT_OUTPUT_ROOT="${GWP_MOT_OUTPUT_ROOT:-/inspire/hdd/project/robot-dna/sunmingyang-240108120101/wam_locomanip/2_data_ckpt_cache/loco_manip/experiments}"

    [ -x "$CONDA_ENV/bin/python" ] || { echo "ERROR: missing $CONDA_ENV/bin/python" >&2; exit 1; }
    [ -x "$CONDA_ENV/bin/accelerate" ] || { echo "ERROR: missing $CONDA_ENV/bin/accelerate" >&2; exit 1; }

    export CONDA_PREFIX="$CONDA_ENV"
    export PATH="$CONDA_ENV/bin:$PATH"

    local py_nvidia_libs
    py_nvidia_libs="$(gwp_python_nvidia_libs)"
    export LD_LIBRARY_PATH="${py_nvidia_libs:+$py_nvidia_libs:}$CONDA_ENV/lib:$CONDA_ENV/lib64:${LD_LIBRARY_PATH:-}"

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

    if [ "${GWP_SETUP_RDMA:-1}" = "1" ] && [ -d /sys/class/infiniband ]; then
        local rdma_ifs
        rdma_ifs="$(ls /sys/class/infiniband 2>/dev/null | tr '\n' ',' | sed 's/,$//' || true)"
        if [ -n "$rdma_ifs" ]; then
            export NCCL_IB_DISABLE=0
            export NCCL_IB_HCA="$rdma_ifs"
            export NCCL_NET_GDR_LEVEL=2
            echo "  NCCL: RDMA enabled, IB HCA=$rdma_ifs"
        fi
    fi
}

gwp_launch() {
    local task="$1"
    shift

    gwp_setup_env
    cd "$REPO_ROOT"

    local log_dir="$GWP_MOT_OUTPUT_ROOT/logs/$task"
    local log_file="$log_dir/${date}_node${NODE_RANK}.log"
    mkdir -p "$log_dir"

    export TMPDIR="${GWP_MOT_TMPDIR:-/tmp/gwp/n${NODE_RANK}_$$}"
    export TMP="$TMPDIR"
    export TEMP="$TMPDIR"
    mkdir -p "$TMPDIR"

    local accel_config="${ACCEL_CONFIG:-scripts/accelerate_configs/config_deepspeed_zero2.json}"
    local -a gpu_args=()
    [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && gpu_args=(--gpu_ids "$CUDA_VISIBLE_DEVICES")

    echo "=== gwp-mot launch: task=$task ==="
    echo "  NPROC_PER_NODE=$NPROC_PER_NODE TOTAL_PROCS=$TOTAL_PROCS NODE_RANK=$NODE_RANK/$NUM_NODES"
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
    echo "  MASTER=$MASTER_ADDR:$MASTER_PORT"
    echo "  CONDA_ENV=$CONDA_ENV"
    echo "  PICK_PLACE_ROOT=${PICK_PLACE_ROOT:-<config default>}"
    echo "  WAN22_DIFFUSERS_PATH=${WAN22_DIFFUSERS_PATH:-<config default>}"
    echo "  LOG=$log_file"
    echo "================================"

    "$CONDA_ENV/bin/accelerate" launch \
        --config_file "$accel_config" \
        "${gpu_args[@]}" \
        --num_processes "$TOTAL_PROCS" \
        --num_machines "$NUM_NODES" \
        --machine_rank "$NODE_RANK" \
        --main_process_ip "$MASTER_ADDR" \
        --main_process_port "$MASTER_PORT" \
        scripts/train_hydra.py task="$task" "$@" \
        2>&1 | tee "$log_file"
}
