#!/usr/bin/env bash
set -euo pipefail

export date="${date:-$(date +%m%d_%H%M)}"
OUTPUT_ROOT="${GWP_MOT_OUTPUT_ROOT:-/shared_disk/users/hengtao.li/codex/gwp-mot}"
export GWP_MOT_OUTPUT_ROOT="$OUTPUT_ROOT"
export TMPDIR="${GWP_MOT_TMPDIR:-/tmp/gwp-mot/all_tshape_${date}_node${MLP_ROLE_INDEX:-${NODE_RANK:-0}}_$$}"
log_dir="$OUTPUT_ROOT/logs/robocasa_all_tshape_mot"
mkdir -p "$log_dir" "$TMPDIR"

cd /mnt/pfs/users/hengtao.li/varl/gwp-mot

# Activate conda env (needed when launched from platform)
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate /mnt/pfs/users/hengtao.li/conda_envs/gwpmot 2>/dev/null || true

# --- Multi-node config ---
NUM_NODES="${MLP_WORKER_NUM:-${NUM_NODES:-1}}"
NODE_RANK="${MLP_ROLE_INDEX:-${NODE_RANK:-0}}"
NPROC_PER_NODE="${MLP_WORKER_GPU:-${NPROC_PER_NODE:-8}}"
TOTAL_PROCS=$((NUM_NODES * NPROC_PER_NODE))

export MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}"
export MASTER_PORT="${MLP_WORKER_0_PORT:-${MASTER_PORT:-29500}}"

# NCCL config
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"

RDMA_IFS=$(ls /sys/class/infiniband/ 2>/dev/null | tr '\n' ',' | sed 's/,$//')
if [ -n "$RDMA_IFS" ]; then
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA="$RDMA_IFS"
    export NCCL_NET_GDR_LEVEL=2
    echo "  NCCL: RDMA enabled, IB HCA=$RDMA_IFS"
else
    RDMA_NET=$(ip link show | grep -E 'rdma|roce|ib' | awk -F: '{print $2}' | tr -d ' ' | head -1)
    if [ -n "$RDMA_NET" ]; then
        export NCCL_IB_DISABLE=0
        export NCCL_SOCKET_IFNAME="$RDMA_NET"
        echo "  NCCL: RDMA net interface=$RDMA_NET"
    else
        echo "  NCCL: No RDMA detected, using default TCP"
        ip -brief addr show 2>/dev/null || ifconfig -a 2>/dev/null | grep -E "^[a-z]|inet "
    fi
fi

CONFIG="configs.robocasa_all_tshape_mot.config"
ACCEL_CONFIG="scripts/accelerate_configs/config_deepspeed_zero2.json"

echo "=== Training RoboCasa All T-Shape MoT ==="
echo "  NUM_NODES:      $NUM_NODES"
echo "  NODE_RANK:      $NODE_RANK"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"
echo "  TOTAL_PROCS:    $TOTAL_PROCS"
echo "  MASTER_ADDR:    $MASTER_ADDR"
echo "  MASTER_PORT:    $MASTER_PORT"
echo "  CONFIG:         $CONFIG"
echo "  Layout:         T-shape (head=agentview_right 320x256 + 2x wrist 160x128)"
echo "================================"

accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    --num_processes "$TOTAL_PROCS" \
    --num_machines "$NUM_NODES" \
    --machine_rank "$NODE_RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    scripts/train.py --config "$CONFIG" \
    2>&1 | tee "$log_dir/${date}_node${NODE_RANK}.log"
