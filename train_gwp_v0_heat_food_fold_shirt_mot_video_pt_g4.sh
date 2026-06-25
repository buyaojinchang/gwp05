#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export date="${date:-$(date +%m%d_%H%M)}"

OUTPUT_ROOT="${GWP_MOT_OUTPUT_ROOT:-/shared_disk/users/hengtao.li/codex/gwp-mot}"
export GWP_MOT_OUTPUT_ROOT="$OUTPUT_ROOT"
export TMPDIR="${GWP_MOT_TMPDIR:-/tmp/gwp-mot/gwp_v0_video_pt_${date}_node${MLP_ROLE_INDEX:-${NODE_RANK:-0}}_$$}"
log_dir="$OUTPUT_ROOT/logs/gwp_v0_heat_food_fold_shirt_mot_video_pt"
mkdir -p "$log_dir" "$TMPDIR"

cd /mnt/pfs/users/hengtao.li/varl/gwp-mot

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate /mnt/pfs/users/hengtao.li/conda_envs/gwpmot 2>/dev/null || true

NUM_NODES="${MLP_WORKER_NUM:-${NUM_NODES:-1}}"
NODE_RANK="${MLP_ROLE_INDEX:-${NODE_RANK:-0}}"
NPROC_PER_NODE="${MLP_WORKER_GPU:-${NPROC_PER_NODE:-4}}"
TOTAL_PROCS=$((NUM_NODES * NPROC_PER_NODE))

export MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}"
export MASTER_PORT="${MLP_WORKER_0_PORT:-${MASTER_PORT:-29674}}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export WANDB_MODE="${WANDB_MODE:-offline}"

CONFIG="configs.gwp_v0_heat_food_fold_shirt_mot_video_pt.config"
ACCEL_CONFIG="scripts/accelerate_configs/config_deepspeed_zero2.json"
log_file="$log_dir/${date}_node${NODE_RANK}.log"

echo "=== GWP v0 heat_food + fold_shirt MoT video-pt 1 epoch, 4 GPUs ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE TOTAL_PROCS=$TOTAL_PROCS"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "CONFIG=$CONFIG"
echo "LOG=$log_file"

accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    --gpu_ids "$CUDA_VISIBLE_DEVICES" \
    --num_processes "$TOTAL_PROCS" \
    --num_machines "$NUM_NODES" \
    --machine_rank "$NODE_RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    scripts/train.py --config "$CONFIG" \
    2>&1 | tee "$log_file"
