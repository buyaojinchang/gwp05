#!/usr/bin/env bash
set -euo pipefail

# Fresh heat_food MoT joint/action training from the completed video-pretrain EMA.
# 8-GPU default launcher. This starts a clean run; it does not resume h20/h20_r2.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export date="${date:-heat_food_joint5ep_restart_g8_$(date +%m%d_%H%M)}"

OUTPUT_ROOT="${GWP_MOT_OUTPUT_ROOT:-/shared_disk/users/hengtao.li/codex/gwp-mot}"
export GWP_MOT_OUTPUT_ROOT="$OUTPUT_ROOT"
export TMPDIR="${GWP_MOT_TMPDIR:-/tmp/gwp_heat_food_joint_restart_g8_${MLP_ROLE_INDEX:-${NODE_RANK:-0}}_$$}"

log_dir="$OUTPUT_ROOT/logs/gwp_v0_heat_food_mot_joint_from_videopt_5ep_restart"
mkdir -p "$log_dir" "$TMPDIR"

cd /mnt/pfs/users/hengtao.li/varl/gwp-mot

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate /mnt/pfs/users/hengtao.li/conda_envs/gwpmot 2>/dev/null || true

NUM_NODES="${MLP_WORKER_NUM:-${NUM_NODES:-1}}"
NODE_RANK="${MLP_ROLE_INDEX:-${NODE_RANK:-0}}"
NPROC_PER_NODE="${MLP_WORKER_GPU:-${NPROC_PER_NODE:-8}}"
TOTAL_PROCS=$((NUM_NODES * NPROC_PER_NODE))

export MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}"
export MASTER_PORT="${MLP_WORKER_0_PORT:-${MASTER_PORT:-29694}}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export TORCH_NCCL_TIMEOUT_SEC="${TORCH_NCCL_TIMEOUT_SEC:-3600}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export TORCH_DISTRIBUTED_TIMEOUT_SEC="${TORCH_DISTRIBUTED_TIMEOUT_SEC:-3600}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"
export LEROBOT_SAMPLE_TIMEOUT_SEC="${LEROBOT_SAMPLE_TIMEOUT_SEC:-120}"
export LEROBOT_MAX_SAMPLE_RETRIES="${LEROBOT_MAX_SAMPLE_RETRIES:-5}"
export LEROBOT_DATALOADER_TIMEOUT_SEC="${LEROBOT_DATALOADER_TIMEOUT_SEC:-300}"
export LEROBOT_NUM_WORKERS="${LEROBOT_NUM_WORKERS:-8}"
export LEROBOT_PREFETCH_FACTOR="${LEROBOT_PREFETCH_FACTOR:-4}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_PROJECT="${WANDB_PROJECT:-gwp-mot}"
export WANDB_NAME="${WANDB_NAME:-gwp_v0_heat_food_mot_joint_from_videopt_5ep_${date}}"

export MOT_STAGE1_CHECKPOINT="${MOT_STAGE1_CHECKPOINT:-/shared_disk/users/hengtao.li/codex/gwp-mot/experiments/gwp_v0_heat_food_fold_shirt_mot_video_pt_0530_videopt_e1_g8_h20_online/checkpoint-17890/model_ema.pt}"

CONFIG="configs.gwp_v0_heat_food_mot_joint_from_videopt_5ep_restart.config"
ACCEL_CONFIG="${ACCEL_CONFIG:-scripts/accelerate_configs/config_deepspeed_zero2.json}"
log_file="$log_dir/${date}_node${NODE_RANK}.log"

echo "=== GWP v0 heat_food MoT joint/action 5 epochs from video-pt restart, 8 GPUs ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE TOTAL_PROCS=$TOTAL_PROCS"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "CONFIG=$CONFIG"
echo "ACCEL_CONFIG=$ACCEL_CONFIG"
echo "WANDB_MODE=$WANDB_MODE WANDB_PROJECT=$WANDB_PROJECT WANDB_NAME=$WANDB_NAME"
echo "MOT_STAGE1_CHECKPOINT=$MOT_STAGE1_CHECKPOINT"
echo "LEROBOT_VIDEO_BACKEND=$LEROBOT_VIDEO_BACKEND"
echo "LEROBOT_SAMPLE_TIMEOUT_SEC=$LEROBOT_SAMPLE_TIMEOUT_SEC LEROBOT_MAX_SAMPLE_RETRIES=$LEROBOT_MAX_SAMPLE_RETRIES"
echo "LEROBOT_DATALOADER_TIMEOUT_SEC=$LEROBOT_DATALOADER_TIMEOUT_SEC LEROBOT_NUM_WORKERS=$LEROBOT_NUM_WORKERS LEROBOT_PREFETCH_FACTOR=$LEROBOT_PREFETCH_FACTOR"
echo "GWP_MOT_OUTPUT_ROOT=$GWP_MOT_OUTPUT_ROOT"
echo "TMPDIR=$TMPDIR"
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
