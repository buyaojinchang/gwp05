#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export date="${date:-$(date +%m%d_%H%M)}"

OUTPUT_ROOT="${GWP_MOT_OUTPUT_ROOT:-/shared_disk/users/hengtao.li/codex/gwp-mot}"
export GWP_MOT_OUTPUT_ROOT="$OUTPUT_ROOT"
export TMPDIR="${GWP_MOT_TMPDIR:-/tmp/gwp_crop320_${date}_n${MLP_ROLE_INDEX:-${NODE_RANK:-0}}_$$}"

log_dir="$OUTPUT_ROOT/logs/robocasa_all_tshape_mot_video_pt_epoch1_randomcrop"
mkdir -p "$log_dir" "$TMPDIR"

cd /mnt/pfs/users/hengtao.li/varl/gwp-mot

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate /mnt/pfs/users/hengtao.li/conda_envs/gwpmot 2>/dev/null || true

NUM_NODES="${MLP_WORKER_NUM:-${NUM_NODES:-1}}"
NODE_RANK="${MLP_ROLE_INDEX:-${NODE_RANK:-0}}"
if [ -z "${NPROC_PER_NODE:-}" ]; then
    IFS=',' read -r -a _visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
    NPROC_PER_NODE="${#_visible_gpus[@]}"
fi
TOTAL_PROCS=$((NUM_NODES * NPROC_PER_NODE))

export MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}"
export MASTER_PORT="${MASTER_PORT:-29540}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export TORCH_DISTRIBUTED_TIMEOUT_SEC="${TORCH_DISTRIBUTED_TIMEOUT_SEC:-3600}"
# DeepSpeed interprets DEEPSPEED_TIMEOUT in minutes.
export DEEPSPEED_TIMEOUT="${DEEPSPEED_TIMEOUT:-60}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="configs.robocasa_all_tshape_mot_video_pt_epoch1_randomcrop.config"
ACCEL_CONFIG="${ACCEL_CONFIG:-scripts/accelerate_configs/config_deepspeed_zero2_video_pt_timeout.json}"
log_file="$log_dir/${date}_node${NODE_RANK}.log"
exec > >(tee "$log_file") 2>&1

echo "=== Training RoboCasa all-data T-shape MoT video-pt epoch1 random-crop ==="
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  NUM_NODES:            $NUM_NODES"
echo "  NODE_RANK:            $NODE_RANK"
echo "  NPROC_PER_NODE:       $NPROC_PER_NODE"
echo "  TOTAL_PROCS:          $TOTAL_PROCS"
echo "  MASTER_ADDR:          $MASTER_ADDR"
echo "  MASTER_PORT:          $MASTER_PORT"
echo "  TORCH_DIST_TIMEOUT:   $TORCH_DISTRIBUTED_TIMEOUT_SEC"
echo "  DEEPSPEED_TIMEOUT:    $DEEPSPEED_TIMEOUT minutes"
echo "  ACCEL_CONFIG:         $ACCEL_CONFIG"
echo "  GWP_RESUME:           ${GWP_RESUME:-0}"
echo "  CONFIG:               $CONFIG"
echo "  OUTPUT_ROOT:          $OUTPUT_ROOT"
echo "  TMPDIR:               $TMPDIR"
echo "  LOG:                  $log_file"
echo "================================"

accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    --gpu_ids "$CUDA_VISIBLE_DEVICES" \
    --num_processes "$TOTAL_PROCS" \
    --num_machines "$NUM_NODES" \
    --machine_rank "$NODE_RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    scripts/train.py --config "$CONFIG"
