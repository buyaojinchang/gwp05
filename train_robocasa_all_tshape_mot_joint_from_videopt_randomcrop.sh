#!/usr/bin/env bash
set -euo pipefail

# Fresh RoboCasa all-data MoT joint/action training from video-pretrain EMA.
# This initializes model weights from GWP_VIDEO_PT_CKPT and starts a new phase.

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ -n "${MLP_WORKER_GPU:-}" ]; then
        CUDA_VISIBLE_DEVICES=""
        for ((i = 0; i < MLP_WORKER_GPU; i++)); do
            if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
                CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES},${i}"
            else
                CUDA_VISIBLE_DEVICES="${i}"
            fi
        done
    else
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    fi
fi
export CUDA_VISIBLE_DEVICES
export date="${date:-robocasa_all_videopt_action_joint_crop320_$(date +%m%d_%H%M)}"

OUTPUT_ROOT="${GWP_MOT_OUTPUT_ROOT:-/shared_disk/users/hengtao.li/codex/gwp-mot}"
export GWP_MOT_OUTPUT_ROOT="$OUTPUT_ROOT"
export TMPDIR="${GWP_MOT_TMPDIR:-/tmp/gjv${MLP_ROLE_INDEX:-${NODE_RANK:-0}}_$$}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

log_dir="$OUTPUT_ROOT/logs/robocasa_all_tshape_mot_joint_from_videopt_randomcrop"
mkdir -p "$log_dir" "$TMPDIR"

cd /mnt/pfs/users/hengtao.li/varl/gwp-mot

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate /mnt/pfs/users/hengtao.li/conda_envs/gwpmot 2>/dev/null || true

VIDEO_PT_DIR="${GWP_VIDEO_PT_DIR:-$OUTPUT_ROOT/experiments/robocasa_all_tshape_mot_video_pt_epoch1_randomcrop_robocasa_all_videopt_crop320_0603_0212}"
if [ -z "${GWP_VIDEO_PT_CKPT:-}" ]; then
    GWP_VIDEO_PT_CKPT=""
    mapfile -t _video_pt_ckpts < <(find "$VIDEO_PT_DIR" -mindepth 2 -maxdepth 2 -type f -path "$VIDEO_PT_DIR/checkpoint-*/model_ema.pt" | sort -Vr)
    for _ckpt in "${_video_pt_ckpts[@]}"; do
        _ckpt_dir="$(dirname "$_ckpt")"
        if [ ! -f "$_ckpt_dir/training_state.pt" ]; then
            echo "Skipping incomplete checkpoint without training_state.pt: $_ckpt"
            continue
        fi
        if python3 - "$_ckpt" <<'PY'
import sys
import zipfile

path = sys.argv[1]
try:
    with zipfile.ZipFile(path) as zf:
        zf.namelist()
except Exception as exc:
    print(f"Skipping unreadable checkpoint {path}: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
        then
            GWP_VIDEO_PT_CKPT="$_ckpt"
            break
        fi
    done
fi
if [ -z "${GWP_VIDEO_PT_CKPT:-}" ] || [ ! -f "$GWP_VIDEO_PT_CKPT" ]; then
    echo "ERROR: no video-pretrain EMA checkpoint found."
    echo "  VIDEO_PT_DIR=$VIDEO_PT_DIR"
    echo "  Set GWP_VIDEO_PT_CKPT=/path/to/checkpoint-*/model_ema.pt to override."
    exit 1
fi
export GWP_VIDEO_PT_CKPT

NUM_NODES="${MLP_WORKER_NUM:-${NUM_NODES:-1}}"
NODE_RANK="${MLP_ROLE_INDEX:-${NODE_RANK:-0}}"
if [ -z "${NPROC_PER_NODE:-}" ]; then
    IFS=',' read -r -a _visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
    NPROC_PER_NODE="${#_visible_gpus[@]}"
fi
TOTAL_PROCS=$((NUM_NODES * NPROC_PER_NODE))

export MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}"
export MASTER_PORT="${MLP_WORKER_0_PORT:-${MASTER_PORT:-29542}}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export TORCH_NCCL_TIMEOUT_SEC="${TORCH_NCCL_TIMEOUT_SEC:-3600}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export TORCH_DISTRIBUTED_TIMEOUT_SEC="${TORCH_DISTRIBUTED_TIMEOUT_SEC:-3600}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# DeepSpeed interprets DEEPSPEED_TIMEOUT in minutes.
export DEEPSPEED_TIMEOUT="${DEEPSPEED_TIMEOUT:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-decord}"
export LEROBOT_SAMPLE_TIMEOUT_SEC="${LEROBOT_SAMPLE_TIMEOUT_SEC:-120}"
export LEROBOT_MAX_SAMPLE_RETRIES="${LEROBOT_MAX_SAMPLE_RETRIES:-5}"
export LEROBOT_DATALOADER_TIMEOUT_SEC="${LEROBOT_DATALOADER_TIMEOUT_SEC:-300}"
export LEROBOT_NUM_WORKERS="${LEROBOT_NUM_WORKERS:-8}"
export LEROBOT_PREFETCH_FACTOR="${LEROBOT_PREFETCH_FACTOR:-4}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_PROJECT="${WANDB_PROJECT:-gwp-mot}"
export GWP_MAX_EPOCHS="${GWP_MAX_EPOCHS:-5}"
export GWP_DECAY_EPOCHS="${GWP_DECAY_EPOCHS:-$GWP_MAX_EPOCHS}"
export GWP_ACTION_LOSS_WEIGHT="${GWP_ACTION_LOSS_WEIGHT:-1.0}"
export GWP_VISUAL_LOSS_WEIGHT="${GWP_VISUAL_LOSS_WEIGHT:-1.0}"
export GWP_USE_GT_ACTION_FOR_VIDEO="${GWP_USE_GT_ACTION_FOR_VIDEO:-0}"
if [ -z "${GWP_BATCH_SIZE_PER_GPU:-}" ]; then
    if [ "$NPROC_PER_NODE" -le 2 ]; then
        # Two-GPU smoke has much less ZeRO-2 optimizer-state sharding than the
        # real 8-GPU run. Keep it tiny so the first optimizer step can allocate
        # CAME state tensors.
        export GWP_BATCH_SIZE_PER_GPU=1
    else
        export GWP_BATCH_SIZE_PER_GPU=4
    fi
fi
if [ -z "${GWP_GRAD_ACCUM_STEPS:-}" ]; then
    if [ "$NPROC_PER_NODE" -le 2 ]; then
        export GWP_GRAD_ACCUM_STEPS=1
    else
        export GWP_GRAD_ACCUM_STEPS=4
    fi
fi
export GWP_VIEW_INTERVAL="${GWP_VIEW_INTERVAL:-1000000}"

CONFIG="configs.robocasa_all_tshape_mot_joint_from_videopt_randomcrop.config"
ACCEL_CONFIG="${ACCEL_CONFIG:-scripts/accelerate_configs/config_deepspeed_zero2_video_pt_timeout.json}"
log_file="$log_dir/${date}_node${NODE_RANK}.log"
exec > >(tee "$log_file") 2>&1

echo "=== RoboCasa all-data T-shape MoT joint/action from video-pt random-crop ==="
echo "  CUDA_VISIBLE_DEVICES:       $CUDA_VISIBLE_DEVICES"
echo "  NUM_NODES:                  $NUM_NODES"
echo "  NODE_RANK:                  $NODE_RANK"
echo "  NPROC_PER_NODE:             $NPROC_PER_NODE"
echo "  TOTAL_PROCS:                $TOTAL_PROCS"
echo "  MASTER_ADDR:                $MASTER_ADDR"
echo "  MASTER_PORT:                $MASTER_PORT"
echo "  TORCH_DIST_TIMEOUT:         $TORCH_DISTRIBUTED_TIMEOUT_SEC"
echo "  DEEPSPEED_TIMEOUT:          $DEEPSPEED_TIMEOUT minutes"
echo "  ACCEL_CONFIG:               $ACCEL_CONFIG"
echo "  CONFIG:                     $CONFIG"
echo "  VIDEO_PT_DIR:               $VIDEO_PT_DIR"
echo "  GWP_VIDEO_PT_CKPT:          $GWP_VIDEO_PT_CKPT"
echo "  GWP_RESUME:                 ${GWP_RESUME:-0}"
echo "  GWP_MAX_EPOCHS:             $GWP_MAX_EPOCHS"
echo "  GWP_ACTION_LOSS_WEIGHT:     $GWP_ACTION_LOSS_WEIGHT"
echo "  GWP_VISUAL_LOSS_WEIGHT:     $GWP_VISUAL_LOSS_WEIGHT"
echo "  GWP_USE_GT_ACTION_FOR_VIDEO:$GWP_USE_GT_ACTION_FOR_VIDEO"
echo "  GWP_BATCH_SIZE_PER_GPU:     $GWP_BATCH_SIZE_PER_GPU"
echo "  GWP_GRAD_ACCUM_STEPS:       $GWP_GRAD_ACCUM_STEPS"
echo "  GWP_VIEW_INTERVAL:          $GWP_VIEW_INTERVAL"
echo "  LEROBOT_VIDEO_BACKEND:      $LEROBOT_VIDEO_BACKEND"
echo "  LEROBOT_SAMPLE_TIMEOUT_SEC: $LEROBOT_SAMPLE_TIMEOUT_SEC"
echo "  LEROBOT_MAX_SAMPLE_RETRIES: $LEROBOT_MAX_SAMPLE_RETRIES"
echo "  LEROBOT_DATALOADER_TIMEOUT:$LEROBOT_DATALOADER_TIMEOUT_SEC"
echo "  LEROBOT_NUM_WORKERS:        $LEROBOT_NUM_WORKERS"
echo "  LEROBOT_PREFETCH_FACTOR:    $LEROBOT_PREFETCH_FACTOR"
echo "  OUTPUT_ROOT:                $OUTPUT_ROOT"
echo "  TMPDIR:                     $TMPDIR"
echo "  LOG:                        $log_file"
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
