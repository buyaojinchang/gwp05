#!/bin/bash
# MoT open-loop evaluation on target-domain RoboCasa T-shape datasets.
# Dataset layout: <DATA_ROOT>/<task>/lerobot
# Usage: bash openloop_tshape.sh <CHECKPOINT_PATH> [GPU_ID] [MAX_DATASETS] [EPISODE_IDX]
# Example:
#   bash openloop_tshape.sh /path/to/checkpoint/model_ema.pt 0 18 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CHECKPOINT="${1:?Usage: bash openloop_tshape.sh <CHECKPOINT_PATH> [GPU_ID] [MAX_DATASETS] [EPISODE_IDX]}"
GPU="${2:-0}"
MAX_DATASETS="${3:-18}"
EPISODE_IDX="${4:-0}"

DATA_ROOT="${DATA_ROOT:-/shared_disk/users/hengtao.li/robocasa_datasets/collected/epoch0}"
STATS_PATH="${STATS_PATH:-/shared_disk/users/hengtao.li/robocasa_datasets/v1.0/pretrain_gwp/norm_stats_delta.json}"
DST_W="${DST_W:-320}"
DST_H="${DST_H:-256}"
NUM_FRAMES="${NUM_FRAMES:-24}"
ACTION_CHUNK="${ACTION_CHUNK:-20}"
REPLAN_STEPS="${REPLAN_STEPS:-20}"
NUM_STEPS="${NUM_STEPS:-10}"
ACTION_FLOW_SHIFT="${ACTION_FLOW_SHIFT:-5.0}"
TSHAPE_HEAD_INDEX="${TSHAPE_HEAD_INDEX:-2}"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "  MoT Open-Loop Evaluation [Target T-shape]"
echo "  Checkpoint:        $CHECKPOINT"
echo "  GPU:               $GPU"
echo "  Max datasets:      $MAX_DATASETS"
echo "  Episode:           $EPISODE_IDX"
echo "  Data root:         $DATA_ROOT"
echo "  Stats path:        $STATS_PATH"
echo "  Num frames:        $NUM_FRAMES"
echo "  Action chunk:      $ACTION_CHUNK"
echo "  Replan steps:      $REPLAN_STEPS"
echo "  dst_size (WxH):    ${DST_W} x ${DST_H}"
echo "  Head index:        $TSHAPE_HEAD_INDEX"
echo "============================================================"

CUDA_VISIBLE_DEVICES="$GPU" python -u experiment/openloop/openloop_eval.py \
    --checkpoint_path "$CHECKPOINT" \
    --data_root "$DATA_ROOT" \
    --stats_path "$STATS_PATH" \
    --max_datasets "$MAX_DATASETS" \
    --episode_idx "$EPISODE_IDX" \
    --num_frames "$NUM_FRAMES" \
    --action_chunk "$ACTION_CHUNK" \
    --replan_steps "$REPLAN_STEPS" \
    --num_steps "$NUM_STEPS" \
    --action_flow_shift "$ACTION_FLOW_SHIFT" \
    --dst_size "$DST_W" "$DST_H" \
    --tshape \
    --tshape_head_index "$TSHAPE_HEAD_INDEX"
