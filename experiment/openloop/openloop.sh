#!/bin/bash
# MoT open-loop evaluation on RoboCasa atomic-seen training datasets.
# Usage: bash openloop.sh <CHECKPOINT_PATH> [GPU_ID] [MAX_DATASETS] [EPISODE_IDX]
# Example:
#   bash openloop.sh /path/to/checkpoint/model_ema.pt 0 18 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CHECKPOINT="${1:?Usage: bash openloop.sh <CHECKPOINT_PATH> [GPU_ID] [MAX_DATASETS] [EPISODE_IDX]}"
GPU="${2:-0}"
MAX_DATASETS="${3:-18}"
EPISODE_IDX="${4:-0}"

DATA_ROOT="${ATOMIC_SEEN_DATA_ROOT:-/shared_disk/users/hengtao.li/robocasa_datasets/v1.0/pretrain_gwp}"
STATS_PATH="${STATS_PATH:-/shared_disk/users/hengtao.li/robocasa_datasets/v1.0/pretrain_gwp/norm_stats_delta.json}"
DST_W="${DST_W:-320}"
DST_H="${DST_H:-256}"
NUM_FRAMES="${NUM_FRAMES:-24}"
ACTION_CHUNK="${ACTION_CHUNK:-24}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
NUM_STEPS="${NUM_STEPS:-10}"
ACTION_FLOW_SHIFT="${ACTION_FLOW_SHIFT:-5.0}"
TSHAPE_HEAD_INDEX="${TSHAPE_HEAD_INDEX:-2}"
TASK_SET="${TASK_SET:-atomic_seen}"
ONE_PER_TASK="${ONE_PER_TASK:-1}"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "  MoT Open-Loop Evaluation [RoboCasa T-shape]"
echo "  Checkpoint:        $CHECKPOINT"
echo "  GPU:               $GPU"
echo "  Max datasets:      $MAX_DATASETS"
echo "  Task set:          $TASK_SET"
echo "  One per task:      $ONE_PER_TASK"
echo "  Episode:           $EPISODE_IDX"
echo "  Data root:         $DATA_ROOT"
echo "  Stats path:        $STATS_PATH"
echo "  Num frames:        $NUM_FRAMES"
echo "  Action chunk:      $ACTION_CHUNK"
echo "  Replan steps:      $REPLAN_STEPS"
echo "  dst_size (WxH):    ${DST_W} x ${DST_H}"
echo "  Head index:        $TSHAPE_HEAD_INDEX"
echo "  Dataset intent:    training pretrain_gwp atomic seen, one episode per task"
echo "============================================================"

CUDA_VISIBLE_DEVICES="$GPU" python -u experiment/openloop/openloop_eval.py \
    --checkpoint_path "$CHECKPOINT" \
    --data_root "$DATA_ROOT" \
    --stats_path "$STATS_PATH" \
    --max_datasets "$MAX_DATASETS" \
    --task_set "$TASK_SET" \
    --episode_idx "$EPISODE_IDX" \
    --num_frames "$NUM_FRAMES" \
    --action_chunk "$ACTION_CHUNK" \
    --replan_steps "$REPLAN_STEPS" \
    --num_steps "$NUM_STEPS" \
    --action_flow_shift "$ACTION_FLOW_SHIFT" \
    --dst_size "$DST_W" "$DST_H" \
    $(if [[ "$ONE_PER_TASK" == "1" ]]; then echo "--one_per_task"; fi) \
    --tshape \
    --tshape_head_index "$TSHAPE_HEAD_INDEX"
