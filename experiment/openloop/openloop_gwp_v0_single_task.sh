#!/usr/bin/env bash
# MoT open-loop evaluation for one GWP-V0 real-world task.
#
# Usage:
#   bash experiment/openloop/openloop_gwp_v0_single_task.sh <CHECKPOINT_PATH> [TASK_NAME] [GPU_ID] [EPISODE_IDX]
#
# Example:
#   bash experiment/openloop/openloop_gwp_v0_single_task.sh \
#     /shared_disk/users/hengtao.li/codex/gwp-mot/experiments/.../checkpoint-110000/model_ema.pt \
#     heat_food 0 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CHECKPOINT="${1:?Usage: bash openloop_gwp_v0_single_task.sh <CHECKPOINT_PATH> [TASK_NAME] [GPU_ID] [EPISODE_IDX]}"
TASK_NAME="${2:-${TASK_NAME:-heat_food}}"
GPU="${3:-${GPU:-0}}"
EPISODE_IDX="${4:-${EPISODE_IDX:-0}}"

GWP_V0_ROOT="${GWP_V0_ROOT:-/shared_disk/users/hengtao.li/giga_real_data/gwp_v0}"
TASK_DATA_ROOT="${TASK_DATA_ROOT:-$GWP_V0_ROOT/$TASK_NAME}"
STATS_PATH="${STATS_PATH:-$TASK_DATA_ROOT/norm_stats_delta.json}"

# openloop_eval.py discovers datasets as <data_root>/<task>/lerobot.
# GWP-V0 single-task dirs are themselves LeRobot repos, so create a lightweight
# symlink adapter instead of copying data.
OPENLOOP_ROOT="${GWP_V0_OPENLOOP_ROOT:-/tmp/gwpv0_openloop_root}"
ADAPTER_TASK_DIR="$OPENLOOP_ROOT/$TASK_NAME"
ADAPTER_LEROBOT_DIR="$ADAPTER_TASK_DIR/lerobot"

DST_W="${DST_W:-320}"
DST_H="${DST_H:-256}"
NUM_FRAMES="${NUM_FRAMES:-36}"
ACTION_CHUNK="${ACTION_CHUNK:-30}"
REPLAN_STEPS="${REPLAN_STEPS:-30}"
NUM_STEPS="${NUM_STEPS:-10}"
ACTION_DIM="${ACTION_DIM:-14}"
STATE_DIM="${STATE_DIM:-14}"
ACTION_FLOW_SHIFT="${ACTION_FLOW_SHIFT:-5.0}"
SEED="${SEED:-42}"
TSHAPE_HEAD_INDEX="${TSHAPE_HEAD_INDEX:-0}"
MAX_DATASETS="${MAX_DATASETS:-1}"
ONE_PER_TASK="${ONE_PER_TASK:-1}"
INPUT_VIEW_MODE="${INPUT_VIEW_MODE:-auto}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/pfs/users/hengtao.li/conda_envs/gwpmot/bin/python}"

cd "$PROJECT_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python)"
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi
if [[ ! -d "$TASK_DATA_ROOT" ]]; then
    echo "Task data root not found: $TASK_DATA_ROOT" >&2
    exit 1
fi
if [[ ! -f "$STATS_PATH" ]]; then
    echo "Norm stats not found: $STATS_PATH" >&2
    exit 1
fi

mkdir -p "$ADAPTER_TASK_DIR"
ln -sfn "$TASK_DATA_ROOT" "$ADAPTER_LEROBOT_DIR"

echo "============================================================"
echo "  MoT Open-Loop Evaluation [GWP-V0 single task]"
echo "  Checkpoint:        $CHECKPOINT"
echo "  GPU:               $GPU"
echo "  Task:              $TASK_NAME"
echo "  Task data root:    $TASK_DATA_ROOT"
echo "  Adapter root:      $OPENLOOP_ROOT"
echo "  Stats path:        $STATS_PATH"
echo "  Episode:           $EPISODE_IDX"
echo "  Num frames:        $NUM_FRAMES"
echo "  Action chunk:      $ACTION_CHUNK"
echo "  Replan steps:      $REPLAN_STEPS"
echo "  Compare:           raw_from_delta"
echo "  GT delta mode:     agilex_cobot_magic"
echo "  Eval prefix only:  yes"
echo "  Action/state dim:  $ACTION_DIM / $STATE_DIM"
echo "  dst_size (WxH):    ${DST_W} x ${DST_H}"
echo "  Head index:        $TSHAPE_HEAD_INDEX"
echo "  Input view mode:   $INPUT_VIEW_MODE"
echo "  Python:            $PYTHON_BIN"
echo "============================================================"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u experiment/openloop/openloop_eval.py \
    --checkpoint_path "$CHECKPOINT" \
    --data_root "$OPENLOOP_ROOT" \
    --stats_path "$STATS_PATH" \
    --max_datasets "$MAX_DATASETS" \
    --task_set all \
    --episode_idx "$EPISODE_IDX" \
    --num_frames "$NUM_FRAMES" \
    --action_chunk "$ACTION_CHUNK" \
    --replan_steps "$REPLAN_STEPS" \
    --num_steps "$NUM_STEPS" \
    --action_dim "$ACTION_DIM" \
    --state_dim "$STATE_DIM" \
    --seed "$SEED" \
    --action_flow_shift "$ACTION_FLOW_SHIFT" \
    --dst_size "$DST_W" "$DST_H" \
    --comparison_space raw_from_delta \
    --gt_delta_mode agilex_cobot_magic \
    --eval_replan_prefix_only \
    --input_view_mode "$INPUT_VIEW_MODE" \
    --tshape \
    --tshape_head_index "$TSHAPE_HEAD_INDEX" \
    $(if [[ "$ONE_PER_TASK" == "1" ]]; then echo "--one_per_task"; fi)
