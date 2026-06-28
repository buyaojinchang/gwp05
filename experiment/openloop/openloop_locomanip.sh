#!/bin/bash
# MoT open-loop evaluation for the locomanip pick_place (G1 sonic) task.
#
# Predicts the 66-d sonic latent action from a single ego_view frame + 2-token
# state (43-d joint + 66-d sonic latent), and compares against the GT latent.
#
# Usage:
#   bash experiment/openloop/openloop_locomanip.sh <CHECKPOINT_PATH> [GPU_ID] [MAX_EPISODES] [EPISODE_IDX...]
# Examples:
#   bash experiment/openloop/openloop_locomanip.sh \
#     /shared_disk/users/hengtao.li/locomanip/exp/debug/model_ema.pt 0 3
#   # specific episodes:
#   bash experiment/openloop/openloop_locomanip.sh \
#     /shared_disk/users/hengtao.li/locomanip/exp/debug/model_ema.pt 0 0 0 1 5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CHECKPOINT="${1:?Usage: bash openloop_locomanip.sh <CHECKPOINT_PATH> [GPU_ID] [MAX_EPISODES] [EPISODE_IDX...]}"
GPU="${2:-0}"
MAX_EPISODES="${3:-3}"
shift $(( $# < 3 ? $# : 3 )) || true
EPISODE_INDICES="$*"   # remaining args, optional explicit episode indices

DATA_ROOT="${PICK_PLACE_ROOT:-/shared_disk/users/hengtao.li/locomanip/data/pick_place_gwp}"
STATS_PATH="${STATS_PATH:-$DATA_ROOT/norm_stats_delta.json}"
MODEL_ID="${WAN22_DIFFUSERS_PATH:-/shared_disk/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers}"
DST_W="${DST_W:-320}"
DST_H="${DST_H:-256}"
NUM_FRAMES="${NUM_FRAMES:-56}"
REPLAN_STEPS="${REPLAN_STEPS:-50}"
NUM_STEPS="${NUM_STEPS:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
ACTION_DIM="${ACTION_DIM:-66}"
STATE_DIM="${STATE_DIM:-66}"
JOINT_STATE_DIM="${JOINT_STATE_DIM:-43}"
LATENT_STATE_DIM="${LATENT_STATE_DIM:-66}"
ACTION_FLOW_SHIFT="${ACTION_FLOW_SHIFT:-5.0}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/pfs/users/hengtao.li/conda_envs/gwpmot/bin/python}"

cd "$PROJECT_ROOT"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python)"
fi

setup_python_cuda_libs() {
    local python_bin="$1"
    local env_root
    env_root="$(cd "$(dirname "$python_bin")/.." && pwd)"

    local py_nvidia_libs
    py_nvidia_libs="$("$python_bin" - <<'PY' 2>/dev/null || true
from pathlib import Path
import site

libs = []
for sp in site.getsitepackages():
    root = Path(sp) / "nvidia"
    if root.exists():
        libs += [str(p) for p in root.glob("*/lib") if p.is_dir()]

print(":".join(libs))
PY
)"

    export CONDA_PREFIX="${CONDA_PREFIX:-$env_root}"
    export PATH="$env_root/bin:$PATH"
    export LD_LIBRARY_PATH="${py_nvidia_libs:+$py_nvidia_libs:}$env_root/lib:$env_root/lib64:${LD_LIBRARY_PATH:-}"
}

setup_python_cuda_libs "$PYTHON_BIN"

echo "============================================================"
echo "  MoT Open-Loop Evaluation [locomanip pick_place G1 sonic]"
echo "  Checkpoint:        $CHECKPOINT"
echo "  Base model:        $MODEL_ID"
echo "  GPU:               $GPU"
echo "  Data root:         $DATA_ROOT"
echo "  Stats path:        $STATS_PATH"
echo "  Max episodes:      $MAX_EPISODES"
echo "  Episode indices:   ${EPISODE_INDICES:-<auto>}"
echo "  Num frames:        $NUM_FRAMES"
echo "  Replan steps:      $REPLAN_STEPS"
echo "  Num steps:         $NUM_STEPS"
echo "  Num samples:       $NUM_SAMPLES (averaged per window)"
echo "  Action/state dim:  $ACTION_DIM / $STATE_DIM (joint $JOINT_STATE_DIM, latent $LATENT_STATE_DIM)"
echo "  dst_size (WxH):    ${DST_W} x ${DST_H}"
echo "  Python:            $PYTHON_BIN"
echo "============================================================"

CMD=( "$PYTHON_BIN" -u experiment/openloop/openloop_locomanip.py
    --checkpoint_path "$CHECKPOINT"
    --model_id "$MODEL_ID"
    --data_root "$DATA_ROOT"
    --stats_path "$STATS_PATH"
    --max_episodes "$MAX_EPISODES"
    --num_frames "$NUM_FRAMES"
    --replan_steps "$REPLAN_STEPS"
    --num_steps "$NUM_STEPS"
    --num_samples "$NUM_SAMPLES"
    --action_dim "$ACTION_DIM"
    --state_dim "$STATE_DIM"
    --joint_state_dim "$JOINT_STATE_DIM"
    --latent_state_dim "$LATENT_STATE_DIM"
    --action_flow_shift "$ACTION_FLOW_SHIFT"
    --seed "$SEED"
    --dst_size "$DST_W" "$DST_H" )

if [[ -n "$EPISODE_INDICES" ]]; then
    CMD+=( --episode_indices $EPISODE_INDICES )
fi

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"
