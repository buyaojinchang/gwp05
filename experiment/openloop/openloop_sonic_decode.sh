#!/bin/bash
# Sonic-decode open-loop eval: MoT sonic-latent -> SONIC decoder -> joint action.
#
# Predicts the 66-d sonic latent (motion_token[64] + hand_binary[2]) with the MoT
# model, feeds the predicted token + teacher-forced proprioception through the
# SONIC decoder ONNX (model_decoder.onnx), and compares the decoded body joints
# against the ground-truth action.wbc (+ hand_binary).
#
# Requires:
#   * a RAW pick_place dataset that still carries action.wbc / observation.state /
#     observation.projected_gravity / observation.root_orientation (NOT the
#     repacked pick_place_gwp). Set DATA_ROOT to where you upload it.
#   * model_decoder.onnx from nvidia/GEAR-SONIC (download_from_hf.py). onnxruntime.
#
# Usage:
#   bash experiment/openloop/openloop_sonic_decode.sh <CHECKPOINT_PATH> [GPU_ID] [MAX_EPISODES] [EPISODE_IDX...]
#   # inspect the decoder ONNX I/O only (no model load, no data needed):
#   INSPECT_ONNX=1 bash experiment/openloop/openloop_sonic_decode.sh dummy 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CHECKPOINT="${1:?Usage: bash openloop_sonic_decode.sh <CHECKPOINT_PATH> [GPU_ID] [MAX_EPISODES] [EPISODE_IDX...]}"
GPU="${2:-0}"
MAX_EPISODES="${3:-3}"
shift $(( $# < 3 ? $# : 3 )) || true
EPISODE_INDICES="$*"

# RAW dataset (with action.wbc + proprio). Upload it here, or override DATA_ROOT.
DATA_ROOT="${PICK_PLACE_RAW_ROOT:-/shared_disk/users/hengtao.li/locomanip/data/pick_place}"
# MoT state-token norm stats (joint mean/std) — same file as the latent open-loop.
STATS_PATH="${STATS_PATH:-/shared_disk/users/hengtao.li/locomanip/data/pick_place_gwp/norm_stats_delta.json}"
# T5 language embeddings (raw pick_place lacks meta/t5_text_embeds.pt; use gwp's).
T5_ROOT="${T5_ROOT:-/shared_disk/users/hengtao.li/locomanip/data/pick_place_gwp}"
DECODER_ONNX="${DECODER_ONNX:-/mnt/pfs/users/hengtao.li/locomanip/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx}"
MODEL_ID="${WAN22_DIFFUSERS_PATH:-/shared_disk/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers}"

DST_W="${DST_W:-320}"
DST_H="${DST_H:-256}"
NUM_FRAMES="${NUM_FRAMES:-56}"
REPLAN_STEPS="${REPLAN_STEPS:-50}"
NUM_STEPS="${NUM_STEPS:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
ACTION_FLOW_SHIFT="${ACTION_FLOW_SHIFT:-5.0}"
SEED="${SEED:-42}"

# SONIC decoder proprioception config (validate against the real ONNX!).
FPS="${FPS:-50}"
HISTORY_LENGTH="${HISTORY_LENGTH:-10}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/pfs/users/hengtao.li/conda_envs/gwpmot/bin/python}"

cd "$PROJECT_ROOT"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python)"
fi

echo "============================================================"
echo "  Sonic-decode Open-Loop [latent -> decoder -> joint vs wbc]"
echo "  Checkpoint:    $CHECKPOINT"
echo "  Decoder ONNX:  $DECODER_ONNX"
echo "  Raw data:      $DATA_ROOT"
echo "  Stats path:    $STATS_PATH"
echo "  GPU:           $GPU"
echo "  Max episodes:  $MAX_EPISODES"
echo "  History len:   $HISTORY_LENGTH   fps: $FPS"
echo "  Python:        $PYTHON_BIN"
echo "============================================================"

# Inspect-only mode: print decoder ONNX I/O and exit (no checkpoint/data needed).
if [[ "${INSPECT_ONNX:-0}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u experiment/openloop/openloop_sonic_decode.py \
        --checkpoint_path "$CHECKPOINT" --decoder_onnx "$DECODER_ONNX" --inspect_onnx
    exit 0
fi

CMD=( "$PYTHON_BIN" -u experiment/openloop/openloop_sonic_decode.py
    --checkpoint_path "$CHECKPOINT"
    --decoder_onnx "$DECODER_ONNX"
    --model_id "$MODEL_ID"
    --data_root "$DATA_ROOT"
    --stats_path "$STATS_PATH"
    --t5_root "$T5_ROOT"
    --max_episodes "$MAX_EPISODES"
    --num_frames "$NUM_FRAMES"
    --replan_steps "$REPLAN_STEPS"
    --num_steps "$NUM_STEPS"
    --num_samples "$NUM_SAMPLES"
    --action_flow_shift "$ACTION_FLOW_SHIFT"
    --seed "$SEED"
    --fps "$FPS"
    --history_length "$HISTORY_LENGTH"
    --dst_size "$DST_W" "$DST_H" )

if [[ -n "$EPISODE_INDICES" ]]; then
    CMD+=( --episode_indices $EPISODE_INDICES )
fi

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"
