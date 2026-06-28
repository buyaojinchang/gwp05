#!/usr/bin/env bash
# Launch the G1 locomanip pick_place (g1_sonic) GWP-MoT inference server.
#
# Isaac-GR00T PolicyClient compatible (msgpack ZMQ REQ/REP, port 5550 by
# default), so GR00T-WholeBodyControl's run_vla_inference.py can drive it.
# get_action returns raw motion_token[1,T,64] + hand_binary[1,T,2]
# (the client decodes hand_binary -> 7-DoF joints via its own IK solver).
#
# 默认值：在下方「可编辑默认值」里直接改，或运行前 export 覆盖，例如：
#   export CHECKPOINT=/path/to/model_ema.pt
#   export STATS=/path/to/norm_stats_delta.json
#   bash inference_server.sh
#
# 命令行会覆盖默认值：
#   bash inference_server.sh --checkpoint /m.pt --stats /s.json
#   bash inference_server.sh /path/to/model_ema.pt /path/to/norm_stats_delta.json

set -euo pipefail

# ── 可编辑默认值（也可在运行前 export 同名变量覆盖）────────────────────────
export SERVER_PYTHON="${SERVER_PYTHON:-/mnt/pfs/users/hengtao.li/conda_envs/gwpmot/bin/python}"

export CHECKPOINT="${CHECKPOINT:-/shared_disk/users/hengtao.li/locomanip/exp/debug/model_ema.pt}"
export DATA_ROOT="${PICK_PLACE_ROOT:-/shared_disk/users/hengtao.li/locomanip/data/pick_place_gwp}"
export STATS="${STATS:-${DATA_ROOT}/norm_stats_delta.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export NUM_FRAMES="${NUM_FRAMES:-56}"
export NUM_STEPS="${NUM_STEPS:-10}"
export NUM_SAMPLES="${NUM_SAMPLES:-1}"
export REPLAN_STEPS="${REPLAN_STEPS:-50}"         # generate NUM_FRAMES, return first N (0=all)
export ACTION_FLOW_SHIFT="${ACTION_FLOW_SHIFT:-5.0}"

export ACTION_DIM="${ACTION_DIM:-66}"
export STATE_DIM="${STATE_DIM:-66}"
export JOINT_STATE_DIM="${JOINT_STATE_DIM:-43}"
export LATENT_STATE_DIM="${LATENT_STATE_DIM:-66}"
export DST_W="${DST_W:-320}"
export DST_H="${DST_H:-256}"

export TEXT_MODE="${TEXT_MODE:-t5}"               # t5 (encode prompt online) | precomputed (load TEXT_CONTEXT_FILE)
export TEXT_CONTEXT_FILE="${TEXT_CONTEXT_FILE:-}"
export REQUIRE_LATENT_STATE="${REQUIRE_LATENT_STATE:-false}"
export DEBUG_PRINT_STATS="${DEBUG_PRINT_STATS:-false}"

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-5550}"
export PRETRAINED_PATH="${PRETRAINED_PATH:-/shared_disk/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers}"
export MOT_CHECKPOINT_MIXED_ATTN="${MOT_CHECKPOINT_MIXED_ATTN:-true}"
export SEED="${SEED:-}"
# ────────────────────────────────────────────────────────────────────────────

usage() {
    cat <<'EOF'
Usage:
  bash inference_server.sh [--checkpoint PATH] [--stats PATH] [extra tyro options...]
  bash inference_server.sh CHECKPOINT [STATS_JSON] [extra tyro options...]

Defaults (edit in script or export before run):
  CHECKPOINT, STATS, NUM_FRAMES, NUM_STEPS, NUM_SAMPLES, REPLAN_STEPS,
  ACTION_FLOW_SHIFT, ACTION_DIM, STATE_DIM, JOINT_STATE_DIM, LATENT_STATE_DIM,
  DST_W, DST_H, TEXT_MODE, TEXT_CONTEXT_FILE, HOST, PORT, PRETRAINED_PATH,
  CUDA_VISIBLE_DEVICES, SERVER_PYTHON, MOT_CHECKPOINT_MIXED_ATTN, SEED

CLI aliases -> tyro:
  --checkpoint, --ckpt   -> --checkpoint-path
  --stats, --norm-stats  -> --stats-path
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GWP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CLI_CHECKPOINT=""
CLI_STATS=""
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --checkpoint|--ckpt|--checkpoint-path)
            CLI_CHECKPOINT="${2:?missing value for $1}"; shift 2 ;;
        --checkpoint=*|--ckpt=*|--checkpoint-path=*)
            CLI_CHECKPOINT="${1#*=}"; shift ;;
        --stats|--norm-stats|--stats-path)
            CLI_STATS="${2:?missing value for $1}"; shift 2 ;;
        --stats=*|--norm-stats=*|--stats-path=*)
            CLI_STATS="${1#*=}"; shift ;;
        --)
            shift; EXTRA_ARGS+=("$@"); break ;;
        -*)
            EXTRA_ARGS+=("$1"); shift ;;
        *)
            if [ -z "${CLI_CHECKPOINT}" ]; then CLI_CHECKPOINT="$1"
            elif [ -z "${CLI_STATS}" ]; then CLI_STATS="$1"
            else EXTRA_ARGS+=("$1"); fi
            shift ;;
    esac
done

CHECKPOINT="${CLI_CHECKPOINT:-${CHECKPOINT}}"
STATS="${CLI_STATS:-${STATS}}"

if [ -z "${CHECKPOINT}" ]; then
    echo "ERROR: CHECKPOINT is empty. Set it in the script, export CHECKPOINT=..., or pass --checkpoint." >&2
    echo >&2; usage >&2; exit 1
fi
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2; exit 1
fi
if [ -n "${STATS}" ] && [ ! -f "${STATS}" ]; then
    echo "ERROR: norm stats file not found: ${STATS}" >&2; exit 1
fi
if [ ! -x "${SERVER_PYTHON}" ]; then
    echo "ERROR: python not found: ${SERVER_PYTHON}" >&2; exit 1
fi

PY_ARGS=(
    --checkpoint-path "${CHECKPOINT}"
    --stats-path "${STATS}"
    --pretrained-path "${PRETRAINED_PATH}"
    --num-frames "${NUM_FRAMES}"
    --num-steps "${NUM_STEPS}"
    --num-samples "${NUM_SAMPLES}"
    --replan-steps "${REPLAN_STEPS}"
    --action-flow-shift "${ACTION_FLOW_SHIFT}"
    --action-dim "${ACTION_DIM}"
    --state-dim "${STATE_DIM}"
    --joint-state-dim "${JOINT_STATE_DIM}"
    --latent-state-dim "${LATENT_STATE_DIM}"
    --dst-size "${DST_W}" "${DST_H}"
    --text-mode "${TEXT_MODE}"
    --host "${HOST}"
    --port "${PORT}"
)
if [ "${MOT_CHECKPOINT_MIXED_ATTN}" = "false" ] || [ "${MOT_CHECKPOINT_MIXED_ATTN}" = "0" ]; then
    PY_ARGS+=(--no-mot-checkpoint-mixed-attn)
fi
if [ -n "${TEXT_CONTEXT_FILE}" ]; then
    PY_ARGS+=(--text-context-file "${TEXT_CONTEXT_FILE}")
fi
if [ "${REQUIRE_LATENT_STATE}" = "true" ] || [ "${REQUIRE_LATENT_STATE}" = "1" ]; then
    PY_ARGS+=(--require-latent-state)
fi
if [ "${DEBUG_PRINT_STATS}" = "true" ] || [ "${DEBUG_PRINT_STATS}" = "1" ]; then
    PY_ARGS+=(--debug-print-stats)
fi
if [ -n "${SEED}" ]; then
    PY_ARGS+=(--seed "${SEED}")
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    PY_ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "============================================================"
echo "  G1 Sonic GWP-MoT Inference Server (GR00T PolicyClient compatible)"
echo "  Python     : ${SERVER_PYTHON}"
echo "  GWP        : ${GWP_ROOT}"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Norm stats : ${STATS}"
echo "  GPU        : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  Frames     : num_frames=${NUM_FRAMES}, replan_steps=${REPLAN_STEPS} (gen ${NUM_FRAMES}, return first ${REPLAN_STEPS})"
echo "  Sampling   : num_steps=${NUM_STEPS}, num_samples=${NUM_SAMPLES}, flow_shift=${ACTION_FLOW_SHIFT}"
echo "  Action     : motion_token[64] + hand_binary[2] (raw; client-side IK decode)"
echo "  State      : [${JOINT_STATE_DIM} joint, ${LATENT_STATE_DIM} latent] -> ${STATE_DIM}"
echo "  dst (WxH)  : ${DST_W} x ${DST_H}"
echo "  Text       : ${TEXT_MODE}${TEXT_CONTEXT_FILE:+ (${TEXT_CONTEXT_FILE})}"
echo "  Host:Port  : ${HOST}:${PORT}"
echo "  MoT mixed  : ${MOT_CHECKPOINT_MIXED_ATTN}"
echo "  Seed       : ${SEED:-<none>}"
echo "  Extra args : ${EXTRA_ARGS[*]:-<none>}"
echo "============================================================"

cd "${GWP_ROOT}"
exec "${SERVER_PYTHON}" -u -m experiment.g1.inference_server "${PY_ARGS[@]}"
