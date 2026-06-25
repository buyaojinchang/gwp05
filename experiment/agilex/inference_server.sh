#!/usr/bin/env bash
# Launch AgileX GWP-MoT inference server (ZMQ, giga-brain compatible).
#
# 默认值：在下方「可编辑默认值」里直接改，或运行前 export 覆盖，例如：
#   export CHECKPOINT=/path/to/model_ema.pt
#   export STATS=/path/to/norm_stats_delta.json
#   bash inference_server.sh
#
# 命令行会覆盖上面的默认值：
#   bash inference_server.sh --checkpoint /other/model.pt --stats /other/stats.json
#   bash inference_server.sh /path/to/model.pt /path/to/norm_stats_delta.json

set -euo pipefail

# ── 可编辑默认值（也可在运行前 export 同名变量覆盖）────────────────────────
export SERVER_PYTHON="${SERVER_PYTHON:-/mnt/pfs/users/hengtao.li/conda_envs/gwpmot/bin/python}"

export CHECKPOINT="${CHECKPOINT:-/shared_disk/users/hengtao.li/codex/gwp-mot/experiments/gwp_v0_heat_food_mot_joint_from_videopt_5ep_heat_food_joint5ep_restart_g8_0604_pyav/checkpoint-58664/model_ema.pt}"
export STATS="${STATS:-/shared_disk/users/hengtao.li/giga_real_data/gwp_v0/heat_food/norm_stats_delta.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

export ACTION_CHUNK="${ACTION_CHUNK:-36}"
export REPLAN_STEPS="${REPLAN_STEPS:-30}"
export NUM_FRAMES="${NUM_FRAMES:-36}"
export NUM_STEPS="${NUM_STEPS:-10}"

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-11411}"
export PRETRAINED_PATH="${PRETRAINED_PATH:-/shared_disk/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers}"
export MOT_CHECKPOINT_MIXED_ATTN="${MOT_CHECKPOINT_MIXED_ATTN:-true}"
export SEED="${SEED:-}"
# ────────────────────────────────────────────────────────────────────────────

usage() {
    cat <<'EOF'
Usage:
  bash inference_server.sh [--checkpoint PATH] [--stats PATH] [extra options...]
  bash inference_server.sh CHECKPOINT [STATS_JSON] [extra options...]

Defaults (edit in script or export before run):
  CHECKPOINT, STATS, ACTION_CHUNK, REPLAN_STEPS, NUM_FRAMES, CUDA_VISIBLE_DEVICES,
  HOST, PORT, PRETRAINED_PATH, SERVER_PYTHON, MOT_CHECKPOINT_MIXED_ATTN, SEED

CLI aliases → Python:
  --checkpoint, --ckpt              → --checkpoint-path
  --stats, --norm-stats             → --stats-path

Examples:
  export CHECKPOINT=/path/to/model_ema.pt
  bash inference_server.sh --port 11411

  pip install pyzmq  # if missing
  pip install -e /mnt/pfs/users/hengtao.li/varl/gwp-mot
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
            CLI_CHECKPOINT="${2:?missing value for $1}"
            shift 2
            ;;
        --checkpoint=*|--ckpt=*|--checkpoint-path=*)
            CLI_CHECKPOINT="${1#*=}"
            shift
            ;;
        --stats|--norm-stats|--stats-path)
            CLI_STATS="${2:?missing value for $1}"
            shift 2
            ;;
        --stats=*|--norm-stats=*|--stats-path=*)
            CLI_STATS="${1#*=}"
            shift
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        -*)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        *)
            if [ -z "${CLI_CHECKPOINT}" ]; then
                CLI_CHECKPOINT="$1"
            elif [ -z "${CLI_STATS}" ]; then
                CLI_STATS="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

CHECKPOINT="${CLI_CHECKPOINT:-${CHECKPOINT}}"
STATS="${CLI_STATS:-${STATS}}"

if [ -z "${CHECKPOINT}" ]; then
    echo "ERROR: CHECKPOINT is empty. Set it in the script, export CHECKPOINT=..., or pass --checkpoint." >&2
    echo >&2
    usage >&2
    exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

if [ -n "${STATS}" ] && [ ! -f "${STATS}" ]; then
    echo "ERROR: norm stats file not found: ${STATS}" >&2
    exit 1
fi

if [ ! -x "${SERVER_PYTHON}" ]; then
    echo "ERROR: python not found: ${SERVER_PYTHON}" >&2
    exit 1
fi

if ! "${SERVER_PYTHON}" -c "
import zmq, tyro
from world_action_model.models.transformer_wa_mot import MoTWorldActionTransformer
" >/dev/null 2>&1; then
    echo "ERROR: ${SERVER_PYTHON} missing deps (gwpmot + gwp-mot MoT + pyzmq). Install:" >&2
    echo "  ${SERVER_PYTHON} -m pip install -e ${GWP_ROOT} pyzmq tyro" >&2
    exit 1
fi

PY_ARGS=(
    --checkpoint-path "${CHECKPOINT}"
    --stats-path "${STATS}"
    --num-frames "${NUM_FRAMES}"
    --action-chunk "${ACTION_CHUNK}"
    --replan-steps "${REPLAN_STEPS}"
    --num-steps "${NUM_STEPS}"
    --host "${HOST}"
    --port "${PORT}"
    --pretrained-path "${PRETRAINED_PATH}"
)
if [ "${MOT_CHECKPOINT_MIXED_ATTN}" = "false" ] || [ "${MOT_CHECKPOINT_MIXED_ATTN}" = "0" ]; then
    PY_ARGS+=(--no-mot-checkpoint-mixed-attn)
fi
if [ -n "${SEED}" ]; then
    PY_ARGS+=(--seed "${SEED}")
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    PY_ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "============================================================"
echo "  AgileX GWP-MoT Inference Server"
echo "  Python     : ${SERVER_PYTHON}"
echo "  GWP        : ${GWP_ROOT}"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Norm stats : ${STATS}"
echo "  GPU        : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  Frames     : num_frames=${NUM_FRAMES}"
echo "  Chunk      : action_chunk=${ACTION_CHUNK}, replan_steps=${REPLAN_STEPS}"
echo "  Host:Port  : ${HOST}:${PORT}"
echo "  Pretrained : ${PRETRAINED_PATH}"
echo "  MoT mixed  : ${MOT_CHECKPOINT_MIXED_ATTN}"
echo "  Seed       : ${SEED:-<none>}"
echo "  Task       : from client (dynamic T5)"
echo "  Extra args : ${EXTRA_ARGS[*]:-<none>}"
echo "============================================================"

cd "${GWP_ROOT}"
exec "${SERVER_PYTHON}" -u -m experiment.agilex.inference_server "${PY_ARGS[@]}"
