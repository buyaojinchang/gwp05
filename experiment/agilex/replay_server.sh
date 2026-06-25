#!/usr/bin/env bash
# Launch AgileX dataset action replay server (ZMQ, giga-brain compatible).
#
# Replays actions from a LeRobot episode instead of running GWP-MoT inference.
# Remote client code is unchanged — only server-side responses differ.
#
# Defaults (edit below or export before run):
#   export DATASET_ROOT=/shared_disk/users/hengtao.li/giga_real_data/gwp_v0/heat_food
#   export EPISODE_IDX=0
#   bash replay_server.sh
#
# CLI examples:
#   bash replay_server.sh --episode-idx 3 --port 11411
#   bash replay_server.sh /path/to/lerobot_dataset --start-frame 100

set -euo pipefail

# ── 可编辑默认值（也可在运行前 export 同名变量覆盖）────────────────────────
export SERVER_PYTHON="${SERVER_PYTHON:-/mnt/pfs/users/hengtao.li/conda_envs/gwpmot/bin/python}"

export DATASET_ROOT="${DATASET_ROOT:-/shared_disk/users/hengtao.li/giga_real_data/gwp_v0/heat_food}"
export EPISODE_IDX="${EPISODE_IDX:-5}"
export START_FRAME="${START_FRAME:-0}"
export LOOP="${LOOP:-false}"

export ACTION_CHUNK="${ACTION_CHUNK:-36}"
# Match client pos_lookahead_step (default 30 in inference_agilex_client_gwp.py).
export REPLAN_STEPS="${REPLAN_STEPS:-36}"
export ACTION_FORMAT="${ACTION_FORMAT:-absolute}"

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-11412}"
# ────────────────────────────────────────────────────────────────────────────

usage() {
    cat <<'EOF'
Usage:
  bash replay_server.sh [--dataset-root PATH] [--episode-idx N] [extra options...]
  bash replay_server.sh DATASET_ROOT [EPISODE_IDX] [extra options...]

Defaults (edit in script or export before run):
  DATASET_ROOT, EPISODE_IDX, START_FRAME, LOOP, ACTION_CHUNK, REPLAN_STEPS,
  ACTION_FORMAT, HOST, PORT, SERVER_PYTHON

CLI aliases → Python:
  --dataset, --dataset-root, --lerobot-dir  → --dataset-root
  --episode, --episode-idx                  → --episode-idx
  --start-frame                             → --start-frame
  --action-format                           → --action-format (absolute|delta)

Examples:
  export DATASET_ROOT=/shared_disk/users/hengtao.li/giga_real_data/gwp_v0/heat_food
  bash replay_server.sh --episode-idx 0 --port 11411

  # Client (unchanged): set SERVER_HOST/SERVER_PORT to match, then run client script.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GWP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CLI_DATASET=""
CLI_EPISODE=""
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --dataset|--dataset-root|--lerobot-dir)
            CLI_DATASET="${2:?missing value for $1}"
            shift 2
            ;;
        --dataset=*|--dataset-root=*|--lerobot-dir=*)
            CLI_DATASET="${1#*=}"
            shift
            ;;
        --episode|--episode-idx)
            CLI_EPISODE="${2:?missing value for $1}"
            shift 2
            ;;
        --episode=*|--episode-idx=*)
            CLI_EPISODE="${1#*=}"
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
            if [ -z "${CLI_DATASET}" ]; then
                CLI_DATASET="$1"
            elif [ -z "${CLI_EPISODE}" ]; then
                CLI_EPISODE="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

DATASET_ROOT="${CLI_DATASET:-${DATASET_ROOT}}"
EPISODE_IDX="${CLI_EPISODE:-${EPISODE_IDX}}"

if [ -z "${DATASET_ROOT}" ]; then
    echo "ERROR: DATASET_ROOT is empty. Set it in the script, export DATASET_ROOT=..., or pass --dataset-root." >&2
    echo >&2
    usage >&2
    exit 1
fi

if [ ! -d "${DATASET_ROOT}" ]; then
    echo "ERROR: dataset root not found: ${DATASET_ROOT}" >&2
    exit 1
fi

if [ ! -d "${DATASET_ROOT}/data" ]; then
    echo "ERROR: LeRobot data/ dir not found under: ${DATASET_ROOT}" >&2
    exit 1
fi

if [ ! -x "${SERVER_PYTHON}" ]; then
    echo "ERROR: python not found: ${SERVER_PYTHON}" >&2
    exit 1
fi

if ! "${SERVER_PYTHON}" -c "import zmq, tyro, pandas" >/dev/null 2>&1; then
    echo "ERROR: ${SERVER_PYTHON} missing deps (pyzmq, tyro, pandas). Install:" >&2
    echo "  ${SERVER_PYTHON} -m pip install -e ${GWP_ROOT} pyzmq tyro pandas pyarrow" >&2
    exit 1
fi

PY_ARGS=(
    --dataset-root "${DATASET_ROOT}"
    --episode-idx "${EPISODE_IDX}"
    --action-chunk "${ACTION_CHUNK}"
    --replan-steps "${REPLAN_STEPS}"
    --start-frame "${START_FRAME}"
    --action-format "${ACTION_FORMAT}"
    --host "${HOST}"
    --port "${PORT}"
)
if [ "${LOOP}" = "true" ] || [ "${LOOP}" = "1" ]; then
    PY_ARGS+=(--loop)
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    PY_ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "============================================================"
echo "  AgileX Dataset Action Replay Server"
echo "  Python       : ${SERVER_PYTHON}"
echo "  GWP          : ${GWP_ROOT}"
echo "  Dataset      : ${DATASET_ROOT}"
echo "  Episode      : ${EPISODE_IDX}"
echo "  Start frame  : ${START_FRAME}"
echo "  Loop         : ${LOOP}"
echo "  Chunk        : action_chunk=${ACTION_CHUNK}, replan_steps=${REPLAN_STEPS}"
echo "  Action format: ${ACTION_FORMAT}"
echo "  Host:Port    : ${HOST}:${PORT}"
echo "  Extra args   : ${EXTRA_ARGS[*]:-<none>}"
echo "============================================================"

cd "${GWP_ROOT}"
exec "${SERVER_PYTHON}" -u -m experiment.agilex.replay_server "${PY_ARGS[@]}"
