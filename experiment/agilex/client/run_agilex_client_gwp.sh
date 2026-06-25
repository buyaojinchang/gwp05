#!/usr/bin/env bash
# Launch Agilex GWP client (joint-space / ROS machine). Point SERVER_HOST to the GPU server IP.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-11411}"
TASK_NAME="${TASK_NAME:-heat the food}"
CONFIRM_EACH_CHUNK="${CONFIRM_EACH_CHUNK:-true}"
REPLAY_MODE="${REPLAY_MODE:-true}"

export SERVER_HOST SERVER_PORT TASK_NAME CONFIRM_EACH_CHUNK REPLAY_MODE

echo "Server:  ${SERVER_HOST}:${SERVER_PORT}"
echo "Task:    ${TASK_NAME}"
echo "Confirm each chunk: ${CONFIRM_EACH_CHUNK} (set CONFIRM_EACH_CHUNK=true to enable)"
echo "Replay mode: ${REPLAY_MODE} (true=blank frames on sync fail; false=report error and retry)"

python "${SCRIPT_DIR}/inference_agilex_client_gwp.py"
