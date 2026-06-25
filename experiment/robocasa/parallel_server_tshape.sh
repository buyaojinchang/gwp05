#!/bin/bash
# Launch MoT T-shape inference servers.
# Usage: bash parallel_server_tshape.sh <CHECKPOINT_PATH> [ACTION_CHUNK] [SEED]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

FPS="${FPS:-20}"
MODEL_ID="${MODEL_ID:-/shared_disk/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers}"
CHECKPOINT="${1:?Usage: bash parallel_server_tshape.sh <CHECKPOINT_PATH> [ACTION_CHUNK] [SEED]}"
STATS_PATH="${STATS_PATH:-/shared_disk/users/hengtao.li/robocasa_datasets/v1.0/pretrain_gwp/norm_stats_delta.json}"
NUM_STEPS="${NUM_STEPS:-10}"
NUM_FRAMES="${NUM_FRAMES:-24}"
ACTION_CHUNK="${2:-24}"
SEED="${3:-42}"
BASE_PORT="${BASE_PORT:-19055}"
NUM_SERVERS="${NUM_SERVERS:-${NUM_WORKERS:-4}}"
NUM_WORKERS="$NUM_SERVERS"
GPU_OFFSET="${GPU_OFFSET:-4}"
OUTPUT_ROOT="${GWP_MOT_OUTPUT_ROOT:-/shared_disk/users/hengtao.li/codex/gwp-mot}"
RUNTIME_DIR="$OUTPUT_ROOT/robocasa_eval"

cd "$PROJECT_ROOT"

TIMESTAMP=$(date +%m%d_%H%M)
SERVER_LOG_DIR="$RUNTIME_DIR/server_tshape/${TIMESTAMP}"
mkdir -p "$SERVER_LOG_DIR"

if [ "$ACTION_CHUNK" -gt "$NUM_FRAMES" ]; then
    echo "ERROR: ACTION_CHUNK ($ACTION_CHUNK) must be <= NUM_FRAMES ($NUM_FRAMES)"
    exit 1
fi

echo "============================================================"
echo "  Launching $NUM_WORKERS MoT T-shape servers"
echo "  Checkpoint:   $CHECKPOINT"
echo "  FPS:          $FPS"
echo "  Num frames:   $NUM_FRAMES"
echo "  Action chunk: $ACTION_CHUNK (~$(awk "BEGIN{printf \"%.2f\", $ACTION_CHUNK/$FPS}")s)"
echo "  Ports:        $BASE_PORT - $((BASE_PORT + NUM_WORKERS - 1))"
echo "  GPUs:         $GPU_OFFSET - $((GPU_OFFSET + NUM_WORKERS - 1))"
echo "  Layout:       T-shape (head=agentview_right)"
echo "  Seed:         $SEED"
echo "  Logs:         $SERVER_LOG_DIR"
echo "============================================================"

PIDS=()
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    PORT=$((BASE_PORT + i))
    GPU=$((GPU_OFFSET + i))
    LOG="${SERVER_LOG_DIR}/server_${i}.log"

    echo "  [Server $i] GPU=$GPU  Port=$PORT  Log=$LOG"

    CUDA_VISIBLE_DEVICES=$GPU python -u experiment/robocasa/inference_server.py \
        --model_id "$MODEL_ID" \
        --checkpoint_path "$CHECKPOINT" \
        --stats_path "$STATS_PATH" \
        --port $PORT \
        --num_steps $NUM_STEPS \
        --num_frames $NUM_FRAMES \
        --action_chunk $ACTION_CHUNK \
        --action_only \
        --dst_size 320 256 \
        --tshape \
        --tshape_head_index 2 \
        --zero_action_dims 3 \
        --ctrl_mode_dim 4 \
        --seed $SEED \
        > "$LOG" 2>&1 &

    PIDS+=($!)
done

PID_FILE="${SERVER_LOG_DIR}/pids.txt"
echo "kill ${PIDS[*]}" > "$PID_FILE"
echo "# Server PIDs (MoT T-shape) - $(date)" >> "$PID_FILE"
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "server_${i}: ${PIDS[$i]}" >> "$PID_FILE"
done

echo ""
echo "Server PIDs: ${PIDS[*]}"
echo "PIDs saved to: $PID_FILE"
echo "Waiting for all servers to be ready..."

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    LOG="${SERVER_LOG_DIR}/server_${i}.log"
    while ! grep -q "Server listening" "$LOG" 2>/dev/null; do
        sleep 2
    done
    echo "  [Server $i] Ready."
done

echo ""
echo "All $NUM_WORKERS servers are ready."
echo "To stop: kill ${PIDS[*]}"
echo ""

INFO_FILE="$RUNTIME_DIR/.server_tshape_info"
mkdir -p "$(dirname "$INFO_FILE")"
echo "CHECKPOINT=${CHECKPOINT}" > "$INFO_FILE"
echo "SERVER_LOG_DIR=${SERVER_LOG_DIR}" >> "$INFO_FILE"
echo "BASE_PORT=${BASE_PORT}" >> "$INFO_FILE"
echo "NUM_SERVERS=${NUM_SERVERS}" >> "$INFO_FILE"
echo "NUM_WORKERS=${NUM_WORKERS}" >> "$INFO_FILE"
echo "GPU_OFFSET=${GPU_OFFSET}" >> "$INFO_FILE"
echo "FPS=${FPS}" >> "$INFO_FILE"
echo "NUM_FRAMES=${NUM_FRAMES}" >> "$INFO_FILE"
echo "ACTION_CHUNK=${ACTION_CHUNK}" >> "$INFO_FILE"
echo "TSHAPE=1" >> "$INFO_FILE"
echo "SEED=${SEED}" >> "$INFO_FILE"
echo "GWP_MOT_OUTPUT_ROOT=${OUTPUT_ROOT}" >> "$INFO_FILE"
echo "Server info written to: $INFO_FILE"

wait
