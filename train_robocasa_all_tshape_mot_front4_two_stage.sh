#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export date="${date:-$(date +%m%d_%H%M)}"

OUTPUT_ROOT="${GWP_MOT_OUTPUT_ROOT:-/shared_disk/users/hengtao.li/codex/gwp-mot}"
export GWP_MOT_OUTPUT_ROOT="$OUTPUT_ROOT"
export TMPDIR="${GWP_MOT_TMPDIR:-/tmp/gwp-mot/front4_${date}_node${MLP_ROLE_INDEX:-${NODE_RANK:-0}}_$$}"
log_dir="$OUTPUT_ROOT/logs/robocasa_atomic_seen_tshape_mot_front4_two_stage"
mkdir -p "$log_dir" "$TMPDIR"

cd /mnt/pfs/users/hengtao.li/varl/gwp-mot

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate /mnt/pfs/users/hengtao.li/conda_envs/gwpmot 2>/dev/null || true

NUM_NODES="${MLP_WORKER_NUM:-${NUM_NODES:-1}}"
NODE_RANK="${MLP_ROLE_INDEX:-${NODE_RANK:-0}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
TOTAL_PROCS=$((NUM_NODES * NPROC_PER_NODE))

export MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}"
export MASTER_PORT="${MASTER_PORT:-29510}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"

ACCEL_CONFIG="scripts/accelerate_configs/config_deepspeed_zero2.json"
STAGE1_CONFIG="configs.robocasa_all_tshape_mot_front4_video_pt.config"
STAGE2_CONFIG="configs.robocasa_all_tshape_mot_front4_joint_stage2.config"
STAGE1_EXP="robocasa_atomic_seen_tshape_mot_front4_video_pt"
STAGE1_PROJECT_DIR="$OUTPUT_ROOT/experiments/${STAGE1_EXP}_${date}"
run_stage() {
    local stage_name="$1"
    local config="$2"
    local log_file="$log_dir/${date}_${stage_name}_node${NODE_RANK}.log"
    echo "=== Training ${stage_name} ==="
    echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "  NUM_NODES:            $NUM_NODES"
    echo "  NODE_RANK:            $NODE_RANK"
    echo "  NPROC_PER_NODE:       $NPROC_PER_NODE"
    echo "  TOTAL_PROCS:          $TOTAL_PROCS"
    echo "  MASTER_ADDR:          $MASTER_ADDR"
    echo "  MASTER_PORT:          $MASTER_PORT"
    echo "  CONFIG:               $config"
    if [ -n "${MOT_STAGE1_CHECKPOINT:-}" ]; then
        echo "  MOT_STAGE1_CHECKPOINT: $MOT_STAGE1_CHECKPOINT"
    fi
    echo "  LOG:                  $log_file"
    echo "================================"
    accelerate launch         --config_file "$ACCEL_CONFIG"         --gpu_ids "$CUDA_VISIBLE_DEVICES"         --num_processes "$TOTAL_PROCS"         --num_machines "$NUM_NODES"         --machine_rank "$NODE_RANK"         --main_process_ip "$MASTER_ADDR"         --main_process_port "$MASTER_PORT"         scripts/train.py --config "$config"         2>&1 | tee "$log_file"
}

find_latest_stage_checkpoint() {
    local project_dir="$1"
    local latest_step=-1
    local latest_checkpoint=""
    local checkpoint_dir checkpoint_name checkpoint_step checkpoint_path

    shopt -s nullglob
    for checkpoint_dir in "$project_dir"/checkpoint-*; do
        [ -d "$checkpoint_dir" ] || continue
        checkpoint_name="${checkpoint_dir##*/}"
        checkpoint_step="${checkpoint_name#checkpoint-}"
        [[ "$checkpoint_step" =~ ^[0-9]+$ ]] || continue
        checkpoint_path="$checkpoint_dir/model.pt"
        [ -f "$checkpoint_path" ] || continue
        if (( checkpoint_step > latest_step )); then
            latest_step="$checkpoint_step"
            latest_checkpoint="$checkpoint_path"
        fi
    done
    shopt -u nullglob

    [ -n "$latest_checkpoint" ] || return 1
    printf '%s\n' "$latest_checkpoint"
}

run_stage "front4_video_pt" "$STAGE1_CONFIG"

if ! MOT_STAGE1_CHECKPOINT="$(find_latest_stage_checkpoint "$STAGE1_PROJECT_DIR")"; then
    echo "Missing stage1 checkpoint under: $STAGE1_PROJECT_DIR" >&2
    exit 1
fi
export MOT_STAGE1_CHECKPOINT
echo "Resolved stage1 checkpoint: $MOT_STAGE1_CHECKPOINT"

run_stage "front4_joint_stage2" "$STAGE2_CONFIG"
