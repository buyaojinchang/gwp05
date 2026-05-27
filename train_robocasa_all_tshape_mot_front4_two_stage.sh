#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export date="${date:-$(date +%m%d_%H%M)}"

OUTPUT_ROOT="${GWP_MOT_OUTPUT_ROOT:-/shared_disk/users/hengtao.li/codex/gwp-mot}"
export GWP_MOT_OUTPUT_ROOT="$OUTPUT_ROOT"
export TMPDIR="${GWP_MOT_TMPDIR:-/tmp/gwp-mot/front4_${date}_node${MLP_ROLE_INDEX:-${NODE_RANK:-0}}_$$}"
log_dir="$OUTPUT_ROOT/logs/robocasa_all_tshape_mot_front4_two_stage"
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
STAGE1_EXP="robocasa_all_tshape_mot_front4_video_pt"
STAGE1_PROJECT_DIR="$OUTPUT_ROOT/experiments/${STAGE1_EXP}_${date}"
export MOT_STAGE1_CHECKPOINT="${STAGE1_PROJECT_DIR}/checkpoint-10000/model.pt"

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
    echo "  LOG:                  $log_file"
    echo "================================"
    accelerate launch         --config_file "$ACCEL_CONFIG"         --gpu_ids "$CUDA_VISIBLE_DEVICES"         --num_processes "$TOTAL_PROCS"         --num_machines "$NUM_NODES"         --machine_rank "$NODE_RANK"         --main_process_ip "$MASTER_ADDR"         --main_process_port "$MASTER_PORT"         scripts/train.py --config "$config"         2>&1 | tee "$log_file"
}

run_stage "front4_video_pt" "$STAGE1_CONFIG"

if [ ! -f "$MOT_STAGE1_CHECKPOINT" ]; then
    echo "Missing stage1 checkpoint: $MOT_STAGE1_CHECKPOINT" >&2
    exit 1
fi

run_stage "front4_joint_stage2" "$STAGE2_CONFIG"
