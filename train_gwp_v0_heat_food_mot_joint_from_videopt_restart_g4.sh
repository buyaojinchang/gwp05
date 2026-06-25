#!/usr/bin/env bash
# GWP-V0 heat_food MoT joint/action, fresh 5-epoch run from the video-pretrain
# EMA checkpoint (MOT_STAGE1_CHECKPOINT), 4 GPUs. Does not resume old runs.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GWP_DEFAULT_NPROC=4
export MASTER_PORT="${MASTER_PORT:-29694}"
export WANDB_MODE="${WANDB_MODE:-online}"
export date="${date:-heat_food_joint5ep_restart_$(date +%m%d_%H%M)}"

source "$(dirname "${BASH_SOURCE[0]}")/scripts/launch_lib.sh"
gwp_launch gwp_v0_heat_food_joint_from_videopt_5ep "$@"
