#!/usr/bin/env bash
# GWP-V0 heat_food + fold_shirt MoT video-pretrain, 1 epoch, 4 GPUs.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GWP_DEFAULT_NPROC=4
export MASTER_PORT="${MASTER_PORT:-29674}"
export WANDB_MODE="${WANDB_MODE:-offline}"

source "$(dirname "${BASH_SOURCE[0]}")/scripts/launch_lib.sh"
gwp_launch gwp_v0_heat_food_fold_shirt_mot_video_pt "$@"
