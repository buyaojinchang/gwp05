#!/usr/bin/env bash
set -euo pipefail

export PS1="${PS1:-}"

export CONDA_ENV="${CONDA_ENV:-/inspire/hdd/project/robot-dna/sunmingyang-240108120101/wam_locomanip/0_conda_env/gwp05}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export GWP_DEFAULT_NPROC=8
export MASTER_PORT="${MASTER_PORT:-29500}"
export WANDB_MODE="${WANDB_MODE:-offline}"

source "$(dirname "${BASH_SOURCE[0]}")/scripts/launch_lib.sh"

gwp_launch pick_place_g1_sonic_mot_video_pt "$@"