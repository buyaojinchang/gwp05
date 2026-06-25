#!/usr/bin/env bash
# RoboCasa atomic-seen T-shape MoT back4 joint training (4 GPUs).
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export GWP_DEFAULT_NPROC=4
export MASTER_PORT="${MASTER_PORT:-29520}"

source "$(dirname "${BASH_SOURCE[0]}")/scripts/launch_lib.sh"
gwp_launch robocasa_atomic_seen_back4_joint "$@"
