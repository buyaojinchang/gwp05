#!/usr/bin/env bash
# RoboCasa all-data T-shape MoT (head=agentview_right 320x256 + 2x wrist 160x128).
set -euo pipefail

export GWP_DEFAULT_NPROC=8
export MASTER_PORT="${MASTER_PORT:-29500}"

source "$(dirname "${BASH_SOURCE[0]}")/scripts/launch_lib.sh"
gwp_launch robocasa_all_tshape_mot "$@"
