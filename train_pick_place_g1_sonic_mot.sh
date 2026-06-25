#!/usr/bin/env bash
# Locomanip pick_place (G1 cola task) MoT training with 66-d sonic latent actions.
#
# Prereq (run once): build the gwp-mot-compatible dataset from the raw pick_place:
#   python scripts/prepare_pick_place_gwp.py
#
# Single-GPU smoke run example:
#   CUDA_VISIBLE_DEVICES=0 GWP_DEFAULT_NPROC=1 bash train_pick_place_g1_sonic_mot.sh \
#     train.max_epochs=1 data.batch_size_per_gpu=2
set -euo pipefail

export GWP_DEFAULT_NPROC="${GWP_DEFAULT_NPROC:-8}"
export MASTER_PORT="${MASTER_PORT:-29500}"

source "$(dirname "${BASH_SOURCE[0]}")/scripts/launch_lib.sh"
gwp_launch pick_place_g1_sonic_mot "$@"
