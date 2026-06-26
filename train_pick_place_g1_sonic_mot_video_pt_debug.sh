#!/usr/bin/env bash
# DEBUG: single-GPU smoke run of stage-1 (pick_place G1 sonic) video pretrain.
#
# Tiny batch, no grad-accum, capped at a handful of steps (see
# configs/task/pick_place_g1_sonic_mot_video_pt_debug.yaml). Use this to verify
# the data pipeline / model / EMA / checkpointing end-to-end before the 8-GPU run.
#
# Override anything via Hydra on the CLI, e.g.:
#   bash train_pick_place_g1_sonic_mot_video_pt_debug.sh train.max_steps=5 data.batch_size_per_gpu=1
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GWP_DEFAULT_NPROC=1
export MASTER_PORT="${MASTER_PORT:-29510}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export date="${date:-debug_$(date +%m%d_%H%M)}"

source "$(dirname "${BASH_SOURCE[0]}")/scripts/launch_lib.sh"
gwp_launch pick_place_g1_sonic_mot_video_pt_debug "$@"
