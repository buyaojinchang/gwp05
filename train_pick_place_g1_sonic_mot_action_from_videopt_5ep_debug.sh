#!/usr/bin/env bash
# DEBUG: single-GPU smoke run of stage-2 (pick_place G1 sonic) action/joint training.
#
# Tiny batch, no grad-accum, capped at a handful of steps (see
# configs/task/pick_place_g1_sonic_mot_action_from_videopt_5ep_debug.yaml).
# Still requires the stage-1 video-pretrain EMA checkpoint:
#   export MOT_STAGE1_CHECKPOINT=$GWP_MOT_OUTPUT_ROOT/experiments/pick_place_g1_sonic_mot_video_pt_<date>/checkpoint-<step>/model_ema.pt
#
# Override anything via Hydra on the CLI, e.g.:
#   bash train_pick_place_g1_sonic_mot_action_from_videopt_5ep_debug.sh train.max_steps=5
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GWP_DEFAULT_NPROC=1
export MASTER_PORT="${MASTER_PORT:-29511}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export date="${date:-debug_$(date +%m%d_%H%M)}"

if [[ -z "${MOT_STAGE1_CHECKPOINT:-}" ]]; then
  echo "ERROR: set MOT_STAGE1_CHECKPOINT to the stage-1 video-pretrain EMA checkpoint (model_ema.pt)." >&2
  echo "  e.g. export MOT_STAGE1_CHECKPOINT=\$GWP_MOT_OUTPUT_ROOT/experiments/pick_place_g1_sonic_mot_video_pt_<date>/checkpoint-<step>/model_ema.pt" >&2
  exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/scripts/launch_lib.sh"
gwp_launch pick_place_g1_sonic_mot_action_from_videopt_5ep_debug "$@"
