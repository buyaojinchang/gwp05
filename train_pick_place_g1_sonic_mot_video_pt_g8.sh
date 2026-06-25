#!/usr/bin/env bash
# Stage 1: pick_place (G1 sonic) MoT video-pretrain, 1 epoch, 8 GPUs.
#
# Trains the video/world-model branch only (action expert frozen,
# action_loss_weight=0), starting from the Wan2.2-TI2V-5B pretrained transformer.
# The stage-2 action run initializes from the EMA checkpoint produced here:
#   $GWP_MOT_OUTPUT_ROOT/experiments/pick_place_g1_sonic_mot_video_pt_<date>/checkpoint-<step>/model_ema.pt
#
# Prereq (run once): build the prepared dataset + T5 embeds:
#   python scripts/prepare_pick_place_gwp.py
#   python scripts/generate_t5_embeddings.py --data_root <pick_place_gwp> --wan_model_path <wan>
#
# Override anything via Hydra on the CLI, e.g.:
#   bash train_pick_place_g1_sonic_mot_video_pt_g8.sh data.batch_size_per_gpu=2
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export GWP_DEFAULT_NPROC=8
export MASTER_PORT="${MASTER_PORT:-29500}"
export WANDB_MODE="${WANDB_MODE:-offline}"

source "$(dirname "${BASH_SOURCE[0]}")/scripts/launch_lib.sh"
gwp_launch pick_place_g1_sonic_mot_video_pt "$@"
