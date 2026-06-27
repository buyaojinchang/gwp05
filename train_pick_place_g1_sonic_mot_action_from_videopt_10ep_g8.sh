#!/usr/bin/env bash
# Stage 2: pick_place (G1 sonic) MoT action/joint training, 5 epochs, 8 GPUs,
# initialized from the stage-1 video-pretrain EMA checkpoint (weights only;
# this is a fresh run, not an accelerate-state resume).
#
# REQUIRED: point MOT_STAGE1_CHECKPOINT at the stage-1 EMA checkpoint, e.g.:
#   export MOT_STAGE1_CHECKPOINT=$GWP_MOT_OUTPUT_ROOT/experiments/pick_place_g1_sonic_mot_video_pt_<date>/checkpoint-<step>/model_ema.pt
#
# Override anything via Hydra on the CLI, e.g.:
#   bash train_pick_place_g1_sonic_mot_action_from_videopt_5ep_g8.sh train.max_epochs=5
set -euo pipefail

export PS1="${PS1:-}"

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

export MOT_STAGE1_CHECKPOINT=/inspire/hdd/project/robot-dna/sunmingyang-240108120101/wam_locomanip/2_data_ckpt_cache/loco_manip/experiments/experiments/pick_place_g1_sonic_mot_video_pt_0626_1457/checkpoint-8901/model_ema.pt

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export GWP_DEFAULT_NPROC=8
export MASTER_PORT="${MASTER_PORT:-29501}"
export date="${date:-action5ep_g8_$(date +%m%d_%H%M)}"

if [[ -z "${MOT_STAGE1_CHECKPOINT:-}" ]]; then
  echo "ERROR: set MOT_STAGE1_CHECKPOINT to the stage-1 video-pretrain EMA checkpoint (model_ema.pt)." >&2
  echo "  e.g. export MOT_STAGE1_CHECKPOINT=\$GWP_MOT_OUTPUT_ROOT/experiments/pick_place_g1_sonic_mot_video_pt_<date>/checkpoint-<step>/model_ema.pt" >&2
  exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/scripts/launch_lib.sh"
gwp_launch pick_place_g1_sonic_mot_action_from_videopt_10ep "$@"

