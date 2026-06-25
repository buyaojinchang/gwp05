import copy
import os

from .robocasa_all_tshape_mot import config as _base_config


exp_name = "robocasa_all_tshape_mot_video_pt_epoch1_randomcrop"
date_str = os.environ.get("date", "default")
output_root = os.environ.get("GWP_MOT_OUTPUT_ROOT", "/shared_disk/users/hengtao.li/codex/gwp-mot")
project_dir = f"{output_root}/experiments/{exp_name}_{date_str}"

config = copy.deepcopy(_base_config)
config["project_dir"] = project_dir
config["wandb"].update(
    project="gwp-mot",
    name=f"{exp_name}_{date_str}",
    mode=os.environ.get("WANDB_MODE", "online"),
    init_timeout=int(os.environ.get("WANDB_INIT_TIMEOUT", "300")),
)

# Full RoboCasa pretrain data, video-only pretraining.
# Same 320x384 T-shape layout as the baseline:
#   head: 320x256, wrists: 160x128 + 160x128.
# With resize_mode="crop" and is_train=True, the transform uses random
# resize-to-fill crops; eval/inference should use deterministic center-crop.
config["dataloaders"]["train"]["transform"].update(
    dst_size=(320, 256),
    resize_mode="crop",
    random_shift_pad=0,
)
config["dataloaders"]["train"].update(
    # If a video decode / shared-disk read stalls, retry another sample inside
    # the same __getitem__ so distributed ranks keep the same number of steps.
    sample_timeout_sec=int(os.environ.get("LEROBOT_SAMPLE_TIMEOUT_SEC", "120")),
    max_sample_retries=int(os.environ.get("LEROBOT_MAX_SAMPLE_RETRIES", "5")),
    # Hard DataLoader guard: fail fast instead of waiting for NCCL watchdog.
    timeout=int(os.environ.get("LEROBOT_DATALOADER_TIMEOUT_SEC", "300")),
    # PyAV has repeatedly left one rank waiting on video reads. Decord avoids
    # that path while keeping enough workers to preserve training throughput.
    num_workers=int(os.environ.get("LEROBOT_NUM_WORKERS", "8")),
    prefetch_factor=int(os.environ.get("LEROBOT_PREFETCH_FACTOR", "4")),
)
for _dc in config["dataloaders"]["train"].get("data_or_config", []):
    _dc["video_backend"] = os.environ.get("LEROBOT_VIDEO_BACKEND", "decord")

config["models"]["view_dir"] = project_dir
config["models"].update(
    dict(
        checkpoint=None,
        freeze_action=True,
        use_gt_action_for_video=True,
        action_loss_weight=0.0,
        visual_loss_weight=1.0,
    )
)

config["schedulers"].update(
    decay_epochs=1,
)

config["train"].update(
    resume=os.environ.get("GWP_RESUME", "0") == "1",
    max_epochs=1,
    max_steps=0,
    process_group_timeout_sec=3600,
    checkpoint_interval=5000,
    checkpoint_epoch_interval=1,
    with_ema=True,
    ema=dict(
        enabled=True,
        decay=0.995,
        device="model",
    ),
)
