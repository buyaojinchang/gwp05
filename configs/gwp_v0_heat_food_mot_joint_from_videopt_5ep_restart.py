"""Heat-food MoT joint/action training from the completed video-pretrain checkpoint.

This config is intentionally explicit for the heat_food restart run:
- 36-frame GWP-V0 AgileX T-shape input is inherited from the base config.
- Data/norm stats are task-local: /shared_disk/.../gwp_v0/heat_food.
- The model is initialized from the heat_food+fold_shirt video-pt EMA checkpoint.
- Training starts a fresh joint/action run; it does not resume an old failed run.
"""

import copy
import os

from . import gwp_v0_heat_food_fold_shirt_mot as base


task_name = "heat_food"
date_str = os.environ.get("date", "heat_food_joint5ep_restart")
exp_name = f"gwp_v0_{task_name}_mot_joint_from_videopt_5ep"
output_root = os.environ.get("GWP_MOT_OUTPUT_ROOT", "/shared_disk/users/hengtao.li/codex/gwp-mot")
project_dir = f"{output_root}/experiments/{exp_name}_{date_str}"
data_root = os.environ.get("GWP_V0_DATA_ROOT", "/shared_disk/users/hengtao.li/giga_real_data/gwp_v0")
data_path = os.path.join(data_root, task_name)
norm_path = os.path.join(data_path, "norm_stats_delta.json")

checkpoint = os.environ.get(
    "MOT_STAGE1_CHECKPOINT",
    "/shared_disk/users/hengtao.li/codex/gwp-mot/experiments/"
    "gwp_v0_heat_food_fold_shirt_mot_video_pt_0530_videopt_e1_g8_h20_online/"
    "checkpoint-17890/model_ema.pt",
)

assert os.path.isfile(os.path.join(data_path, "meta", "info.json")), f"Missing LeRobot dataset: {data_path}"
assert os.path.isfile(norm_path), f"Missing per-task norm stats: {norm_path}"
assert os.path.isfile(checkpoint), f"Missing video pretrain checkpoint: {checkpoint}"

config = copy.deepcopy(base.config)
config["project_dir"] = project_dir
config["wandb"].update(
    project=os.environ.get("WANDB_PROJECT", "gwp-mot"),
    name=os.environ.get("WANDB_NAME", f"{exp_name}_{date_str}"),
    mode=os.environ.get("WANDB_MODE", "online"),
    init_timeout=int(os.environ.get("WANDB_INIT_TIMEOUT", "300")),
)

config["dataloaders"]["train"]["data_or_config"] = [
    dict(
        _class_name="LeRobotDataset",
        data_path=data_path,
        data_size=None,
        delta_info={"action": base.num_frames},
        delta_frames={k: base.image_frame_offsets for k in base.view_keys},
        # GWP-V0 heat_food MP4s currently fail in decord with "cannot find video stream".
        video_backend=os.environ.get("LEROBOT_VIDEO_BACKEND", "pyav"),
        robotype="agilex_cobot_magic",
    )
]
config["dataloaders"]["train"].update(
    sample_timeout_sec=int(os.environ.get("LEROBOT_SAMPLE_TIMEOUT_SEC", "120")),
    max_sample_retries=int(os.environ.get("LEROBOT_MAX_SAMPLE_RETRIES", "5")),
    timeout=int(os.environ.get("LEROBOT_DATALOADER_TIMEOUT_SEC", "300")),
    num_workers=int(os.environ.get("LEROBOT_NUM_WORKERS", "8")),
    prefetch_factor=int(os.environ.get("LEROBOT_PREFETCH_FACTOR", "4")),
)
config["dataloaders"]["train"]["transform"]["norm_path"] = norm_path

config["models"].update(
    checkpoint=checkpoint,
    strict=False,
    freeze_action=False,
    use_gt_action_for_video=False,
    action_loss_weight=1.0,
    visual_loss_weight=1.0,
    view_dir=project_dir,
)

config["schedulers"].update(
    decay_epochs=5,
)

config["train"].update(
    resume=False,
    max_epochs=int(os.environ.get("GWP_MAX_EPOCHS", "5")),
    max_steps=int(os.environ.get("GWP_MAX_STEPS", "0")),
    checkpoint_interval=int(os.environ.get("GWP_CHECKPOINT_INTERVAL", "5000")),
    checkpoint_epoch_interval=1,
    process_group_timeout_sec=int(os.environ.get("TORCH_DISTRIBUTED_TIMEOUT_SEC", "3600")),
    with_ema=True,
    ema=dict(
        enabled=True,
        decay=0.995,
        device="model",
    ),
)
