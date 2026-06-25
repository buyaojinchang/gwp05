import copy
import os

from .robocasa_all_tshape_mot import config as _base_config
from .robocasa_task_sets import is_atomic_seen_data_path

exp_name = "robocasa_atomic_seen_tshape_mot_front4_joint_stage2"
date_str = os.environ.get("date", "default")
output_root = os.environ.get("GWP_MOT_OUTPUT_ROOT", "/shared_disk/users/hengtao.li/codex/gwp-mot")
project_dir = f"{output_root}/experiments/{exp_name}_{date_str}"
stage1_checkpoint = os.environ.get("MOT_STAGE1_CHECKPOINT")
if not stage1_checkpoint:
    raise RuntimeError(
        "MOT_STAGE1_CHECKPOINT must point to the completed front4 stage1 checkpoint "
        "before importing robocasa_all_tshape_mot_front4_joint_stage2.config"
    )

config = copy.deepcopy(_base_config)
config["dataloaders"]["train"]["data_or_config"] = [
    d for d in config["dataloaders"]["train"]["data_or_config"]
    if is_atomic_seen_data_path(d["data_path"])
]
config["project_dir"] = project_dir
config["wandb"].update(
    project="gwp-mot",
    name=f"{exp_name}_{date_str}",
)
config["models"]["view_dir"] = project_dir
config["models"].update(
    checkpoint=stage1_checkpoint,
    freeze_action=False,
    use_gt_action_for_video=False,
    action_loss_weight=1.0,
    visual_loss_weight=1.0,
)
config["train"].update(
    resume=False,
    max_epochs=2,
    max_steps=0,
    checkpoint_interval=5000,
    checkpoint_epoch_interval=1,
    with_ema=True,
    ema=dict(
        enabled=True,
        decay=0.995,
        device="model",
    ),
)
