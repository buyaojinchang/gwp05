import copy
import os

from .robocasa_all_tshape_mot import config as _base_config
from .robocasa_task_sets import is_atomic_seen_data_path

exp_name = 'robocasa_atomic_seen_tshape_mot_front4_video_pt'
date_str = os.environ.get("date", "default")
output_root = os.environ.get("GWP_MOT_OUTPUT_ROOT", "/shared_disk/users/hengtao.li/codex/gwp-mot")
project_dir = f"{output_root}/experiments/{exp_name}_{date_str}"

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
config["models"].update(dict(
    checkpoint=None,
    freeze_action=True,
    use_gt_action_for_video=True,
    action_loss_weight=0.0,
    visual_loss_weight=1.0,
))
config["train"].update(
    resume=False,
    max_epochs=1,
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
