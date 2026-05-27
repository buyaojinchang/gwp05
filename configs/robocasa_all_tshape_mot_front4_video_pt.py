import copy
import os

from .robocasa_all_tshape_mot import config as _base_config

exp_name = 'robocasa_all_tshape_mot_front4_video_pt'
date_str = os.environ.get("date", "default")
output_root = os.environ.get("GWP_MOT_OUTPUT_ROOT", "/shared_disk/users/hengtao.li/codex/gwp-mot")
project_dir = f"{output_root}/experiments/{exp_name}_{date_str}"

config = copy.deepcopy(_base_config)
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
    max_epochs=0,
    max_steps=10000,
    checkpoint_interval=10000,
    with_ema=True,
    ema=dict(
        enabled=True,
        decay=0.995,
        device="model",
    ),
)
