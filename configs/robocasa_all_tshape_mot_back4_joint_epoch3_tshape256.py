import copy
import os

from .robocasa_all_tshape_mot_back4_joint_epoch3 import config as _base_config


exp_name = "robocasa_atomic_seen_tshape_mot_back4_joint_epoch3_tshape256"
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
config["models"]["view_dir"] = project_dir
config["dataloaders"]["train"]["transform"].update(
    dst_size=(256, 256),
    random_shift_pad=0,
)
