import copy

from .robocasa_all_tshape_mot_back4_joint_epoch3_tshape320_stretch import config as _base_config


config = copy.deepcopy(_base_config)

config["train"].update(
    resume=True,
    max_epochs=3,
    max_steps=0,
)
config["wandb"].update(
    name=config["wandb"]["name"] + "_resume_lrfix",
)
