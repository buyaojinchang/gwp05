import copy

from .robocasa_all_tshape_mot_back4_joint import config as _base_config


config = copy.deepcopy(_base_config)

config["wandb"].update(
    name=f"{config['wandb']['name']}_fromscratch_epoch3",
)
config["models"].update(
    checkpoint=None,
)
config["train"].update(
    resume=False,
    max_epochs=3,
    max_steps=0,
)
