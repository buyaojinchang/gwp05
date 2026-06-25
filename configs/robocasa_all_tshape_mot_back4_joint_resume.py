import copy
import os

from .robocasa_all_tshape_mot_back4_joint import config as _base_config


config = copy.deepcopy(_base_config)

# Resume from the latest checkpoint in the same project_dir. max_epochs is the
# total target epoch count, so setting 3 continues the existing 2-epoch Exp2
# run for one more epoch without controlling training by a fixed step count.
config["train"].update(
    resume=True,
    max_epochs=int(os.environ.get("RESUME_TOTAL_EPOCHS", "3")),
    max_steps=0,
)
config["wandb"].update(
    name=f"{config['wandb']['name']}_resume_to_epoch{config['train']['max_epochs']}",
)
