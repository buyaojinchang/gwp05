import copy
import os

from . import gwp_v0_heat_food_fold_shirt_mot as base


task_name = os.environ.get("GWP_V0_TASK_NAME")
if task_name not in {"heat_food", "fold_shirt"}:
    raise ValueError(
        "GWP_V0_TASK_NAME must be one of: heat_food, fold_shirt "
        f"(got {task_name!r})"
    )

date_str = os.environ.get("date", "default")
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
    project="gwp-mot",
    name=f"{exp_name}_{date_str}",
    mode=os.environ.get("WANDB_MODE", "online"),
)

config["dataloaders"]["train"]["data_or_config"] = [
    dict(
        _class_name="LeRobotDataset",
        data_path=data_path,
        data_size=None,
        delta_info={"action": base.num_frames},
        delta_frames={k: base.image_frame_offsets for k in base.view_keys},
        video_backend="pyav",
        robotype="agilex_cobot_magic",
    )
]
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
    max_epochs=5,
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
