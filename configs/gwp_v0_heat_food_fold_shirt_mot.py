import os

# T-shape layout: cam_high 320x256 on top, two wrist views 160x128 each on bottom.
# Final image: 320w x 384h, all dims divisible by 16.
# NOTE: dst_size is (width, height) per WATransformsLerobot convention.
dst_size = (320, 256)
num_frames = 36
action_dim = 14
state_dim = 14

exp_name = "gwp_v0_heat_food_fold_shirt_mot"
date_str = os.environ.get("date", "default")
output_root = os.environ.get("GWP_MOT_OUTPUT_ROOT", "/shared_disk/users/hengtao.li/codex/gwp-mot")
project_dir = f"{output_root}/experiments/{exp_name}_{date_str}"
data_root = os.environ.get("GWP_V0_DATA_ROOT", "/shared_disk/users/hengtao.li/giga_real_data/gwp_v0")
pretrained_path = os.environ.get(
    "WAN22_DIFFUSERS_PATH",
    "/shared_disk/users/xuancheng.xu/models/Wan2.2-TI2V-5B-step11000-diffusers",
)
task_names = ["heat_food", "fold_shirt"]

debug_single = os.environ.get("WA_DEBUG_SINGLE_GPU", "0") == "1"
gpu_ids = [0] if debug_single else [0, 1, 2, 3, 4, 5, 6, 7]

view_keys = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
image_frame_offsets = [0, num_frames // 4, num_frames // 2, (3 * num_frames) // 4, num_frames]

lerobot_data_paths = [os.path.join(data_root, name) for name in task_names]
for p in lerobot_data_paths:
    assert os.path.isfile(os.path.join(p, "meta", "info.json")), f"Missing LeRobot dataset: {p}"

norm_path = os.path.join(data_root, "norm_stats_delta.json")
assert os.path.isfile(norm_path), f"Missing combined norm stats: {norm_path}"

data_or_config = [
    dict(
        _class_name="LeRobotDataset",
        data_path=p,
        data_size=None,
        delta_info={"action": num_frames},
        delta_frames={k: image_frame_offsets for k in view_keys},
        video_backend="pyav",
        robotype="agilex_cobot_magic",
    )
    for p in lerobot_data_paths
]

config = dict(
    project_dir=project_dir,
    runners=["MoTCasualWATrainerPretrain"],
    wandb=dict(
        project="gwp-mot",
        name=f"{exp_name}_{date_str}",
        mode=os.environ.get("WANDB_MODE", "offline"),
        init_timeout=int(os.environ.get("WANDB_INIT_TIMEOUT", "300")),
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=data_or_config,
            batch_size_per_gpu=8,
            num_workers=8,
            transform=dict(
                type="WATransformsLerobot",
                dst_size=dst_size,
                num_frames=num_frames,
                is_train=True,
                norm_path=norm_path,
                model_action_dim=action_dim,
                model_state_dim=state_dim,
                num_views=3,
                view_keys=view_keys,
                t5_len=64,
                robotype_to_embed_id={
                    "aloha": 0,
                    "agilex": 0,
                    "agilex_cobot_magic": 0,
                    "w1": 0,
                    "arx5": 0,
                    "ur5": 3,
                    "robocasa": 2,
                },
                image_cfg=dict(
                    mask_generator=dict(
                        max_ref_frames=1,
                        start=1,
                        factor=4,
                    ),
                ),
                skip_action_norm=False,
                tshape=True,
                tshape_head_index=0,
            ),
        ),
        test=dict(),
    ),
    models=dict(
        pretrained=pretrained_path,
        checkpoint=None,
        strict=False,
        action_dim=action_dim,
        state_dim=state_dim,
        type="mot",
        mot_checkpoint_mixed_attn=True,
        video_attention_mask_mode="gwp_casual",
        action_expert=dict(
            hidden_dim=1024,
            ffn_dim=4096,
        ),
        flow_shift=2.0,
        action_flow_shift=5.0,
        expand_timesteps=True,
        action_loss_weight=1.0,
        visual_loss_weight=1.0,
        freeze_action=False,
        use_gt_action_for_video=False,
        view_dir=project_dir,
        state_repeats=1,
        view_interval=200,
    ),
    optimizers=dict(
        type="CAME8Bit",
        lr=4e-5,
        weight_decay=1e-2,
    ),
    schedulers=dict(
        type="CosineScheduler",
        warmup_steps=1000,
        decay_epochs=5,
        decay_lr=4e-6,
    ),
    train=dict(
        resume=False,
        max_epochs=5,
        max_steps=0,
        gradient_accumulation_steps=2,
        mixed_precision="bf16",
        checkpoint_interval=5000,
        checkpoint_total_limit=-1,
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_interval=2,
        with_ema=True,
        ema=dict(
            enabled=True,
            decay=0.995,
            device="model",
        ),
        activation_checkpointing=False,
        activation_class_names=["WanAttention"],
    ),
    test=dict(),
)
