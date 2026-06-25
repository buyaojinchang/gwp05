"""RoboCasa all-data MoT joint/action training from video-pretrain EMA.

This file is intentionally self-contained for auditability. It starts a fresh
5-epoch action+video joint phase from a completed crop320 video-pretrain EMA.
It does not resume the video-pretrain optimizer/scheduler state.
"""

import os


# T-shape layout: head view is 320w x 256h, two wrist views are 160w x 128h.
# Final image: 384h x 320w. WATransformsLerobot uses dst_size=(width, height).
dst_size = (320, 256)
num_frames = 24
action_dim = 12  # RoboCasa PandaOmron action dimension.
state_dim = 16   # RoboCasa PandaOmron state dimension.

exp_name = "robocasa_all_tshape_mot_joint_from_videopt_randomcrop"
date_str = os.environ.get("date", "default")
output_root = os.environ.get("GWP_MOT_OUTPUT_ROOT", "/shared_disk/users/hengtao.li/codex/gwp-mot")
project_dir = f"{output_root}/experiments/{exp_name}_{date_str}"

data_root = os.environ.get(
    "ROBOCASA_PRETRAIN_DATA_ROOT",
    "/shared_disk/users/hengtao.li/robocasa_datasets/v1.0/pretrain_gwp",
)
norm_path = os.path.join(data_root, "norm_stats_delta.json")

default_video_pt_ckpt = (
    f"{output_root}/experiments/"
    "robocasa_all_tshape_mot_video_pt_epoch1_randomcrop_robocasa_all_videopt_crop320_0603_0212/"
    "checkpoint-442267/model_ema.pt"
)
video_pt_ckpt = os.environ.get("GWP_VIDEO_PT_CKPT", default_video_pt_ckpt)
assert os.path.isfile(video_pt_ckpt), f"Missing video-pretrain EMA checkpoint: {video_pt_ckpt}"
assert os.path.isfile(norm_path), f"Missing norm stats: {norm_path}"

joint_epochs = int(os.environ.get("GWP_MAX_EPOCHS", "5"))

view_keys = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_eye_in_hand",
    "observation.images.robot0_agentview_right",
]
image_frame_offsets = [0, num_frames // 4, num_frames // 2, (3 * num_frames) // 4, num_frames]

_lerobot_dirs = []
for category in ("atomic", "composite"):
    cat_dir = os.path.join(data_root, category)
    if not os.path.isdir(cat_dir):
        continue
    for task_name in sorted(os.listdir(cat_dir)):
        task_dir = os.path.join(cat_dir, task_name)
        if not os.path.isdir(task_dir):
            continue
        for date_dir in sorted(os.listdir(task_dir)):
            lerobot_dir = os.path.join(task_dir, date_dir, "lerobot")
            if os.path.isdir(lerobot_dir):
                _lerobot_dirs.append(lerobot_dir)

data_or_config = [
    dict(
        _class_name="LeRobotDataset",
        data_path=p,
        data_size=None,
        delta_info={"action": num_frames},
        delta_frames={k: image_frame_offsets for k in view_keys},
        video_backend=os.environ.get("LEROBOT_VIDEO_BACKEND", "decord"),
        robotype="robocasa",
    )
    for p in _lerobot_dirs
]
assert data_or_config, f"No RoboCasa LeRobot datasets found under {data_root}"

config = dict(
    project_dir=project_dir,
    runners=["MoTCasualWATrainerPretrain"],
    wandb=dict(
        project=os.environ.get("WANDB_PROJECT", "gwp-mot"),
        name=os.environ.get("WANDB_NAME", f"{exp_name}_{date_str}"),
        mode=os.environ.get("WANDB_MODE", "online"),
        init_timeout=int(os.environ.get("WANDB_INIT_TIMEOUT", "300")),
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=data_or_config,
            batch_size_per_gpu=int(os.environ.get("GWP_BATCH_SIZE_PER_GPU", "4")),
            num_workers=int(os.environ.get("LEROBOT_NUM_WORKERS", "8")),
            prefetch_factor=int(os.environ.get("LEROBOT_PREFETCH_FACTOR", "4")),
            sample_timeout_sec=int(os.environ.get("LEROBOT_SAMPLE_TIMEOUT_SEC", "120")),
            max_sample_retries=int(os.environ.get("LEROBOT_MAX_SAMPLE_RETRIES", "5")),
            timeout=int(os.environ.get("LEROBOT_DATALOADER_TIMEOUT_SEC", "300")),
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
                tshape_head_index=2,
                resize_mode="crop",
                random_shift_pad=0,
            ),
        ),
        test=dict(),
    ),
    models=dict(
        pretrained=os.environ.get(
            "WAN_PRETRAINED_DIR",
            "/shared_disk/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers",
        ),
        checkpoint=video_pt_ckpt,
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
        freeze_action=False,
        use_gt_action_for_video=os.environ.get("GWP_USE_GT_ACTION_FOR_VIDEO", "0") == "1",
        action_loss_weight=float(os.environ.get("GWP_ACTION_LOSS_WEIGHT", "1.0")),
        visual_loss_weight=float(os.environ.get("GWP_VISUAL_LOSS_WEIGHT", "1.0")),
        view_dir=project_dir,
        state_repeats=1,
        view_interval=int(os.environ.get("GWP_VIEW_INTERVAL", "1000000")),
    ),
    optimizers=dict(
        type="CAME8Bit",
        lr=2 ** (-14.5),
        weight_decay=1e-2,
    ),
    schedulers=dict(
        type="CosineScheduler",
        warmup_steps=int(os.environ.get("GWP_WARMUP_STEPS", "1000")),
        decay_epochs=int(os.environ.get("GWP_DECAY_EPOCHS", str(joint_epochs))),
        decay_lr=float(os.environ.get("GWP_DECAY_LR", "4e-6")),
    ),
    train=dict(
        resume=os.environ.get("GWP_RESUME", "0") == "1",
        max_epochs=joint_epochs,
        max_steps=int(os.environ.get("GWP_MAX_STEPS", "0")),
        gradient_accumulation_steps=int(os.environ.get("GWP_GRAD_ACCUM_STEPS", "4")),
        mixed_precision=os.environ.get("GWP_MIXED_PRECISION", "bf16"),
        process_group_timeout_sec=int(os.environ.get("TORCH_DISTRIBUTED_TIMEOUT_SEC", "3600")),
        checkpoint_interval=int(os.environ.get("GWP_CHECKPOINT_INTERVAL", "5000")),
        checkpoint_epoch_interval=1,
        checkpoint_total_limit=-1,
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_interval=int(os.environ.get("GWP_LOG_INTERVAL", "2")),
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
