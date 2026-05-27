import os

# T-shape layout: head view (agent_right) 320w x 256h on top, two wrist views 160w x 128h each on bottom
# Final image: 384h x 320w (H x W), ~1:1 ratio — within Wan pretrain bucket range, all dims divisible by 16
# NOTE: dst_size is (width, height) per WATransformsLerobot convention
dst_size = (320, 256)
num_frames = 24
action_dim = 12  # robocasa PandaOmron: action is 12d
state_dim = 16   # robocasa PandaOmron: state is 16d

exp_name = "robocasa_all_tshape_mot"
date_str = os.environ.get("date", "default")
output_root = os.environ.get("GWP_MOT_OUTPUT_ROOT", "/shared_disk/users/hengtao.li/codex/gwp-mot")
project_dir = f"{output_root}/experiments/{exp_name}_{date_str}"
data_root = "/shared_disk/users/hengtao.li/robocasa_datasets/v1.0/pretrain_gwp"

debug_single = os.environ.get("WA_DEBUG_SINGLE_GPU", "0") == "1"
gpu_ids = [0] if debug_single else [0, 1, 2, 3, 4, 5, 6, 7]

view_keys = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_eye_in_hand",
    "observation.images.robot0_agentview_right",
]
image_frame_offsets = [0, num_frames // 4, num_frames // 2, (3 * num_frames) // 4, num_frames]

# --- All datasets (atomic + composite) ---
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
        video_backend="pyav",
        robotype="robocasa",
    )
    for p in _lerobot_dirs
]

config = dict(
    project_dir=project_dir,
    runners=["MoTCasualWATrainerPretrain"],
    wandb=dict(
        project="gwp-mot",
        name=f"robocasa_all_tshape_mot_{date_str}",
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
                norm_path=os.path.join(data_root, "norm_stats_delta.json"),
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
                tshape_head_index=2,  # agentview_right as head (full size)
            ),
        ),
        test=dict(),
    ),
    models=dict(
        pretrained="/shared_disk/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers",
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
        flow_shift=2.0,            # video 用更小 shift，精细去噪提供更好的梯度监督
        action_flow_shift=5.0,     # action 保持 5.0
        expand_timesteps=True,
        action_loss_weight=1.0,
        visual_loss_weight=1.0,
        view_dir=project_dir,
        state_repeats=1,
        view_interval=200,
    ),
    optimizers=dict(
        type="CAME8Bit",
        lr=2 ** (-14.5),
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
