"""MoT-only open-loop evaluation for GigaWorld-Policy on RoboCasa datasets.

Loads a checkpoint, feeds dataset images/states to the model, and compares
predicted actions against ground-truth. Produces per-dimension action plots
(gt vs pred) similar to training visualizations.
"""

import argparse
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

logger = logging.getLogger(__name__)

RAW_VIEW_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_eye_in_hand",
    "observation.images.robot0_agentview_right",
]
GWP_V0_RAW_VIEW_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
RAW_VIEW_KEY_GROUPS = [RAW_VIEW_KEYS, GWP_V0_RAW_VIEW_KEYS]
TSHAPE_VIEW_KEY = "observation.images.tshape"

ACTION_DIM_NAMES = [
    "base_x",
    "base_y",
    "base_yaw",
    "ee_rot_rx_zero_std",
    "ctrl_mode",
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_rot_x",
    "ee_rot_y",
    "ee_rot_z",
    "gripper",
]

try:
    from configs.robocasa_task_sets import ATOMIC_SEEN_TASKS
except Exception:
    ATOMIC_SEEN_TASKS = frozenset({
        "CloseBlenderLid",
        "CloseFridge",
        "CloseToasterOvenDoor",
        "CoffeeSetupMug",
        "NavigateKitchen",
        "OpenCabinet",
        "OpenDrawer",
        "OpenStandMixerHead",
        "PickPlaceCounterToCabinet",
        "PickPlaceCounterToStove",
        "PickPlaceDrawerToCounter",
        "PickPlaceSinkToCounter",
        "PickPlaceToasterToCounter",
        "SlideDishwasherRack",
        "TurnOffStove",
        "TurnOnElectricKettle",
        "TurnOnMicrowave",
        "TurnOnSinkFaucet",
    })

# -- Model loading ---------------------------------------------------------

def build_model(pretrained_path, checkpoint_path, action_dim=12, state_dim=16,
                flow_shift=5.0, device="cuda", dtype=torch.bfloat16,
                action_expert_hidden_dim=1024, action_expert_ffn_dim=4096,
                mot_checkpoint_mixed_attn=True):
    from diffusers.models import AutoencoderKLWan
    from world_action_model.models.transformer_wa_mot import MoTWorldActionTransformer
    from world_action_model.trainers.wa_trainer import get_model_path, process_transformer

    pretrained = get_model_path(pretrained_path)

    vae = AutoencoderKLWan.from_pretrained(os.path.join(pretrained, "vae"))
    vae.requires_grad_(False).eval().to(device, dtype=dtype)

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype=dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype=dtype)

    transformer = MoTWorldActionTransformer.from_pretrained_video(
        transformer_pretrained=os.path.join(pretrained, "transformer"),
        torch_dtype=dtype,
        action_dim=action_dim,
        state_dim=state_dim,
        action_expert={
            "hidden_dim": int(action_expert_hidden_dim),
            "ffn_dim": int(action_expert_ffn_dim),
        },
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_attention_mask_mode="gwp_casual",
    )
    process_transformer(transformer.video_expert, {})
    transformer.to(device, dtype=dtype)

    print(f"Loading checkpoint: {checkpoint_path}")
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    elif "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    keys = tuple(sd.keys())
    if not any(k.startswith("transformer.mot.") or k.startswith("mot.") or ".mot." in k for k in keys):
        raise ValueError(
            "Open-loop evaluation is MoT-only in this project. "
            f"Checkpoint has no MoT keys: {checkpoint_path}"
        )
    tf_state = {}
    for k, v in sd.items():
        tf_state[k.removeprefix("transformer.")] = v
    missing, unexpected = transformer.load_state_dict(tf_state, strict=False)
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    transformer.eval()

    return dict(vae=vae, transformer=transformer, latents_mean=latents_mean,
                latents_std=latents_std, flow_shift=flow_shift,
                action_flow_shift=flow_shift,  # will be overridden by CLI arg
                action_dim=action_dim, state_dim=state_dim, device=device, dtype=dtype)


# ── Normalization ────────────────────────────────────────────────────────

def load_norm_stats(stats_path, state_dim, action_dim, device):
    with open(stats_path) as f:
        ns = json.load(f)["norm_stats"]
    def _t(arr, dim):
        t = torch.tensor(arr, dtype=torch.float32, device=device).flatten()[:dim]
        if t.numel() < dim:
            t = torch.nn.functional.pad(t, (0, dim - t.numel()))
        return t
    return dict(
        state_mean=_t(ns["observation.state"]["mean"], state_dim),
        state_std=_t(ns["observation.state"]["std"], state_dim),
        action_mean=_t(ns["action"]["mean"], action_dim),
        action_std=_t(ns["action"]["std"], action_dim),
    )

def normalize_state(state, norm):
    return (state - norm["state_mean"]) / norm["state_std"].clamp_min(1e-8)

def denormalize_action(action, norm):
    return action * norm["action_std"].clamp_min(1e-8) + norm["action_mean"]


def active_action_dims(norm, actual_dim, include_zero_std_dims=False, threshold=1e-4):
    """Return action dims used for metrics/primary plots, matching training mask."""
    if include_zero_std_dims:
        return list(range(actual_dim)), []
    action_std = norm["action_std"][:actual_dim].detach().float().cpu().numpy()
    active = [int(i) for i, std in enumerate(action_std) if float(std) >= threshold]
    ignored = [int(i) for i, std in enumerate(action_std) if float(std) < threshold]
    if len(active) == 0:
        active = list(range(actual_dim))
        ignored = []
    return active, ignored


def action_stats(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return {"abs_sum": 0.0, "max_abs": 0.0, "mean_abs_dim": [], "std_dim": []}
    return {
        "abs_sum": float(np.abs(arr).sum()),
        "max_abs": float(np.abs(arr).max()),
        "mean_abs_dim": np.abs(arr).mean(axis=0).astype(float).tolist(),
        "std_dim": arr.std(axis=0).astype(float).tolist(),
    }


def action_delta_mask(mode, action_dim):
    """Return dims that are represented as action-state deltas in training."""
    if mode == "raw":
        return None
    if mode != "agilex_cobot_magic":
        raise ValueError(f"Unsupported gt_delta_mode: {mode}")
    base = np.asarray(
        [True, True, True, True, True, True, False,
         True, True, True, True, True, True, False],
        dtype=bool,
    )
    if action_dim <= base.shape[0]:
        return base[:action_dim]
    return np.pad(base, (0, action_dim - base.shape[0]), constant_values=False)


def state_base_for_action(states, start, action_dim):
    base = np.zeros((action_dim,), dtype=states.dtype)
    if states.shape[0] > start:
        n = min(action_dim, states.shape[-1])
        base[:n] = states[start, :n]
    return base


def raw_to_delta_action_chunk(raw_actions, states, start, end, mode):
    chunk = raw_actions[start:end].copy()
    mask = action_delta_mask(mode, chunk.shape[-1])
    if mask is None:
        return chunk
    base = state_base_for_action(states, start, chunk.shape[-1])
    chunk[:, mask] = chunk[:, mask] - base[mask]
    return chunk


def delta_to_raw_action_chunk(delta_actions, states, start, mode):
    chunk = delta_actions.copy()
    mask = action_delta_mask(mode, chunk.shape[-1])
    if mask is None:
        return chunk
    base = state_base_for_action(states, start, chunk.shape[-1])
    chunk[:, mask] = chunk[:, mask] + base[mask]
    return chunk


def offset_mae_stats(gt, pr, frame_mask, step_interval, active_dims):
    if step_interval <= 0:
        return []
    err = np.abs(gt[:, active_dims] - pr[:, active_dims]).mean(axis=1)
    frame_idx = np.arange(gt.shape[0])
    rows = []
    for offset in range(step_interval):
        m = (frame_idx % step_interval == offset) & frame_mask
        rows.append({
            "offset": int(offset),
            "num_frames": int(m.sum()),
            "mae": float(err[m].mean()) if m.any() else None,
        })
    return rows


def block_mae_stats(gt, pr, frame_mask, step_interval, active_dims, max_blocks=20):
    if step_interval <= 0:
        return []
    err = np.abs(gt[:, active_dims] - pr[:, active_dims]).mean(axis=1)
    rows = []
    for block_idx, start in enumerate(range(0, gt.shape[0], step_interval)):
        if block_idx >= max_blocks:
            break
        end = min(start + step_interval, gt.shape[0])
        m = frame_mask[start:end]
        rows.append({
            "block": int(block_idx),
            "start": int(start),
            "end": int(end),
            "num_frames": int(m.sum()),
            "mae": float(err[start:end][m].mean()) if m.any() else None,
        })
    return rows


def hold_state_baseline(states, action_dim, step_interval):
    """Raw-action baseline that keeps the state at each replan boundary."""
    baseline = np.zeros((states.shape[0], action_dim), dtype=np.float32)
    state_dim = min(action_dim, states.shape[-1])
    for start in range(0, states.shape[0], step_interval):
        end = min(start + step_interval, states.shape[0])
        baseline[start:end, :state_dim] = states[start, :state_dim]
    return baseline


def per_dim_diagnostics(gt, pr, states, frame_mask, step_interval, action_dim_names):
    action_dim = gt.shape[-1]
    baseline = hold_state_baseline(states, action_dim, step_interval)
    rows = []
    frame_idx = np.arange(gt.shape[0])
    for dim in range(action_dim):
        valid = frame_mask
        gt_d = gt[valid, dim]
        pr_d = pr[valid, dim]
        base_d = baseline[valid, dim]
        mae = float(np.mean(np.abs(pr_d - gt_d)))
        mse = float(np.mean((pr_d - gt_d) ** 2))
        hold_mae = float(np.mean(np.abs(base_d - gt_d)))
        offset0_mask = valid & ((frame_idx % step_interval) == 0)
        offset_last_mask = valid & ((frame_idx % step_interval) == (step_interval - 1))
        rows.append({
            "dim": int(dim),
            "name": action_dim_name(action_dim_names, dim),
            "mae": mae,
            "mse": mse,
            "hold_state_mae": hold_mae,
            "model_over_hold": float(mae / max(hold_mae, 1e-12)),
            "bias": float(np.mean(pr_d - gt_d)),
            "gt_std": float(np.std(gt_d)),
            "pred_std": float(np.std(pr_d)),
            "offset0_mae": float(np.mean(np.abs(pr[offset0_mask, dim] - gt[offset0_mask, dim]))) if offset0_mask.any() else None,
            "offset_last_mae": float(np.mean(np.abs(pr[offset_last_mask, dim] - gt[offset_last_mask, dim]))) if offset_last_mask.any() else None,
        })
    return rows


def action_dim_name(action_dim_names, dim):
    if action_dim_names is not None and dim < len(action_dim_names):
        return action_dim_names[dim]
    if dim < len(ACTION_DIM_NAMES):
        return ACTION_DIM_NAMES[dim]
    return f"dim_{dim}"


def plot_action_comparison(save_path, task_name, gt, pr, dims, mse, mae,
                           title_suffix="", action_dim_names=None):
    if len(dims) == 0:
        return
    T = gt.shape[0]
    cols = 4
    rows = (len(dims) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).flatten()

    for ax_idx, d in enumerate(dims):
        ax = axes[ax_idx]
        label = action_dim_name(action_dim_names, d)
        ax.plot(range(T), gt[:, d], label="gt", color="tab:blue", linewidth=1.2)
        ax.plot(range(T), pr[:, d], label="pred", color="tab:orange", linewidth=1.2, alpha=0.8)
        ax.set_title(f"{d}: {label}", fontsize=10)
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    for ax_idx in range(len(dims), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(f"{task_name}{title_suffix}  (MSE={mse:.4f}, MAE={mae:.4f})", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ── Image preprocessing ─────────────────────────────────────────────────

def preprocess_image(img_uint8, dst_size=(320, 256)):
    dst_w, dst_h = dst_size
    img = Image.fromarray(img_uint8)
    w, h = img.size
    if float(dst_h) / h < float(dst_w) / w:
        new_h = int(round(float(dst_w) / w * h))
        new_w = dst_w
    else:
        new_h = dst_h
        new_w = int(round(float(dst_h) / h * w))
    img_t = TF.to_tensor(img).unsqueeze(0)
    img_t = TF.resize(img_t, (new_h, new_w), InterpolationMode.BILINEAR)
    x1, y1 = (new_w - dst_w) // 2, (new_h - dst_h) // 2
    img_t = TF.crop(img_t, y1, x1, dst_h, dst_w)
    return img_t * 2.0 - 1.0  # 1,C,H,W


# ── Flow matching sampler (using diffusers scheduler, matching ref impl) ─

@torch.no_grad()
def sample_action(md, ref_latents, prompt_embeds, state, num_steps=10, num_frames=24):
    """Sample actions from a MoT checkpoint using the action-only path."""
    device, dtype = md["device"], md["dtype"]
    transformer = md["transformer"]
    if hasattr(transformer, "clear_action_only_cache"):
        transformer.clear_action_only_cache()
    flow_shift = md.get("action_flow_shift", md["flow_shift"])
    ad = md["action_dim"]
    bs = ref_latents.shape[0]

    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps

    noisy_action = torch.randn(bs, num_frames, ad, device=device, dtype=dtype)

    for i, t in enumerate(timesteps):
        ns_tok = state.shape[1]
        lh, lw = ref_latents.shape[-2], ref_latents.shape[-1]
        fpt = lh * lw // 4
        total = ns_tok + fpt + num_frames
        timestep = torch.zeros(bs, total, device=device, dtype=dtype)
        # token order for MoT action-only: [state | ref_video | action]
        noise_t = t.float()
        timestep[:, ns_tok + fpt:] = noise_t

        action_pred = transformer(
            ref_latents=ref_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            action=noisy_action,
            state=state,
            action_only=True,
        )
        noisy_action = scheduler.step(action_pred, t, noisy_action, return_dict=False)[0]

    return noisy_action


# ── Dataset helpers ──────────────────────────────────────────────────────

def load_episode_data(lerobot_dir, episode_idx, view_keys):
    """Load one episode's parquet + video frames."""
    import pandas as pd

    parquet = os.path.join(lerobot_dir, "data", "chunk-000",
                           f"episode_{episode_idx:06d}.parquet")
    df = pd.read_parquet(parquet)

    actions = np.stack([df["action"].iloc[i] for i in range(len(df))]).astype(np.float32)
    states = np.stack([df["observation.state"].iloc[i] for i in range(len(df))]).astype(np.float32)

    # Load video frames
    frames_per_view = {}
    for vk in view_keys:
        video_path = os.path.join(lerobot_dir, "videos", "chunk-000", vk,
                                  f"episode_{episode_idx:06d}.mp4")
        if os.path.exists(video_path):
            import imageio
            reader = imageio.get_reader(video_path)
            frames = [np.array(f) for f in reader]
            reader.close()
            frames_per_view[vk] = frames

    task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns else 0

    return dict(actions=actions, states=states, frames=frames_per_view,
                num_frames=len(df), task_index=task_index)


def load_action_dim_names(lerobot_dir, actual_dim):
    """Prefer dataset metadata names over the RoboCasa fallback labels."""
    info_path = os.path.join(lerobot_dir, "meta", "info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, "r") as f:
                info = json.load(f)
            names = info.get("features", {}).get("action", {}).get("names")
            if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
                names = names[0]
            if isinstance(names, list) and len(names) >= actual_dim:
                return [str(n) for n in names[:actual_dim]]
        except Exception as exc:
            logger.warning("Failed to read action names from %s: %s", info_path, exc)
    return [action_dim_name(None, i) for i in range(actual_dim)]


# ── T5 embedding loader ─────────────────────────────────────────────

T5_MAX_LEN = 64

def load_t5_embedding(lerobot_dir, episode_idx, device, dtype, task_index=0):
    """Load t5 embedding from dataset, pad to T5_MAX_LEN.

    Searches in order:
      1. <lerobot_dir>/t5_embedding/episode_XXXXXX.pt  (per-episode)
      2. <lerobot_dir>/meta/t5_text_embeds.pt          (dict: task_index -> tensor)
    """
    def _pad_and_return(t5):
        if not isinstance(t5, torch.Tensor):
            t5 = torch.as_tensor(t5)
        t5 = t5.float()[:T5_MAX_LEN]
        if t5.shape[0] < T5_MAX_LEN:
            t5 = torch.nn.functional.pad(t5, (0, 0, 0, T5_MAX_LEN - t5.shape[0]))
        return t5.unsqueeze(0).to(device, dtype=dtype)

    # 1) per-episode file
    per_ep_path = os.path.join(lerobot_dir, "t5_embedding",
                               f"episode_{episode_idx:06d}.pt")
    if os.path.exists(per_ep_path):
        t5 = torch.load(per_ep_path, map_location="cpu", weights_only=False)
        return _pad_and_return(t5)

    # 2) meta/t5_text_embeds.pt (dict: task_index -> tensor)
    for name in ("t5_text_embeds.pt", "text_embeddings.pt"):
        meta_path = os.path.join(lerobot_dir, "meta", name)
        if os.path.exists(meta_path):
            data = torch.load(meta_path, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                t5 = data.get(task_index, next(iter(data.values())))
                return _pad_and_return(t5)
            elif isinstance(data, torch.Tensor):
                return _pad_and_return(data)

    return None


# ── Open-loop evaluation ────────────────────────────────────────────────

def discover_lerobot_dirs(data_root):
    """Find LeRobot dataset dirs under pretrain_gwp or flat collected layouts."""
    lerobot_dirs = []
    nested_found = False
    for cat in ("atomic", "composite"):
        cat_dir = os.path.join(data_root, cat)
        if not os.path.isdir(cat_dir):
            continue
        nested_found = True
        for task in sorted(os.listdir(cat_dir)):
            task_dir = os.path.join(cat_dir, task)
            if not os.path.isdir(task_dir):
                continue
            for date_dir in sorted(os.listdir(task_dir)):
                ld = os.path.join(task_dir, date_dir, "lerobot")
                if os.path.isdir(ld):
                    lerobot_dirs.append((task, ld))
    if not nested_found:
        for task in sorted(os.listdir(data_root)):
            task_dir = os.path.join(data_root, task)
            if not os.path.isdir(task_dir):
                continue
            ld = os.path.join(task_dir, "lerobot")
            if os.path.isdir(ld):
                lerobot_dirs.append((task, ld))
    return lerobot_dirs


def select_lerobot_dirs(lerobot_dirs, args):
    task_filter = None
    if args.task_set == "atomic_seen":
        task_filter = set(ATOMIC_SEEN_TASKS)
    if args.task_names:
        explicit = set(args.task_names)
        task_filter = explicit if task_filter is None else task_filter & explicit

    if task_filter is not None:
        available_before_filter = {task for task, _ in lerobot_dirs}
        missing_before_filter = sorted(task_filter - available_before_filter)
        if missing_before_filter and args.task_set == "atomic_seen" and not args.task_names:
            raise RuntimeError(f"Missing atomic_seen tasks under data_root: {missing_before_filter}")
        lerobot_dirs = [(task, ld) for task, ld in lerobot_dirs if task in task_filter]

    if args.one_per_task:
        selected = []
        seen = set()
        for task, ld in lerobot_dirs:
            if task in seen:
                continue
            selected.append((task, ld))
            seen.add(task)
        lerobot_dirs = selected

    if args.max_datasets > 0:
        lerobot_dirs = lerobot_dirs[:args.max_datasets]

    return lerobot_dirs


@torch.no_grad()
def run_openloop(args, md, norm, t5_emb):
    device, dtype = md["device"], md["dtype"]
    vae = md["vae"]
    latents_mean, latents_std = md["latents_mean"], md["latents_std"]
    action_chunk = args.action_chunk
    num_frames = args.num_frames
    action_dim = args.action_dim

    view_keys = [TSHAPE_VIEW_KEY] + [vk for group in RAW_VIEW_KEY_GROUPS for vk in group]
    lerobot_dirs = select_lerobot_dirs(discover_lerobot_dirs(args.data_root), args)
    selected_tasks = [task for task, _ in lerobot_dirs]

    print(f"Found {len(lerobot_dirs)} datasets, evaluating episode {args.episode_idx} from each")
    print(f"Task set: {args.task_set}; one_per_task={args.one_per_task}")
    print(f"Selected tasks: {selected_tasks}")

    # fallback t5 embedding (zeros)
    fallback_t5 = t5_emb.unsqueeze(0).to(device, dtype=dtype)

    # Per-task metrics for final summary table
    results = []  # list of dict(task, num_frames, mse, mae)

    for task_name, ld in lerobot_dirs:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"Dataset: {ld}")

        ep = load_episode_data(ld, args.episode_idx, view_keys)

        # Load per-dataset t5 embedding
        prompt_embeds = load_t5_embedding(ld, args.episode_idx, device, dtype,
                                          task_index=ep.get("task_index", 0))
        if prompt_embeds is None:
            print(f"  No t5 embedding found, using fallback zeros")
            prompt_embeds = fallback_t5
        else:
            print(f"  Loaded t5 embedding: {prompt_embeds.shape}")

        T = ep["num_frames"]
        raw_gt_actions = ep["actions"]  # T, action_dim
        states_np = ep["states"]
        print(f"  Episode length: {T} frames, action_dim: {raw_gt_actions.shape[-1]}")

        # Sliding window: step through episode, generate num_frames actions,
        # then accumulate the first action_chunk predictions.
        all_pred = np.zeros_like(raw_gt_actions)  # T, action_dim
        all_gt = np.zeros_like(raw_gt_actions)  # T, action_dim in the requested comparison space
        pred_counts = np.zeros(T)
        gt_counts = np.zeros(T)

        available_views = sorted(ep["frames"].keys())
        view_mode = args.input_view_mode
        if view_mode == "auto":
            view_mode = "tshape" if TSHAPE_VIEW_KEY in ep["frames"] else "raw"
        print(f"  Loaded video views: {available_views if available_views else 'none'}")
        print(f"  Input view mode: {view_mode}")

        step_interval = args.replan_steps
        for start in range(0, T, step_interval):
            dst_w, dst_h = tuple(args.dst_size)
            if view_mode == "tshape":
                frames = ep["frames"].get(TSHAPE_VIEW_KEY)
                if frames is None or start >= len(frames):
                    print(f"  Skipping frame {start}: missing {TSHAPE_VIEW_KEY}")
                    continue
                ref_image = preprocess_image(frames[start], (dst_w, dst_h)).to(device, dtype=dtype)
            elif view_mode == "raw":
                half_w, half_h = dst_w // 2, dst_h // 2
                head_idx = args.tshape_head_index
                head = None
                others = []
                for raw_view_keys in RAW_VIEW_KEY_GROUPS:
                    ts_images = []
                    for vi, vk in enumerate(raw_view_keys):
                        if vk not in ep["frames"] or start >= len(ep["frames"][vk]):
                            continue
                        raw = ep["frames"][vk][start]
                        size = (dst_w, dst_h) if vi == head_idx else (half_w, half_h)
                        ts_images.append((vi, preprocess_image(raw, size)))

                    head = next((img for i, img in ts_images if i == head_idx), None)
                    others = [img for i, img in ts_images if i != head_idx]
                    if head is not None and others:
                        break
                if head is None or not others:
                    print(f"  Skipping frame {start}: missing raw T-shape views")
                    continue

                wrist_row = torch.cat(others, dim=-1)
                if wrist_row.shape[-1] < head.shape[-1]:
                    wrist_row = torch.nn.functional.pad(
                        wrist_row, (0, head.shape[-1] - wrist_row.shape[-1]))
                elif wrist_row.shape[-1] > head.shape[-1]:
                    wrist_row = wrist_row[..., :head.shape[-1]]
                ref_image = torch.cat([head, wrist_row], dim=-2).to(device, dtype=dtype)
            else:
                raise ValueError(f"Unsupported input_view_mode: {view_mode}")
            ref_image_5d = ref_image.unsqueeze(2)
            ref_latents = vae.encode(ref_image_5d).latent_dist.mode()
            ref_latents = (ref_latents - latents_mean) * latents_std

            # State
            state = torch.from_numpy(ep["states"][start:start+1]).to(device, dtype=dtype)
            sd = args.state_dim
            if state.shape[-1] > sd:
                state = state[..., :sd]
            elif state.shape[-1] < sd:
                state = torch.nn.functional.pad(state, (0, sd - state.shape[-1]))
            state = normalize_state(state.float(), norm).to(dtype=dtype).unsqueeze(0)

            pred = sample_action(
                md,
                ref_latents,
                prompt_embeds,
                state,
                num_steps=args.num_steps,
                num_frames=num_frames,
            )
            if args.skip_action_denorm:
                pred = pred.float().squeeze(0).cpu().numpy()
            else:
                pred = denormalize_action(pred.float().squeeze(0), norm).cpu().numpy()

            # Accumulate predictions. Closed-loop deployment keeps only the
            # replan prefix from each chunk; evaluating the discarded suffix can
            # make long-horizon actions dominate open-loop diagnostics.
            eval_chunk = min(action_chunk, args.replan_steps) if args.eval_replan_prefix_only else action_chunk
            end = min(start + eval_chunk, T)
            chunk_len = end - start
            pred_chunk = pred[:chunk_len]
            if args.comparison_space == "raw_from_delta":
                pred_chunk = delta_to_raw_action_chunk(
                    pred_chunk,
                    states_np,
                    start,
                    args.gt_delta_mode,
                )
                gt_chunk = raw_gt_actions[start:end]
            elif args.comparison_space == "window_delta":
                gt_chunk = raw_to_delta_action_chunk(
                    raw_gt_actions,
                    states_np,
                    start,
                    end,
                    args.gt_delta_mode,
                )[:chunk_len]
            elif args.comparison_space == "raw":
                gt_chunk = raw_gt_actions[start:end]
            else:
                raise ValueError(f"Unsupported comparison_space: {args.comparison_space}")

            all_pred[start:end] += pred_chunk
            all_gt[start:end] += gt_chunk
            pred_counts[start:end] += 1
            gt_counts[start:end] += 1

        # Average overlapping predictions
        mask = pred_counts > 0
        covered_frames = int(mask.sum())
        if covered_frames == 0:
            raise RuntimeError(
                f"No predictions were generated for {task_name}. "
                f"Available video views: {sorted(ep['frames'].keys())}. "
                "Check --input_view_mode and dataset video keys."
            )
        if covered_frames < T:
            print(f"  Warning: predictions cover {covered_frames}/{T} frames; metrics use covered frames only")
        all_pred[mask] /= pred_counts[mask, None]
        all_gt[mask] /= gt_counts[mask, None]

        # Truncate to actual action dim
        actual_dim = min(action_dim, raw_gt_actions.shape[-1])
        action_dim_names = load_action_dim_names(ld, actual_dim)
        gt = all_gt[:, :actual_dim]
        pr = all_pred[:, :actual_dim]
        active_dims, ignored_dims = active_action_dims(
            norm,
            actual_dim,
            include_zero_std_dims=args.include_zero_std_dims,
            threshold=args.zero_std_threshold,
        )

        gt_metric = gt[mask][:, active_dims]
        pr_metric = pr[mask][:, active_dims]
        mse = float(np.mean((gt_metric - pr_metric) ** 2))
        mae = float(np.mean(np.abs(gt_metric - pr_metric)))
        mse_all = float(np.mean((gt[mask] - pr[mask]) ** 2))
        mae_all = float(np.mean(np.abs(gt[mask] - pr[mask])))
        gt_stats = action_stats(gt[mask])
        pred_stats = action_stats(pr[mask])
        offset_mae = offset_mae_stats(gt, pr, mask, step_interval, active_dims)
        block_mae = block_mae_stats(gt, pr, mask, step_interval, active_dims)
        per_dim = per_dim_diagnostics(gt, pr, states_np, mask, step_interval, action_dim_names)

        print(f"  Active dims: {active_dims}; ignored zero-std dims: {ignored_dims}")
        print(f"  GT abs_sum: {gt_stats['abs_sum']:.6f}, Pred abs_sum: {pred_stats['abs_sum']:.6f}")
        print(f"  MSE(active): {mse:.6f}, MAE(active): {mae:.6f}; MSE(all): {mse_all:.6f}, MAE(all): {mae_all:.6f}")
        if offset_mae:
            preview = ", ".join(
                f"{row['offset']}:{row['mae']:.4f}" for row in offset_mae[:min(10, len(offset_mae))]
                if row["mae"] is not None
            )
            print(f"  Offset MAE preview: {preview}")
        top_dims = sorted(per_dim, key=lambda row: row["mae"], reverse=True)[:5]
        print("  Top dim MAE: " + ", ".join(
            f"{row['dim']}:{row['name']}={row['mae']:.4f}(hold={row['hold_state_mae']:.4f})"
            for row in top_dims
        ))
        results.append(dict(
            task=task_name,
            data_path=ld,
            num_frames=int(T),
            pred_covered_frames=covered_frames,
            view_mode=view_mode,
            comparison_space=args.comparison_space,
            gt_delta_mode=args.gt_delta_mode,
            eval_replan_prefix_only=args.eval_replan_prefix_only,
            eval_chunk=int(eval_chunk),
            active_dims=active_dims,
            ignored_dims=ignored_dims,
            action_dim_names=action_dim_names,
            mse=mse,
            mae=mae,
            mse_all_dims=mse_all,
            mae_all_dims=mae_all,
            gt_abs_sum=gt_stats["abs_sum"],
            pred_abs_sum=pred_stats["abs_sum"],
            offset_mae=offset_mae,
            block_mae_first20=block_mae,
            per_dim_diagnostics=per_dim,
        ))

        # Plot and diagnostics
        save_dir = os.path.join(args.output_dir, task_name)
        os.makedirs(save_dir, exist_ok=True)
        plot_action_comparison(
            os.path.join(save_dir, f"ep{args.episode_idx:03d}_actions.png"),
            task_name,
            gt,
            pr,
            active_dims,
            mse,
            mae,
            title_suffix=" active dims",
            action_dim_names=action_dim_names,
        )
        if not args.skip_all_dims_plot:
            plot_action_comparison(
                os.path.join(save_dir, f"ep{args.episode_idx:03d}_actions_all_dims.png"),
                task_name,
                gt,
                pr,
                list(range(actual_dim)),
                mse_all,
                mae_all,
                title_suffix=" all dims",
                action_dim_names=action_dim_names,
            )

        np.savez_compressed(
            os.path.join(save_dir, f"ep{args.episode_idx:03d}_arrays.npz"),
            gt=gt,
            pred=pr,
            pred_counts=pred_counts,
            gt_counts=gt_counts,
            active_dims=np.asarray(active_dims, dtype=np.int64),
            ignored_dims=np.asarray(ignored_dims, dtype=np.int64),
            action_dim_names=np.asarray(action_dim_names, dtype="U64"),
        )

        metrics = {
            "task": task_name,
            "episode": args.episode_idx,
            "num_frames": int(T),
            "pred_covered_frames": covered_frames,
            "view_mode": view_mode,
            "comparison_space": args.comparison_space,
            "gt_delta_mode": args.gt_delta_mode,
            "eval_replan_prefix_only": args.eval_replan_prefix_only,
            "eval_chunk": int(eval_chunk),
            "active_dims": active_dims,
            "ignored_dims": ignored_dims,
            "action_dim_names": action_dim_names,
            "mse": float(mse),
            "mae": float(mae),
            "mse_all_dims": float(mse_all),
            "mae_all_dims": float(mae_all),
            "gt_stats": gt_stats,
            "pred_stats": pred_stats,
            "offset_mae": offset_mae,
            "block_mae_first20": block_mae,
            "per_dim_diagnostics": per_dim,
        }
        with open(os.path.join(save_dir, f"ep{args.episode_idx:03d}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  Saved to {save_dir}")

    # Summary across all tasks
    print(f"\n{'='*60}")
    print(f"All plots saved to: {args.output_dir}")

    if len(results) == 0:
        print("No datasets evaluated, skipping summary table.")
        return

    task_col_w = max(len("Task"), max(len(r["task"]) for r in results))
    header = f"  {'#':>3}  {'Task':<{task_col_w}}  {'Frames':>7}  {'MSE':>12}  {'MAE':>12}"
    sep = "  " + "-" * (len(header) - 2)
    print(f"\n{'='*len(header)}")
    print(f"  Open-Loop Summary  (episode={args.episode_idx}, tasks={len(results)})")
    print("=" * len(header))
    print(header)
    print(sep)
    for i, r in enumerate(results):
        print(f"  {i+1:>3}  {r['task']:<{task_col_w}}  {r['num_frames']:>7d}"
              f"  {r['mse']:>12.6f}  {r['mae']:>12.6f}")
    print(sep)

    # Aggregated metrics
    total_frames = sum(r["num_frames"] for r in results)
    mean_mse = float(np.mean([r["mse"] for r in results]))
    mean_mae = float(np.mean([r["mae"] for r in results]))
    # Frame-weighted average (accounts for episode length differences)
    wmean_mse = float(sum(r["mse"] * r["num_frames"] for r in results) / max(total_frames, 1))
    wmean_mae = float(sum(r["mae"] * r["num_frames"] for r in results) / max(total_frames, 1))

    print(f"  {'':>3}  {'Mean (unweighted)':<{task_col_w}}  {total_frames:>7d}"
          f"  {mean_mse:>12.6f}  {mean_mae:>12.6f}")
    print(f"  {'':>3}  {'Mean (frame-weighted)':<{task_col_w}}  {total_frames:>7d}"
          f"  {wmean_mse:>12.6f}  {wmean_mae:>12.6f}")
    print("=" * len(header))

    # Save summary
    summary = {
        "checkpoint": args.checkpoint_path,
        "data_root": args.data_root,
        "episode": args.episode_idx,
        "task_set": args.task_set,
        "task_names": args.task_names,
        "one_per_task": args.one_per_task,
        "comparison_space": args.comparison_space,
        "gt_delta_mode": args.gt_delta_mode,
        "eval_replan_prefix_only": args.eval_replan_prefix_only,
        "eval_chunk": int(min(args.action_chunk, args.replan_steps) if args.eval_replan_prefix_only else args.action_chunk),
        "mot_checkpoint_mixed_attn": args.mot_checkpoint_mixed_attn,
        "seed": args.seed,
        "selected_datasets": [{"task": task, "data_path": ld} for task, ld in lerobot_dirs],
        "action_dim_names": results[0].get("action_dim_names") if results else None,
        "num_tasks": len(results),
        "total_frames": total_frames,
        "mean_mse": mean_mse,
        "mean_mae": mean_mae,
        "frame_weighted_mse": wmean_mse,
        "frame_weighted_mae": wmean_mae,
        "per_task": results,
    }
    summary_path = os.path.join(args.output_dir, f"summary_ep{args.episode_idx:03d}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MoT-only open-loop evaluation")
    p.add_argument("--model_id", type=str,
                   default="/shared_disk/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers")
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--stats_path", type=str,
                   default="/shared_disk/users/hengtao.li/robocasa_datasets/v1.0/pretrain_gwp/norm_stats_delta.json")
    p.add_argument("--data_root", type=str,
                   default="/shared_disk/users/hengtao.li/robocasa_datasets/v1.0/pretrain_gwp")
    p.add_argument("--t5_embedding_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output dir (auto-derived from checkpoint if not set)")
    p.add_argument("--action_dim", type=int, default=12)
    p.add_argument("--state_dim", type=int, default=16)
    p.add_argument("--num_frames", type=int, default=24,
                   help="Number of action tokens generated by the MoT model")
    p.add_argument("--action_chunk", type=int, default=24)
    p.add_argument("--num_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for action sampling noise")
    p.add_argument("--action_flow_shift", type=float, default=5.0,
                   help="Flow shift for action sampling (default 5.0, matching training)")
    p.add_argument("--replan_steps", type=int, default=24, help="Steps between re-predictions")
    p.add_argument("--dst_size", type=int, nargs=2, default=[320, 256])
    p.add_argument("--episode_idx", type=int, default=0, help="Which episode to evaluate")
    p.add_argument("--max_datasets", type=int, default=5,
                   help="Max number of datasets to evaluate (0=all)")
    p.add_argument("--task_set", choices=["all", "atomic_seen"], default="all",
                   help="Dataset task subset. atomic_seen matches the training configs.")
    p.add_argument("--task_names", nargs="*", default=None,
                   help="Optional explicit task names to evaluate after task_set filtering.")
    p.add_argument("--one_per_task", action="store_true", default=False,
                   help="Keep only the first sorted LeRobot dataset/date for each task.")
    p.add_argument("--comparison_space", choices=["raw", "raw_from_delta", "window_delta"], default="raw",
                   help="Metric space: legacy raw, denormalized delta + state vs raw GT, or training window-delta.")
    p.add_argument("--gt_delta_mode", choices=["raw", "agilex_cobot_magic"], default="raw",
                   help="Delta mask used by raw_from_delta/window_delta comparisons.")
    p.add_argument("--eval_replan_prefix_only", action="store_true", default=False,
                   help="Evaluate only the prefix kept by deployment: min(action_chunk, replan_steps).")
    p.add_argument("--skip_action_denorm", action="store_true", default=False,
                   help="Skip action denormalization (use when trained with skip_action_norm=True)")
    p.add_argument("--input_view_mode", choices=["auto", "tshape", "raw"], default="auto",
                   help="Video input mode: auto uses observation.images.tshape when present, otherwise raw camera views")
    p.add_argument("--include_zero_std_dims", action="store_true", default=False,
                   help="Include zero-std action dims in primary metrics/plot instead of matching the training mask")
    p.add_argument("--zero_std_threshold", type=float, default=1e-4,
                   help="Action std threshold used to ignore zero-std dims in primary metrics/plot")
    p.add_argument("--skip_all_dims_plot", action="store_true", default=False,
                   help="Skip the diagnostic all-dim plot")
    p.add_argument("--tshape", action="store_true", default=True,
                   help="Compatibility flag; MoT open-loop always uses T-shape layout")
    p.add_argument("--tshape_head_index", type=int, default=2,
                   help="Which view (index into view_keys) is the head (full size). Default 2 = agentview_right")
    p.add_argument("--action_expert_hidden_dim", type=int, default=1024)
    p.add_argument("--action_expert_ffn_dim", type=int, default=4096)
    p.add_argument("--mot_checkpoint_mixed_attn", action=argparse.BooleanOptionalAction, default=True,
                   help="Build the MoT wrapper with the same mixed-attn mode as training/deployment configs.")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()
    if args.action_chunk > args.num_frames:
        raise ValueError(f"action_chunk ({args.action_chunk}) must be <= num_frames ({args.num_frames})")

    # Auto-derive output_dir from checkpoint path
    if args.output_dir is None:
        ckpt_dir = os.path.dirname(args.checkpoint_path)
        exp_name = os.path.basename(os.path.dirname(ckpt_dir))
        ckpt_name = os.path.basename(ckpt_dir)
        model_name = os.path.basename(args.checkpoint_path).replace(".pt", "")
        args.output_dir = os.path.join(
            "/shared_disk/users/hengtao.li/codex/gwp-mot/openloop",
            exp_name, ckpt_name, model_name)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("MoT Open-Loop Evaluation")
    print("=" * 60)
    print(f"  Checkpoint:   {args.checkpoint_path}")
    print(f"  Data root:    {args.data_root}")
    print(f"  Output:       {args.output_dir}")
    print(f"  Episode:      {args.episode_idx}")
    print(f"  Num frames:   {args.num_frames}")
    print(f"  Action chunk: {args.action_chunk}")
    print(f"  Replan steps: {args.replan_steps}")
    print(f"  Eval prefix:  {args.eval_replan_prefix_only}")
    print(f"  Seed:         {args.seed}")
    print(f"  Max datasets: {args.max_datasets}")
    print(f"  Task set:     {args.task_set}")
    print(f"  One per task: {args.one_per_task}")
    print(f"  Compare:      {args.comparison_space}")
    print(f"  GT mode:      {args.gt_delta_mode}")
    print(f"  View mode:    {args.input_view_mode}")
    print(f"  Layout:       T-shape (head_index={args.tshape_head_index})")
    print(f"  MoT mixed:    {args.mot_checkpoint_mixed_attn}")
    print("=" * 60)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    md = build_model(
        args.model_id,
        args.checkpoint_path,
        args.action_dim,
        args.state_dim,
        device="cuda",
        dtype=torch.bfloat16,
        action_expert_hidden_dim=args.action_expert_hidden_dim,
        action_expert_ffn_dim=args.action_expert_ffn_dim,
        mot_checkpoint_mixed_attn=args.mot_checkpoint_mixed_attn,
    )
    md["action_flow_shift"] = args.action_flow_shift
    norm = load_norm_stats(args.stats_path, args.state_dim, args.action_dim, "cuda")
    active_dims_preview, ignored_dims_preview = active_action_dims(
        norm,
        args.action_dim,
        include_zero_std_dims=args.include_zero_std_dims,
        threshold=args.zero_std_threshold,
    )
    print(f"  Metric dims:  {active_dims_preview} (ignored zero-std: {ignored_dims_preview})")

    if args.t5_embedding_path and os.path.exists(args.t5_embedding_path):
        t5 = torch.load(args.t5_embedding_path, map_location="cpu")
        if not isinstance(t5, torch.Tensor):
            t5 = torch.as_tensor(t5)
        t5 = t5[:64]
        if t5.shape[0] < 64:
            t5 = torch.nn.functional.pad(t5, (0, 0, 0, 64 - t5.shape[0]))
    else:
        t5 = torch.zeros(64, 4096, dtype=torch.float32)

    run_openloop(args, md, norm, t5)


if __name__ == "__main__":
    main()
