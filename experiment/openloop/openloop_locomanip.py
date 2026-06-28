"""MoT open-loop evaluation for the locomanip pick_place (G1 sonic) task.

Unlike the RoboCasa open-loop (see ``openloop_eval.py``), this task:

  * predicts a 66-d sonic latent action (motion_token[64] + hand_binary[2]);
  * feeds a 2-token state sequence to the action expert
    (token0 = 43-d joint state z-score normalized, token1 = 66-d sonic latent
    kept raw), matching ``configs/data/pick_place_g1_sonic.yaml``;
  * uses a single ``observation.images.ego_view`` camera (no T-shape);
  * trains with ``skip_action_norm=True`` so the action is compared raw
    (no denormalization).

It loads a checkpoint, feeds dataset frames/states to the model, samples actions
with the action-only flow-matching path, and compares against the ground-truth
sonic latent with per-dimension plots and MSE/MAE (overall + motion/hand groups).

This mirrors the preprocessing in ``WATransformsLerobot`` for the
``g1_sonic`` robotype (delta template 3: 66-d, no delta, all dims supervised).
"""

import argparse
import json
import logging
import os
import sys

# Resolve ``world_action_model`` to this repo's ``src`` (matching
# scripts/train_hydra.py) instead of any older editable install on the
# environment, so the multi-state-token ActionStateDiT is available.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, _PROJECT_ROOT)
os.environ.setdefault("HF_HOME", "/shared_disk/models/huggingface")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

# Reuse the shared image preprocessing / plotting helpers from the RoboCasa
# open-loop module (same directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openloop_eval import preprocess_image, action_stats, load_t5_embedding  # noqa: E402

logger = logging.getLogger(__name__)

EGO_VIEW_KEY = "observation.images.ego_view"
JOINT_STATE_KEY = "observation.state"
MOTION_LATENT_KEY = "observation.motion_latent"
ACTION_KEY = "action"

MOTION_TOKEN_DIM = 64  # action[:64]
HAND_BINARY_DIM = 2    # action[64:66]


# ── Model loading ─────────────────────────────────────────────────────────

def build_model(pretrained_path, checkpoint_path, action_dim=66, state_dim=66,
                state_token_dims=(43, 66), flow_shift=5.0, device="cuda",
                dtype=torch.bfloat16, action_expert_hidden_dim=1024,
                action_expert_ffn_dim=4096, mot_checkpoint_mixed_attn=True):
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
            "state_token_dims": [int(x) for x in state_token_dims],
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
    if unexpected:
        print(f"  First unexpected keys: {list(unexpected)[:5]}")
    transformer.eval()

    return dict(vae=vae, transformer=transformer, latents_mean=latents_mean,
                latents_std=latents_std, flow_shift=flow_shift,
                action_flow_shift=flow_shift, action_dim=action_dim,
                state_dim=state_dim, device=device, dtype=dtype)


# ── Normalization (mirrors WATransformsLerobot for g1_sonic) ───────────────

def load_norm_stats(stats_path):
    with open(stats_path) as f:
        return json.load(f)["norm_stats"]


def _padded_1d(arr, target_dim, pad_value):
    t = np.asarray(arr, dtype=np.float32).flatten()
    if t.shape[0] >= target_dim:
        return t[:target_dim]
    out = np.full((target_dim,), float(pad_value), dtype=np.float32)
    out[: t.shape[0]] = t
    return out


def _pad_truncate_last(x, target_dim):
    if x.shape[-1] > target_dim:
        return x[..., :target_dim]
    if x.shape[-1] < target_dim:
        return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, target_dim - x.shape[-1])])
    return x


def normalize_state_token(token, stats, norm_key, token_dim, state_dim):
    """Replicate WATransformsLerobot._normalize_state_token + pad to state_dim."""
    token = _pad_truncate_last(np.asarray(token, dtype=np.float32), token_dim)
    field = stats.get(norm_key, None)
    if field is None:
        mean = np.zeros(token_dim, dtype=np.float32)
        std = np.ones(token_dim, dtype=np.float32)
    else:
        mean = _padded_1d(field.get("mean", []), token_dim, 0.0)
        std = _padded_1d(field.get("std", []), token_dim, 1.0)
    zero_mask = std < 1e-4
    norm = (token - mean) / np.clip(std, 1e-8, None)
    norm[..., zero_mask] = 0.0
    return _pad_truncate_last(norm, state_dim)


def build_state(joint_row, latent_row, stats, state_token_dims, state_dim):
    """Build the 2-token state [joint, latent] -> (num_tokens, state_dim)."""
    joint_tok = normalize_state_token(joint_row, stats, JOINT_STATE_KEY,
                                      state_token_dims[0], state_dim)
    latent_tok = normalize_state_token(latent_row, stats, MOTION_LATENT_KEY,
                                       state_token_dims[1], state_dim)
    return np.stack([joint_tok, latent_tok], axis=0)


# ── Flow-matching action sampler (action-only path) ────────────────────────

@torch.no_grad()
def sample_action(md, ref_latents, prompt_embeds, state, state_mask,
                  num_steps=10, num_frames=56):
    device, dtype = md["device"], md["dtype"]
    transformer = md["transformer"]
    if hasattr(transformer, "clear_action_only_cache"):
        transformer.clear_action_only_cache()
    flow_shift = md.get("action_flow_shift", md["flow_shift"])
    ad = md["action_dim"]
    bs = ref_latents.shape[0]

    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    scheduler.set_timesteps(num_steps, device=device)

    noisy_action = torch.randn(bs, num_frames, ad, device=device, dtype=dtype)
    for t in scheduler.timesteps:
        # 1-D timestep: the model assigns 0 to state/ref tokens and ``t`` to the
        # action tokens (matches training's action_noise_t = sigma * 1000).
        timestep = t.float().to(device).expand(bs)
        action_pred = transformer(
            ref_latents=ref_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            action=noisy_action,
            state=state,
            state_mask=state_mask,
            action_only=True,
        )
        noisy_action = scheduler.step(action_pred, t, noisy_action, return_dict=False)[0]

    return noisy_action


# ── Dataset helpers ────────────────────────────────────────────────────────

def load_episode_data(lerobot_dir, episode_idx):
    import pandas as pd

    parquet = os.path.join(lerobot_dir, "data", "chunk-000",
                           f"episode_{episode_idx:06d}.parquet")
    df = pd.read_parquet(parquet)

    def _stack(col):
        return np.stack([df[col].iloc[i] for i in range(len(df))]).astype(np.float32)

    actions = _stack(ACTION_KEY)
    joints = _stack(JOINT_STATE_KEY)
    latents = _stack(MOTION_LATENT_KEY) if MOTION_LATENT_KEY in df.columns else actions
    task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns else 0

    video_path = os.path.join(lerobot_dir, "videos", "chunk-000", EGO_VIEW_KEY,
                              f"episode_{episode_idx:06d}.mp4")
    frames = None
    if os.path.exists(video_path):
        import imageio
        reader = imageio.get_reader(video_path)
        frames = [np.array(f) for f in reader]
        reader.close()

    return dict(actions=actions, joints=joints, latents=latents, frames=frames,
                num_frames=len(df), task_index=task_index)


def discover_episodes(lerobot_dir):
    data_dir = os.path.join(lerobot_dir, "data", "chunk-000")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    eps = []
    for fn in sorted(os.listdir(data_dir)):
        if fn.startswith("episode_") and fn.endswith(".parquet"):
            eps.append(int(fn[len("episode_"):-len(".parquet")]))
    return eps


# ── Plotting ───────────────────────────────────────────────────────────────

def action_dim_names(action_dim):
    names = [f"motion_{i}" for i in range(min(MOTION_TOKEN_DIM, action_dim))]
    for i in range(max(0, action_dim - MOTION_TOKEN_DIM)):
        names.append(f"hand_{i}")
    return names[:action_dim]


def plot_action_comparison(save_path, task_name, gt, pr, dims, mse, mae,
                           title_suffix="", names=None):
    if len(dims) == 0:
        return
    T = gt.shape[0]
    cols = 4
    rows = (len(dims) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.6))
    axes = np.array(axes).flatten()
    for ax_idx, d in enumerate(dims):
        ax = axes[ax_idx]
        label = names[d] if names is not None and d < len(names) else f"dim_{d}"
        ax.plot(range(T), gt[:, d], label="gt", color="tab:blue", linewidth=1.0)
        ax.plot(range(T), pr[:, d], label="pred", color="tab:orange", linewidth=1.0, alpha=0.8)
        ax.set_title(f"{d}: {label}", fontsize=8)
        ax.tick_params(labelsize=6)
        if ax_idx == 0:
            ax.legend(fontsize=6)
    for ax_idx in range(len(dims), len(axes)):
        axes[ax_idx].set_visible(False)
    fig.suptitle(f"{task_name}{title_suffix}  (MSE={mse:.4f}, MAE={mae:.4f})", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def group_metrics(gt, pr, action_dim):
    motion_dims = list(range(0, min(MOTION_TOKEN_DIM, action_dim)))
    hand_dims = list(range(MOTION_TOKEN_DIM, action_dim))

    def _mm(dims):
        if not dims:
            return None, None
        d = np.asarray(dims)
        diff = gt[:, d] - pr[:, d]
        return float(np.mean(diff ** 2)), float(np.mean(np.abs(diff)))

    motion_mse, motion_mae = _mm(motion_dims)
    hand_mse, hand_mae = _mm(hand_dims)
    return {
        "motion_dims": motion_dims,
        "hand_dims": hand_dims,
        "motion_mse": motion_mse,
        "motion_mae": motion_mae,
        "hand_mse": hand_mse,
        "hand_mae": hand_mae,
    }


# ── Open-loop evaluation ────────────────────────────────────────────────────

@torch.no_grad()
def run_openloop(args, md, stats):
    device, dtype = md["device"], md["dtype"]
    fallback_t5 = torch.zeros(1, 64, 4096, dtype=dtype, device=device)
    vae = md["vae"]
    latents_mean, latents_std = md["latents_mean"], md["latents_std"]
    action_dim = args.action_dim
    num_frames = args.num_frames
    step_interval = args.replan_steps
    names = action_dim_names(action_dim)
    state_token_dims = [int(args.joint_state_dim), int(args.latent_state_dim)]

    all_eps = discover_episodes(args.data_root)
    if args.episode_indices:
        episodes = [e for e in args.episode_indices if e in all_eps]
        missing = [e for e in args.episode_indices if e not in all_eps]
        if missing:
            print(f"  Warning: requested episodes not found and skipped: {missing}")
    else:
        episodes = all_eps[: args.max_episodes] if args.max_episodes > 0 else all_eps

    print(f"Found {len(all_eps)} episodes under {args.data_root}; evaluating {len(episodes)}: {episodes}")

    results = []
    for episode_idx in episodes:
        task_name = f"ep{episode_idx:06d}"
        print(f"\n{'='*60}\nEpisode: {episode_idx}")
        ep = load_episode_data(args.data_root, episode_idx)
        if ep["frames"] is None:
            raise RuntimeError(f"Missing ego_view video for episode {episode_idx}")

        T = ep["num_frames"]
        raw_gt = ep["actions"]
        joints = ep["joints"]
        latents = ep["latents"]
        n_video = len(ep["frames"])
        print(f"  Episode length: {T} frames (video {n_video}), action_dim {raw_gt.shape[-1]}")

        # Load the real T5 language embedding the model was trained with
        # (meta/t5_text_embeds.pt keyed by task_index); fall back to zeros.
        prompt_embeds = load_t5_embedding(args.data_root, episode_idx, device, dtype,
                                          task_index=ep.get("task_index", 0))
        if prompt_embeds is None:
            print("  No T5 embedding found, using zeros fallback")
            prompt_embeds = fallback_t5
        else:
            print(f"  Loaded T5 embedding: {tuple(prompt_embeds.shape)}")

        all_pred = np.zeros((T, action_dim), dtype=np.float32)
        all_gt = np.zeros((T, action_dim), dtype=np.float32)
        pred_counts = np.zeros(T, dtype=np.float32)
        # Per-horizon-offset error: offset k = action predicted k steps after the
        # ref frame. Lets us see whether the immediate action is accurate (good
        # conditioning) vs error growing with the open-loop horizon.
        off_err_sum = np.zeros(num_frames, dtype=np.float64)
        off_cnt = np.zeros(num_frames, dtype=np.float64)

        dst_w, dst_h = tuple(args.dst_size)
        for start in range(0, T, step_interval):
            frame_idx = min(start, n_video - 1)
            ref_image = preprocess_image(ep["frames"][frame_idx], (dst_w, dst_h)).to(device, dtype=dtype)
            ref_image_5d = ref_image.unsqueeze(2)
            ref_latents = vae.encode(ref_image_5d).latent_dist.mode()
            ref_latents = (ref_latents - latents_mean) * latents_std

            state_np = build_state(joints[start], latents[start], stats,
                                   state_token_dims, args.state_dim)
            state = torch.from_numpy(state_np).to(device, dtype=dtype).unsqueeze(0)
            state_mask = torch.ones((1, state_np.shape[0]), dtype=torch.bool, device=device)

            # A single diffusion sample scatters around the conditional mean, so
            # --num_samples>1 averages several samples to estimate that mean
            # (the fair accuracy lens for a stochastic policy in open-loop).
            sample_acc = None
            for s_i in range(args.num_samples):
                if args.num_samples > 1:
                    torch.manual_seed(args.seed + 1000 * s_i + start)
                one = sample_action(md, ref_latents, prompt_embeds, state, state_mask,
                                    num_steps=args.num_steps, num_frames=num_frames)
                one = one.float().squeeze(0)
                sample_acc = one if sample_acc is None else sample_acc + one
            pred = (sample_acc / args.num_samples).cpu().numpy()  # skip_action_norm -> raw

            end = min(start + num_frames, T)
            chunk_len = end - start
            all_pred[start:end] += pred[:chunk_len]
            all_gt[start:end] += raw_gt[start:end, :action_dim]
            pred_counts[start:end] += 1

            chunk_err = np.abs(pred[:chunk_len] - raw_gt[start:end, :action_dim]).mean(axis=1)
            off_err_sum[:chunk_len] += chunk_err
            off_cnt[:chunk_len] += 1

        mask = pred_counts > 0
        covered = int(mask.sum())
        if covered == 0:
            raise RuntimeError(f"No predictions generated for episode {episode_idx}")
        all_pred[mask] /= pred_counts[mask, None]
        all_gt[mask] /= pred_counts[mask, None]

        gt = all_gt
        pr = all_pred
        gt_m, pr_m = gt[mask], pr[mask]
        mse = float(np.mean((gt_m - pr_m) ** 2))
        mae = float(np.mean(np.abs(gt_m - pr_m)))
        grp = group_metrics(gt_m, pr_m, action_dim)
        gt_stats = action_stats(gt_m)
        pred_stats = action_stats(pr_m)

        # Trivial baselines (computed on covered frames) to contextualize MAE.
        zeros_mae = float(np.mean(np.abs(gt_m)))
        mean_mae = float(np.mean(np.abs(gt_m - gt_m.mean(axis=0, keepdims=True))))
        hold0_mae = float(np.mean(np.abs(gt_m - gt_m[0:1])))
        # Per-offset MAE (averaged over all dims and windows).
        valid_off = off_cnt > 0
        offset_mae = (off_err_sum[valid_off] / off_cnt[valid_off]).tolist()

        print(f"  Covered {covered}/{T} frames")
        print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")
        print(f"  motion[0:{MOTION_TOKEN_DIM}] MSE={grp['motion_mse']}, MAE={grp['motion_mae']}")
        print(f"  hand[{MOTION_TOKEN_DIM}:{action_dim}] MSE={grp['hand_mse']}, MAE={grp['hand_mae']}")
        print(f"  Baselines MAE -> zeros: {zeros_mae:.4f}, const-mean: {mean_mae:.4f}, hold-first: {hold0_mae:.4f}")
        if offset_mae:
            preview_idx = [i for i in (0, 1, 2, 4, 8, 16, 24, 32, 48, len(offset_mae) - 1) if i < len(offset_mae)]
            preview = ", ".join(f"{i}:{offset_mae[i]:.3f}" for i in sorted(set(preview_idx)))
            print(f"  Per-offset MAE (step-from-ref): {preview}")
        print(f"  GT abs_sum {gt_stats['abs_sum']:.4f}, Pred abs_sum {pred_stats['abs_sum']:.4f}")

        save_dir = os.path.join(args.output_dir, task_name)
        os.makedirs(save_dir, exist_ok=True)
        if not args.skip_motion_plot:
            plot_action_comparison(
                os.path.join(save_dir, f"{task_name}_motion.png"), task_name,
                gt, pr, list(range(min(MOTION_TOKEN_DIM, action_dim))),
                grp["motion_mse"] or 0.0, grp["motion_mae"] or 0.0,
                title_suffix=" motion tokens", names=names)
        plot_action_comparison(
            os.path.join(save_dir, f"{task_name}_hand.png"), task_name,
            gt, pr, list(range(MOTION_TOKEN_DIM, action_dim)),
            grp["hand_mse"] or 0.0, grp["hand_mae"] or 0.0,
            title_suffix=" hand binary", names=names)
        np.savez_compressed(
            os.path.join(save_dir, f"{task_name}_arrays.npz"),
            gt=gt, pred=pr, pred_counts=pred_counts,
            action_dim_names=np.asarray(names, dtype="U32"))

        metrics = {
            "episode": episode_idx,
            "num_frames": int(T),
            "covered_frames": covered,
            "action_dim": action_dim,
            "num_frames_per_chunk": num_frames,
            "replan_steps": step_interval,
            "mse": mse,
            "mae": mae,
            "baseline_zeros_mae": zeros_mae,
            "baseline_const_mean_mae": mean_mae,
            "baseline_hold_first_mae": hold0_mae,
            "offset_mae": offset_mae,
            "gt_stats": gt_stats,
            "pred_stats": pred_stats,
            **{k: v for k, v in grp.items() if k not in ("motion_dims", "hand_dims")},
        }
        with open(os.path.join(save_dir, f"{task_name}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved to {save_dir}")
        results.append(dict(episode=episode_idx, num_frames=int(T), mse=mse, mae=mae,
                            motion_mae=grp["motion_mae"], hand_mae=grp["hand_mae"]))

    if not results:
        print("No episodes evaluated.")
        return

    print(f"\n{'='*64}")
    print(f"  Locomanip Open-Loop Summary (episodes={len(results)})")
    print(f"{'='*64}")
    header = f"  {'#':>3}  {'Episode':>8}  {'Frames':>7}  {'MSE':>11}  {'MAE':>11}  {'motionMAE':>11}  {'handMAE':>11}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, r in enumerate(results):
        print(f"  {i+1:>3}  {r['episode']:>8d}  {r['num_frames']:>7d}  {r['mse']:>11.6f}"
              f"  {r['mae']:>11.6f}  {(r['motion_mae'] or 0.0):>11.6f}  {(r['hand_mae'] or 0.0):>11.6f}")
    print("  " + "-" * (len(header) - 2))
    mean_mse = float(np.mean([r["mse"] for r in results]))
    mean_mae = float(np.mean([r["mae"] for r in results]))
    print(f"  {'':>3}  {'mean':>8}  {'':>7}  {mean_mse:>11.6f}  {mean_mae:>11.6f}")

    summary = {
        "checkpoint": args.checkpoint_path,
        "data_root": args.data_root,
        "episodes": [r["episode"] for r in results],
        "num_frames_per_chunk": num_frames,
        "replan_steps": step_interval,
        "num_steps": args.num_steps,
        "num_samples": args.num_samples,
        "action_flow_shift": args.action_flow_shift,
        "mean_mse": mean_mse,
        "mean_mae": mean_mae,
        "per_episode": results,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MoT open-loop eval for locomanip pick_place (G1 sonic)")
    p.add_argument("--model_id", type=str,
                   default=os.environ.get("WAN22_DIFFUSERS_PATH",
                                          "/shared_disk/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers"))
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--data_root", type=str,
                   default="/shared_disk/users/hengtao.li/locomanip/data/pick_place_gwp",
                   help="LeRobot dataset root containing data/, videos/, norm_stats_delta.json")
    p.add_argument("--stats_path", type=str, default=None,
                   help="Defaults to <data_root>/norm_stats_delta.json")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--action_dim", type=int, default=66)
    p.add_argument("--state_dim", type=int, default=66)
    p.add_argument("--joint_state_dim", type=int, default=43)
    p.add_argument("--latent_state_dim", type=int, default=66)
    p.add_argument("--num_frames", type=int, default=56,
                   help="Number of action tokens generated per chunk (training num_frames)")
    p.add_argument("--num_steps", type=int, default=10)
    p.add_argument("--num_samples", type=int, default=1,
                   help="Average this many diffusion samples per window to estimate the "
                        "conditional mean (reduces single-sample variance; try 8).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--action_flow_shift", type=float, default=5.0)
    p.add_argument("--replan_steps", type=int, default=56, help="Frames between re-predictions")
    p.add_argument("--dst_size", type=int, nargs=2, default=[320, 256])
    p.add_argument("--episode_indices", type=int, nargs="*", default=None,
                   help="Explicit episode indices to evaluate (overrides --max_episodes)")
    p.add_argument("--max_episodes", type=int, default=3, help="Max episodes to evaluate (0=all)")
    p.add_argument("--action_expert_hidden_dim", type=int, default=1024)
    p.add_argument("--action_expert_ffn_dim", type=int, default=4096)
    p.add_argument("--mot_checkpoint_mixed_attn", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip_motion_plot", action="store_true", default=False,
                   help="Skip the 64-dim motion-token plot (keep only the hand plot)")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    if args.stats_path is None:
        args.stats_path = os.path.join(args.data_root, "norm_stats_delta.json")

    if args.output_dir is None:
        ckpt_dir = os.path.dirname(args.checkpoint_path)
        exp_name = os.path.basename(os.path.dirname(ckpt_dir)) or "exp"
        ckpt_name = os.path.basename(ckpt_dir) or "ckpt"
        model_name = os.path.basename(args.checkpoint_path).replace(".pt", "")
        args.output_dir = os.path.join(
            "/shared_disk/users/hengtao.li/locomanip/exp/openloop",
            exp_name, ckpt_name, model_name)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("MoT Open-Loop Evaluation [locomanip pick_place G1 sonic]")
    print("=" * 60)
    print(f"  Checkpoint:   {args.checkpoint_path}")
    print(f"  Base model:   {args.model_id}")
    print(f"  Data root:    {args.data_root}")
    print(f"  Stats path:   {args.stats_path}")
    print(f"  Output:       {args.output_dir}")
    print(f"  Action dim:   {args.action_dim} (motion {MOTION_TOKEN_DIM} + hand {HAND_BINARY_DIM})")
    print(f"  State tokens: [{args.joint_state_dim} joint, {args.latent_state_dim} latent] -> dim {args.state_dim}")
    print(f"  Num frames:   {args.num_frames}")
    print(f"  Replan steps: {args.replan_steps}")
    print(f"  Num steps:    {args.num_steps}")
    print(f"  Num samples:  {args.num_samples} (averaged per window)")
    print(f"  Flow shift:   {args.action_flow_shift}")
    print(f"  Seed:         {args.seed}")
    print("=" * 60)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    md = build_model(
        args.model_id,
        args.checkpoint_path,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        state_token_dims=(args.joint_state_dim, args.latent_state_dim),
        device="cuda",
        dtype=torch.bfloat16,
        action_expert_hidden_dim=args.action_expert_hidden_dim,
        action_expert_ffn_dim=args.action_expert_ffn_dim,
        mot_checkpoint_mixed_attn=args.mot_checkpoint_mixed_attn,
    )
    md["action_flow_shift"] = args.action_flow_shift
    stats = load_norm_stats(args.stats_path)

    run_openloop(args, md, stats)


if __name__ == "__main__":
    main()
