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

# -- Model loading ---------------------------------------------------------

def build_model(pretrained_path, checkpoint_path, action_dim=12, state_dim=16,
                flow_shift=5.0, device="cuda", dtype=torch.bfloat16,
                action_expert_hidden_dim=1024, action_expert_ffn_dim=4096):
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
        mot_checkpoint_mixed_attn=False,
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

@torch.no_grad()
def run_openloop(args, md, norm, t5_emb):
    device, dtype = md["device"], md["dtype"]
    vae = md["vae"]
    latents_mean, latents_std = md["latents_mean"], md["latents_std"]
    action_chunk = args.action_chunk
    num_frames = args.num_frames
    action_dim = args.action_dim

    view_keys = [
        "observation.images.robot0_agentview_left",
        "observation.images.robot0_eye_in_hand",
        "observation.images.robot0_agentview_right",
    ]

    # Find lerobot dirs. Supports two layouts:
    #   1) pretrain_gwp: <data_root>/{atomic,composite}/<task>/<date_dir>/lerobot
    #   2) collected:    <data_root>/<task>/lerobot                (flat)
    lerobot_dirs = []
    nested_found = False
    for cat in ("atomic", "composite"):
        cat_dir = os.path.join(args.data_root, cat)
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
        for task in sorted(os.listdir(args.data_root)):
            task_dir = os.path.join(args.data_root, task)
            if not os.path.isdir(task_dir):
                continue
            ld = os.path.join(task_dir, "lerobot")
            if os.path.isdir(ld):
                lerobot_dirs.append((task, ld))

    if args.max_datasets > 0:
        lerobot_dirs = lerobot_dirs[:args.max_datasets]

    print(f"Found {len(lerobot_dirs)} datasets, evaluating episode {args.episode_idx} from each")

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
        gt_actions = ep["actions"]  # T, action_dim
        print(f"  Episode length: {T} frames, action_dim: {gt_actions.shape[-1]}")

        # Sliding window: step through episode, generate num_frames actions,
        # then accumulate the first action_chunk predictions.
        all_pred = np.zeros_like(gt_actions)  # T, action_dim
        pred_counts = np.zeros(T)

        step_interval = args.replan_steps
        for start in range(0, T, step_interval):
            # Build the MoT T-shape observation: head view at full size, wrist
            # views half-sized and concatenated below it.
            dst_w, dst_h = tuple(args.dst_size)
            half_w, half_h = dst_w // 2, dst_h // 2
            head_idx = args.tshape_head_index
            ts_images = []
            for vi, vk in enumerate(view_keys):
                if vk not in ep["frames"] or start >= len(ep["frames"][vk]):
                    continue
                raw = ep["frames"][vk][start]
                size = (dst_w, dst_h) if vi == head_idx else (half_w, half_h)
                ts_images.append((vi, preprocess_image(raw, size)))

            head = next((img for i, img in ts_images if i == head_idx), None)
            others = [img for i, img in ts_images if i != head_idx]
            if head is None or not others:
                print("  Skipping frame: missing T-shape views")
                continue

            wrist_row = torch.cat(others, dim=-1)
            if wrist_row.shape[-1] < head.shape[-1]:
                wrist_row = torch.nn.functional.pad(
                    wrist_row, (0, head.shape[-1] - wrist_row.shape[-1]))
            elif wrist_row.shape[-1] > head.shape[-1]:
                wrist_row = wrist_row[..., :head.shape[-1]]
            ref_image = torch.cat([head, wrist_row], dim=-2).to(device, dtype=dtype)
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

            # Accumulate predictions (average overlapping chunks)
            end = min(start + action_chunk, T)
            chunk_len = end - start
            all_pred[start:end] += pred[:chunk_len]
            pred_counts[start:end] += 1

        # Average overlapping predictions
        mask = pred_counts > 0
        all_pred[mask] /= pred_counts[mask, None]

        # Truncate to actual action dim
        actual_dim = min(action_dim, gt_actions.shape[-1])
        gt = gt_actions[:, :actual_dim]
        pr = all_pred[:, :actual_dim]

        # Compute metrics
        mse = float(np.mean((gt - pr) ** 2))
        mae = float(np.mean(np.abs(gt - pr)))
        print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")
        results.append(dict(task=task_name, num_frames=int(T), mse=mse, mae=mae))

        # Plot
        save_dir = os.path.join(args.output_dir, task_name)
        os.makedirs(save_dir, exist_ok=True)

        cols = 4
        rows = (actual_dim + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = np.array(axes).flatten()

        for d in range(actual_dim):
            ax = axes[d]
            ax.plot(range(T), gt[:, d], label="gt", color="tab:blue", linewidth=1.2)
            ax.plot(range(T), pr[:, d], label="pred", color="tab:orange", linewidth=1.2, alpha=0.8)
            ax.set_title(f"Action Dimension {d}", fontsize=10)
            ax.set_xlabel("Frame", fontsize=8)
            ax.set_ylabel("Value", fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

        # Hide unused subplots
        for d in range(actual_dim, len(axes)):
            axes[d].set_visible(False)

        fig.suptitle(f"{task_name}  (MSE={mse:.4f}, MAE={mae:.4f})", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"ep{args.episode_idx:03d}_actions.png"), dpi=150)
        plt.close(fig)

        # Save metrics
        with open(os.path.join(save_dir, f"ep{args.episode_idx:03d}_metrics.json"), "w") as f:
            json.dump({"task": task_name, "episode": args.episode_idx,
                       "num_frames": T, "mse": float(mse), "mae": float(mae)}, f, indent=2)

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
    p.add_argument("--action_flow_shift", type=float, default=5.0,
                   help="Flow shift for action sampling (default 5.0, matching training)")
    p.add_argument("--replan_steps", type=int, default=24, help="Steps between re-predictions")
    p.add_argument("--dst_size", type=int, nargs=2, default=[320, 256])
    p.add_argument("--episode_idx", type=int, default=0, help="Which episode to evaluate")
    p.add_argument("--max_datasets", type=int, default=5,
                   help="Max number of datasets to evaluate (0=all)")
    p.add_argument("--skip_action_denorm", action="store_true", default=False,
                   help="Skip action denormalization (use when trained with skip_action_norm=True)")
    p.add_argument("--tshape", action="store_true", default=True,
                   help="Compatibility flag; MoT open-loop always uses T-shape layout")
    p.add_argument("--tshape_head_index", type=int, default=2,
                   help="Which view (index into view_keys) is the head (full size). Default 2 = agentview_right")
    p.add_argument("--action_expert_hidden_dim", type=int, default=1024)
    p.add_argument("--action_expert_ffn_dim", type=int, default=4096)
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
    print(f"  Max datasets: {args.max_datasets}")
    print(f"  Layout:       T-shape (head_index={args.tshape_head_index})")
    print("=" * 60)

    md = build_model(
        args.model_id,
        args.checkpoint_path,
        args.action_dim,
        args.state_dim,
        device="cuda",
        dtype=torch.bfloat16,
        action_expert_hidden_dim=args.action_expert_hidden_dim,
        action_expert_ffn_dim=args.action_expert_ffn_dim,
    )
    md["action_flow_shift"] = args.action_flow_shift
    norm = load_norm_stats(args.stats_path, args.state_dim, args.action_dim, "cuda")

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
