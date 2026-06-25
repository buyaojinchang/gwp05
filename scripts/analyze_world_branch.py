#!/usr/bin/env python3
"""Forward-only diagnostics for the gwp-mot world branch.

The script intentionally mirrors CasualWATrainerPretrain.forward_step for
latent/noise/timestep construction, but never backpropagates or writes model
weights.  It is meant for a few-batch mechanism readout on an existing EMA
checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", "/shared_disk/models/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/shared_disk/models/huggingface")
os.environ.setdefault("WANDB_MODE", "disabled")

from world_action_model.models.mot import LayoutSegment  # noqa: E402
from world_action_model.runtime import load_config, resolve_runner  # noqa: E402
from world_action_model.trainer import DictConfig  # noqa: E402


def _to_device(value: Any, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(v, device) for v in value)
    return value


def _task_key_from_path(path: str) -> str:
    parts = Path(path).parts
    for category in ("atomic", "composite"):
        if category in parts:
            idx = parts.index(category)
            if idx + 1 < len(parts):
                return f"{category}/{parts[idx + 1]}"
    return Path(path).parent.parent.name


def _pick_task_configs(data_or_config: list[dict], max_tasks: int, seed: int) -> list[tuple[str, dict]]:
    by_task: dict[str, dict] = {}
    for item in data_or_config:
        path = item.get("data_path", "")
        task = _task_key_from_path(path)
        by_task.setdefault(task, item)
    tasks = sorted(by_task)
    rng = random.Random(seed)
    rng.shuffle(tasks)
    tasks = sorted(tasks[:max_tasks])
    return [(task, copy.deepcopy(by_task[task])) for task in tasks]


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = _mean(values)
    return float((sum((x - m) ** 2 for x in values) / (len(values) - 1)) ** 0.5)


def _metric_pair(base: torch.Tensor, other: torch.Tensor) -> dict[str, torch.Tensor]:
    base_f = base.float().reshape(base.shape[0], -1)
    other_f = other.float().reshape(other.shape[0], -1)
    diff = base_f - other_f
    rel_l2 = diff.norm(dim=1) / (base_f.norm(dim=1) + 1e-8)
    cosine = F.cosine_similarity(base_f, other_f, dim=1)
    return {"rel_l2": rel_l2.detach().cpu(), "cosine": cosine.detach().cpu()}


def _sigma_bucket(sigma: float) -> str:
    if sigma < 1.0 / 3.0:
        return "low[0,.33)"
    if sigma < 2.0 / 3.0:
        return "mid[.33,.66)"
    return "high[.66,1]"


def _append_values(store: dict[str, list[float]], prefix: str, values: dict[str, torch.Tensor]):
    for name, tensor in values.items():
        store[f"{prefix}_{name}"].extend(float(x) for x in tensor.flatten().tolist())


def _summarize_store(store: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {
        key: {"mean": _mean(vals), "std": _std(vals), "n": len(vals)}
        for key, vals in sorted(store.items())
    }


@torch.no_grad()
def _build_context(trainer, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    images = batch["images"]
    bs = images.shape[0]
    prompt_embeds = batch["prompt_embeds"].to(trainer.dtype)
    action = batch["action"]
    state = batch["state"]

    timestep, sigma = trainer.get_timestep_and_sigma(bs, images.ndim)
    _, action_sigma_5d = trainer.get_timestep_and_sigma(bs, images.ndim, flow_shift=trainer.action_flow_shift)
    action_sigma = action_sigma_5d.squeeze(-1).squeeze(-1)
    action_noise_t = torch.round(action_sigma[:, 0, 0].unsqueeze(-1) * 1000).to(
        dtype=sigma.dtype, device=sigma.device
    )

    if trainer.state_repeats > 1:
        state = state.repeat(1, trainer.state_repeats, 1)
    if trainer.action_repeats > 1:
        action = action.repeat(1, trainer.action_repeats, 1)

    visual_latents = trainer.forward_vae(images)
    visual_noise = torch.randn_like(visual_latents)
    noisy_latents_all = visual_noise * sigma + visual_latents * (1 - sigma)

    action_noise = torch.randn_like(action)
    noisy_action = action_noise * action_sigma + action * (1 - action_sigma)

    if not trainer.expand_timesteps:
        raise RuntimeError("This diagnostic currently expects expand_timesteps=True, matching the target cfg.")

    num_latent_frames = visual_latents.shape[2]
    latent_height = visual_latents.shape[-2]
    latent_width = visual_latents.shape[-1]
    ref_images = batch["ref_images"][:, :1]
    ref_latents_raw = trainer.forward_vae(ref_images)
    first_frame_mask = torch.ones(
        bs,
        1,
        num_latent_frames,
        latent_height,
        latent_width,
        dtype=visual_latents.dtype,
        device=visual_latents.device,
    )
    first_frame_mask[:, :, 0] = 0
    insert_noisy_latents = (1 - first_frame_mask) * ref_latents_raw + first_frame_mask * noisy_latents_all
    temp_ts = (first_frame_mask[:, :, :, ::2, ::2] * timestep[:, None, None, None, None]).reshape(bs, -1)
    noise_t = temp_ts[:, -2:-1]

    insert_noisy_latents = insert_noisy_latents.to(trainer.dtype)
    state = state.to(trainer.dtype)
    noisy_action = noisy_action.to(trainer.dtype)
    clean_action = action.to(trainer.dtype)
    ref_latents = insert_noisy_latents[:, :, :1]
    noisy_latents = insert_noisy_latents[:, :, 1:]

    frame_per_tokens = first_frame_mask.shape[-1] * first_frame_mask.shape[-2] // 4
    num_latent_tokens = frame_per_tokens * first_frame_mask.shape[2]
    num_clean_latent_tokens = frame_per_tokens
    num_state_tokens = state.shape[1]
    num_action_tokens = action.shape[1]
    action_start = num_state_tokens + num_clean_latent_tokens
    action_end = action_start + num_action_tokens

    def make_timestep(gt_action: bool) -> torch.Tensor:
        out = torch.zeros(
            bs,
            num_state_tokens + num_action_tokens + num_latent_tokens,
            device=noisy_latents.device,
            dtype=noisy_latents.dtype,
        )
        out[:, action_start:action_end] = 0 if gt_action else action_noise_t
        out[:, action_end:] = noise_t
        return out

    return {
        "prompt_embeds": prompt_embeds,
        "state": state,
        "clean_action": clean_action,
        "noisy_action": noisy_action,
        "action_sigma": action_sigma.to(trainer.dtype),
        "visual_sigma": sigma.flatten().to(torch.float32),
        "ref_latents": ref_latents,
        "noisy_latents": noisy_latents,
        "timestep_noisy_action": make_timestep(False),
        "timestep_gt_action": make_timestep(True),
    }


@torch.no_grad()
def _forward(transformer, ctx: dict[str, torch.Tensor], action: torch.Tensor, timestep: torch.Tensor, action_only: bool = False):
    if hasattr(transformer, "clear_action_only_cache") and action_only:
        transformer.clear_action_only_cache()
    return transformer(
        ref_latents=ctx["ref_latents"],
        noisy_latents=ctx["noisy_latents"],
        timestep=timestep,
        encoder_hidden_states=ctx["prompt_embeds"],
        return_dict=False,
        action=action,
        state=ctx["state"],
        action_only=action_only,
    )


@torch.no_grad()
def _attention_masses(transformer, ctx: dict[str, torch.Tensor], action: torch.Tensor, timestep: torch.Tensor) -> dict[str, Any]:
    hidden_states = torch.cat([ctx["ref_latents"], ctx["noisy_latents"]], dim=2)
    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = transformer.config.patch_size
    post_patch_height = height // p_h
    post_patch_width = width // p_w
    num_state = ctx["state"].shape[1]
    num_action = action.shape[1]
    num_ref = (ctx["ref_latents"].shape[2] // p_t) * post_patch_height * post_patch_width
    num_noisy = (num_frames // p_t) * post_patch_height * post_patch_width - num_ref

    state_ts, ref_ts, action_ts, noisy_ts = transformer._split_timesteps(
        timestep,
        batch_size=batch_size,
        num_state_tokens=num_state,
        num_ref_tokens=num_ref,
        num_action_tokens=num_action,
        num_noisy_tokens=num_noisy,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    video_pre = transformer._build_video_pre(
        ref_latents=ctx["ref_latents"],
        noisy_latents=ctx["noisy_latents"],
        video_timestep=torch.cat([ref_ts, noisy_ts], dim=1),
        encoder_hidden_states=ctx["prompt_embeds"],
    )
    action_pre = transformer.action_expert.pre_dit(
        state=ctx["state"].to(dtype=video_pre["tokens"].dtype),
        action=action.to(dtype=video_pre["tokens"].dtype),
        state_timestep=state_ts,
        action_timestep=action_ts,
        encoder_hidden_states=ctx["prompt_embeds"],
    )
    layout = [
        LayoutSegment("action", 0, num_state),
        LayoutSegment("video", 0, num_ref),
        LayoutSegment("action", num_state, num_state + num_action),
        LayoutSegment("video", num_ref, num_ref + num_noisy),
    ]
    attention_mask = transformer.build_gwp_casual_mask(
        num_state_tokens=num_state,
        num_ref_tokens=num_ref,
        num_action_tokens=num_action,
        num_noisy_tokens=num_noisy,
        device=video_pre["tokens"].device,
        dtype=video_pre["tokens"].dtype,
    )

    tokens_all = {"video": video_pre["tokens"], "action": action_pre["tokens"]}
    rotary_all = {"video": video_pre["rotary_emb"], "action": action_pre["rotary_emb"]}
    context_all = {"video": video_pre["context"], "action": action_pre["context"]}
    t_mod_all = {"video": video_pre["t_mod"], "action": action_pre["t_mod"]}
    mot = transformer.mot

    s = slice(0, num_state)
    r = slice(num_state, num_state + num_ref)
    a = slice(num_state + num_ref, num_state + num_ref + num_action)
    f = slice(num_state + num_ref + num_action, num_state + num_ref + num_action + num_noisy)
    segments = {"state": s, "ref": r, "action": a, "future": f}

    layer_rows = []
    for layer_idx in range(mot.num_layers):
        q_all, k_all, v_all, cached_all = {}, {}, {}, {}
        for name in mot.expert_order:
            expert = mot.mixtures[name]
            block = expert.blocks[layer_idx]
            io = mot._build_attention_io(
                block=block,
                hidden_states=tokens_all[name],
                temb=t_mod_all[name],
                rotary_emb=rotary_all[name],
            )
            q_all[name], k_all[name], v_all[name] = io["q"], io["k"], io["v"]
            cached_all[name] = io
        q_cat = mot._assemble_from_layout(q_all, layout)
        k_cat = mot._assemble_from_layout(k_all, layout)
        v_cat = mot._assemble_from_layout(v_all, layout)

        q = q_cat.unflatten(2, (mot.num_heads, mot.attn_head_dim)).transpose(1, 2).float()
        k = k_cat.unflatten(2, (mot.num_heads, mot.attn_head_dim)).transpose(1, 2).float()
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(mot.attn_head_dim)
        scores = scores + attention_mask[None, None].float()
        probs = torch.softmax(scores, dim=-1)

        def mass(row_slice, col_slice) -> float:
            return float(probs[:, :, row_slice, col_slice].sum(dim=-1).mean().detach().cpu())

        row = {"layer": layer_idx}
        for name, col_slice in segments.items():
            row[f"action_to_{name}"] = mass(a, col_slice)
            row[f"future_to_{name}"] = mass(f, col_slice)
        layer_rows.append(row)

        mixed = mot._mixed_attention(q_cat, k_cat, v_cat, attention_mask)
        mixed_by_expert = mot._scatter_to_experts(mixed, layout, q_all)
        for name in mot.expert_order:
            block = mot.mixtures[name].blocks[layer_idx]
            tokens_all[name] = mot._apply_post_block(
                block=block,
                mixed_slice=mixed_by_expert[name],
                cached=cached_all[name],
                context=context_all.get(name),
            )

    mean_row = {}
    for key in layer_rows[0]:
        if key == "layer":
            continue
        mean_row[key] = _mean([row[key] for row in layer_rows])
    return {"mean": mean_row, "layers": layer_rows}


def _format_summary(result: dict[str, Any]) -> str:
    lines = []
    lines.append("")
    lines.append(f"## {result['title']}")
    lines.append("")
    lines.append(f"- Command: `{result['command']}`")
    lines.append(f"- Checkpoint: `{result['checkpoint']}`")
    lines.append(f"- Config: `{result['config']}`")
    lines.append(f"- Batches: {result['num_batches']} across {len(result['tasks'])} task paths; batch_size={result['batch_size']}")
    lines.append(f"- Output JSON: `{result['output_json']}`")
    lines.append("")

    overall = result["overall"]
    lines.append("### A1 action-swap -> future prediction")
    for key in (
        "a1_noisy_swap_rel_l2",
        "a1_noisy_swap_cosine",
        "a1_noisy_zero_rel_l2",
        "a1_noisy_zero_cosine",
        "a1_gt_swap_rel_l2",
        "a1_gt_swap_cosine",
        "a1_gt_zero_rel_l2",
        "a1_gt_zero_cosine",
    ):
        if key in overall:
            row = overall[key]
            lines.append(f"- {key}: mean={row['mean']:.6f}, std={row['std']:.6f}, n={row['n']}")
    lines.append("")
    lines.append("Sigma buckets for noisy-action swap:")
    lines.append("")
    lines.append("| sigma bucket | rel_l2 | cosine | n |")
    lines.append("|---|---:|---:|---:|")
    for bucket, row in result["sigma_buckets"].items():
        rel = row.get("a1_noisy_swap_rel_l2", {})
        cos = row.get("a1_noisy_swap_cosine", {})
        lines.append(f"| {bucket} | {rel.get('mean', float('nan')):.6f} | {cos.get('mean', float('nan')):.6f} | {int(rel.get('n', 0))} |")
    lines.append("")

    lines.append("### B2 action_only vs full action")
    for key in ("b2_actiononly_full_max_abs", "b2_actiononly_full_mean_abs", "b2_actiononly_full_rel_l2", "b2_actiononly_full_cosine"):
        if key in overall:
            row = overall[key]
            lines.append(f"- {key}: mean={row['mean']:.8f}, std={row['std']:.8f}, n={row['n']}")
    lines.append("")

    lines.append("### C1 ref ablation -> action prediction")
    for key in (
        "c1_ref_zero_rel_l2",
        "c1_ref_zero_cosine",
        "c1_ref_shuffle_rel_l2",
        "c1_ref_shuffle_cosine",
    ):
        if key in overall:
            row = overall[key]
            lines.append(f"- {key}: mean={row['mean']:.6f}, std={row['std']:.6f}, n={row['n']}")
    lines.append("")

    if result.get("attention"):
        att = result["attention"]["mean"]
        lines.append("### B1/A2 attention mass (first analyzed batch only)")
        lines.append("")
        lines.append("| query block | to state | to ref | to action | to future |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.append(
            "| action | "
            f"{att['action_to_state']:.6f} | {att['action_to_ref']:.6f} | "
            f"{att['action_to_action']:.6f} | {att['action_to_future']:.6f} |"
        )
        lines.append(
            "| future | "
            f"{att['future_to_state']:.6f} | {att['future_to_ref']:.6f} | "
            f"{att['future_to_action']:.6f} | {att['future_to_future']:.6f} |"
        )
        lines.append("")

    lines.append("### Per-task sample table")
    lines.append("")
    lines.append("| task | noisy swap rel_l2 | noisy swap cosine | B2 max_abs | C1 zero-ref rel_l2 | n |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for task, store in result["per_task"].items():
        rel = store.get("a1_noisy_swap_rel_l2", {}).get("mean", float("nan"))
        cos = store.get("a1_noisy_swap_cosine", {}).get("mean", float("nan"))
        b2 = store.get("b2_actiononly_full_max_abs", {}).get("mean", float("nan"))
        c1 = store.get("c1_ref_zero_rel_l2", {}).get("mean", float("nan"))
        n = store.get("a1_noisy_swap_rel_l2", {}).get("n", 0)
        lines.append(f"| {task} | {rel:.6f} | {cos:.6f} | {b2:.8f} | {c1:.6f} | {int(n)} |")
    lines.append("")

    lines.append("### Mechanism readout")
    noisy_rel = overall.get("a1_noisy_swap_rel_l2", {}).get("mean", float("nan"))
    gt_rel = overall.get("a1_gt_swap_rel_l2", {}).get("mean", float("nan"))
    b2_abs = overall.get("b2_actiononly_full_max_abs", {}).get("mean", float("nan"))
    c1_rel = overall.get("c1_ref_zero_rel_l2", {}).get("mean", float("nan"))
    lines.append(f"- World action-conditioning: action swap changes visual velocity by rel_l2={noisy_rel:.4f} with noisy action and {gt_rel:.4f} with clean GT action.")
    lines.append(f"- Inference-time relevance: full/action_only action predictions differ by max_abs={b2_abs:.8f}; future tokens are therefore numerically absent from the action path at inference.")
    lines.append(f"- Current-observation/ref relevance: zeroing ref changes action prediction by rel_l2={c1_rel:.4f}.")
    if result.get("attention"):
        att = result["attention"]["mean"]
        lines.append(f"- Attention route: action queries allocate {att['action_to_ref']:.3f} mass to ref tokens and {att['action_to_action']:.3f} to action tokens; future queries allocate {att['future_to_action']:.3f} mass to action tokens.")
    lines.append("- D1 causal training ablation was not run in this pass; this report is forward-only.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs.robocasa_all_tshape_mot_joint_from_videopt_randomcrop.config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-task-paths", type=int, default=8)
    parser.add_argument("--batches-per-task", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--attention-batches", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--append-md", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.environ.setdefault("GWP_VIDEO_PT_CKPT", args.checkpoint)
    config = copy.deepcopy(load_config(args.config))
    config.setdefault("wandb", {})["mode"] = "disabled"
    config["project_dir"] = "/tmp/gwp_world_branch_analysis"
    config.setdefault("train", {})["max_steps"] = 1
    model_cfg = config.setdefault("models", {})
    model_cfg["checkpoint"] = args.checkpoint
    model_cfg["strict"] = True
    model_cfg["view_interval"] = 10**12

    dl_cfg = config["dataloaders"]["train"]
    all_data_configs = copy.deepcopy(dl_cfg.get("data_or_config") or [])
    task_configs = _pick_task_configs(all_data_configs, max_tasks=args.num_task_paths, seed=args.seed)
    dl_cfg["batch_size_per_gpu"] = args.batch_size
    dl_cfg["num_workers"] = 0
    dl_cfg["timeout"] = 0

    runner_cls = resolve_runner(config["runners"][0])
    trainer = runner_cls(config)
    trainer.cur_step = 10**9
    trainer.model = trainer.get_models(DictConfig(config["models"]))
    trainer.model.eval()
    transformer = trainer.model["transformer"]
    transformer.eval()

    overall_raw: dict[str, list[float]] = defaultdict(list)
    task_raw: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sigma_raw: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    attention = None
    attention_done = 0
    num_batches = 0
    selected_tasks = []

    for task, data_cfg in task_configs:
        selected_tasks.append({"task": task, "data_path": data_cfg.get("data_path")})
        trainer.config["dataloaders"]["train"]["data_or_config"] = [data_cfg]
        dataset = trainer._build_dataset()
        dataloader = trainer._build_dataloader(dataset)
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.batches_per_task:
                break
            batch = _to_device(batch, trainer.device)
            ctx = _build_context(trainer, batch)
            bs = ctx["clean_action"].shape[0]
            if bs < 2:
                perm = torch.arange(bs, device=ctx["clean_action"].device)
            else:
                perm = torch.arange(bs, device=ctx["clean_action"].device).roll(1)

            # A1: future prediction under action swaps.
            visual_noisy, action_full = _forward(
                transformer,
                ctx,
                action=ctx["noisy_action"],
                timestep=ctx["timestep_noisy_action"],
                action_only=False,
            )
            visual_noisy_swap, _ = _forward(
                transformer,
                ctx,
                action=ctx["noisy_action"][perm],
                timestep=ctx["timestep_noisy_action"],
                action_only=False,
            )
            visual_noisy_zero, _ = _forward(
                transformer,
                ctx,
                action=torch.zeros_like(ctx["noisy_action"]),
                timestep=ctx["timestep_noisy_action"],
                action_only=False,
            )
            _append_values(overall_raw, "a1_noisy_swap", _metric_pair(visual_noisy, visual_noisy_swap))
            _append_values(overall_raw, "a1_noisy_zero", _metric_pair(visual_noisy, visual_noisy_zero))
            _append_values(task_raw[task], "a1_noisy_swap", _metric_pair(visual_noisy, visual_noisy_swap))
            _append_values(task_raw[task], "a1_noisy_zero", _metric_pair(visual_noisy, visual_noisy_zero))

            visual_gt, _ = _forward(
                transformer,
                ctx,
                action=ctx["clean_action"],
                timestep=ctx["timestep_gt_action"],
                action_only=False,
            )
            visual_gt_swap, _ = _forward(
                transformer,
                ctx,
                action=ctx["clean_action"][perm],
                timestep=ctx["timestep_gt_action"],
                action_only=False,
            )
            visual_gt_zero, _ = _forward(
                transformer,
                ctx,
                action=torch.zeros_like(ctx["clean_action"]),
                timestep=ctx["timestep_gt_action"],
                action_only=False,
            )
            _append_values(overall_raw, "a1_gt_swap", _metric_pair(visual_gt, visual_gt_swap))
            _append_values(overall_raw, "a1_gt_zero", _metric_pair(visual_gt, visual_gt_zero))
            _append_values(task_raw[task], "a1_gt_swap", _metric_pair(visual_gt, visual_gt_swap))
            _append_values(task_raw[task], "a1_gt_zero", _metric_pair(visual_gt, visual_gt_zero))

            noisy_pair = _metric_pair(visual_noisy, visual_noisy_swap)
            for i, sigma in enumerate(ctx["visual_sigma"].detach().cpu().flatten().tolist()[:bs]):
                bucket = _sigma_bucket(float(sigma))
                for name, tensor in noisy_pair.items():
                    sigma_raw[bucket][f"a1_noisy_swap_{name}"].append(float(tensor.flatten()[i]))

            # B2: full vs action-only action path.
            action_only = _forward(
                transformer,
                ctx,
                action=ctx["noisy_action"],
                timestep=ctx["timestep_noisy_action"],
                action_only=True,
            )
            diff = (action_full.float() - action_only.float()).reshape(bs, -1)
            base = action_full.float().reshape(bs, -1)
            b2_vals = {
                "max_abs": diff.abs().max(dim=1).values.detach().cpu(),
                "mean_abs": diff.abs().mean(dim=1).detach().cpu(),
                "rel_l2": (diff.norm(dim=1) / (base.norm(dim=1) + 1e-8)).detach().cpu(),
                "cosine": F.cosine_similarity(base, action_only.float().reshape(bs, -1), dim=1).detach().cpu(),
            }
            _append_values(overall_raw, "b2_actiononly_full", b2_vals)
            _append_values(task_raw[task], "b2_actiononly_full", b2_vals)

            # C1: ref ablation on action prediction.
            original_ref = ctx["ref_latents"]
            ctx["ref_latents"] = torch.zeros_like(original_ref)
            action_zero_ref = _forward(
                transformer,
                ctx,
                action=ctx["noisy_action"],
                timestep=ctx["timestep_noisy_action"],
                action_only=True,
            )
            ctx["ref_latents"] = original_ref[perm]
            action_shuffle_ref = _forward(
                transformer,
                ctx,
                action=ctx["noisy_action"],
                timestep=ctx["timestep_noisy_action"],
                action_only=True,
            )
            ctx["ref_latents"] = original_ref
            _append_values(overall_raw, "c1_ref_zero", _metric_pair(action_only, action_zero_ref))
            _append_values(overall_raw, "c1_ref_shuffle", _metric_pair(action_only, action_shuffle_ref))
            _append_values(task_raw[task], "c1_ref_zero", _metric_pair(action_only, action_zero_ref))
            _append_values(task_raw[task], "c1_ref_shuffle", _metric_pair(action_only, action_shuffle_ref))

            if args.attention_batches > 0 and attention_done < args.attention_batches:
                attention = _attention_masses(
                    transformer,
                    ctx,
                    action=ctx["noisy_action"],
                    timestep=ctx["timestep_noisy_action"],
                )
                attention_done += 1

            num_batches += 1
            torch.cuda.empty_cache()

    if args.output_json is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_json = f"/shared_disk/users/hengtao.li/codex/gwp-mot/analysis/world_branch_{timestamp}.json"
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    result = {
        "title": "GWP-MoT world-branch forward analysis",
        "command": " ".join(sys.argv),
        "checkpoint": args.checkpoint,
        "config": args.config,
        "batch_size": args.batch_size,
        "num_batches": num_batches,
        "tasks": selected_tasks,
        "overall": _summarize_store(overall_raw),
        "per_task": {task: _summarize_store(store) for task, store in sorted(task_raw.items())},
        "sigma_buckets": {bucket: _summarize_store(store) for bucket, store in sorted(sigma_raw.items())},
        "attention": attention,
        "output_json": args.output_json,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    md = _format_summary(result)
    print(md)
    if args.append_md:
        with open(args.append_md, "a", encoding="utf-8") as f:
            f.write(md)


if __name__ == "__main__":
    main()
