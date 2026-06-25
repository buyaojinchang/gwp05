#!/usr/bin/env python3
"""Gradient attribution between visual_loss and action_loss for gwp-mot.

This is a forward/backward-only diagnostic: no optimizer, no parameter update.
It measures whether the future-video FM objective sends meaningful and aligned
gradients into the ref/current-observation representation and the action-path
parameters that the policy consumes.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

os.environ.setdefault("HF_HOME", "/shared_disk/models/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/shared_disk/models/huggingface")
os.environ.setdefault("WANDB_MODE", "disabled")

from analyze_world_branch import _mean, _pick_task_configs, _std, _to_device  # noqa: E402
from world_action_model.models.mot import LayoutSegment  # noqa: E402
from world_action_model.runtime import load_config, resolve_runner  # noqa: E402
from world_action_model.trainer import DictConfig  # noqa: E402
from world_action_model.trainers.wa_trainer_pretrain import _as_dim_mask, masked_mse  # noqa: E402


def _make_context(trainer, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
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

    with torch.no_grad():
        visual_latents = trainer.forward_vae(images)
        ref_latents_raw = trainer.forward_vae(batch["ref_images"][:, :1])

    visual_noise = torch.randn_like(visual_latents)
    visual_target = visual_noise - visual_latents
    noisy_latents_all = visual_noise * sigma + visual_latents * (1 - sigma)

    action_noise = torch.randn_like(action)
    action_target = action_noise - action
    noisy_action = action_noise * action_sigma + action * (1 - action_sigma)
    input_action = action if trainer.use_gt_action_for_video else noisy_action

    if not trainer.expand_timesteps:
        raise RuntimeError("This diagnostic expects expand_timesteps=True for the target config.")

    num_latent_frames = visual_latents.shape[2]
    latent_height = visual_latents.shape[-2]
    latent_width = visual_latents.shape[-1]
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
    ref_latents = insert_noisy_latents[:, :, :1]
    noisy_latents = insert_noisy_latents[:, :, 1:]

    frame_per_tokens = first_frame_mask.shape[-1] * first_frame_mask.shape[-2] // 4
    num_latent_tokens = frame_per_tokens * first_frame_mask.shape[2]
    num_clean_latent_tokens = frame_per_tokens
    num_state_tokens = state.shape[1]
    num_action_tokens = action.shape[1]
    action_start = num_state_tokens + num_clean_latent_tokens
    action_end = action_start + num_action_tokens
    model_timestep = torch.zeros(
        bs,
        num_state_tokens + num_action_tokens + num_latent_tokens,
        device=noisy_latents.device,
        dtype=noisy_latents.dtype,
    )
    if trainer.use_gt_action_for_video:
        model_timestep[:, action_start:action_end] = 0
    else:
        model_timestep[:, action_start:action_end] = action_noise_t
    model_timestep[:, action_end:] = noise_t

    dim_mask = None
    if "action_dim_mask" in batch:
        dim_mask = _as_dim_mask(
            batch["action_dim_mask"],
            batch_size=bs,
            seq_len=action.shape[1],
            dim=action.shape[2],
            device=noisy_latents.device,
        )

    return {
        "prompt_embeds": prompt_embeds,
        "state": state.to(trainer.dtype),
        "action": input_action.to(trainer.dtype),
        "timestep": model_timestep,
        "ref_latents": ref_latents,
        "noisy_latents": noisy_latents,
        "visual_target": visual_target.to(trainer.dtype),
        "first_frame_mask": first_frame_mask.to(trainer.dtype),
        "action_target": action_target.to(trainer.dtype),
        "dim_mask": dim_mask,
    }


def _retain_ref(ref_tensors: list[tuple[str, torch.Tensor, int]], name: str, tensor: torch.Tensor, num_ref: int):
    if tensor.requires_grad:
        tensor.retain_grad()
        ref_tensors.append((name, tensor, num_ref))


def _forward_with_ref_capture(transformer, ctx: dict[str, torch.Tensor]):
    hidden_states = torch.cat([ctx["ref_latents"], ctx["noisy_latents"]], dim=2)
    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = transformer.config.patch_size
    post_patch_height = height // p_h
    post_patch_width = width // p_w
    num_state = ctx["state"].shape[1]
    num_action = ctx["action"].shape[1]
    num_ref = (ctx["ref_latents"].shape[2] // p_t) * post_patch_height * post_patch_width
    num_noisy = (num_frames // p_t) * post_patch_height * post_patch_width - num_ref

    state_ts, ref_ts, action_ts, noisy_ts = transformer._split_timesteps(
        ctx["timestep"],
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
        action=ctx["action"].to(dtype=video_pre["tokens"].dtype),
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

    ref_tensors: list[tuple[str, torch.Tensor, int]] = []
    _retain_ref(ref_tensors, "input", tokens_all["video"], num_ref)

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
        _retain_ref(ref_tensors, f"layer_{layer_idx:02d}", tokens_all["video"], num_ref)

    visual_pred = transformer._post_video(tokens_all["video"], video_pre)
    action_tokens = tokens_all["action"][:, num_state : num_state + num_action]
    action_pred = transformer.action_expert.post_action(action_tokens)
    return visual_pred, action_pred, ref_tensors


def _compute_losses(trainer, transformer, ctx: dict[str, torch.Tensor]):
    visual_pred, action_pred, refs = _forward_with_ref_capture(transformer, ctx)
    visual_loss = ((visual_pred.float() - ctx["visual_target"].float()) * ctx["first_frame_mask"].float()).pow(2).mean()
    action_loss = masked_mse(action_pred.float(), ctx["action_target"].float(), dim_mask=ctx["dim_mask"], time_mask=None)
    return visual_loss, action_loss, refs


_LAYER_RE = re.compile(r"blocks\.(\d+)")


def _layer_from_name(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _groups_for_param(name: str) -> list[str]:
    groups = []
    if name.startswith("mot.mixtures.action."):
        groups.append("action_all")
        if ".attn1.to_q" in name or ".attn1.to_k" in name or ".attn1.to_v" in name or ".attn1.to_qkv" in name:
            groups.append("action_attn_qkv")
        if ".ffn." in name:
            groups.append("action_ffn")
    if name.startswith("mot.mixtures.video."):
        if ".attn1.to_k" in name or ".attn1.to_v" in name or ".attn1.to_qkv" in name:
            groups.append("video_attn_kv_or_qkv")
        if ".patch_embedding" not in name:
            groups.append("video_ref_backbone")
    return groups


def _collect_param_norm2(transformer) -> tuple[dict[str, float], dict[str, dict[int, float]]]:
    group_norm2: dict[str, float] = defaultdict(float)
    layer_norm2: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for name, param in transformer.named_parameters():
        if param.grad is None:
            continue
        val = float(param.grad.detach().float().pow(2).sum().cpu())
        if val == 0.0:
            continue
        layer = _layer_from_name(name)
        for group in _groups_for_param(name):
            group_norm2[group] += val
            if layer is not None:
                layer_norm2[group][layer] += val
    return dict(group_norm2), {g: dict(v) for g, v in layer_norm2.items()}


def _collect_ref_norms(refs: list[tuple[str, torch.Tensor, int]]) -> dict[str, float]:
    out = {}
    for name, tensor, num_ref in refs:
        if tensor.grad is None:
            continue
        grad = tensor.grad[:, :num_ref]
        out[name] = float(grad.detach().float().norm().cpu())
    return out


def _accum(dst: dict[str, float], src: dict[str, float], scale: float = 1.0):
    for key, value in src.items():
        dst[key] += value * scale


def _accum_layer(dst: dict[str, dict[int, float]], src: dict[str, dict[int, float]], scale: float = 1.0):
    for group, per_layer in src.items():
        for layer, value in per_layer.items():
            dst[group][layer] += value * scale


def _run_backward_mode(trainer, transformer, ctx: dict[str, torch.Tensor], mode: str):
    trainer.model.zero_grad(set_to_none=True)
    visual_loss, action_loss, refs = _compute_losses(trainer, transformer, ctx)
    if mode == "visual":
        loss = visual_loss
    elif mode == "action":
        loss = action_loss
    elif mode == "sum":
        loss = visual_loss + action_loss
    else:
        raise ValueError(mode)
    loss.backward()
    group_norm2, layer_norm2 = _collect_param_norm2(transformer)
    ref_norms = _collect_ref_norms(refs)
    losses = {"visual_loss": float(visual_loss.detach().cpu()), "action_loss": float(action_loss.detach().cpu())}
    trainer.model.zero_grad(set_to_none=True)
    del visual_loss, action_loss, loss, refs
    torch.cuda.empty_cache()
    return group_norm2, layer_norm2, ref_norms, losses


def _safe_cos(n_vis: float, n_act: float, n_sum: float) -> float:
    denom = (n_vis * n_act) ** 0.5
    if denom <= 0:
        return float("nan")
    dot = 0.5 * (n_sum - n_vis - n_act)
    return float(dot / denom)


def _summarize_result(
    result: dict[str, Any],
    append_path: str | None,
):
    lines = []
    lines.append("")
    lines.append("#### 6.5.1 Gradient attribution result")
    lines.append("")
    lines.append(f"- Command: `{result['command']}`")
    lines.append(f"- Checkpoint: `{result['checkpoint']}`")
    lines.append(f"- Config: `{result['config']}`")
    lines.append(f"- Batches: {result['num_batches']} across {len(result['tasks'])} task paths; batch_size={result['batch_size']}")
    lines.append(f"- Output JSON: `{result['output_json']}`")
    lines.append("")
    lines.append("##### Loss scale")
    lines.append("")
    lines.append(f"- visual_loss mean={result['losses']['visual_loss_mean']:.6f}")
    lines.append(f"- action_loss mean={result['losses']['action_loss_mean']:.6f}")
    lines.append("")
    lines.append("##### Ref/current-obs hidden-state gradient norms")
    lines.append("")
    lines.append("| layer | ||d visual / d h_ref|| | ||d action / d h_ref|| | ratio visual/action |")
    lines.append("|---|---:|---:|---:|")
    for row in result["ref_grad_layers"]:
        lines.append(f"| {row['layer']} | {row['visual_norm']:.6e} | {row['action_norm']:.6e} | {row['ratio']:.4f} |")
    lines.append("")
    lines.append("##### Parameter gradient alignment")
    lines.append("")
    lines.append("| group | ||g_visual|| | ||g_action|| | visual/action | cos(g_visual,g_action) |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in result["param_groups"]:
        lines.append(
            f"| {row['group']} | {row['visual_norm']:.6e} | {row['action_norm']:.6e} | "
            f"{row['ratio']:.4f} | {row['cosine']:.4f} |"
        )
    lines.append("")
    lines.append("##### Largest visual/action ratios by layer")
    lines.append("")
    lines.append("| group | layer | ||g_visual|| | ||g_action|| | ratio | cosine |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in result["top_layers"]:
        lines.append(
            f"| {row['group']} | {row['layer']} | {row['visual_norm']:.6e} | {row['action_norm']:.6e} | "
            f"{row['ratio']:.4f} | {row['cosine']:.4f} |"
        )
    lines.append("")
    lines.append("##### Infra-level readout")
    lines.append("")
    lines.append(result["readout"])
    lines.append("")
    md = "\n".join(lines)
    print(md)
    if append_path:
        with open(append_path, "a", encoding="utf-8") as f:
            f.write(md)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs.robocasa_all_tshape_mot_joint_from_videopt_randomcrop.config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-task-paths", type=int, default=8)
    parser.add_argument("--batches-per-task", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--append-md", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.environ.setdefault("GWP_VIDEO_PT_CKPT", args.checkpoint)
    config = copy.deepcopy(load_config(args.config))
    config.setdefault("wandb", {})["mode"] = "disabled"
    config["project_dir"] = "/tmp/gwp_visual_action_gradients"
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
    for param in transformer.parameters():
        param.requires_grad_(True)

    total_vis: dict[str, float] = defaultdict(float)
    total_act: dict[str, float] = defaultdict(float)
    total_sum: dict[str, float] = defaultdict(float)
    layer_vis: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    layer_act: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    layer_sum: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    ref_vis: dict[str, float] = defaultdict(float)
    ref_act: dict[str, float] = defaultdict(float)
    visual_losses, action_losses = [], []
    selected_tasks, num_batches = [], 0

    for task, data_cfg in task_configs:
        selected_tasks.append({"task": task, "data_path": data_cfg.get("data_path")})
        trainer.config["dataloaders"]["train"]["data_or_config"] = [data_cfg]
        dataset = trainer._build_dataset()
        dataloader = trainer._build_dataloader(dataset)
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.batches_per_task:
                break
            batch = _to_device(batch, trainer.device)
            ctx = _make_context(trainer, batch)

            g_vis, l_vis, h_vis, losses = _run_backward_mode(trainer, transformer, ctx, "visual")
            _accum(total_vis, g_vis)
            _accum_layer(layer_vis, l_vis)
            _accum(ref_vis, h_vis)
            visual_losses.append(losses["visual_loss"])
            action_losses.append(losses["action_loss"])

            g_act, l_act, h_act, _ = _run_backward_mode(trainer, transformer, ctx, "action")
            _accum(total_act, g_act)
            _accum_layer(layer_act, l_act)
            _accum(ref_act, h_act)

            g_sum, l_sum, _, _ = _run_backward_mode(trainer, transformer, ctx, "sum")
            _accum(total_sum, g_sum)
            _accum_layer(layer_sum, l_sum)

            num_batches += 1

    ref_rows = []
    ref_layers = sorted(set(ref_vis) | set(ref_act), key=lambda x: (-1 if x == "input" else int(x.split("_")[-1])))
    for layer in ref_layers:
        v = ref_vis.get(layer, 0.0) / max(1, num_batches)
        a = ref_act.get(layer, 0.0) / max(1, num_batches)
        ref_rows.append({"layer": layer, "visual_norm": v, "action_norm": a, "ratio": v / (a + 1e-12)})

    group_rows = []
    for group in sorted(set(total_vis) | set(total_act) | set(total_sum)):
        nv, na, ns = total_vis.get(group, 0.0), total_act.get(group, 0.0), total_sum.get(group, 0.0)
        visual_norm, action_norm = nv**0.5, na**0.5
        group_rows.append(
            {
                "group": group,
                "visual_norm": visual_norm,
                "action_norm": action_norm,
                "ratio": visual_norm / (action_norm + 1e-12),
                "cosine": _safe_cos(nv, na, ns),
            }
        )

    top_layers = []
    for group in sorted(set(layer_vis) | set(layer_act) | set(layer_sum)):
        rows = []
        for layer in sorted(set(layer_vis[group]) | set(layer_act[group]) | set(layer_sum[group])):
            nv, na, ns = layer_vis[group].get(layer, 0.0), layer_act[group].get(layer, 0.0), layer_sum[group].get(layer, 0.0)
            vn, an = nv**0.5, na**0.5
            rows.append(
                {
                    "group": group,
                    "layer": layer,
                    "visual_norm": vn,
                    "action_norm": an,
                    "ratio": vn / (an + 1e-12),
                    "cosine": _safe_cos(nv, na, ns),
                }
            )
        rows.sort(key=lambda r: r["ratio"], reverse=True)
        top_layers.extend(rows[:5])
    top_layers.sort(key=lambda r: (r["group"], -r["ratio"]))

    action_group = next((r for r in group_rows if r["group"] == "action_all"), None)
    video_kv = next((r for r in group_rows if r["group"] == "video_attn_kv_or_qkv"), None)
    ref_mid = ref_rows[min(len(ref_rows) - 1, max(0, len(ref_rows) // 2))] if ref_rows else None
    readout_bits = []
    if ref_mid:
        readout_bits.append(
            f"- Ref hidden gradients are nonzero on the action-consumed current-observation path; around {ref_mid['layer']} the visual/action norm ratio is {ref_mid['ratio']:.3f}."
        )
    if action_group:
        readout_bits.append(
            f"- On action expert parameters, visual/action grad-norm ratio is {action_group['ratio']:.3f} with cosine {action_group['cosine']:.3f}."
        )
    if video_kv:
        readout_bits.append(
            f"- On video K/V (the ref keys/values read by action attention), visual/action grad-norm ratio is {video_kv['ratio']:.3f} with cosine {video_kv['cosine']:.3f}."
        )
    readout_bits.append("- No optimizer step was run; this is an infra-level gradient attribution, not a training conclusion.")

    if args.output_json is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_json = f"/shared_disk/users/hengtao.li/codex/gwp-mot/analysis/visual_action_gradients_{timestamp}.json"
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    result = {
        "command": " ".join(sys.argv),
        "checkpoint": args.checkpoint,
        "config": args.config,
        "batch_size": args.batch_size,
        "num_batches": num_batches,
        "tasks": selected_tasks,
        "output_json": args.output_json,
        "losses": {
            "visual_loss_mean": _mean(visual_losses),
            "visual_loss_std": _std(visual_losses),
            "action_loss_mean": _mean(action_losses),
            "action_loss_std": _std(action_losses),
        },
        "ref_grad_layers": ref_rows,
        "param_groups": group_rows,
        "top_layers": top_layers,
        "readout": "\n".join(readout_bits),
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    _summarize_result(result, args.append_md)


if __name__ == "__main__":
    main()
