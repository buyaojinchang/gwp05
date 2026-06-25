#!/usr/bin/env python3
"""Action-regression probe for gwp-mot Wan VAE current-frame latents.

The protocol intentionally mirrors smolwam/scripts/probe_tokenizer_a.py:
same LeRobot sample contract, same MLP probe, same action standardization.
Only the feature extractor changes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import AutoencoderKLWan
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from world_action_model.datasets.lerobot_dataset import LeRobotDataset


class ProbeDataset(Dataset):
    def __init__(self, dataset: Dataset, view_keys: list[str]):
        self.dataset = dataset
        self.view_keys = view_keys

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        out = {key: item[key] for key in self.view_keys}
        out["action"] = item["action"]
        return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", action="append", required=True)
    parser.add_argument("--view-keys", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--vae-pretrained",
        default="/shared_disk/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/vae",
    )
    parser.add_argument("--clip-frames", type=int, default=1)
    parser.add_argument("--future-offset", type=int, default=24)
    parser.add_argument("--action-horizon", type=int, default=24)
    parser.add_argument("--video-backend", default="pyav", choices=["pyav", "decord"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--feature-mode", default="pooled", choices=["pooled", "flatten"])
    parser.add_argument("--dst-size", nargs=2, type=int, default=[320, 256], metavar=("W", "H"))
    parser.add_argument("--tshape-head-index", type=int, default=2)
    parser.add_argument("--resize-mode", default="crop", choices=["crop", "stretch"])
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def dtype_from_arg(name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def build_dataset(args):
    offsets = list(range(args.clip_frames)) + list(range(args.future_offset, args.future_offset + args.clip_frames))
    datasets = []
    for path in args.data_path:
        ds = LeRobotDataset(
            data_path=path,
            delta_info={"action": args.action_horizon},
            delta_frames={key: offsets for key in args.view_keys},
            video_backend=args.video_backend,
        )
        datasets.append(ProbeDataset(ds, args.view_keys))
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def split_curr_frames(batch: dict[str, torch.Tensor], view_keys: list[str], clip_frames: int) -> torch.Tensor:
    views = []
    for key in view_keys:
        value = batch[key]
        if value.ndim != 5:
            raise ValueError(f"Expected {key} batch shape [B,T,H,W,C], got {tuple(value.shape)}")
        views.append(value[:, :clip_frames])
    return torch.stack(views, dim=1)  # [B,V,T,H,W,C]


def _resize_crop_nchw(x: torch.Tensor, dst_width: int, dst_height: int, mode: str) -> torch.Tensor:
    if x.dtype != torch.uint8:
        x_f = x.float()
        if float(x_f.max().item()) <= 1.0:
            x_f = x_f * 255.0
        x = x_f.clamp(0, 255).to(torch.uint8)
    x = x.float() / 255.0
    if mode == "stretch":
        x = TVF.resize(x, (dst_height, dst_width), InterpolationMode.BILINEAR)
    else:
        height = int(x.shape[-2])
        width = int(x.shape[-1])
        if float(dst_height) / height < float(dst_width) / width:
            new_height = int(round(float(dst_width) / width * height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / height * width))
        x = TVF.resize(x, (new_height, new_width), InterpolationMode.BILINEAR)
        left = max(0, (new_width - dst_width) // 2)
        top = max(0, (new_height - dst_height) // 2)
        x = TVF.crop(x, top, left, dst_height, dst_width)
    return (x - 0.5) / 0.5


def make_tshape_images(
    frames: torch.Tensor,
    *,
    dst_width: int,
    dst_height: int,
    head_index: int,
    resize_mode: str,
) -> torch.Tensor:
    """Convert [B,V,T,H,W,C] RGB uint8 frames to gwp T-shape [B,T,C,H,W]."""
    batch_size, num_views, num_frames, _, _, _ = frames.shape
    if not (0 <= head_index < num_views):
        raise ValueError(f"head_index={head_index} out of range for {num_views} views")

    processed = []
    for view_i in range(num_views):
        x = frames[:, view_i].permute(0, 1, 4, 2, 3).reshape(batch_size * num_frames, 3, frames.shape[3], frames.shape[4])
        if view_i == head_index:
            y = _resize_crop_nchw(x, dst_width, dst_height, resize_mode)
        else:
            y = _resize_crop_nchw(x, dst_width // 2, dst_height // 2, resize_mode)
        y = y.reshape(batch_size, num_frames, 3, y.shape[-2], y.shape[-1])
        processed.append(y)

    head = processed[head_index]
    others = [v for i, v in enumerate(processed) if i != head_index]
    wrist_row = torch.cat(others, dim=-1)
    if wrist_row.shape[-1] < head.shape[-1]:
        wrist_row = F.pad(wrist_row, (0, head.shape[-1] - wrist_row.shape[-1]))
    elif wrist_row.shape[-1] > head.shape[-1]:
        wrist_row = wrist_row[..., : head.shape[-1]]
    return torch.cat([head, wrist_row], dim=-2).contiguous()


def encode_vae_features(
    vae: AutoencoderKLWan,
    images: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    feature_mode: str,
) -> torch.Tensor:
    images = images.to(device=latents_mean.device, dtype=vae.dtype)
    with torch.no_grad():
        latents = vae.encode(images.permute(0, 2, 1, 3, 4)).latent_dist.mode()
        latents = (latents - latents_mean) * latents_std
    if feature_mode == "pooled":
        return latents.float().mean(dim=(2, 3, 4))
    if feature_mode == "flatten":
        return latents.float().flatten(1)
    raise ValueError(f"Unsupported feature_mode={feature_mode}")


def build_probe(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


def evaluate(model, loader, y_mean, y_std, device):
    model.eval()
    total = 0.0
    raw_total = 0.0
    sse = 0.0
    y_sum = None
    y_sq_sum = None
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            norm_mse = (pred - y).pow(2).mean(dim=1)
            raw_pred = pred * y_std.to(device) + y_mean.to(device)
            raw_y = y * y_std.to(device) + y_mean.to(device)
            raw_mse = (raw_pred - raw_y).pow(2).mean(dim=1)
            total += norm_mse.sum().item()
            raw_total += raw_mse.sum().item()
            sse += (pred - y).pow(2).sum().item()
            y_cpu = y.detach().float().cpu()
            cur_sum = y_cpu.sum(dim=0)
            cur_sq_sum = y_cpu.pow(2).sum(dim=0)
            y_sum = cur_sum if y_sum is None else y_sum + cur_sum
            y_sq_sum = cur_sq_sum if y_sq_sum is None else y_sq_sum + cur_sq_sum
            count += x.shape[0]
    if y_sum is None or y_sq_sum is None or count <= 0:
        sst = 0.0
    else:
        sst = float((y_sq_sum - y_sum.pow(2) / float(count)).sum().item())
    r2 = 1.0 - sse / max(sst, 1e-12)
    return total / max(1, count), raw_total / max(1, count), r2


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    vae_dtype = dtype_from_arg(args.dtype) if args.device.startswith("cuda") else torch.float32
    vae = AutoencoderKLWan.from_pretrained(args.vae_pretrained)
    vae.requires_grad_(False)
    vae.to(args.device, dtype=vae_dtype)
    vae.eval()
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(args.device, dtype=vae_dtype)
    latents_std = (1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)).to(args.device, dtype=vae_dtype)

    features = []
    targets = []
    seen = 0
    total_hint = min(len(loader), max(1, (args.max_samples + args.batch_size - 1) // args.batch_size))
    with torch.no_grad():
        for batch in tqdm(loader, total=total_hint):
            if seen >= args.max_samples:
                break
            curr_frames = split_curr_frames(batch, args.view_keys, args.clip_frames)
            images = make_tshape_images(
                curr_frames,
                dst_width=int(args.dst_size[0]),
                dst_height=int(args.dst_size[1]),
                head_index=args.tshape_head_index,
                resize_mode=args.resize_mode,
            )
            x = encode_vae_features(vae, images, latents_mean, latents_std, args.feature_mode).cpu().float()
            y = batch["action"].reshape(batch["action"].shape[0], -1).cpu().float()
            features.append(x)
            targets.append(y)
            seen += x.shape[0]

    if not features:
        raise RuntimeError("No probe samples collected")
    x = torch.cat(features, dim=0)[: args.max_samples]
    y = torch.cat(targets, dim=0)[: args.max_samples]
    if x.shape[0] < 4:
        raise RuntimeError("Need at least 4 samples for train/test probe")

    perm = torch.randperm(x.shape[0])
    x = x[perm]
    y = y[perm]
    split = max(1, min(x.shape[0] - 1, int(args.train_ratio * x.shape[0])))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std

    train_ds = TensorDataset(x_train, y_train_norm)
    test_ds = TensorDataset(x_test, y_test_norm)
    train_loader = DataLoader(train_ds, batch_size=args.probe_batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.probe_batch_size, shuffle=False)

    model = build_probe(x.shape[1], y.shape[1], args.hidden_dim).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(args.epochs):
        model.train()
        for bx, by in train_loader:
            bx = bx.to(args.device)
            by = by.to(args.device)
            pred = model(bx)
            loss = (pred - by).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        train_norm, train_raw, train_r2 = evaluate(model, train_loader, y_mean, y_std, args.device)
        test_norm, test_raw, test_r2 = evaluate(model, test_loader, y_mean, y_std, args.device)
        row = {
            "epoch": epoch + 1,
            "train_norm_mse": train_norm,
            "test_norm_mse": test_norm,
            "train_raw_mse": train_raw,
            "test_raw_mse": test_raw,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "norm_mse_gap": test_norm - train_norm,
        }
        history.append(row)
        print(json.dumps(row))

    result = {
        "num_samples": int(x.shape[0]),
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "feature_source": "wan_vae",
        "feature_mode": args.feature_mode,
        "feature_dim": int(x.shape[1]),
        "action_dim": int(y.shape[1]),
        "future_offset": args.future_offset,
        "action_horizon": args.action_horizon,
        "vae_pretrained": args.vae_pretrained,
        "dst_size": args.dst_size,
        "tshape_head_index": args.tshape_head_index,
        "resize_mode": args.resize_mode,
        "final": history[-1],
        "history": history,
    }
    (output_dir / "probe_results.json").write_text(json.dumps(result, indent=2))
    print("Wrote", output_dir / "probe_results.json")


if __name__ == "__main__":
    main()
