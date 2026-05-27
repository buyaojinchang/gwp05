#!/usr/bin/env python3
"""Run one MoT training step without saving a full checkpoint."""

from __future__ import annotations

import argparse
import copy
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_HOME", "/shared_disk/models/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/shared_disk/models/huggingface")
os.environ.setdefault("WANDB_MODE", "disabled")

from world_action_model.runtime import load_config, resolve_runner  # noqa: E402
from world_action_model.trainer import DictConfig, EMA  # noqa: E402


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(v, device) for v in value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs.robocasa_all_tshape_mot.config")
    parser.add_argument("--data-index", type=int, default=0)
    parser.add_argument("--with-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--ema-device", default="model")
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    config = copy.deepcopy(load_config(args.config))
    train_cfg = config.setdefault("train", {})
    train_cfg.update(
        max_epochs=0,
        max_steps=1,
        gradient_accumulation_steps=1,
        with_ema=args.with_ema,
        ema=dict(
            enabled=args.with_ema,
            decay=args.ema_decay,
            device=args.ema_device,
        ),
        log_interval=1,
        checkpoint_interval=10**9,
    )
    config.setdefault("wandb", {})["mode"] = "disabled"
    config["project_dir"] = os.path.join(PROJECT_ROOT, "logs", "mot_training_smoke")

    model_cfg = config.setdefault("models", {})
    model_cfg["checkpoint"] = None
    model_cfg["view_interval"] = 10**9
    dl_cfg = config["dataloaders"]["train"]
    data_configs = dl_cfg.get("data_or_config") or []
    if not data_configs:
        raise RuntimeError("No training datasets were discovered by the config.")
    if args.data_index >= len(data_configs):
        raise IndexError(f"--data-index {args.data_index} out of range for {len(data_configs)} datasets")
    dl_cfg["data_or_config"] = [data_configs[args.data_index]]
    dl_cfg["batch_size_per_gpu"] = 1
    dl_cfg["num_workers"] = 0

    runner_cls = resolve_runner(config["runners"][0])
    trainer = runner_cls(config)
    trainer.cur_step = 2
    trainer.model = trainer.get_models(DictConfig(config["models"]))
    ema = EMA(trainer.model, decay=args.ema_decay, device=args.ema_device) if args.with_ema else None
    dataset = trainer._build_dataset()
    dataloader = trainer._build_dataloader(dataset)
    optimizer = trainer._build_optimizer(trainer.model)
    trainer._resolved_max_steps = 1
    scheduler = trainer._build_scheduler(optimizer)

    batch = next(iter(dataloader))
    batch = _to_device(batch, trainer.device)

    trainer.model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = trainer.forward_step(batch)
    total_loss = sum(losses.values())
    trainer.accelerator.backward(total_loss)
    grad_norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0e9)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    if ema is not None:
        ema.update(trainer.model)
    if args.save_checkpoint:
        trainer.cur_step = 1
        trainer._save_checkpoint(config["project_dir"], ema)

    loss_str = ", ".join(f"{k}={float(v.detach().float().cpu()):.6f}" for k, v in losses.items())
    data_path = dl_cfg["data_or_config"][0]["data_path"]
    ema_str = ""
    if ema is not None:
        ema_dtypes = sorted({str(t.dtype) for t in ema.shadow.values()})
        ema_str = f" ema_updates={ema.updates} ema_tracked={len(ema.shadow)} ema_dtypes={ema_dtypes}"
    print(
        "real MoT training smoke ok "
        f"device={trainer.device} dataset={data_path} total_loss={float(total_loss.detach().float().cpu()):.6f} "
        f"grad_norm={float(grad_norm.detach().float().cpu()):.6f} {loss_str}{ema_str}"
    )


if __name__ == "__main__":
    main()
