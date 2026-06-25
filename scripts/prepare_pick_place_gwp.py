#!/usr/bin/env python3
"""Prepare the locomanip `pick_place` (G1 cola task) for gwp-mot training.

gwp-mot's LeRobotDataset hardcodes two parquet columns: ``action`` and
``observation.state``. The raw pick_place stores the sonic latent split across
``action.motion_token`` (64-d) and ``action.hand_binary`` (2-d), and the raw
joint state in ``observation.state``. This script writes a sibling LeRobot
dataset ``pick_place_gwp/`` where:

  * ``action``            = concat(motion_token[64], hand_binary[2]) -> 66-d/frame
  * ``observation.state`` = raw joint state -> 43-d/frame
  * ``observation.motion_latent`` = the same 66-d latent used for action

It symlinks ``videos/`` and ``meta/`` from the source (no video copy) and writes
``norm_stats_delta.json`` with real joint-state mean/std and identity stats for
action / motion latent (combined with ``skip_action_norm: true`` in the data
config).

Usage:
    python scripts/prepare_pick_place_gwp.py \
        --src /home/s4090/hengtao.li/2_data_ckpt_cache/locomanip/data/pick_place \
        --dst /home/s4090/hengtao.li/2_data_ckpt_cache/locomanip/data/pick_place_gwp
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

MOTION_TOKEN_COL = "action.motion_token"
HAND_BINARY_COL = "action.hand_binary"
JOINT_STATE_COL = "observation.state"
MOTION_LATENT_COL = "observation.motion_latent"
ACTION_DIM = 66  # 64 motion token + 2 hand binary
JOINT_STATE_DIM = 43


def _stack_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise KeyError(f"Column {col!r} not found. Available: {list(df.columns)}")
    return np.stack(df[col].values).astype(np.float32)


def _build_action(df: pd.DataFrame) -> np.ndarray:
    motion = _stack_col(df, MOTION_TOKEN_COL)            # [T, 64]
    hand = _stack_col(df, HAND_BINARY_COL)               # [T, 2]
    action = np.concatenate([motion, hand], axis=1)      # [T, 66]
    if action.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM}-d action, got {action.shape[1]}")
    lo, hi = float(action.min()), float(action.max())
    if lo < -1.0 - 1e-3 or hi > 1.0 + 1e-3:
        print(f"  [warn] action range [{lo:.4f}, {hi:.4f}] outside [-1, 1]")
    return action


def _build_joint_state(df: pd.DataFrame) -> np.ndarray:
    joint = _stack_col(df, JOINT_STATE_COL)
    if joint.shape[1] != JOINT_STATE_DIM:
        raise ValueError(f"Expected {JOINT_STATE_DIM}-d joint state, got {joint.shape[1]}")
    return joint


def _symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        default="/home/s4090/hengtao.li/2_data_ckpt_cache/locomanip/data/pick_place",
    )
    parser.add_argument(
        "--dst",
        default="/home/s4090/hengtao.li/2_data_ckpt_cache/locomanip/data/pick_place_gwp",
    )
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    src_data = src / "data"
    if not src_data.is_dir():
        raise FileNotFoundError(f"Missing source data dir: {src_data}")

    parquet_files = sorted(src_data.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {src_data}")

    dst.mkdir(parents=True, exist_ok=True)
    _symlink(src / "videos", dst / "videos")
    _symlink(src / "meta", dst / "meta")

    joint_state_chunks = []
    n_written = 0
    for pf in parquet_files:
        rel = pf.relative_to(src_data)            # e.g. chunk-000/episode_000000.parquet
        out_path = dst / "data" / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(pf)
        action = _build_action(df)                # [T, 66]
        joint_state = _build_joint_state(df)      # [T, 43]
        joint_state_chunks.append(joint_state)

        out = pd.DataFrame()
        out["action"] = list(action)
        out[JOINT_STATE_COL] = list(joint_state)
        out[MOTION_LATENT_COL] = list(action)
        out["episode_index"] = (
            df["episode_index"].values
            if "episode_index" in df.columns
            else np.zeros(len(df), dtype=np.int64)
        )
        if "task_index" in df.columns:
            out["task_index"] = df["task_index"].values
        if "frame_index" in df.columns:
            out["frame_index"] = df["frame_index"].values
        if "timestamp" in df.columns:
            out["timestamp"] = df["timestamp"].values

        out.to_parquet(out_path)
        n_written += 1

    all_joint = np.concatenate(joint_state_chunks, axis=0)
    joint_mean = all_joint.mean(axis=0).astype(np.float64)
    joint_std = all_joint.std(axis=0).astype(np.float64)

    stats = {
        "norm_stats": {
            JOINT_STATE_COL: {"mean": joint_mean.tolist(), "std": joint_std.tolist()},
            MOTION_LATENT_COL: {"mean": [0.0] * ACTION_DIM, "std": [1.0] * ACTION_DIM},
            "action": {"mean": [0.0] * ACTION_DIM, "std": [1.0] * ACTION_DIM},
        }
    }
    norm_path = dst / "norm_stats_delta.json"
    norm_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Wrote {n_written} episodes to {dst / 'data'}")
    print(f"Symlinked videos -> {dst / 'videos'}, meta -> {dst / 'meta'}")
    print(f"Wrote joint/action norm stats -> {norm_path}")


if __name__ == "__main__":
    main()
