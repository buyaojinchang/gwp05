"""AgileX dataset action replay server (ZMQ, giga-brain compatible).

Replays joint targets from a LeRobot episode parquet instead of model inference.
The remote client protocol is unchanged: observations in, (action_chunk, 14) tensor out.
"""

from __future__ import annotations

import glob
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import tyro

from experiment.agilex.sockets import RobotInferenceServer

ACTION_DIM = 14
STATE_DIM = 14
ACTION_CHUNK = 36
REPLAN_STEPS = 30

# Same delta mask as inference_server / agilex training (joint dims delta, grippers absolute).
DELTA_MASK = np.array(
    [True, True, True, True, True, True, False,
     True, True, True, True, True, True, False],
    dtype=bool,
)


def find_episode_parquet(lerobot_dir: str, episode_idx: int) -> str:
    pattern = os.path.join(lerobot_dir, "data", "**", f"episode_{episode_idx:06d}.parquet")
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(
            f"No parquet for episode {episode_idx} under {lerobot_dir}/data "
            f"(pattern: {pattern})"
        )
    if len(matches) > 1:
        print(f"Warning: multiple parquet files for episode {episode_idx}, using {matches[0]}")
    return matches[0]


def load_episode(
    lerobot_dir: str,
    episode_idx: int,
    action_format: str = "absolute",
) -> dict[str, Any]:
    """Load one episode's actions and states from LeRobot parquet."""
    parquet_path = find_episode_parquet(lerobot_dir, episode_idx)
    df = pd.read_parquet(parquet_path)

    actions = np.stack(df["action"].values).astype(np.float32)
    states = np.stack(df["observation.state"].values).astype(np.float32)

    if action_format == "delta":
        mask = DELTA_MASK[: min(ACTION_DIM, actions.shape[-1])]
        for t in range(len(actions)):
            actions[t, mask] = actions[t, mask] + states[t, mask]
    elif action_format != "absolute":
        raise ValueError(f"Unsupported action_format: {action_format}")

    task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns else 0
    return {
        "parquet_path": parquet_path,
        "actions": actions,
        "states": states,
        "length": len(df),
        "task_index": task_index,
    }


class AgilexReplayPolicy:
    """Replay absolute joint targets from a LeRobot episode."""

    def __init__(
        self,
        dataset_root: str,
        episode_idx: int = 0,
        action_chunk: int = ACTION_CHUNK,
        replan_steps: int | None = None,
        start_frame: int = 0,
        loop: bool = False,
        action_format: str = "absolute",
    ):
        self.action_chunk = action_chunk
        self.replan_steps = replan_steps or action_chunk
        self.loop = loop
        self.action_format = action_format

        print(f"Loading episode {episode_idx} from {dataset_root}")
        ep = load_episode(dataset_root, episode_idx, action_format=action_format)
        self.actions = ep["actions"]
        self.states = ep["states"]
        self.episode_len = ep["length"]
        self.parquet_path = ep["parquet_path"]

        if start_frame < 0 or start_frame >= self.episode_len:
            raise ValueError(
                f"start_frame={start_frame} out of range for episode length {self.episode_len}"
            )
        self.step_idx = start_frame
        self._done = False

        print(
            f"AgileX Replay Policy ready: {self.parquet_path}, "
            f"length={self.episode_len}, start_frame={start_frame}, "
            f"action_chunk={self.action_chunk}, replan_steps={self.replan_steps}, "
            f"action_format={action_format}, loop={loop}"
        )

    def _slice_chunk(self) -> np.ndarray:
        if self._done and not self.loop:
            last = self.actions[min(self.step_idx, self.episode_len - 1)]
            return np.tile(last, (self.replan_steps, 1))

        start = self.step_idx
        end = min(start + self.action_chunk, self.episode_len)
        chunk = self.actions[start:end].copy()

        execute_steps = min(self.action_chunk, self.replan_steps, max(chunk.shape[0], 1))
        if chunk.shape[0] < execute_steps:
            pad_row = chunk[-1] if chunk.shape[0] > 0 else self.actions[0]
            pad = np.tile(pad_row, (execute_steps - chunk.shape[0], 1))
            chunk = np.concatenate([chunk, pad], axis=0)
        chunk = chunk[:execute_steps]

        next_idx = self.step_idx + self.replan_steps
        if next_idx >= self.episode_len:
            if self.loop:
                self.step_idx = next_idx % self.episode_len
            else:
                self.step_idx = self.episode_len
                self._done = True
        else:
            self.step_idx = next_idx

        return chunk

    def inference(self, data: dict[str, Any]) -> torch.Tensor:
        start_t = time.time()
        chunk = self._slice_chunk()
        elapsed = time.time() - start_t
        status = "done" if self._done else "running"
        print(
            f"  Replay {elapsed * 1000:.0f}ms, step→{self.step_idx}/{self.episode_len}, "
            f"shape={chunk.shape}, status={status}, action[0]={chunk[0]}"
        )
        return torch.from_numpy(chunk.astype(np.float32))


def run_server(
    dataset_root: str,
    episode_idx: int = 0,
    action_chunk: int = ACTION_CHUNK,
    replan_steps: int = REPLAN_STEPS,
    start_frame: int = 0,
    loop: bool = False,
    action_format: str = "absolute",
    host: str = "127.0.0.1",
    port: int = 11411,
) -> None:
    policy = AgilexReplayPolicy(
        dataset_root=dataset_root,
        episode_idx=episode_idx,
        action_chunk=action_chunk,
        replan_steps=replan_steps,
        start_frame=start_frame,
        loop=loop,
        action_format=action_format,
    )
    server = RobotInferenceServer(policy, host=host, port=port)
    server.run()


if __name__ == "__main__":
    tyro.cli(run_server)
