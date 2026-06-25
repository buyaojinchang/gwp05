#!/usr/bin/env python3
"""One-step RoboCasa closed-loop smoke test against a running MoT server."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiment.robocasa.inference_client import WebsocketClient  # noqa: E402


def _reorder_lerobot_to_hdf5(action: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            action[5:8],
            action[8:11],
            action[11:12],
            action[0:4],
            action[4:5],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19055)
    parser.add_argument("--env_name", default="CloseBlenderLid")
    parser.add_argument("--split", default="pretrain", choices=["pretrain", "target"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--expected_action_chunk", type=int, default=None)
    parser.add_argument("--log_dir", required=True)
    args = parser.parse_args()

    import gymnasium as gym
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action

    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    horizon = int(get_task_horizon(args.env_name))
    env = gym.make(f"robocasa/{args.env_name}", split=args.split, seed=args.seed)
    client = WebsocketClient(args.host, args.port)

    obs, _ = env.reset()
    state = np.concatenate(
        [
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.gripper_qpos"],
        ],
        axis=0,
    )
    element = {
        "observation/image": np.ascontiguousarray(obs["video.robot0_agentview_left"]),
        "observation/wrist_image": np.ascontiguousarray(obs["video.robot0_eye_in_hand"]),
        "observation/state": state,
        "prompt": obs["annotation.human.task_description"],
    }
    if "video.robot0_agentview_right" in obs:
        element["observation/right_image"] = np.ascontiguousarray(obs["video.robot0_agentview_right"])

    result = client.infer(element)
    actions = np.asarray(result["actions"])
    if args.expected_action_chunk is not None and len(actions) != args.expected_action_chunk:
        raise RuntimeError(f"Expected {args.expected_action_chunk} actions, got {len(actions)}")

    env_action = convert_action(_reorder_lerobot_to_hdf5(actions[0]))
    _, reward, done, truncated, info = env.step(env_action)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "env_name": args.env_name,
        "split": args.split,
        "seed": args.seed,
        "horizon": horizon,
        "action_chunk": int(len(actions)),
        "action_shape": list(actions.shape),
        "first_action_finite": bool(np.isfinite(actions[0]).all()),
        "reward": float(reward),
        "done": bool(done),
        "truncated": bool(truncated),
        "success": bool(info.get("success", False)),
    }
    out_path = pathlib.Path(args.log_dir) / "closed_loop_smoke_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Summary saved to: {out_path}")

    client.close()
    env.env.close()


if __name__ == "__main__":
    main()
