"""GigaWorld-Policy inference client for RoboCasa evaluation.

Connects to the inference server and runs closed-loop evaluation
in RoboCasa environments, following the same protocol as openpi.
"""

import argparse
import collections
import dataclasses
import json
import logging
import os
import pathlib
import time
from datetime import datetime

import imageio
import numpy as np
import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Websocket client (compatible with openpi_client protocol)
# ---------------------------------------------------------------------------
import functools
import msgpack


def _pack_array(obj):
    """Serialize numpy arrays in openpi_client format."""
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    """Deserialize numpy arrays from openpi_client format."""
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_packb = functools.partial(msgpack.packb, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class WebsocketClient:
    """Persistent websocket client with auto-reconnect."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._ws = None
        self._metadata = None

    def _connect(self):
        import websockets.sync.client as ws_client
        url = f"ws://{self.host}:{self.port}"
        logger.info(f"Connecting to {url}...")
        while True:
            try:
                self._ws = ws_client.connect(
                    url, max_size=None, compression=None,
                    ping_interval=120, ping_timeout=600,
                )
                # Receive server metadata
                self._metadata = _unpackb(self._ws.recv())
                logger.info(f"Connected. Server metadata: {self._metadata}")
                return
            except (ConnectionRefusedError, OSError):
                time.sleep(2)

    def infer(self, obs: dict) -> dict:
        if self._ws is None:
            self._connect()
        try:
            self._ws.send(_packb(obs))
            raw = self._ws.recv()
            if isinstance(raw, str):
                raise RuntimeError(f"Server error:\n{raw}")
            return _unpackb(raw)
        except Exception:
            logger.warning("Connection lost, reconnecting...")
            self._ws = None
            self._connect()
            self._ws.send(_packb(obs))
            raw = self._ws.recv()
            if isinstance(raw, str):
                raise RuntimeError(f"Server error:\n{raw}")
            return _unpackb(raw)

    def close(self):
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None


# ---------------------------------------------------------------------------
# Image utils
# ---------------------------------------------------------------------------

def resize_with_pad(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize image maintaining aspect ratio with padding."""
    from PIL import Image
    pil_img = Image.fromarray(img)
    w, h = pil_img.size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    result = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    result.paste(pil_img, (paste_x, paste_y))
    return np.array(result)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_env(
    env_name: str,
    split: str,
    log_dir: str,
    num_trials: int,
    replan_steps: int,
    client: WebsocketClient,
    seed: int,
    expected_action_chunk: int | None = None,
):
    import gymnasium as gym
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action

    # Model outputs actions in lerobot ordering; convert_action expects HDF5 ordering.
    # lerobot: [base_motion(4), control_mode(1), ee_pos(3), ee_rot(3), gripper(1)]
    # HDF5:    [ee_pos(3), ee_rot(3), gripper(1), base_motion(4), control_mode(1)]
    def reorder_lerobot_to_hdf5(action):
        return np.concatenate([
            action[5:8],    # ee_position
            action[8:11],   # ee_rotation
            action[11:12],  # gripper_close
            action[0:4],    # base_motion
            action[4:5],    # control_mode
        ])

    # RoboCasa 1.0.1 stores the longer benchmark horizon directly in the registry.
    horizon = int(get_task_horizon(env_name))

    now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_path = f"{log_dir}/{env_name}/{now}"

    # Skip if already evaluated
    for root, dirs, files in os.walk(os.path.dirname(log_path)):
        if "stats.json" in files:
            print(f"{env_name}/{split}, stats path exists, skipping.")
            return

    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)

    env = gym.make(f"robocasa/{env_name}", split=split, seed=seed)

    total_episodes, total_successes = 0, 0
    task_episodes, task_successes = 0, 0
    chunk_len_checked = False

    for episode_idx in tqdm.tqdm(range(num_trials), desc=env_name):
        obs, info = env.reset()
        task_lang = obs["annotation.human.task_description"]
        action_plan = collections.deque()

        t = 0
        replay_images = []

        logger.info(f"Starting episode {task_episodes + 1}, task: {task_lang}")

        while t < horizon:
            # Get images
            img_left = np.ascontiguousarray(obs["video.robot0_agentview_left"])
            img_wrist = np.ascontiguousarray(obs["video.robot0_eye_in_hand"])

            if not action_plan:
                # Build state vector (must match lerobot dataset ordering from PandaOmron_modality.json)
                # [base_pos(3), base_rot(4), ee_pos_rel(3), ee_rot_rel(4), gripper_qpos(2)] = 16d
                state = np.concatenate([
                    obs["state.base_position"],
                    obs["state.base_rotation"],
                    obs["state.end_effector_position_relative"],
                    obs["state.end_effector_rotation_relative"],
                    obs["state.gripper_qpos"],
                ], axis=0)

                # Also try to get right view
                element = {
                    "observation/image": img_left,
                    "observation/wrist_image": img_wrist,
                    "observation/state": state,
                    "prompt": task_lang,
                }

                if "video.robot0_agentview_right" in obs:
                    img_right = np.ascontiguousarray(obs["video.robot0_agentview_right"])
                    element["observation/right_image"] = img_right

                # Query model
                result = client.infer(element)
                action_chunk = result["actions"]
                if expected_action_chunk is not None and not chunk_len_checked:
                    got = len(action_chunk)
                    if got != expected_action_chunk:
                        raise RuntimeError(
                            f"Server returned {got} actions but --action_chunk expects {expected_action_chunk}. "
                            "Restart parallel_server_tshape.sh with matching ACTION_CHUNK or fix .server_tshape_info."
                        )
                    chunk_len_checked = True
                assert len(action_chunk) >= replan_steps, \
                    f"Want to replan every {replan_steps} steps, but got {len(action_chunk)} actions"
                action_plan.extend(action_chunk[:replan_steps])

            action = action_plan.popleft()
            action = reorder_lerobot_to_hdf5(action)
            action = convert_action(action)

            obs, reward, done, truncated, info = env.step(action)
            done = info["success"]

            replay_img = env.render()
            replay_img = np.ascontiguousarray(replay_img)
            if t % 2 == 0 or t == horizon - 1 or done:
                replay_images.append(replay_img)

            if done:
                task_successes += 1
                total_successes += 1
                break
            t += 1

        task_episodes += 1
        total_episodes += 1

        # Save replay video
        suffix = "success" if done else "failure"
        imageio.mimwrite(
            pathlib.Path(log_path) / f"rollout_{episode_idx}_{suffix}.mp4",
            [np.asarray(x) for x in replay_images],
            fps=20,
        )

        logger.info(f"Success: {done}")
        logger.info(f"Episodes: {total_episodes}, Successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

    logger.info(f"[{env_name}] Success rate: {float(total_successes) / float(total_episodes):.3f}")

    with open(os.path.join(log_path, "stats.json"), "w") as f:
        json.dump({
            "num_episodes": total_episodes,
            "success_rate": float(total_successes) / float(total_episodes),
        }, f, indent=4)

    env.env.close()
    del env.env
    del env


def parse_args():
    parser = argparse.ArgumentParser(description="GigaWorld-Policy RoboCasa Evaluation Client")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18055)
    parser.add_argument("--task_set", type=str, nargs="+",
                        default=["atomic_seen", "composite_seen", "composite_unseen"])
    parser.add_argument("--split", type=str, default="pretrain", choices=["pretrain", "target"])
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Directory to save evaluation logs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--worker_id", type=int, default=0, help="Worker index for parallel eval (0-based)")
    parser.add_argument("--num_workers", type=int, default=1, help="Total number of parallel workers")
    parser.add_argument(
        "--action_chunk",
        type=int,
        default=None,
        help="Expected number of actions per server response (from parallel_server_tshape / .server_tshape_info); "
        "validates first inference matches.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    np.random.seed(args.seed)

    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

    all_env_names = []
    for task in args.task_set:
        env_names = TASK_SET_REGISTRY[task]
        all_env_names.extend(env_names)

    # Shard tasks across workers
    my_env_names = [name for i, name in enumerate(all_env_names) if i % args.num_workers == args.worker_id]

    print("=" * 60)
    print(f"GigaWorld-Policy RoboCasa Client [worker {args.worker_id}/{args.num_workers}]")
    print("=" * 60)
    print(f"  Server:      {args.host}:{args.port}")
    print(f"  Task sets:   {args.task_set}")
    print(f"  Split:       {args.split}")
    print(f"  Num trials:  {args.num_trials}")
    print(f"  Replan:      every {args.replan_steps} steps")
    if args.action_chunk is not None:
        print(f"  Action chunk (expected): {args.action_chunk}")
    print(f"  Log dir:     {args.log_dir}")
    print(f"  Tasks total: {len(all_env_names)}, this worker: {len(my_env_names)}")
    print(f"  My tasks:    {my_env_names}")
    print("=" * 60)

    client = WebsocketClient(args.host, args.port)

    for env_name in my_env_names:
        try:
            eval_env(
                env_name=env_name,
                split=args.split,
                log_dir=args.log_dir,
                num_trials=args.num_trials,
                replan_steps=args.replan_steps,
                client=client,
                seed=args.seed,
                expected_action_chunk=args.action_chunk,
            )
        except Exception as e:
            logger.error(f"Error evaluating {env_name}: {e}")
            import traceback
            traceback.print_exc()

    client.close()
    print(f"Worker {args.worker_id} complete.")


if __name__ == "__main__":
    main()
