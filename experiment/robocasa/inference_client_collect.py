"""GigaWorld-Policy inference client for RoboCasa evaluation with optional data collection.

Connects to the inference server and runs closed-loop evaluation
in RoboCasa environments, following the same protocol as openpi.

When --collect_data is set, saves rollout trajectories in lerobot v2.1 format:
  data/chunk-000/episode_XXXXXX.parquet
  videos/chunk-000/{camera}/episode_XXXXXX.mp4
  meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json, embodiment.json, ...}
"""

import argparse
import collections
import json
import logging
import os
import pathlib
import time
from datetime import datetime

import imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
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
# Lerobot data collection helpers
# ---------------------------------------------------------------------------

VIDEO_CAMERAS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
]

IMG_SIZE = 256  # target resolution for saved videos
FPS = 20


def _encode_video_ffmpeg(frames: list[np.ndarray], out_path: str, fps: int = FPS):
    """Encode a list of RGB uint8 frames to mp4 video."""
    if not frames:
        logger.warning(f"No frames to encode for {out_path}")
        return
    frames = [np.ascontiguousarray(f, dtype=np.uint8) for f in frames]
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps)


def _build_info_json(total_episodes: int, total_frames: int, total_tasks: int) -> dict:
    """Build info.json matching lerobot v2.1 format."""
    video_info = {
        "video.fps": FPS, "video.codec": "h264", "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False, "has_audio": False,
    }
    cam_feature = lambda: {
        "dtype": "video",
        "shape": [IMG_SIZE, IMG_SIZE, 3],
        "names": ["height", "width", "channel"],
        "video_info": video_info,
        "info": {
            "video.height": IMG_SIZE, "video.width": IMG_SIZE,
            "video.codec": "h264", "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False, "video.fps": FPS,
            "video.channels": 3, "has_audio": False,
        },
    }
    return {
        "codebase_version": "v2.1",
        "robot_type": "PandaOmron",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(VIDEO_CAMERAS),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.robot0_eye_in_hand": cam_feature(),
            "observation.images.robot0_agentview_left": cam_feature(),
            "observation.images.robot0_agentview_right": cam_feature(),
            "annotation.human.task_description": {"dtype": "int64", "shape": [1]},
            "annotation.human.task_name": {"dtype": "int64", "shape": [1]},
            "observation.state": {"dtype": "float64", "shape": [16]},
            "action": {"dtype": "float64", "shape": [12]},
            "next.reward": {"dtype": "float32", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }


MODALITY_JSON = {
    "state": {
        "base_position": {"original_key": "observation.state", "start": 0, "end": 3},
        "base_rotation": {"original_key": "observation.state", "start": 3, "end": 7},
        "end_effector_position_relative": {"original_key": "observation.state", "start": 7, "end": 10},
        "end_effector_rotation_relative": {"original_key": "observation.state", "start": 10, "end": 14},
        "gripper_qpos": {"original_key": "observation.state", "start": 14, "end": 16},
    },
    "action": {
        "base_motion": {"original_key": "action", "start": 0, "end": 4},
        "control_mode": {"original_key": "action", "start": 4, "end": 5},
        "end_effector_position": {"original_key": "action", "start": 5, "end": 8},
        "end_effector_rotation": {"original_key": "action", "start": 8, "end": 11},
        "gripper_close": {"original_key": "action", "start": 11, "end": 12},
    },
    "video": {
        "robot0_eye_in_hand": {"original_key": "observation.images.robot0_eye_in_hand"},
        "robot0_agentview_left": {"original_key": "observation.images.robot0_agentview_left"},
        "robot0_agentview_right": {"original_key": "observation.images.robot0_agentview_right"},
    },
    "annotation": {
        "human.task_description": {"original_key": "annotation.human.task_description"},
    },
}

EMBODIMENT_JSON = {
    "robot_name": "PandaOmron",
    "robot_type": "PandaOmron",
    "record_frequency": 20.0,
    "body_controller_frequency": 20.0,
    "hand_controller_frequency": 20.0,
    "embodiment_tag": "robocasa_panda_omron",
}


def _save_episode_parquet(
    out_dir: str,
    episode_index: int,
    global_index_offset: int,
    task_index: int,
    task_name_index: int,
    states: list[np.ndarray],
    actions: list[np.ndarray],
    rewards: list[float],
    dones: list[bool],
):
    """Save one episode as a parquet file."""
    n = len(states)
    data = {
        "observation.state": [s.tolist() for s in states],
        "action": [a.tolist() for a in actions],
        "next.reward": [np.float32(r) for r in rewards],
        "next.done": dones,
        "annotation.human.task_description": [np.int64(task_index)] * n,
        "annotation.human.task_name": [np.int64(task_name_index)] * n,
        "timestamp": [np.float32(i / FPS) for i in range(n)],
        "frame_index": list(range(n)),
        "episode_index": [episode_index] * n,
        "index": list(range(global_index_offset, global_index_offset + n)),
        "task_index": [task_index] * n,
    }
    table = pa.table(data)
    parquet_dir = os.path.join(out_dir, "data", "chunk-000")
    os.makedirs(parquet_dir, exist_ok=True)
    pq.write_table(table, os.path.join(parquet_dir, f"episode_{episode_index:06d}.parquet"))


def _compute_episode_stats(states, actions, rewards, dones, n_frames,
                           episode_index, global_index_offset, task_index, task_name_index,
                           img_stats: dict) -> dict:
    """Compute per-episode statistics matching episodes_stats.jsonl format."""
    def _arr_stats(arr, count):
        arr = np.array(arr)
        return {
            "min": arr.min(axis=0).tolist() if arr.ndim > 1 else [arr.min().item()],
            "max": arr.max(axis=0).tolist() if arr.ndim > 1 else [arr.max().item()],
            "mean": arr.mean(axis=0).tolist() if arr.ndim > 1 else [arr.mean().item()],
            "std": arr.std(axis=0).tolist() if arr.ndim > 1 else [arr.std().item()],
            "count": [count],
        }

    n = n_frames
    stats = {}
    # Image stats (subsampled)
    for cam_key, cam_stats in img_stats.items():
        stats[cam_key] = cam_stats
    # Annotations
    stats["annotation.human.task_description"] = _arr_stats([task_index] * n, n)
    stats["annotation.human.task_name"] = _arr_stats([task_name_index] * n, n)
    # State & action
    stats["observation.state"] = _arr_stats(states, n)
    stats["action"] = _arr_stats(actions, n)
    # Reward & done
    stats["next.reward"] = _arr_stats(rewards, n)
    stats["next.done"] = {
        "min": [False], "max": [any(dones)],
        "mean": [sum(dones) / n], "std": [np.array(dones, dtype=float).std().item()],
        "count": [n],
    }
    # Indices
    stats["timestamp"] = _arr_stats([i / FPS for i in range(n)], n)
    stats["frame_index"] = _arr_stats(list(range(n)), n)
    stats["episode_index"] = _arr_stats([episode_index] * n, n)
    stats["index"] = _arr_stats(list(range(global_index_offset, global_index_offset + n)), n)
    stats["task_index"] = _arr_stats([task_index] * n, n)
    return {"episode_index": episode_index, "stats": stats}


def _compute_image_stats_subsampled(frames: list[np.ndarray], n_sample: int = 100) -> dict:
    """Compute image stats from a subsample of frames (normalized to [0,1])."""
    indices = np.linspace(0, len(frames) - 1, min(n_sample, len(frames)), dtype=int)
    sampled = np.array([frames[i] for i in indices], dtype=np.float32) / 255.0
    # Shape: (N, H, W, 3) -> per-channel stats
    per_channel_min = sampled.min(axis=(0, 1, 2))  # (3,)
    per_channel_max = sampled.max(axis=(0, 1, 2))
    per_channel_mean = sampled.mean(axis=(0, 1, 2))
    per_channel_std = sampled.std(axis=(0, 1, 2))
    return {
        "min": [[[v]] for v in per_channel_min.tolist()],
        "max": [[[v]] for v in per_channel_max.tolist()],
        "mean": [[[v]] for v in per_channel_mean.tolist()],
        "std": [[[v]] for v in per_channel_std.tolist()],
        "count": [min(n_sample, len(frames))],
    }


class LerobotCollector:
    """Accumulates episodes and writes lerobot v2.1 dataset at the end."""

    def __init__(self, out_dir: str, env_name: str):
        self.out_dir = out_dir
        self.env_name = env_name
        self.episodes = []  # list of episode metadata dicts
        self.tasks = {}     # task_description -> task_index
        self.task_name_index = None  # index for env_name in tasks
        self.global_frame_count = 0

    def _get_task_index(self, task_desc: str) -> int:
        if task_desc not in self.tasks:
            self.tasks[task_desc] = len(self.tasks)
        return self.tasks[task_desc]

    def save_episode(
        self,
        episode_index: int,
        task_lang: str,
        states: list[np.ndarray],
        actions: list[np.ndarray],
        rewards: list[float],
        dones: list[bool],
        images_left: list[np.ndarray],
        images_right: list[np.ndarray],
        images_wrist: list[np.ndarray],
    ):
        n = len(states)
        task_index = self._get_task_index(task_lang)
        # Ensure env_name is also registered as a task
        if self.task_name_index is None:
            self.task_name_index = len(self.tasks)
            self.tasks[self.env_name] = self.task_name_index
        task_name_idx = self.task_name_index

        offset = self.global_frame_count

        # 1. Save parquet
        _save_episode_parquet(
            self.out_dir, episode_index, offset, task_index, task_name_idx,
            states, actions, rewards, dones,
        )

        # 2. Save videos (h264)
        cam_frames = {
            "observation.images.robot0_agentview_left": images_left,
            "observation.images.robot0_agentview_right": images_right,
            "observation.images.robot0_eye_in_hand": images_wrist,
        }
        img_stats = {}
        for cam_key, frames in cam_frames.items():
            video_path = os.path.join(
                self.out_dir, "videos", "chunk-000", cam_key,
                f"episode_{episode_index:06d}.mp4",
            )
            _encode_video_ffmpeg(frames, video_path, fps=FPS)
            img_stats[cam_key] = _compute_image_stats_subsampled(frames)

        # 3. Compute episode stats
        ep_stats = _compute_episode_stats(
            states, actions, rewards, dones, n,
            episode_index, offset, task_index, task_name_idx, img_stats,
        )

        self.episodes.append({
            "episode_index": episode_index,
            "tasks": [task_lang],
            "length": n,
            "stats": ep_stats["stats"],
        })
        self.global_frame_count += n
        logger.info(f"[Collect] Saved episode {episode_index} ({n} frames) to {self.out_dir}")

    def finalize(self):
        """Write all meta files after all episodes are saved."""
        # Guard: don't clobber an existing dataset's meta when this run produced
        # no new episodes (e.g. the env was skipped because stats.json already
        # exists). Without this, re-running with --collect_data would overwrite
        # info.json / episodes.jsonl / etc. with empty manifests, breaking the
        # previously collected lerobot dataset on disk.
        if not self.episodes:
            logger.info(
                f"[Collect] Skip finalize for {self.env_name}: no new episodes "
                f"(preserving any existing dataset at {self.out_dir})"
            )
            return

        meta_dir = os.path.join(self.out_dir, "meta")
        os.makedirs(meta_dir, exist_ok=True)

        total_episodes = len(self.episodes)
        total_frames = self.global_frame_count
        total_tasks = len(self.tasks)

        # info.json
        with open(os.path.join(meta_dir, "info.json"), "w") as f:
            json.dump(_build_info_json(total_episodes, total_frames, total_tasks), f, indent=4)

        # modality.json
        with open(os.path.join(meta_dir, "modality.json"), "w") as f:
            json.dump(MODALITY_JSON, f, indent=4)

        # embodiment.json
        with open(os.path.join(meta_dir, "embodiment.json"), "w") as f:
            json.dump(EMBODIMENT_JSON, f, indent=4)

        # tasks.jsonl
        with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
            for task_desc, task_idx in sorted(self.tasks.items(), key=lambda x: x[1]):
                f.write(json.dumps({"task_index": task_idx, "task": task_desc}) + "\n")

        # episodes.jsonl
        with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
            for ep in self.episodes:
                f.write(json.dumps({
                    "episode_index": ep["episode_index"],
                    "tasks": ep["tasks"],
                    "length": ep["length"],
                }) + "\n")

        # episodes_stats.jsonl
        with open(os.path.join(meta_dir, "episodes_stats.jsonl"), "w") as f:
            for ep in self.episodes:
                f.write(json.dumps({
                    "episode_index": ep["episode_index"],
                    "stats": ep["stats"],
                }) + "\n")

        # stats.json (global stats across all episodes)
        global_stats = self._compute_global_stats()
        with open(os.path.join(meta_dir, "stats.json"), "w") as f:
            json.dump(global_stats, f, indent=4)

        logger.info(f"[Collect] Finalized dataset: {total_episodes} episodes, {total_frames} frames, {total_tasks} tasks")

    def _compute_global_stats(self) -> dict:
        """Aggregate per-episode stats into global stats."""
        if not self.episodes:
            return {}

        all_keys = self.episodes[0]["stats"].keys()
        global_stats = {}

        for key in all_keys:
            ep_stats_list = [ep["stats"][key] for ep in self.episodes if key in ep["stats"]]
            if not ep_stats_list:
                continue

            # For numeric stats, compute weighted aggregates
            mins = [s["min"] for s in ep_stats_list]
            maxs = [s["max"] for s in ep_stats_list]
            means = [s["mean"] for s in ep_stats_list]
            counts = [s["count"][0] for s in ep_stats_list]

            total_count = sum(counts)
            if total_count == 0:
                continue

            # Element-wise min/max
            g_min = np.minimum.reduce([np.array(m) for m in mins]).tolist()
            g_max = np.maximum.reduce([np.array(m) for m in maxs]).tolist()

            # Weighted mean
            weighted_sum = sum(np.array(m) * c for m, c in zip(means, counts))
            g_mean = (weighted_sum / total_count).tolist()

            # Approximate std (pooled)
            stds = [s["std"] for s in ep_stats_list]
            var_sum = sum(
                (np.array(s) ** 2 + np.array(m) ** 2) * c
                for s, m, c in zip(stds, means, counts)
            )
            g_std = np.sqrt(var_sum / total_count - np.array(g_mean) ** 2).tolist()

            # q01, q99 approximated by min/max
            global_stats[key] = {
                "mean": g_mean, "std": g_std,
                "min": g_min, "max": g_max,
                "q01": g_min, "q99": g_max,
            }

        return global_stats


# ---------------------------------------------------------------------------
# Evaluation (with optional data collection)
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
    collector: LerobotCollector | None = None,
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

    task_horizon = get_task_horizon(env_name)
    horizon = int(task_horizon * 1.5)

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

        # Data collection buffers
        collect_states = []
        collect_actions = []
        collect_rewards = []
        collect_dones = []
        collect_imgs_left = []
        collect_imgs_right = []
        collect_imgs_wrist = []

        logger.info(f"Starting episode {task_episodes + 1}, task: {task_lang}")

        while t < horizon:
            # Get images
            img_left = np.ascontiguousarray(obs["video.robot0_agentview_left"])
            img_wrist = np.ascontiguousarray(obs["video.robot0_eye_in_hand"])

            # Build state vector (must match lerobot dataset ordering from PandaOmron_modality.json)
            # [base_pos(3), base_rot(4), ee_pos_rel(3), ee_rot_rel(4), gripper_qpos(2)] = 16d
            state = np.concatenate([
                obs["state.base_position"],
                obs["state.base_rotation"],
                obs["state.end_effector_position_relative"],
                obs["state.end_effector_rotation_relative"],
                obs["state.gripper_qpos"],
            ], axis=0)

            if not action_plan:
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

            # Get raw lerobot-ordered action BEFORE reorder (12d)
            raw_action = action_plan.popleft()

            # Collect data for this step
            if collector is not None:
                collect_states.append(state.astype(np.float64).copy())
                collect_actions.append(np.array(raw_action, dtype=np.float64).copy())
                # Resize images to 256x256 for collection
                collect_imgs_left.append(resize_with_pad(img_left, IMG_SIZE, IMG_SIZE))
                collect_imgs_wrist.append(resize_with_pad(img_wrist, IMG_SIZE, IMG_SIZE))
                if "video.robot0_agentview_right" in obs:
                    img_right_raw = np.ascontiguousarray(obs["video.robot0_agentview_right"])
                    collect_imgs_right.append(resize_with_pad(img_right_raw, IMG_SIZE, IMG_SIZE))
                else:
                    # Fallback: black image if right camera not available
                    collect_imgs_right.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

            action = reorder_lerobot_to_hdf5(raw_action)
            action = convert_action(action)

            obs, reward, done, truncated, info = env.step(action)
            done = info["success"]

            # Collect reward/done
            if collector is not None:
                collect_rewards.append(float(reward))
                collect_dones.append(bool(done))

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

        # Save replay video (always, same as original)
        suffix = "success" if done else "failure"
        imageio.mimwrite(
            pathlib.Path(log_path) / f"rollout_{episode_idx}_{suffix}.mp4",
            [np.asarray(x) for x in replay_images],
            fps=20,
        )

        # Save collected data
        if collector is not None and len(collect_states) > 0:
            collector.save_episode(
                episode_index=len(collector.episodes),
                task_lang=task_lang,
                states=collect_states,
                actions=collect_actions,
                rewards=collect_rewards,
                dones=collect_dones,
                images_left=collect_imgs_left,
                images_right=collect_imgs_right,
                images_wrist=collect_imgs_wrist,
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
    parser = argparse.ArgumentParser(description="GigaWorld-Policy RoboCasa Evaluation Client (with data collection)")
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
    # Data collection args
    parser.add_argument("--collect_data", action="store_true",
                        help="Enable lerobot v2.1 data collection during evaluation")
    parser.add_argument("--collect_dir", type=str, default=None,
                        help="Root directory for collected lerobot datasets. "
                        "Each env gets a subdirectory: <collect_dir>/<env_name>/lerobot/")
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
    if args.collect_data:
        print(f"  *** DATA COLLECTION ENABLED ***")
    print("=" * 60)
    print(f"  Server:      {args.host}:{args.port}")
    print(f"  Task sets:   {args.task_set}")
    print(f"  Split:       {args.split}")
    print(f"  Num trials:  {args.num_trials}")
    print(f"  Replan:      every {args.replan_steps} steps")
    if args.action_chunk is not None:
        print(f"  Action chunk (expected): {args.action_chunk}")
    print(f"  Log dir:     {args.log_dir}")
    if args.collect_data:
        print(f"  Collect dir: {args.collect_dir}")
    print(f"  Tasks total: {len(all_env_names)}, this worker: {len(my_env_names)}")
    print(f"  My tasks:    {my_env_names}")
    print("=" * 60)

    if args.collect_data and args.collect_dir is None:
        raise ValueError("--collect_dir is required when --collect_data is set")

    client = WebsocketClient(args.host, args.port)

    for env_name in my_env_names:
        collector = None
        if args.collect_data:
            collect_out = os.path.join(args.collect_dir, env_name, "lerobot")
            collector = LerobotCollector(out_dir=collect_out, env_name=env_name)

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
                collector=collector,
            )
        except Exception as e:
            logger.error(f"Error evaluating {env_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if collector is not None:
                try:
                    collector.finalize()
                except Exception as e:
                    logger.error(f"Error finalizing collector for {env_name}: {e}")

    client.close()
    print(f"Worker {args.worker_id} complete.")


if __name__ == "__main__":
    main()

