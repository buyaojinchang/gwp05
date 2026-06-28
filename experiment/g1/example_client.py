"""
VLA inference runner — NO ROS 2 DEPENDENCY.

Runs an Isaac-GR00T VLA policy against the Sonic whole-body control stack.
All communication uses ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (from C++ zmq_output_handler)
  2. Actions out  -> ZMQ PUB (latent protocol v4: motion token + hand joints)
  3. Camera       -> ZMQ/TCP via ComposedCameraClientSensor
  4. Keyboard     -> ZMQ SUB via ZMQKeyboardSubscriber

Uses the Isaac-GR00T PolicyClient (ZMQ REQ/REP) to communicate with a
running PolicyServer.

Keyboard commands (received via ZMQ from the standalone keyboard publisher):
  p  -> pause / resume the policy loop
  k  -> start / stop the C++ control loop
  i  -> send initial pose and switch to POSE mode
  t  -> change prompt at runtime (publisher sends ``prompt:<text>``)
  [  -> toggle left hand open/closed for initial pose
  ]  -> toggle right hand open/closed for initial pose
  c  -> start recording (handled by data exporter if running)
  s  -> stop recording success (handled by data exporter)
  f  -> stop recording failure (handled by data exporter)
"""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any

import numpy as np
import tyro
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.data_collection.keyboard_subscriber import (
    DEFAULT_ZMQ_KEYBOARD_PORT,
    ZMQKeyboardSubscriber,
)
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.data_collection.zmq_state_subscriber import (
    ZMQStateSubscriber,
    poll_robot_config_zmq,
)
from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN
from gear_sonic.utils.inference.vla_utils import (
    calculate_latency_compensated_index,
    concat_action,
    prepare_observation_for_eval,
    should_trigger_new_inference,
)
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)


@dataclass
class InferenceConfig:
    """CLI config for the VLA inference runner."""

    # Policy server (Isaac-GR00T PolicyServer)
    host: str = "localhost"
    """The host address of the Isaac-GR00T PolicyServer."""

    port: int = 5550
    """The port of the Isaac-GR00T PolicyServer."""

    # Control
    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 40
    """Action horizon of the VLA policy (number of future actions per inference)."""

    rate: float = 1 / 1.0
    """Rate at which we run the forward pass of the VLA policy (Hz)."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # ZMQ: Robot state (from C++ zmq_output_handler, g1_debug topic)
    state_zmq_host: str = "localhost"
    """ZMQ host for robot state (g1_debug topic from C++ deploy)."""

    state_zmq_port: int = 5557
    """ZMQ port for robot state (same socket as robot_config topic)."""

    # ZMQ: Action output (latent actions to C++ control loop)
    action_zmq_host: str = "localhost"
    """ZMQ host for action output (PUB socket)."""

    action_zmq_port: int = 5556
    """ZMQ port for action output."""

    # ZMQ: Keyboard input
    keyboard_zmq_host: str = "localhost"
    """ZMQ host for keyboard input."""

    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT
    """ZMQ port for keyboard input."""

    # Embodiment
    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag for policy inference."""

    hand_type: str = "auto"
    """Robot-side hand backend: auto, dex3, or inspire."""

    # Prompt / eval
    prompt: str = "demo"
    """The language prompt for the VLA policy."""

    # Debug
    verbose_timing: bool = False
    """Whether to always print timing info (not just when loop is slow)."""

    debug_save_hand_dir: str | None = None
    """Optional directory for per-frame hand debug captures."""

    debug_save_hand_every: int = 1
    """Save one hand debug capture every N sent action frames."""

    debug_save_hand_max: int = 0
    """Maximum number of hand debug captures to save. 0 means unlimited."""


def print_green(x):
    print(f"\033[92m{x}\033[0m")


class _LocalPolicyClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
        strict: bool = False,
    ):
        del strict
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        if hasattr(self, "socket"):
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    @staticmethod
    def _to_bytes(data: Any) -> bytes:
        import msgpack_numpy as mnp

        return mnp.packb(data, default=mnp.encode)

    @staticmethod
    def _from_bytes(data: bytes) -> Any:
        import msgpack_numpy as mnp

        return mnp.unpackb(data, object_hook=mnp.decode, raw=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        try:
            self.socket.send(self._to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again:
            self._init_socket()
            raise

        if message == b"ERROR":
            raise RuntimeError("Server error. Make sure we are running the correct policy server.")

        response = self._from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()
            return False

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        return tuple(response)

    def close(self):
        self.socket.close(linger=0)
        self.context.term()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _resolve_policy_client_class():
    try:
        from gr00t.policy.server_client import PolicyClient

        return PolicyClient
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] != "gr00t":
            raise
        print("gr00t package not found in current environment; using local ZMQ PolicyClient fallback.")
        return _LocalPolicyClient


# ---------------------------------------------------------------------------
# Action packing (latent protocol v4)
# ---------------------------------------------------------------------------


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray = None,
    right_hand_joints: np.ndarray = None,
) -> bytes:
    """Pack a single motion-token action into a ZMQ message (Protocol v4).

    Args:
        motion_token: Shape ``[64]`` (flat) or ``[1, 64]``.
        frame_index:  Shape ``[1]``.
        left_hand_joints:  Shape ``[7]`` or ``[1, 7]``, optional.
        right_hand_joints: Shape ``[7]`` or ``[1, 7]``, optional.

    Returns:
        Packed ZMQ message bytes.
    """
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)

    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }

    if left_hand_joints is not None:
        left_hand_joints = np.asarray(left_hand_joints, dtype=np.float32)
        if left_hand_joints.ndim == 1:
            if left_hand_joints.shape[0] != 7:
                raise ValueError(
                    f"left_hand_joints must have shape [7], got {left_hand_joints.shape}"
                )
            left_hand_joints = left_hand_joints.reshape(1, 7)
        pose_data["left_hand_joints"] = left_hand_joints

    if right_hand_joints is not None:
        right_hand_joints = np.asarray(right_hand_joints, dtype=np.float32)
        if right_hand_joints.ndim == 1:
            if right_hand_joints.shape[0] != 7:
                raise ValueError(
                    f"right_hand_joints must have shape [7], got {right_hand_joints.shape}"
                )
            right_hand_joints = right_hand_joints.reshape(1, 7)
        pose_data["right_hand_joints"] = right_hand_joints

    return pack_pose_message(pose_data, topic="pose", version=4)


def get_action_field(action_dict: dict, key: str):
    """Get action field from dict, checking both with and without 'action.' prefix."""
    value = action_dict.get(key)
    if value is not None:
        return value
    value = action_dict.get(f"action.{key}")
    if value is not None:
        return value
    raise AssertionError(
        f"Required action field '{key}' (or 'action.{key}') not found in processed_action. "
        f"Available keys: {list(action_dict.keys())}"
    )


def _get_optional_action_field(action_dict: dict, key: str):
    value = action_dict.get(key)
    if value is not None:
        return value
    return action_dict.get(f"action.{key}")


HAND_BINARY_CLOSE_THRESHOLD = 0.5
DEX3_FULL_CLOSE_ABS = 1.5


def _dex3_hand_close_binary(joints) -> float:
    q = np.asarray(joints, dtype=np.float32).reshape(-1)
    if q.size == 0:
        return 0.0
    close_amount = np.clip(np.max(np.abs(q)) / DEX3_FULL_CLOSE_ABS, 0.0, 1.0)
    return float(close_amount >= HAND_BINARY_CLOSE_THRESHOLD)


def _inspire_hand_close_binary(joints) -> float:
    q = np.asarray(joints, dtype=np.float32).reshape(-1)
    if q.size == 0:
        return 0.0
    close_amount = np.clip(np.max(1.0 - np.clip(q[:6], 0.0, 1.0)), 0.0, 1.0)
    return float(close_amount >= HAND_BINARY_CLOSE_THRESHOLD)


def _hand_close_binary(joints, hand_type: str) -> float:
    normalized_hand_type = hand_type.lower()
    if normalized_hand_type == "inspire":
        return _inspire_hand_close_binary(joints)
    return _dex3_hand_close_binary(joints)


def _build_action_state_token(
    token_state_value,
    hand_binary_state: np.ndarray,
) -> np.ndarray:
    token_state = np.asarray(token_state_value, dtype=np.float32).reshape(-1)
    hand_binary_state = np.asarray(hand_binary_state, dtype=np.float32).reshape(-1)

    if hand_binary_state.shape[0] != 2:
        raise ValueError(
            f"hand_binary_state must have shape [2], got {hand_binary_state.shape}"
        )

    if token_state.shape[0] == 64:
        return np.concatenate([token_state, hand_binary_state], axis=0).astype(
            np.float32, copy=False
        )

    if token_state.shape[0] == 66:
        action_state = token_state.astype(np.float32, copy=True)
        action_state[-2:] = hand_binary_state
        return action_state

    raise ValueError(
        f"Expected token_state dim 64 or 66, got {token_state.shape[0]}"
    )


def _select_action_timestep(
    action_value, current_idx: int, expected_dim: int, field_name: str
) -> np.ndarray:
    arr = np.asarray(action_value, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]

    if arr.ndim == 2:
        if arr.shape[-1] != expected_dim:
            raise ValueError(
                f"{field_name} must have trailing dim {expected_dim}, got {arr.shape}"
            )
        arr = arr[min(current_idx, arr.shape[0] - 1)]
    elif arr.ndim == 1:
        if arr.shape[0] != expected_dim:
            raise ValueError(f"{field_name} must have shape [{expected_dim}], got {arr.shape}")
    else:
        raise ValueError(f"{field_name} must be rank 1, 2, or 3, got {arr.ndim}")

    return arr.astype(np.float32)


def _extract_hand_debug_signals(
    processed_action: dict, current_idx: int, hand_type: str
) -> dict[str, Any]:
    left_hand_value = _get_optional_action_field(processed_action, "left_hand_joints")
    right_hand_value = _get_optional_action_field(processed_action, "right_hand_joints")
    if left_hand_value is not None and right_hand_value is not None:
        raw_left_hand_joints = _select_action_timestep(
            left_hand_value, current_idx, 7, "left_hand_joints"
        )
        raw_right_hand_joints = _select_action_timestep(
            right_hand_value, current_idx, 7, "right_hand_joints"
        )
        return (
            {
                "source": "left_hand_joints",
                "raw_hand_binary": None,
                "thresholded_hand_binary": np.array(
                    [
                        _hand_close_binary(raw_left_hand_joints, hand_type),
                        _hand_close_binary(raw_right_hand_joints, hand_type),
                    ],
                    dtype=np.float32,
                ),
                "raw_left_hand_joints": raw_left_hand_joints,
                "raw_right_hand_joints": raw_right_hand_joints,
                "sent_left_hand_joints": raw_left_hand_joints.copy(),
                "sent_right_hand_joints": raw_right_hand_joints.copy(),
            }
        )

    hand_binary_value = None
    hand_binary_field_name = None
    for candidate_key in ("hand_binary_action", "hand_binary"):
        hand_binary_value = _get_optional_action_field(processed_action, candidate_key)
        if hand_binary_value is not None:
            hand_binary_field_name = candidate_key
            break

    if hand_binary_value is None:
        raise AssertionError(
            "Processed action did not contain left/right hand joints, hand_binary_action, "
            f"or hand_binary outputs. Available keys: {list(processed_action.keys())}"
        )

    hand_binary = _select_action_timestep(
        hand_binary_value,
        current_idx,
        2,
        hand_binary_field_name,
    )
    thresholded_hand_binary = (hand_binary >= HAND_BINARY_CLOSE_THRESHOLD).astype(np.float32)
    sent_left_hand_joints = _binary_to_hand_joints(float(hand_binary[0]), "L")
    sent_right_hand_joints = _binary_to_hand_joints(float(hand_binary[1]), "R")
    return {
        "source": hand_binary_field_name,
        "raw_hand_binary": hand_binary,
        "thresholded_hand_binary": thresholded_hand_binary,
        "raw_left_hand_joints": None,
        "raw_right_hand_joints": None,
        "sent_left_hand_joints": sent_left_hand_joints,
        "sent_right_hand_joints": sent_right_hand_joints,
    }


def _extract_hand_joint_actions(
    processed_action: dict, current_idx: int, hand_type: str
) -> tuple[np.ndarray, np.ndarray]:
    hand_debug_signals = _extract_hand_debug_signals(
        processed_action,
        current_idx,
        hand_type,
    )
    return (
        hand_debug_signals["sent_left_hand_joints"],
        hand_debug_signals["sent_right_hand_joints"],
    )


def _collect_state_hand_debug(state_msg: dict | None, hand_type: str) -> dict[str, np.ndarray]:
    if state_msg is None:
        return {}

    left_hand_q = np.asarray(state_msg.get("left_hand_q", np.zeros(7)), dtype=np.float32)
    right_hand_q = np.asarray(state_msg.get("right_hand_q", np.zeros(7)), dtype=np.float32)
    last_left_hand_action = np.asarray(
        state_msg.get("last_left_hand_action", np.zeros(7)),
        dtype=np.float32,
    )
    last_right_hand_action = np.asarray(
        state_msg.get("last_right_hand_action", np.zeros(7)),
        dtype=np.float32,
    )

    return {
        "state_left_hand_q": left_hand_q,
        "state_right_hand_q": right_hand_q,
        "state_last_left_hand_action": last_left_hand_action,
        "state_last_right_hand_action": last_right_hand_action,
        "state_hand_binary_from_q": np.array(
            [
                _hand_close_binary(left_hand_q, hand_type),
                _hand_close_binary(right_hand_q, hand_type),
            ],
            dtype=np.float32,
        ),
        "state_hand_binary_from_last_action": np.array(
            [
                _hand_close_binary(last_left_hand_action, hand_type),
                _hand_close_binary(last_right_hand_action, hand_type),
            ],
            dtype=np.float32,
        ),
    }


def _maybe_save_hand_debug_capture(
    *,
    debug_run_dir: Path | None,
    debug_save_every: int,
    debug_save_max: int,
    debug_saved_count: int,
    frame_index: int,
    current_idx: int,
    hand_type: str,
    prompt: str,
    motion_token: np.ndarray,
    hand_debug_signals: dict[str, Any],
    state_msg: dict | None,
) -> int:
    if debug_run_dir is None:
        return debug_saved_count

    if debug_save_every <= 0:
        return debug_saved_count

    if frame_index % debug_save_every != 0:
        return debug_saved_count

    if debug_save_max > 0 and debug_saved_count >= debug_save_max:
        return debug_saved_count

    arrays_to_save = {
        "motion_token": np.asarray(motion_token, dtype=np.float32),
        "thresholded_hand_binary": np.asarray(
            hand_debug_signals["thresholded_hand_binary"], dtype=np.float32
        ),
        "sent_left_hand_joints": np.asarray(
            hand_debug_signals["sent_left_hand_joints"], dtype=np.float32
        ),
        "sent_right_hand_joints": np.asarray(
            hand_debug_signals["sent_right_hand_joints"], dtype=np.float32
        ),
    }

    raw_hand_binary = hand_debug_signals.get("raw_hand_binary")
    if raw_hand_binary is not None:
        arrays_to_save["raw_hand_binary"] = np.asarray(raw_hand_binary, dtype=np.float32)

    raw_left_hand_joints = hand_debug_signals.get("raw_left_hand_joints")
    if raw_left_hand_joints is not None:
        arrays_to_save["raw_left_hand_joints"] = np.asarray(
            raw_left_hand_joints, dtype=np.float32
        )

    raw_right_hand_joints = hand_debug_signals.get("raw_right_hand_joints")
    if raw_right_hand_joints is not None:
        arrays_to_save["raw_right_hand_joints"] = np.asarray(
            raw_right_hand_joints, dtype=np.float32
        )

    arrays_to_save.update(_collect_state_hand_debug(state_msg, hand_type))

    frame_stem = f"frame_{frame_index:06d}"
    np.savez(debug_run_dir / f"{frame_stem}.npz", **arrays_to_save)

    meta = {
        "frame_index": int(frame_index),
        "current_idx": int(current_idx),
        "hand_type": hand_type,
        "hand_output_source": hand_debug_signals["source"],
        "prompt": prompt,
        "has_state_msg": state_msg is not None,
    }
    (debug_run_dir / f"{frame_stem}.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    return debug_saved_count + 1


# ---------------------------------------------------------------------------
# Observation / inference helpers
# ---------------------------------------------------------------------------


def prepare_observation_from_sensors(
    camera_subscriber,
    state_subscriber,
    robot_model,
    language_prompt: str,
    hand_type: str = "dex3",
    log_errors: bool = False,
):
    """Read sensors and prepare observation for the VLA policy.

    Returns:
        observation dict, or None if sensor data not yet available.
    """
    camera_msg = camera_subscriber.read()
    if camera_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for camera msg..", flush=True)
        return None

    state_msg = state_subscriber.get_msg()
    if state_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for state msg..", flush=True)
        return None

    cam_img = camera_msg["images"]["ego_view"]

    # Copy index finger data to middle finger (hardware coupling)
    state_msg["left_hand_q"][5] = state_msg["left_hand_q"][3]
    state_msg["left_hand_q"][6] = state_msg["left_hand_q"][4]

    qpos = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=state_msg["body_q"],
        left_hand_actuated_joint_values=state_msg["left_hand_q"],
        right_hand_actuated_joint_values=state_msg["right_hand_q"],
    )

    video = {"ego_view": cam_img[np.newaxis, np.newaxis]}
    if "left_wrist" in camera_msg["images"]:
        video["left_wrist"] = camera_msg["images"]["left_wrist"][np.newaxis, np.newaxis]
    if "right_wrist" in camera_msg["images"]:
        video["wrist_view"] = camera_msg["images"]["right_wrist"][np.newaxis, np.newaxis]

    observation = {
        "video": video,
        "state": {},
        "language": {
            "annotation.human.task_description": [[language_prompt]],
        },
        "q": np.asarray(qpos, dtype=np.float32)[np.newaxis, np.newaxis],
        "timestamps": camera_msg["timestamps"]["ego_view"],
    }

    observation = prepare_observation_for_eval(robot_model, observation)

    # Projected gravity for Sonic latent embodiment
    assert "base_quat" in state_msg, "base_quat not found in state_msg"
    base_quat = np.asarray(state_msg["base_quat"], dtype=np.float64)
    assert base_quat.shape == (4,), "base_quat must have shape (4,)"
    projected_gravity = compute_projected_gravity(base_quat)
    observation["state"]["projected_gravity"] = np.asarray(
        projected_gravity, dtype=np.float32
    )[np.newaxis, np.newaxis]

    left_hand_state = np.asarray(
        state_msg.get("left_hand_q", state_msg.get("last_left_hand_action", np.zeros(7))),
        dtype=np.float32,
    )
    right_hand_state = np.asarray(
        state_msg.get("right_hand_q", state_msg.get("last_right_hand_action", np.zeros(7))),
        dtype=np.float32,
    )
    current_hand_binary_state = np.array(
        [
            _hand_close_binary(left_hand_state, hand_type),
            _hand_close_binary(right_hand_state, hand_type),
        ],
        dtype=np.float32,
    )
    observation["state"]["hand_binary_state"] = current_hand_binary_state[
        np.newaxis, np.newaxis
    ]

    token_state_value = state_msg.get("token_state", None)
    if token_state_value is not None:
        observation["state"]["token_state"] = _build_action_state_token(
            token_state_value,
            current_hand_binary_state,
        )[np.newaxis, np.newaxis]

    return observation


def run_policy_inference_and_process(policy, observation, robot_model):
    """Run policy inference via Isaac-GR00T PolicyClient and process results.

    Returns:
        processed_action dict or None on error.
    """
    try:
        action, _info = policy.get_action(observation)

        action.pop("task_progress", None)
        action.pop("action.task_progress", None)

        motion_key = "motion_token" if "motion_token" in action else "action.motion_token"
        if np.abs(action[motion_key]).max() > 1.25:
            print(
                f"[Warning] action['{motion_key}'] max "
                f"({np.abs(action[motion_key]).max():.4f}) > 1.25. "
                "Exceeds action bound, skipping."
            )
            return None

        processed_action = concat_action(robot_model, action)
        return processed_action
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback

        traceback.print_exc()
        return None


def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
):
    """Persistent worker thread for async inference."""
    while not stop_event.is_set():
        try:
            try:
                inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            busy_event.set()
            try:
                observation = prepare_obs_fn()
                if observation is None:
                    print("[DEBUG] Worker thread: Observation is None, skipping", flush=True)
                    continue

                inference_start_time = time.monotonic()
                processed_action = inference_fn(observation)

                if processed_action is not None:
                    try:
                        result_queue.put_nowait((processed_action, inference_start_time))
                    except queue.Full:
                        try:
                            result_queue.get_nowait()
                            result_queue.put_nowait((processed_action, inference_start_time))
                        except queue.Empty:
                            result_queue.put_nowait((processed_action, inference_start_time))
            finally:
                busy_event.clear()
        except Exception as e:
            print(f"Error in inference worker thread: {e}")
            import traceback

            traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def _compute_closed_hand_joints(side: str) -> np.ndarray:
    """Compute closed hand joint positions using G1GripperInverseKinematicsSolver."""
    side_str = "left" if side.upper() == "L" else "right"
    solver = G1GripperInverseKinematicsSolver(side=side_str)
    return solver._get_middle_close_q_desired().astype(np.float32)


def _binary_to_hand_joints(binary_value: float, side: str) -> np.ndarray:
    if binary_value >= HAND_BINARY_CLOSE_THRESHOLD:
        return _compute_closed_hand_joints(side).copy()
    return np.zeros(7, dtype=np.float32)


def _resolve_hand_type(config: InferenceConfig) -> str:
    normalized_hand_type = config.hand_type.lower()
    if normalized_hand_type in {"dex3", "inspire"}:
        return normalized_hand_type

    if normalized_hand_type != "auto":
        raise ValueError(
            f"Unsupported hand_type '{config.hand_type}'. Expected auto, dex3, or inspire."
        )

    try:
        robot_config = poll_robot_config_zmq(
            host=config.state_zmq_host,
            port=config.state_zmq_port,
            timeout_sec=2.5,
        )
        resolved_hand_type = str(robot_config.get("hand_type", "dex3")).lower()
        if resolved_hand_type in {"dex3", "inspire"}:
            print_green(f"Detected hand type from robot_config: {resolved_hand_type}")
            return resolved_hand_type
        print(
            f"Warning: Unsupported robot_config hand_type '{resolved_hand_type}', "
            "defaulting to dex3"
        )
    except Exception as exc:
        print(
            f"Warning: Failed to auto-detect hand type from robot_config: {exc}. "
            "Defaulting to dex3."
        )

    return "dex3"


def main(config: InferenceConfig):
    pause_loop = True
    resolved_hand_type = _resolve_hand_type(config)
    debug_state_subscriber = None
    debug_hand_run_dir = None
    debug_hand_saved_count = 0

    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")

    PolicyClient = _resolve_policy_client_class()

    n1_policy = PolicyClient(host=config.host, port=config.port)

    print(f"Connecting to PolicyServer at {config.host}:{config.port}...")
    if n1_policy.ping():
        print_green("PolicyServer is reachable.")
    else:
        print("WARNING: PolicyServer not reachable. Inference will fail until server is up.")

    state_subscriber = ZMQStateSubscriber(
        host=config.state_zmq_host,
        port=config.state_zmq_port,
    )

    if config.debug_save_hand_dir:
        debug_state_subscriber = ZMQStateSubscriber(
            host=config.state_zmq_host,
            port=config.state_zmq_port,
        )
        debug_hand_run_dir = (
            Path(config.debug_save_hand_dir)
            / time.strftime("hand_debug_%Y%m%d_%H%M%S")
        )
        debug_hand_run_dir.mkdir(parents=True, exist_ok=True)
        print_green(f"Saving hand debug captures to: {debug_hand_run_dir}")

    camera_subscriber = ComposedCameraClientSensor(
        server_ip=config.camera_host, port=config.camera_port
    )

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    time.sleep(0.1)
    print_green(
        f"ZMQ action socket bound to tcp://{config.action_zmq_host}:{config.action_zmq_port}"
    )
    print_green(f"Using embodiment tag: {config.embodiment_tag}")
    print_green(f"Using hand type: {resolved_hand_type}")

    keyboard_listener = ZMQKeyboardSubscriber(
        port=config.keyboard_zmq_port, host=config.keyboard_zmq_host
    )

    telemetry = Telemetry(window_size=100)

    loop_rate = config.action_publish_rate
    loop_period = 1.0 / loop_rate

    # Track C++ control loop state
    cpp_loop_running = False
    cpp_mode = "OFF"  # "OFF", "PLANNER", or "POSE"

    # Track initial pose hand states
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False

    def publish_initial_pose():
        """Publish initial pose command to move robot to starting position."""
        print("Moving to initial pose")
        left_hand = (
            _compute_closed_hand_joints("L")
            if initial_pose_left_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        right_hand = (
            _compute_closed_hand_joints("R")
            if initial_pose_right_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        zmq_message = pack_latent_action_message(
            motion_token=LATENT_INITIAL_MOTION_TOKEN,
            frame_index=np.array([0], dtype=np.int64),
            left_hand_joints=left_hand,
            right_hand_joints=right_hand,
        )
        zmq_socket.send(zmq_message)
        print_green("Sent latent initial pose via ZMQ")
        time.sleep(1.0)
        print("Initial pose done.")

    def send_cpp_control_command(start: bool, planner: bool = False):
        """Send C++ control loop start/stop commands via ZMQ."""
        nonlocal cpp_loop_running, cpp_mode
        try:
            cmd_msg = build_command_message(start=start, stop=not start, planner=planner)
            zmq_socket.send(cmd_msg)
            time.sleep(0.01)
            action_str = "start" if start else "stop"
            mode_str = "planner" if planner else "pose"
            cpp_loop_running = start
            if start:
                cpp_mode = "PLANNER" if planner else "POSE"
            else:
                cpp_mode = "OFF"
            print_green(f"Sent ZMQ command: {action_str} control loop ({mode_str} mode)")
            return True
        except Exception as e:
            action_str = "start" if start else "stop"
            print(f"Warning: Failed to send {action_str} command message: {e}")
            return False

    # Async inference state
    cached_action_chunk = None
    action_chunk_index = 0
    last_inference_time = 0.0
    inference_interval = 1.0 / config.rate

    zmq_frame_counter = 0

    PROMPT_MSG_PREFIX = "prompt:"

    def check_keyboard_input():
        nonlocal pause_loop, cpp_loop_running, cpp_mode
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time
        nonlocal zmq_frame_counter

        key = keyboard_listener.read_msg()
        if key is None:
            return

        if key.startswith(PROMPT_MSG_PREFIX):
            new_prompt = key[len(PROMPT_MSG_PREFIX):]
            if new_prompt:
                old_prompt = language_prompt_ref[0]
                language_prompt_ref[0] = new_prompt
                print_green(f'Inference prompt changed: "{old_prompt}" -> "{new_prompt}"')
            else:
                print("Received empty prompt change -- ignoring.")
            return

        if key == "c":
            print("Keyboard: 'c' (start recording -- handled by data exporter)")
        elif key == "s":
            print("Keyboard: 's' (stop recording success -- handled by data exporter)")
        elif key == "f":
            print("Keyboard: 'f' (stop recording failure -- handled by data exporter)")
        elif key == "i":
            print("Moving to initial pose")
            zmq_frame_counter = 0
            print("Reset ZMQ frame counter")
            publish_initial_pose()
            cached_action_chunk = None
            action_chunk_index = 0
            print("Cleared cached action chunk")
            if cpp_loop_running and cpp_mode == "PLANNER":
                if send_cpp_control_command(start=True, planner=False):
                    print("Switched to POSE mode (from PLANNER mode)")
                else:
                    print("Warning: Failed to switch to POSE mode")
            elif not cpp_loop_running:
                print("Note: C++ loop not running - press 'k' to start")
        elif key == "p":
            pause_loop = not pause_loop
            print(f"{'Paused' if pause_loop else 'Resumed'} policy loop")
            if pause_loop:
                print("Policy loop paused (C++ loop still running - press 'k' to stop)")
            else:
                print("Policy loop resumed")
        elif key == "k":
            if cpp_loop_running:
                current_planner = cpp_mode == "PLANNER"
                print(f"Stopping C++ control loop (from {cpp_mode} mode)...")
                if send_cpp_control_command(start=False, planner=current_planner):
                    print("Stopped C++ control loop")
            else:
                print("Starting C++ control loop in PLANNER mode...")
                if send_cpp_control_command(start=True, planner=True):
                    print("Started C++ control loop in PLANNER mode")
                    print("Press 'i' to send initial pose and switch to POSE mode")
                    if pause_loop:
                        print("Note: Policy loop is paused - press 'p' to resume")
        elif key == "[":
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
            print(
                f"Initial pose left hand: {'closed' if initial_pose_left_hand_closed else 'open'}"
            )
        elif key == "]":
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed
            print(
                f"Initial pose right hand: "
                f"{'closed' if initial_pose_right_hand_closed else 'open'}"
            )

    # Mutable prompt container (single-writer from keyboard, single-reader from inference)
    language_prompt_ref: list[str] = [config.prompt]
    print(f"Starting the policy loop with language prompt: {language_prompt_ref[0]}")

    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    inference_stop_event = threading.Event()
    inference_busy_event = threading.Event()

    inference_worker_thread = threading.Thread(
        target=_inference_worker_loop,
        args=(
            inference_queue,
            result_queue,
            inference_stop_event,
            inference_busy_event,
            lambda: prepare_observation_from_sensors(
                camera_subscriber=camera_subscriber,
                state_subscriber=state_subscriber,
                robot_model=robot_model,
                language_prompt=language_prompt_ref[0],
                hand_type=resolved_hand_type,
                log_errors=True,
            ),
            lambda obs: run_policy_inference_and_process(
                policy=n1_policy,
                observation=obs,
                robot_model=robot_model,
            ),
        ),
        daemon=True,
    )
    inference_worker_thread.start()

    try:
        while True:
            t_start = time.monotonic()
            check_keyboard_input()

            # Consume result first so last_inference_time is fresh before trigger check
            try:
                processed_action, inference_start_time = result_queue.get_nowait()
                inference_delay = time.monotonic() - inference_start_time
                action_chunk_index = calculate_latency_compensated_index(
                    inference_delay, config.action_publish_rate, config.action_horizon
                )
                cached_action_chunk = processed_action
                last_inference_time = time.monotonic()
                print_green(
                    f'New action chunk (prompt: "{language_prompt_ref[0]}", '
                    f"latency: {inference_delay:.3f}s)"
                )
            except queue.Empty:
                pass

            worker_is_busy = inference_busy_event.is_set()
            should_start = should_trigger_new_inference(
                cached_chunk_exists=(cached_action_chunk is not None),
                inference_thread_running=worker_is_busy,
                time_since_last_inference=(time.monotonic() - last_inference_time),
                inference_interval=inference_interval,
            )

            if should_start:
                try:
                    inference_queue.put_nowait(None)
                except queue.Full:
                    pass

            if pause_loop:
                print("Pausing...", end="", flush=True)
                time.sleep(0.2)
                print(".", end="", flush=True)
                continue

            with telemetry.timer("total_loop"):
                if cached_action_chunk is None:
                    print("[DEBUG] No cached chunk yet, waiting...", flush=True)
                    _sleep_remaining(t_start, loop_period)
                    continue

                processed_action = cached_action_chunk

                if processed_action is None or not processed_action:
                    print("[DEBUG] processed_action is None or empty, skipping", flush=True)
                else:
                    motion_token = np.asarray(
                        get_action_field(processed_action, "motion_token"),
                        dtype=np.float32,
                    )

                    # Action arrays arrive as (B, T, D) from the model.
                    # Squeeze batch dim to get (T, D), then index by time step.
                    if motion_token.ndim == 3:
                        motion_token = motion_token[0]

                    horizon = motion_token.shape[0] if motion_token.ndim == 2 else 1
                    current_idx = min(action_chunk_index, horizon - 1)

                    if motion_token.ndim == 2:
                        motion_token = motion_token[current_idx]

                    hand_debug_signals = _extract_hand_debug_signals(
                        processed_action,
                        current_idx,
                        resolved_hand_type,
                    )
                    left_hand_joints = hand_debug_signals["sent_left_hand_joints"]
                    right_hand_joints = hand_debug_signals["sent_right_hand_joints"]

                    frame_index = np.array([zmq_frame_counter], dtype=np.int64)
                    state_debug_msg = None
                    if debug_state_subscriber is not None:
                        state_debug_msg = debug_state_subscriber.get_msg(clear=False)

                    debug_hand_saved_count = _maybe_save_hand_debug_capture(
                        debug_run_dir=debug_hand_run_dir,
                        debug_save_every=config.debug_save_hand_every,
                        debug_save_max=config.debug_save_hand_max,
                        debug_saved_count=debug_hand_saved_count,
                        frame_index=int(frame_index[0]),
                        current_idx=current_idx,
                        hand_type=resolved_hand_type,
                        prompt=language_prompt_ref[0],
                        motion_token=motion_token,
                        hand_debug_signals=hand_debug_signals,
                        state_msg=state_debug_msg,
                    )

                    zmq_frame_counter += 1

                    zmq_message = pack_latent_action_message(
                        motion_token,
                        frame_index,
                        left_hand_joints=left_hand_joints,
                        right_hand_joints=right_hand_joints,
                    )
                    zmq_socket.send(zmq_message)
                    if zmq_frame_counter % 50 == 0:
                        print_green(
                            f"ZMQ: Sent latent action - "
                            f"frame: {frame_index[0]}, "
                            f"token shape: {motion_token.shape}"
                        )

                action_chunk_index = min(action_chunk_index + 1, config.action_horizon - 1)

            end_time = time.monotonic()

            if config.verbose_timing:
                telemetry.log_timing_info(context="VLA Inference Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.rate):
                telemetry.log_timing_info(
                    context="VLA Inference Loop Missed", threshold=0.001
                )

            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        print("VLA inference loop terminated by user")

    finally:
        inference_stop_event.set()
        inference_worker_thread.join(timeout=1.0)
        zmq_socket.close()
        zmq_context.term()
        state_subscriber.close()
        if debug_state_subscriber is not None:
            debug_state_subscriber.close()
        keyboard_listener.close()
        print("Shutdown complete.")


def _sleep_remaining(t_start: float, loop_period: float):
    """Sleep for the remainder of the loop period."""
    elapsed = time.monotonic() - t_start
    remaining = loop_period - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)