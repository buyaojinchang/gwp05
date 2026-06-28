"""GWP-MoT inference server for the G1 locomanip pick_place (g1_sonic) task.

This server is **Isaac-GR00T PolicyClient compatible** so it can be driven by
GR00T-WholeBodyControl's ``gear_sonic/scripts/run_vla_inference.py``
(``from gr00t.policy.server_client import PolicyClient``; default port 5550).

It combines:
  * the GR00T / FastWAM wire protocol (ZMQ REQ/REP + msgpack_numpy; endpoints
    ``ping`` / ``reset`` / ``get_action`` / ``get_modality_config`` / ``kill``;
    ``get_action`` returns an ``(action_dict, info)`` tuple),
  * the *verified* numerical forward path from
    ``experiment/openloop/openloop_locomanip.py`` (imported directly so the
    deployed policy matches open-loop eval exactly),
  * FastWAM-style robustness: tolerant observation parsing with fallbacks,
    hand-binary -> 7-DoF hand-joint decoding, optional debug stats, rich info.

Action contract expected by run_vla_inference.py (B=1):
  * ``motion_token`` [1, T, 64]  raw sonic latent motion token
  * ``hand_binary``  [1, T, 2]   raw left/right hand open-close scalars

The server is intentionally decode-free: it returns the raw 66-d sonic latent
split as motion_token[64] + hand_binary[2]. Mapping hand_binary -> 7-DoF hand
joints is the client's responsibility (it owns the authoritative
G1GripperInverseKinematicsSolver), keeping a single source of truth.

Model contract (matches ``configs/data/pick_place_g1_sonic.yaml``):
  * Action: 66-d sonic latent = motion_token[64] + hand_binary[2], kept RAW.
  * State: 2-token sequence = [43-d joint (z-scored), 66-d sonic latent (raw)].
  * Vision: single ego_view camera.

NOTE (deployment open questions, see warnings at runtime):
  1. The GR00T deploy observation does NOT carry the 66-d sonic latent state
     token; it is read from latent_state/state.token_state if present, else a
     zero token is used (toggle --require-latent-state to hard-fail instead).
  2. The 43-d joint state token is read from flat observation keys if present,
     else built/zeroed; confirm the exact joint ordering for your robot.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
import time
from typing import Any, Literal

import msgpack_numpy as mnp
import numpy as np
import torch
import tyro
import zmq

# GWP-MoT project root (experiment/g1 -> gwp-mot)
GWP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GWP_ROOT not in sys.path:
    sys.path.insert(0, GWP_ROOT)

# Import the verified open-loop forward path by bare module name (same trick
# openloop_locomanip uses to import openloop_eval) for exact numerical parity.
_OPENLOOP_DIR = os.path.join(GWP_ROOT, "experiment", "openloop")
if _OPENLOOP_DIR not in sys.path:
    sys.path.insert(0, _OPENLOOP_DIR)

from openloop_locomanip import (  # noqa: E402
    build_model,
    build_state,
    load_norm_stats,
    sample_action,
    MOTION_TOKEN_DIM,
    HAND_BINARY_DIM,
)
from openloop_eval import preprocess_image  # noqa: E402


# ── Defaults ──────────────────────────────────────────────────────────────
PRETRAINED_PATH = os.environ.get(
    "WAN22_DIFFUSERS_PATH",
    "/shared_disk/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers",
)
DEFAULT_STATS_PATH = os.environ.get(
    "PICK_PLACE_STATS",
    "/shared_disk/users/hengtao.li/locomanip/data/pick_place_gwp/norm_stats_delta.json",
)
DEFAULT_MODEL_SERVER_PORT = 5550  # matches run_vla_inference.py default
T5_MAX_LEN = 64

# Candidate observation keys (first hit wins) — FastWAM-style tolerant parsing.
LATENT_STATE_KEYS = ("latent_state", "motion_latent", "token_state", "sonic_latent")
LATENT_STATE_SUBKEYS = ("latent_state", "token_state", "motion_latent")
JOINT_STATE_KEYS = ("observation.state", "joint_state", "proprio", "q")
JOINT_STATE_SUBKEYS = ("observation.state", "joint_state")

# 43-d joint state token assembly from GR00T-style observation["state"] joint
# groups. Order/dims match the pick_place_gwp dataset meta/info.json
# (observation.state): left_leg, right_leg, waist, left_arm, left_hand,
# right_arm, right_hand = 6+6+3+7+7+7+7 = 43.
G1_JOINT_GROUP_ORDER: tuple[tuple[str, int], ...] = (
    ("left_leg", 6),
    ("right_leg", 6),
    ("waist", 3),
    ("left_arm", 7),
    ("left_hand", 7),
    ("right_arm", 7),
    ("right_hand", 7),
)


class MsgSerializer:
    """msgpack_numpy serializer, wire-compatible with gr00t PolicyClient."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return mnp.packb(data)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return mnp.unpackb(data, raw=False)


@dataclass
class ServerConfig:
    """Configuration for the G1 sonic GWP-MoT inference server."""

    checkpoint_path: str
    """Path to the MoT checkpoint (.pt) to serve."""

    stats_path: str = DEFAULT_STATS_PATH
    """norm_stats_delta.json providing observation.state / observation.motion_latent stats."""

    pretrained_path: str = PRETRAINED_PATH
    """Wan2.2 TI2V-5B diffusers base (vae + transformer + tokenizer + text_encoder)."""

    host: str = "0.0.0.0"
    """Host address to bind."""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port to bind (run_vla_inference.py connects to 5550 by default)."""

    device: str = "cuda"
    """Torch device."""

    # MoT / sampling
    num_frames: int = 56
    """Action tokens generated per chunk (training num_frames)."""

    num_steps: int = 10
    """Flow-matching denoising steps."""

    num_samples: int = 1
    """Average this many diffusion samples per request to estimate the conditional mean."""

    replan_steps: int = 50
    """Return only the first N of the num_frames generated steps (0 = return all)."""

    action_flow_shift: float = 5.0
    """Flow shift for the action-only sampler (matches training)."""

    action_dim: int = 66
    """Sonic latent dim: motion_token(64) + hand_binary(2)."""

    state_dim: int = 66
    """Per-token padded state width."""

    joint_state_dim: int = 43
    """State token0 raw width (joint state)."""

    latent_state_dim: int = 66
    """State token1 raw width (sonic latent)."""

    dst_size: tuple[int, int] = (320, 256)
    """(width, height) cover-crop size, matching WATransformsLerobot."""

    action_expert_hidden_dim: int = 1024
    action_expert_ffn_dim: int = 4096
    mot_checkpoint_mixed_attn: bool = True

    seed: int | None = None
    """Optional fixed seed for sampling noise (reproducible actions)."""

    # Text conditioning (exactly two modes; no zero/empty fallback — missing
    # text is a hard error).
    text_mode: Literal["t5", "precomputed"] = "t5"
    """t5: load a local T5 and encode the prompt online (default);
    precomputed: load a local embedding from --text-context-file."""

    text_context_file: str | None = None
    """Precomputed .pt text embedding [L,4096], required when text_mode=precomputed."""

    # Robustness / debug
    require_latent_state: bool = False
    """Raise instead of silently using a zero sonic-latent state token."""

    debug_print_stats: bool = False
    """Print per-request latent/action stats."""

    debug_print_every: int = 1
    """When debug_print_stats is set, print every Nth request."""


# ── Helpers ───────────────────────────────────────────────────────────────

def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _first_present(data: Any, keys: tuple[str, ...]) -> tuple[Any, str | None]:
    if isinstance(data, dict):
        for key in keys:
            if key in data and data[key] is not None:
                return data[key], key
    return None, None


def _extract_ego_frame(observation: dict[str, Any]) -> np.ndarray:
    """Pull a single HWC uint8 RGB frame from a GR00T-style observation."""
    raw = None
    if isinstance(observation, dict):
        video = observation.get("video")
        if isinstance(video, dict):
            for key in ("ego_view", "ego", "image"):
                if key in video and video[key] is not None:
                    raw = video[key]
                    break
        if raw is None:
            for key in ("observation.images.ego_view", "ego_view", "image"):
                if key in observation and observation[key] is not None:
                    raw = observation[key]
                    break
    if raw is None:
        raise KeyError("observation['video']['ego_view'] (or a flat ego_view key) is required")

    arr = _to_numpy(raw)
    while arr.ndim > 3:  # [B,T,H,W,C] -> [H,W,C]
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"ego_view must be RGB HWC, got shape {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _extract_prompt(observation: dict[str, Any]) -> str:
    lang = observation.get("language", {}) if isinstance(observation, dict) else {}
    candidates = (
        "annotation.human.task_description",
        "task",
        "annotation.language.language_instruction",
    )
    for key in candidates:
        value = lang.get(key) if isinstance(lang, dict) else None
        while isinstance(value, (list, tuple)) and value:
            value = value[0]
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("task", "prompt", "language_instruction"):
        value = observation.get(key) if isinstance(observation, dict) else None
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _to_1d_float(value: Any, expected_dim: int) -> np.ndarray:
    arr = _to_numpy(value).astype(np.float32).reshape(-1)
    if arr.shape[0] == expected_dim:
        return arr
    if arr.shape[0] > expected_dim:
        return arr[:expected_dim]
    out = np.zeros((expected_dim,), dtype=np.float32)
    out[: arr.shape[0]] = arr
    return out


def _extract_state_vector(
    observation: dict[str, Any],
    top_keys: tuple[str, ...],
    sub_keys: tuple[str, ...],
    expected_dim: int,
) -> tuple[np.ndarray | None, str]:
    """Find a 1-D state vector from top-level or nested ``state`` keys."""
    value, key = _first_present(observation, top_keys)
    if value is not None:
        return _to_1d_float(value, expected_dim), key
    state = observation.get("state") if isinstance(observation, dict) else None
    value, key = _first_present(state, sub_keys)
    if value is not None:
        return _to_1d_float(value, expected_dim), f"state.{key}"
    return None, "missing"


def _assemble_joint_from_groups(
    observation: dict[str, Any], expected_dim: int
) -> np.ndarray | None:
    """Build the 43-d joint token by concatenating GR00T state joint groups.

    Matches the deploy observation produced by
    ``gear_sonic ... prepare_observation_for_eval`` (splits whole-body q into
    left_arm/right_arm/waist/left_leg/right_leg/left_hand/right_hand).
    """
    state = observation.get("state") if isinstance(observation, dict) else None
    if not isinstance(state, dict):
        return None
    if not all(group in state and state[group] is not None for group, _ in G1_JOINT_GROUP_ORDER):
        return None
    parts = [_to_1d_float(state[group], dim) for group, dim in G1_JOINT_GROUP_ORDER]
    return _to_1d_float(np.concatenate(parts, axis=0), expected_dim)


def _pad_text_embedding(emb: Any, device: str, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(emb, torch.Tensor):
        emb = torch.as_tensor(emb)
    emb = emb.float()
    if emb.ndim == 3:
        emb = emb[0]
    if emb.ndim != 2:
        raise ValueError(f"text embedding must be [L,D] or [1,L,D], got {tuple(emb.shape)}")
    emb = emb[:T5_MAX_LEN]
    if emb.shape[0] < T5_MAX_LEN:
        emb = torch.nn.functional.pad(emb, (0, 0, 0, T5_MAX_LEN - emb.shape[0]))
    return emb.unsqueeze(0).to(device, dtype=dtype)


def _encode_prompt_t5(text: str, tokenizer, text_encoder, device: str, dtype: torch.dtype) -> torch.Tensor:
    inputs = tokenizer(
        [text], return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = text_encoder(**inputs)
    length = int(inputs.attention_mask[0].sum().item())
    emb = outputs.last_hidden_state[0, :length].detach().float()
    emb = emb[:T5_MAX_LEN]
    if emb.shape[0] < T5_MAX_LEN:
        emb = torch.nn.functional.pad(emb, (0, 0, 0, T5_MAX_LEN - emb.shape[0]))
    return emb.unsqueeze(0).to(device, dtype=dtype)


# ── Policy ────────────────────────────────────────────────────────────────

class G1SonicPolicy:
    """G1 sonic policy: ego_view + 2-token state -> motion_token + hand joints."""

    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.dtype = torch.bfloat16
        self.state_token_dims = [int(cfg.joint_state_dim), int(cfg.latent_state_dim)]
        self.joint_dim = int(cfg.joint_state_dim)
        self.latent_dim = int(cfg.latent_state_dim)

        print(f"[G1Server] Building MoT model on {cfg.device} (bfloat16)...")
        self.md = build_model(
            cfg.pretrained_path,
            cfg.checkpoint_path,
            action_dim=cfg.action_dim,
            state_dim=cfg.state_dim,
            state_token_dims=tuple(self.state_token_dims),
            flow_shift=cfg.action_flow_shift,
            device=cfg.device,
            dtype=self.dtype,
            action_expert_hidden_dim=cfg.action_expert_hidden_dim,
            action_expert_ffn_dim=cfg.action_expert_ffn_dim,
            mot_checkpoint_mixed_attn=cfg.mot_checkpoint_mixed_attn,
        )
        self.md["action_flow_shift"] = cfg.action_flow_shift

        print(f"[G1Server] Loading norm stats from {cfg.stats_path}")
        self.stats = load_norm_stats(cfg.stats_path)

        # Text conditioning: exactly two modes, no zero/empty fallback.
        self.tokenizer = None
        self.text_encoder = None
        self.prompt_cache: dict[str, torch.Tensor] = {}
        self.fixed_text: torch.Tensor | None = None
        if cfg.text_mode == "precomputed":
            if not cfg.text_context_file:
                raise ValueError("text_mode=precomputed requires --text-context-file")
            path = os.path.expanduser(cfg.text_context_file)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"text_context_file not found: {path}")
            payload = torch.load(path, map_location="cpu")
            emb = payload["context"] if isinstance(payload, dict) and "context" in payload else payload
            self.fixed_text = _pad_text_embedding(emb, self.device, self.dtype)
            print(f"[G1Server] Loaded precomputed text embedding from {path}")
        else:  # t5
            self._init_t5()

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)

        self._warned_zero_latent = False
        self._warned_zero_joint = False
        self._warned_flat_joint = False
        self._request_count = 0
        print(
            "[G1Server] Policy ready. "
            f"num_frames={cfg.num_frames}, num_steps={cfg.num_steps}, "
            f"text_mode={cfg.text_mode}, action_dim={cfg.action_dim}"
        )

    def _init_t5(self) -> None:
        from transformers import AutoTokenizer, UMT5EncoderModel
        from world_action_model.trainers.wa_trainer import get_model_path

        pretrained = get_model_path(self.cfg.pretrained_path)
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(pretrained, "tokenizer"))
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            os.path.join(pretrained, "text_encoder"), torch_dtype=torch.float16
        ).to(self.device)
        self.text_encoder.eval()
        print("[G1Server] T5 online prompt encoding enabled.")

    # ── endpoints ────────────────────────────────────────────────────────
    def ping(self) -> dict[str, Any]:
        return {"status": "ok", "message": "G1 sonic server is running"}

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        del options
        return {}

    def get_modality_config(self) -> dict[str, Any]:
        return {
            "action_keys": ["motion_token", "hand_binary"],
            "action_dim": int(self.cfg.action_dim),
            "motion_token_dim": int(MOTION_TOKEN_DIM),
            "hand_binary_dim": int(HAND_BINARY_DIM),
            "action_horizon": int(self.cfg.num_frames),
            "state_token_dims": list(self.state_token_dims),
            "num_state_tokens": 2,
        }

    def _resolve_text(self, observation: dict[str, Any], prompt: str) -> tuple[torch.Tensor, str]:
        del observation
        # Exactly two modes; no zero/empty fallback — missing text is fatal.
        if self.cfg.text_mode == "precomputed":
            return self.fixed_text, "fixed_file"
        # text_mode == "t5": encode the prompt online (cached). Empty prompt is
        # an error rather than silently falling back to zeros.
        if not prompt:
            raise ValueError(
                "text_mode=t5 but observation carries no prompt; "
                "set a task/language instruction or use text_mode=precomputed."
            )
        if prompt in self.prompt_cache:
            return self.prompt_cache[prompt], "t5_cached"
        emb = _encode_prompt_t5(prompt, self.tokenizer, self.text_encoder, self.device, self.dtype)
        self.prompt_cache[prompt] = emb
        return emb, "t5_encoded"

    @torch.no_grad()
    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options
        start_t = time.time()
        self._request_count += 1
        cfg = self.cfg

        # 1) ego image -> ref latents (identical to openloop_locomanip).
        ego_frame = _extract_ego_frame(observation)
        dst_w, dst_h = int(cfg.dst_size[0]), int(cfg.dst_size[1])
        ref_image = preprocess_image(ego_frame, (dst_w, dst_h)).to(self.device, dtype=self.dtype)
        ref_latents = self.md["vae"].encode(ref_image.unsqueeze(2)).latent_dist.mode()
        ref_latents = (ref_latents - self.md["latents_mean"]) * self.md["latents_std"]

        # 2) 2-token state (joint z-scored, sonic latent raw).
        # Prefer assembling the 43-d joint from GR00T state joint groups, then
        # fall back to a flat joint vector, then zeros.
        joint_row = _assemble_joint_from_groups(observation, self.joint_dim)
        joint_src = "state.joint_groups"
        if joint_row is None:
            joint_row, joint_src = _extract_state_vector(
                observation, JOINT_STATE_KEYS, JOINT_STATE_SUBKEYS, self.joint_dim
            )
            if joint_row is not None and not self._warned_flat_joint:
                print(
                    "[G1Server] WARNING: 43-d joint groups not found; "
                    f"falling back to flat key '{joint_src}' for state token0."
                )
                self._warned_flat_joint = True
        if joint_row is None:
            joint_row = np.zeros((self.joint_dim,), dtype=np.float32)
            joint_src = "zeros"
            if not self._warned_zero_joint:
                print("[G1Server] WARNING: no joint state found; using zeros for state token0.")
                self._warned_zero_joint = True

        latent_row, latent_src = _extract_state_vector(
            observation, LATENT_STATE_KEYS, LATENT_STATE_SUBKEYS, self.latent_dim
        )
        if latent_row is None:
            msg = "no sonic latent state found; using zeros for state token1."
            if cfg.require_latent_state:
                raise ValueError(msg)
            latent_row = np.zeros((self.latent_dim,), dtype=np.float32)
            if not self._warned_zero_latent:
                print(f"[G1Server] WARNING: {msg}")
                self._warned_zero_latent = True

        state_np = build_state(joint_row, latent_row, self.stats, self.state_token_dims, cfg.state_dim)
        state = torch.from_numpy(state_np).to(self.device, dtype=self.dtype).unsqueeze(0)
        state_mask = torch.ones((1, state_np.shape[0]), dtype=torch.bool, device=self.device)

        # 3) text conditioning.
        prompt = _extract_prompt(observation)
        prompt_embeds, text_src = self._resolve_text(observation, prompt)

        # 4) action-only flow matching (optionally averaged over samples).
        sample_acc: torch.Tensor | None = None
        for s_i in range(max(1, cfg.num_samples)):
            if cfg.num_samples > 1 and cfg.seed is not None:
                torch.manual_seed(cfg.seed + 1000 * s_i + self._request_count)
            one = sample_action(
                self.md, ref_latents, prompt_embeds, state, state_mask,
                num_steps=cfg.num_steps, num_frames=cfg.num_frames,
            )
            one = one.float().squeeze(0)
            sample_acc = one if sample_acc is None else sample_acc + one
        pred = (sample_acc / max(1, cfg.num_samples)).cpu().numpy()  # [T, 66] raw

        # 4b) replan prefix: generate num_frames, keep the first replan_steps
        # (mirrors open-loop "predict 56, execute first 50").
        if cfg.replan_steps and cfg.replan_steps > 0:
            pred = pred[: min(cfg.replan_steps, pred.shape[0])]

        # 5) split into raw motion_token + raw hand_binary. Decoding hand_binary
        # into 7-DoF joints is the client's job (it owns the authoritative
        # G1GripperInverseKinematicsSolver); the server stays decode-free.
        motion_token = pred[:, :MOTION_TOKEN_DIM].astype(np.float32)
        hand_binary = pred[:, MOTION_TOKEN_DIM:MOTION_TOKEN_DIM + HAND_BINARY_DIM].astype(np.float32)
        if hand_binary.shape[1] < HAND_BINARY_DIM:  # defensive pad
            hand_binary = np.pad(hand_binary, [(0, 0), (0, HAND_BINARY_DIM - hand_binary.shape[1])])

        elapsed = time.time() - start_t
        if cfg.debug_print_stats and cfg.debug_print_every > 0 and (
            self._request_count % cfg.debug_print_every == 0
        ):
            print(
                f"[G1Server] req={self._request_count} {elapsed*1000:.0f}ms "
                f"joint={joint_src} latent={latent_src} text={text_src} "
                f"motion_absmean={np.abs(motion_token).mean():.4f} motion_max={np.abs(motion_token).max():.4f} "
                f"hand_binary_mean=[{hand_binary[:, 0].mean():.3f},{hand_binary[:, 1].mean():.3f}]"
            )

        action = {
            "motion_token": motion_token[None, ...],  # [1, T, 64]
            "hand_binary": hand_binary[None, ...],     # [1, T, 2] (client decodes to joints)
        }
        info = {
            "backend": "gwp-mot-g1-sonic",
            "elapsed_ms": float(elapsed * 1000.0),
            "horizon": int(motion_token.shape[0]),
            "num_frames": int(cfg.num_frames),
            "replan_steps": int(cfg.replan_steps),
            "num_steps": int(cfg.num_steps),
            "num_samples": int(cfg.num_samples),
            "joint_state_source": joint_src,
            "latent_state_source": latent_src,
            "text_source": text_src,
            "prompt": prompt,
            "motion_abs_mean": float(np.abs(motion_token).mean()),
            "motion_max_abs": float(np.abs(motion_token).max()),
        }
        return action, info


# ── GR00T-compatible ZMQ server (hand-rolled, like FastWAM) ─────────────────

class PolicyServer:
    def __init__(self, policy: G1SonicPolicy, host: str, port: int):
        self.policy = policy
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")

    def _handle(self, endpoint: str, data: dict[str, Any] | None) -> Any:
        if endpoint == "ping":
            return self.policy.ping()
        if endpoint == "reset":
            return self.policy.reset(data.get("options") if data else None)
        if endpoint == "get_modality_config":
            return self.policy.get_modality_config()
        if endpoint == "get_action":
            if not data or "observation" not in data:
                raise ValueError("Missing data.observation for get_action")
            return self.policy.get_action(
                observation=data["observation"], options=data.get("options")
            )
        if endpoint == "kill":
            self.running = False
            return {"status": "ok", "message": "Shutting down"}
        raise ValueError(f"Unknown endpoint: {endpoint}")

    def run(self) -> None:
        print("[G1Server] Ready")
        while self.running:
            try:
                message = self.socket.recv()
                request = MsgSerializer.from_bytes(message)
                endpoint = request.get("endpoint", "get_action")
                data = request.get("data", None)
                result = self._handle(endpoint, data)
                self.socket.send(MsgSerializer.to_bytes(result))
            except Exception as exc:  # noqa: BLE001
                import traceback

                print(f"[G1Server] Error: {exc}")
                print(traceback.format_exc())
                self.socket.send(MsgSerializer.to_bytes({"error": str(exc)}))


def run_server(cfg: ServerConfig) -> None:
    if not os.path.isfile(cfg.checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {cfg.checkpoint_path}")
    if cfg.stats_path and not os.path.isfile(cfg.stats_path):
        raise FileNotFoundError(f"norm stats not found: {cfg.stats_path}")

    policy = G1SonicPolicy(cfg)
    server = PolicyServer(policy, host=cfg.host, port=cfg.port)
    print(f"[G1Server] Listening on {cfg.host}:{cfg.port} (GR00T PolicyClient compatible)")
    try:
        server.run()
    except KeyboardInterrupt:
        print("[G1Server] Interrupted, shutting down.")


if __name__ == "__main__":
    run_server(tyro.cli(ServerConfig))
