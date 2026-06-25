"""GWP-MoT inference server for AgileX dual-arm robots (ZMQ, giga-brain compatible).

Expects observations in the same format as giga-brain-0 inference_agilex_client:
  - observation.state: (14,) joint positions
  - observation.images.cam_high / cam_left_wrist / cam_right_wrist: CHW float in [0, 1]
  - task: language instruction string

Returns a (action_chunk, 14) float tensor of absolute joint targets.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import tyro
from PIL import Image
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

# GWP-MoT project root (experiment/agilex -> gwp-mot)
GWP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GWP_ROOT not in sys.path:
    sys.path.insert(0, GWP_ROOT)

from experiment.agilex.sockets import RobotInferenceServer  # noqa: E402

# Defaults aligned with configs/gwp_v0_heat_food_fold_shirt_mot.py
PRETRAINED_PATH = os.environ.get(
    "WAN22_DIFFUSERS_PATH",
    "/shared_disk/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers",
)
NUM_FRAMES = 36
ACTION_CHUNK = 36
NUM_STEPS = 10
ACTION_FLOW_SHIFT = 5.0
ACTION_DIM = 14
STATE_DIM = 14
DST_SIZE = (320, 256)  # (width, height), matches WATransformsLerobot
DELTA_MASK = np.array(
    [True, True, True, True, True, True, False,
     True, True, True, True, True, True, False],
    dtype=bool,
)
DEFAULT_STATS_PATH = os.environ.get(
    "GWP_V0_NORM_STATS",
    "/shared_disk/users/hengtao.li/giga_real_data/gwp_v0/norm_stats_delta.json",
)
REPLAN_STEPS = ACTION_CHUNK
T5_MAX_LEN = 64
IMAGE_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def build_model(
    pretrained_path: str,
    checkpoint_path: str,
    action_dim: int = ACTION_DIM,
    state_dim: int = STATE_DIM,
    action_flow_shift: float = ACTION_FLOW_SHIFT,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    mot_checkpoint_mixed_attn: bool = True,
    action_expert_hidden_dim: int = 1024,
    action_expert_ffn_dim: int = 4096,
):
    from diffusers.models import AutoencoderKLWan
    from world_action_model.models.transformer_wa_mot import MoTWorldActionTransformer
    from world_action_model.trainers.wa_trainer import get_model_path, process_transformer

    pretrained = get_model_path(pretrained_path)

    vae = AutoencoderKLWan.from_pretrained(os.path.join(pretrained, "vae"))
    vae.requires_grad_(False).eval().to(device, dtype=dtype)

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype=dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype=dtype)

    transformer = MoTWorldActionTransformer.from_pretrained_video(
        transformer_pretrained=os.path.join(pretrained, "transformer"),
        torch_dtype=dtype,
        action_dim=action_dim,
        state_dim=state_dim,
        action_expert={
            "hidden_dim": int(action_expert_hidden_dim),
            "ffn_dim": int(action_expert_ffn_dim),
        },
        mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        video_attention_mask_mode="gwp_casual",
    )
    process_transformer(transformer.video_expert, {})
    transformer.to(device, dtype=dtype)

    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    keys = tuple(state_dict.keys())
    if not any(k.startswith("transformer.mot.") or k.startswith("mot.") or ".mot." in k for k in keys):
        raise ValueError(
            "This server only supports gwp-mot MoT checkpoints. "
            f"Checkpoint has no MoT keys: {checkpoint_path}"
        )

    tf_state = {}
    for k, v in state_dict.items():
        tf_state[k.removeprefix("transformer.")] = v
    missing, unexpected = transformer.load_state_dict(tf_state, strict=False)
    print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    transformer.eval()

    return {
        "vae": vae,
        "transformer": transformer,
        "latents_mean": latents_mean,
        "latents_std": latents_std,
        "flow_shift": action_flow_shift,
        "action_flow_shift": action_flow_shift,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "device": device,
        "dtype": dtype,
    }


def load_norm_stats(stats_path: str, state_dim: int, action_dim: int, device: str):
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    ns = stats["norm_stats"]

    def _to_tensor(arr, dim):
        t = torch.tensor(arr, dtype=torch.float32, device=device).flatten()[:dim]
        if t.numel() < dim:
            t = torch.nn.functional.pad(t, (0, dim - t.numel()), value=0.0)
        return t

    return {
        "state_mean": _to_tensor(ns["observation.state"]["mean"], state_dim),
        "state_std": _to_tensor(ns["observation.state"]["std"], state_dim),
        "action_mean": _to_tensor(ns["action"]["mean"], action_dim),
        "action_std": _to_tensor(ns["action"]["std"], action_dim),
    }


def normalize_state(state: torch.Tensor, norm: dict) -> torch.Tensor:
    eps = 1e-8
    return (state - norm["state_mean"]) / norm["state_std"].clamp_min(eps)


def denormalize_action(action: torch.Tensor, norm: dict) -> torch.Tensor:
    eps = 1e-8
    return action * norm["action_std"].clamp_min(eps) + norm["action_mean"]


def encode_prompt_t5(prompt_text: str, tokenizer, text_encoder, device: str, dtype: torch.dtype) -> torch.Tensor:
    inputs = tokenizer(
        [prompt_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        outputs = text_encoder(**inputs)
    length = int(inputs.attention_mask[0].sum().item())
    emb = outputs.last_hidden_state[0, :length].detach().float()
    emb = emb[:T5_MAX_LEN]
    if emb.shape[0] < T5_MAX_LEN:
        emb = torch.nn.functional.pad(emb, (0, 0, 0, T5_MAX_LEN - emb.shape[0]))
    return emb.unsqueeze(0).to(device, dtype=dtype)


@torch.no_grad()
def sample_action(
    model_dict: dict,
    ref_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    state: torch.Tensor,
    num_steps: int = NUM_STEPS,
    action_chunk: int = ACTION_CHUNK,
):
    """MoT action-only flow matching over a fixed number of generated action tokens."""
    device = model_dict["device"]
    dtype = model_dict["dtype"]
    transformer = model_dict["transformer"]
    if hasattr(transformer, "clear_action_only_cache"):
        transformer.clear_action_only_cache()
    flow_shift = model_dict.get("action_flow_shift", model_dict["flow_shift"])
    action_dim = model_dict["action_dim"]
    bs = ref_latents.shape[0]

    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps

    rng = model_dict.get("rng")
    noisy_action = torch.randn(
        bs, action_chunk, action_dim, device=device, dtype=dtype, generator=rng,
    )

    for t in timesteps:
        num_state_tokens = state.shape[1]
        latent_h = ref_latents.shape[-2]
        latent_w = ref_latents.shape[-1]
        frame_per_tokens = latent_h * latent_w // 4
        num_ref_latent_tokens = frame_per_tokens
        total_tokens = num_state_tokens + action_chunk + num_ref_latent_tokens

        timestep = torch.zeros(bs, total_tokens, device=device, dtype=dtype)
        noise_t = t.float()
        timestep[:, num_state_tokens + num_ref_latent_tokens:] = noise_t

        action_pred = transformer(
            ref_latents=ref_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            action=noisy_action,
            state=state,
            action_only=True,
        )
        noisy_action = scheduler.step(action_pred, t, noisy_action, return_dict=False)[0]

    return noisy_action


def _as_chw_float(tensor: torch.Tensor) -> torch.Tensor:
    t = tensor.detach()
    if t.ndim == 4:
        t = t[0]
    if t.ndim != 3:
        raise ValueError(f"expected CHW image tensor, got shape {tuple(tensor.shape)}")
    if t.shape[0] not in (1, 3):
        if t.shape[-1] in (1, 3):
            t = t.permute(2, 0, 1)
        else:
            raise ValueError(f"unsupported image layout: {tuple(tensor.shape)}")
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    else:
        t = t.float()
        if t.max() > 1.5:
            t = t / 255.0
    return t.clamp(0.0, 1.0)


def preprocess_chw01(img_chw: torch.Tensor, dst_size: tuple[int, int]) -> torch.Tensor:
    """Max-scale fill + center crop, then map to [-1, 1] (matches training WATransformsLerobot)."""
    dst_w, dst_h = dst_size
    img = Image.fromarray(
        (_as_chw_float(img_chw).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    )
    w, h = img.size
    img_t = TF.to_tensor(img).unsqueeze(0)
    if dst_h / h < dst_w / w:
        new_h = int(round(dst_w / w * h))
        new_w = dst_w
    else:
        new_h = dst_h
        new_w = int(round(dst_h / h * w))
    img_t = TF.resize(img_t, (new_h, new_w), InterpolationMode.BILINEAR)
    x1 = max(0, (new_w - dst_w) // 2)
    y1 = max(0, (new_h - dst_h) // 2)
    img_t = TF.crop(img_t, y1, x1, dst_h, dst_w)
    return img_t * 2.0 - 1.0


class AgilexGWPMoTPolicy:
    """GWP-MoT policy for AgileX: T-shape 3-view layout, 14D delta-action decoding."""

    def __init__(
        self,
        checkpoint_path: str,
        pretrained_path: str = PRETRAINED_PATH,
        stats_path: str = DEFAULT_STATS_PATH,
        device: str = "cuda",
        num_steps: int = NUM_STEPS,
        num_frames: int = NUM_FRAMES,
        action_chunk: int = ACTION_CHUNK,
        replan_steps: int | None = None,
        mot_checkpoint_mixed_attn: bool = True,
        seed: int | None = None,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.num_steps = num_steps
        self.num_frames = num_frames
        self.action_chunk = action_chunk
        self.replan_steps = replan_steps or action_chunk
        self.last_state: np.ndarray | None = None
        if self.action_chunk > self.num_frames:
            raise ValueError(
                f"action_chunk ({self.action_chunk}) cannot exceed num_frames ({self.num_frames})"
            )

        print("Building GWP-MoT model for AgileX...")
        self.model_dict = build_model(
            pretrained_path,
            checkpoint_path,
            action_dim=ACTION_DIM,
            state_dim=STATE_DIM,
            action_flow_shift=ACTION_FLOW_SHIFT,
            device=device,
            dtype=self.dtype,
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        if seed is not None:
            self.model_dict["rng"] = torch.Generator(device=device).manual_seed(seed)

        print(f"Loading norm stats from {stats_path}")
        self.norm_stats = load_norm_stats(stats_path, STATE_DIM, ACTION_DIM, device)

        from transformers import AutoTokenizer, UMT5EncoderModel
        from world_action_model.trainers.wa_trainer import get_model_path

        pretrained = get_model_path(pretrained_path)
        tok_path = os.path.join(pretrained, "tokenizer")
        te_path = os.path.join(pretrained, "text_encoder")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            te_path, torch_dtype=torch.float16
        ).to(device)
        self.text_encoder.eval()
        print("T5 prompt encoding enabled (task from client)")

        self.prompt_cache: dict[str, torch.Tensor] = {}
        print(
            "AgileX GWP-MoT Policy initialized. "
            f"sample_frames={self.num_frames}, action_chunk={self.action_chunk}, "
            f"replan_steps={self.replan_steps}"
        )

    def _get_prompt_embeds(self, task: str) -> torch.Tensor:
        task = task.strip()
        if not task:
            raise ValueError("client must provide non-empty 'task' in observation")
        if task in self.prompt_cache:
            return self.prompt_cache[task]
        emb = encode_prompt_t5(
            task, self.tokenizer, self.text_encoder, self.device, self.dtype
        )
        self.prompt_cache[task] = emb
        return emb

    @torch.no_grad()
    def inference(self, data: dict[str, Any]) -> torch.Tensor:
        start_t = time.time()

        state = data.get("observation.state")
        if state is None:
            raise KeyError("observation.state is required")
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        state_np = state.detach().float().cpu().numpy().reshape(-1)[:STATE_DIM]
        if state_np.shape[0] < STATE_DIM:
            state_np = np.pad(state_np, (0, STATE_DIM - state_np.shape[0]))
        self.last_state = state_np.copy()

        images = {}
        for key in IMAGE_KEYS:
            if key not in data:
                short = key.split(".")[-1]
                raise KeyError(f"missing image key {key} (or {short})")
            images[key] = _as_chw_float(data[key])

        dst_w, dst_h = DST_SIZE
        half_w, half_h = dst_w // 2, dst_h // 2
        proc_high = preprocess_chw01(images[IMAGE_KEYS[0]], (dst_w, dst_h))
        proc_left = preprocess_chw01(images[IMAGE_KEYS[1]], (half_w, half_h))
        proc_right = preprocess_chw01(images[IMAGE_KEYS[2]], (half_w, half_h))
        wrist_row = torch.cat([proc_left, proc_right], dim=-1)
        if wrist_row.shape[-1] < proc_high.shape[-1]:
            wrist_row = torch.nn.functional.pad(
                wrist_row, (0, proc_high.shape[-1] - wrist_row.shape[-1])
            )
        ref_image = torch.cat([proc_high, wrist_row], dim=-2).to(self.device, dtype=self.dtype)

        vae = self.model_dict["vae"]
        latents_mean = self.model_dict["latents_mean"]
        latents_std = self.model_dict["latents_std"]
        ref_latents = vae.encode(ref_image.unsqueeze(2)).latent_dist.mode()
        ref_latents = (ref_latents - latents_mean) * latents_std

        state_t = torch.from_numpy(state_np).to(self.device, dtype=self.dtype).unsqueeze(0).unsqueeze(0)
        state_f32 = normalize_state(state_t.float().squeeze(0), self.norm_stats).unsqueeze(0)
        state_t = state_f32.to(dtype=self.dtype)

        task = data.get("task", "")
        if isinstance(task, bytes):
            task = task.decode("utf-8", errors="ignore")
        if not isinstance(task, str):
            task = str(task)
        prompt_embeds = self._get_prompt_embeds(task)

        pred_actions = sample_action(
            self.model_dict,
            ref_latents,
            prompt_embeds,
            state_t,
            num_steps=self.num_steps,
            action_chunk=self.num_frames,
        )

        pred_actions = denormalize_action(pred_actions.float().squeeze(0), self.norm_stats)
        actions_np = pred_actions.cpu().numpy()

        for d_idx in range(min(ACTION_DIM, STATE_DIM)):
            if d_idx < len(DELTA_MASK) and DELTA_MASK[d_idx]:
                actions_np[:, d_idx] += self.last_state[d_idx]

        execute_steps = min(self.action_chunk, self.replan_steps, actions_np.shape[0])
        actions_np = actions_np[:execute_steps]
        elapsed = time.time() - start_t
        print(
            f"  Inference {elapsed * 1000:.0f}ms, shape={actions_np.shape}, "
            f"sample_frames={self.num_frames}, action_chunk={self.action_chunk}, "
            f"replan_steps={self.replan_steps}, action[0]={actions_np[0]}"
        )
        return torch.from_numpy(actions_np)


def get_policy(
    checkpoint_path: str,
    pretrained_path: str = PRETRAINED_PATH,
    stats_path: str = DEFAULT_STATS_PATH,
    device: str = "cuda",
    num_steps: int = NUM_STEPS,
    num_frames: int = NUM_FRAMES,
    action_chunk: int = ACTION_CHUNK,
    replan_steps: int = REPLAN_STEPS,
    mot_checkpoint_mixed_attn: bool = True,
    seed: int | None = None,
) -> AgilexGWPMoTPolicy:
    return AgilexGWPMoTPolicy(
        checkpoint_path=checkpoint_path,
        pretrained_path=pretrained_path,
        stats_path=stats_path,
        device=device,
        num_steps=num_steps,
        num_frames=num_frames,
        action_chunk=action_chunk,
        replan_steps=replan_steps,
        mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        seed=seed,
    )


def run_server(
    checkpoint_path: str,
    pretrained_path: str = PRETRAINED_PATH,
    stats_path: str = DEFAULT_STATS_PATH,
    device: str = "cuda",
    num_steps: int = NUM_STEPS,
    num_frames: int = NUM_FRAMES,
    action_chunk: int = ACTION_CHUNK,
    replan_steps: int = REPLAN_STEPS,
    mot_checkpoint_mixed_attn: bool = True,
    seed: int | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    policy = get_policy(
        checkpoint_path=checkpoint_path,
        pretrained_path=pretrained_path,
        stats_path=stats_path,
        device=device,
        num_steps=num_steps,
        num_frames=num_frames,
        action_chunk=action_chunk,
        replan_steps=replan_steps,
        mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        seed=seed,
    )
    server = RobotInferenceServer(policy, host=host, port=port)
    server.run()


if __name__ == "__main__":
    tyro.cli(run_server)
