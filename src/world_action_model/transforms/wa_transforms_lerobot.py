import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as torch_F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from .wa_transforms import WATransforms


class WATransformsLerobot(WATransforms):
    def __init__(
        self,
        is_train=False,
        dst_size=None,
        num_frames=1,
        fps=16,
        norm_path=None,
        robotype_to_embed_id=None,
        robotype_default_embed_id=0,
        model_action_dim=None,
        model_state_dim=None,
        image_cfg=None,
        num_views=1,
        view_keys=None,
        state_key="observation.state",
        state_keys=None,
        state_token_dims=None,
        state_norm_keys=None,
        action_key="action",
        task_key="task",
        t5_len=64,
        skip_action_norm=False,
        tshape=False,
        tshape_head_index=0,
        random_shift_pad=0,
        resize_mode="crop",
    ):
        if norm_path is None:
            raise ValueError("norm_path is None")
        if isinstance(norm_path, (str, os.PathLike)):
            norm_paths = [str(norm_path)]
        else:
            norm_paths = [str(p) for p in norm_path]
            if len(norm_paths) == 0:
                raise ValueError("norm_path list is empty")

        super().__init__(
            is_train=is_train,
            dst_size=dst_size,
            num_frames=num_frames,
            fps=fps,
            norm_path=norm_paths[0],
            image_cfg=image_cfg,
            num_views=num_views,
        )

        self.robotype_default_embed_id = int(robotype_default_embed_id)
        self.robotype_to_embed_id = dict(robotype_to_embed_id)
        self.model_action_dim = None if model_action_dim is None else int(model_action_dim)
        # model_state_dim defaults to model_action_dim for backward compat (aloha/agibot share dim)
        self.model_state_dim = int(model_state_dim) if model_state_dim is not None else self.model_action_dim

        self.norm_paths = norm_paths
        self.stats_dicts = []
        for json_path in self.norm_paths:
            with open(json_path, "r", encoding="utf-8") as f:
                self.stats_dicts.append(json.load(f))
            if os.environ.get("RANK", "0") == "0":
                print("Loading stats dict from:", json_path)

        if view_keys is None:
            view_keys = [
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            ]
        self.view_keys = list(view_keys)
        self.state_key = state_key
        self.state_keys = list(state_keys) if state_keys is not None else [state_key]
        if state_token_dims is None:
            self.state_token_dims = None
        else:
            self.state_token_dims = [int(x) for x in state_token_dims]
            if len(self.state_token_dims) != len(self.state_keys):
                raise ValueError(
                    f"state_token_dims length {len(self.state_token_dims)} must match "
                    f"state_keys length {len(self.state_keys)}"
                )
        self.state_norm_keys = list(state_norm_keys) if state_norm_keys is not None else list(self.state_keys)
        if len(self.state_norm_keys) != len(self.state_keys):
            raise ValueError(
                f"state_norm_keys length {len(self.state_norm_keys)} must match "
                f"state_keys length {len(self.state_keys)}"
            )
        self.multi_state_mode = len(self.state_keys) > 1
        self.action_key = action_key
        self.task_key = task_key
        self.t5_len = int(t5_len)
        self.skip_action_norm = bool(skip_action_norm)
        self.tshape = bool(tshape)
        self.tshape_head_index = int(tshape_head_index)
        self.random_shift_pad = int(random_shift_pad or 0)
        self.resize_mode = str(resize_mode)
        if self.resize_mode not in ("crop", "stretch"):
            raise ValueError(f"Unsupported resize_mode={self.resize_mode!r}; expected 'crop' or 'stretch'")
        self._warned_unknown_robotype = False
        self._warned_stats_fallback = False

    def _parse_robotype(self, robotype):
        if robotype is None:
            return None
        if isinstance(robotype, bytes):
            robotype = robotype.decode("utf-8", errors="ignore")
        if hasattr(robotype, "item"):
            try:
                robotype = robotype.item()
            except Exception:
                pass
        if isinstance(robotype, str):
            robotype = robotype.strip()
        return robotype

    def _get_robotype_embed_id(self, data_dict) -> int:
        robotype = self._parse_robotype(data_dict.get("robotype", None))
        if robotype in self.robotype_to_embed_id:
            return int(self.robotype_to_embed_id[robotype])
        if isinstance(robotype, str):
            robotype_l = robotype.lower()
            if "agibot" in robotype_l and "agibot" in self.robotype_to_embed_id:
                return int(self.robotype_to_embed_id["agibot"])
            if "aloha" in robotype_l and "aloha" in self.robotype_to_embed_id:
                return int(self.robotype_to_embed_id["aloha"])
            if "agilex" in robotype_l and "agilex" in self.robotype_to_embed_id:
                return int(self.robotype_to_embed_id["agilex"])
        if not self._warned_unknown_robotype:
            if os.environ.get("RANK", "0") == "0":
                print(f"Unknown robotype={robotype!r}, fallback to {self.robotype_default_embed_id}")
            self._warned_unknown_robotype = True
        return self.robotype_default_embed_id

    def _get_stats_dict(self, embed_id: int):
        if not self.stats_dicts:
            return self.stats_dict
        # Single norm file → always use it regardless of embed_id
        if len(self.stats_dicts) == 1:
            return self.stats_dicts[0]
        if 0 <= embed_id < len(self.stats_dicts):
            return self.stats_dicts[embed_id]
        if not self._warned_stats_fallback:
            print(f"[RANK {os.environ.get('RANK', '?')}] robotype_embed_id={embed_id} out of range for norm_paths (len={len(self.stats_dicts)}), fallback to 0")
            self._warned_stats_fallback = True
        return self.stats_dicts[0]

    def _to_nchw_uint8(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            if x.shape[0] in (1, 3):
                x = x[None, ...]
            elif x.shape[-1] in (1, 3):
                x = x.permute(2, 0, 1)[None, ...]
            else:
                x = x[None, ...]
        if x.dim() != 4:
            raise ValueError(f"Unexpected image tensor shape: {tuple(x.shape)}")
        if x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dtype != torch.uint8:
            x_f = x.to(dtype=torch.float32)
            x_max = float(x_f.max().item()) if x_f.numel() > 0 else 0.0
            if x_max <= 1.0:
                x_f = x_f * 255.0
            x = x_f.clamp(0.0, 255.0).to(dtype=torch.uint8)
        return x

    def _process_images(self, input_images: torch.Tensor, dst_width: int, dst_height: int) -> torch.Tensor:
        input_images = input_images.to(dtype=torch.float32) / 255.0
        if self.resize_mode == "stretch":
            input_images = F.resize(input_images, (dst_height, dst_width), InterpolationMode.BILINEAR)
        else:
            height = int(input_images.shape[2])
            width = int(input_images.shape[3])
            if float(dst_height) / height < float(dst_width) / width:
                new_height = int(round(float(dst_width) / width * height))
                new_width = dst_width
            else:
                new_height = dst_height
                new_width = int(round(float(dst_height) / height * width))
            input_images = F.resize(input_images, (new_height, new_width), InterpolationMode.BILINEAR)
            if self.is_train:
                x1 = random.randint(0, new_width - dst_width)
                y1 = random.randint(0, new_height - dst_height)
            else:
                x1 = (new_width - dst_width) // 2
                y1 = (new_height - dst_height) // 2
            input_images = F.crop(input_images, y1, x1, dst_height, dst_width)
        if self.is_train and self.random_shift_pad > 0:
            max_pad = min(self.random_shift_pad, (dst_width - 1) // 4, (dst_height - 1) // 4)
            if max_pad > 0:
                inner_width = dst_width - 2 * max_pad
                inner_height = dst_height - 2 * max_pad
                input_images = F.resize(input_images, (inner_height, inner_width), InterpolationMode.BILINEAR)
                pad_left = random.randint(0, 2 * max_pad)
                pad_top = random.randint(0, 2 * max_pad)
                pad_right = dst_width - inner_width - pad_left
                pad_bottom = dst_height - inner_height - pad_top
                input_images = torch_F.pad(
                    input_images,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="replicate",
                )
        input_images = self.normalize(input_images)
        return input_images

    def __call__(self, data_dict):
        if self.dst_size is None:
            raise ValueError("dst_size is required")
        dst_width, dst_height = self.dst_size

        if "robotype" not in data_dict:
            raise KeyError("Missing robotype key")
        robotype_embed_id = self._get_robotype_embed_id(data_dict)
        stats_dict = self._get_stats_dict(robotype_embed_id)

        views = []
        for k in self.view_keys[: self.num_views]:
            if k not in data_dict:
                raise KeyError(f"Missing view key: {k}")
            v = data_dict[k]
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"Unsupported image type for {k}: {type(v)}")
            v = self._to_nchw_uint8(v)
            views.append(v)

        if self.tshape and len(views) > 1:
            # T-shape layout: head view at full size on top, others at half size on bottom
            head = self._process_images(views[self.tshape_head_index], dst_width=dst_width, dst_height=dst_height)
            half_w, half_h = dst_width // 2, dst_height // 2
            others = []
            for i, v in enumerate(views):
                if i == self.tshape_head_index:
                    continue
                others.append(self._process_images(v, dst_width=half_w, dst_height=half_h))
            wrist_row = torch.cat(others, dim=-1)  # (T, C, half_h, half_w * N)
            # Pad wrist_row width to match head width if needed
            if wrist_row.shape[-1] < head.shape[-1]:
                wrist_row = torch_F.pad(wrist_row, (0, head.shape[-1] - wrist_row.shape[-1]))
            elif wrist_row.shape[-1] > head.shape[-1]:
                wrist_row = wrist_row[..., :head.shape[-1]]
            input_images = torch.cat([head, wrist_row], dim=-2)  # vertical concat: head on top
        else:
            for i in range(len(views)):
                views[i] = self._process_images(views[i], dst_width=dst_width, dst_height=dst_height)
            if len(views) == 1:
                input_images = views[0]
            else:
                input_images = torch.cat(views, dim=-1)

        data_dict["input_images"] = input_images

        if self.image_cfg is not None:
            ref_masks, ref_latent_masks = self.mask_generator.get_mask(data_dict["input_images"].shape[0])
            ref_masks = ref_masks[:, None, None, None]
            ref_latent_masks = ref_latent_masks[None, :, None, None]
            ref_images = data_dict["input_images"].clone() * ref_masks
            data_dict["input_ref_images"] = ref_images
            data_dict["input_ref_masks"] = ref_latent_masks

        action = data_dict[self.action_key]
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        action = action.to(dtype=torch.float32)
        if action.dim() == 1:
            action = action[None, :]

        if action.shape[0] != self.num_frames:
            t = int(self.num_frames)
            cur_t = int(action.shape[0])
            if cur_t >= t:
                action = action[:t]
            else:
                pad = torch.zeros((t - cur_t, action.shape[1]), dtype=action.dtype, device=action.device)
                action = torch.cat([action, pad], dim=0)

        assert self.model_action_dim is not None, "model_action_dim must be provided"
        ad = int(self.model_action_dim)
        sd = int(self.model_state_dim) if self.model_state_dim is not None else ad

        def _pad_truncate_last(x: torch.Tensor, target_dim: int) -> torch.Tensor:
            if x.shape[-1] > target_dim:
                return x[..., :target_dim]
            if x.shape[-1] < target_dim:
                return torch_F.pad(x, (0, target_dim - int(x.shape[-1])), value=0.0)
            return x

        def _as_float_2d(value, key: str) -> torch.Tensor:
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Unsupported state type for {key}: {type(value)}")
            value = value.to(dtype=torch.float32)
            if value.dim() == 1:
                value = value[None, :]
            if value.dim() != 2:
                raise ValueError(f"Expected state token {key} as [T,D] or [D], got {tuple(value.shape)}")
            return value[:1]

        def _to_padded_1d(x, target_dim, pad_value: float, device):
            t = torch.as_tensor(x, dtype=torch.float32, device=device).flatten()
            if int(t.numel()) >= target_dim:
                return t[:target_dim]
            out = torch.full((target_dim,), float(pad_value), dtype=torch.float32, device=device)
            if int(t.numel()) > 0:
                out[: t.numel()] = t
            return out

        def _stats_field(norm_key: str):
            return stats_dict.get("norm_stats", {}).get(norm_key, None)

        def _normalize_state_token(token: torch.Tensor, norm_key: str, token_dim: int) -> torch.Tensor:
            token = _pad_truncate_last(token, token_dim)
            field = _stats_field(norm_key)
            if field is None:
                mean = torch.zeros(token_dim, dtype=torch.float32, device=token.device)
                std = torch.ones(token_dim, dtype=torch.float32, device=token.device)
            else:
                mean = _to_padded_1d(field.get("mean", []), token_dim, 0.0, token.device)
                std = _to_padded_1d(field.get("std", []), token_dim, 1.0, token.device)
            zero_mask = std < 1e-4
            norm_token = (token - mean) / std.clamp_min(1e-8)
            norm_token[..., zero_mask] = 0.0
            return norm_token

        # Pad / truncate action to model_action_dim.
        action = _pad_truncate_last(action, ad)

        # Delta mask templates — only used when action_dim == state_dim (dims semantically aligned).
        delta_mask_templates = {
            0: np.array([True, True, True, True, True, True, False, True, True, True, True, True, True, False], dtype=bool),
            1: np.array([True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, False], dtype=bool),
            2: np.array([False] * 16, dtype=bool),  # robocasa PandaOmron: action is already delta-like
            3: np.array([False] * 66, dtype=bool),  # g1 sonic latent: 66-d, no delta, all dims supervised
        }
        base = delta_mask_templates.get(robotype_embed_id, None)
        assert base is not None, f"robotype_embed_id {robotype_embed_id} not found in delta_mask_templates"

        state_for_delta = None
        if self.multi_state_mode:
            state_tokens = []
            state_valid = []
            token_dims = self.state_token_dims or [sd] * len(self.state_keys)
            for key, norm_key, token_dim in zip(self.state_keys, self.state_norm_keys, token_dims):
                if key in data_dict:
                    token = _as_float_2d(data_dict[key], key)
                    token = _normalize_state_token(token, norm_key, int(token_dim))
                    token = _pad_truncate_last(token, sd)
                    state_valid.append(True)
                else:
                    token = torch.zeros((1, sd), dtype=torch.float32, device=action.device)
                    state_valid.append(False)
                state_tokens.append(token.squeeze(0))
            norm_state = torch.stack(state_tokens, dim=0)
            state_mask = torch.as_tensor(state_valid, dtype=torch.bool, device=norm_state.device)
        else:
            if self.state_key not in data_dict:
                raise KeyError(f"Missing state key: {self.state_key}")
            state = _as_float_2d(data_dict[self.state_key], self.state_key)
            state_for_delta = _pad_truncate_last(state, sd)
            norm_state = _normalize_state_token(state_for_delta, self.state_norm_keys[0], sd)
            state_mask = torch.ones((1,), dtype=torch.bool, device=norm_state.device)

        # Delta: action - state only for the legacy single-token aligned state path.
        delta = action.clone()
        if state_for_delta is not None and ad == sd:
            mask = base[:ad] if len(base) >= ad else np.pad(base, (0, ad - len(base)), constant_values=False)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=action.device)
            idx = torch.nonzero(mask_t, as_tuple=False).flatten()
            if idx.numel() > 0:
                delta[:, idx] = action[:, idx] - state_for_delta[:, idx]

        action_field = _stats_field("action") or {}
        delta_mean = _to_padded_1d(action_field.get("mean", []), ad, 0.0, action.device)
        delta_std = _to_padded_1d(action_field.get("std", []), ad, 1.0, action.device)
        action_zero_mask = delta_std < 1e-4

        if self.skip_action_norm:
            norm_delta = delta.clone()
            norm_delta[..., action_zero_mask] = 0.0
        else:
            norm_delta = (delta - delta_mean) / delta_std.clamp_min(1e-8)
            norm_delta[..., action_zero_mask] = 0.0

        prompt = data_dict.get("t5_embedding", None)
        if prompt is None:
            t5_len = int(self.t5_len)
            prompt_embeds = torch.zeros(t5_len, 4096, dtype=torch.float32)
        else:
            if isinstance(prompt, np.ndarray):
                prompt = torch.from_numpy(prompt)
            prompt = prompt.to(dtype=torch.float32)
            t5_len = int(self.t5_len)
            prompt = prompt[:t5_len]
            prompt_embeds = torch_F.pad(prompt, (0, 0, 0, t5_len - prompt.shape[0]), value=0)

        out = {}
        out["fps"] = torch.tensor(self.fps, dtype=torch.float32)
        out["images"] = data_dict["input_images"]
        out["ref_images"] = data_dict.get("input_ref_images", None)
        out["ref_masks"] = data_dict.get("input_ref_masks", None)
        out["prompt_embeds"] = prompt_embeds
        out["action"] = norm_delta
        out["state"] = norm_state
        out["state_mask"] = state_mask
        out["robotype_embed_id"] = torch.tensor(int(robotype_embed_id), dtype=torch.long)

        keys = list(out.keys())
        for k in keys:
            if out[k] is None:
                out.pop(k)

        # action_dim_mask: marks which dims of action are real (not padding)
        effective_action_dim = int(np.asarray(base).shape[0]) if base is not None else ad
        effective_action_dim = min(effective_action_dim, ad)
        out["action_dim_mask"] = (torch.arange(ad, device=out["action"].device) < effective_action_dim).to(dtype=torch.bool)
        # Also mask out zero-std action dims (e.g. robocasa ee_rot_rx)
        out["action_dim_mask"] = out["action_dim_mask"] & (~action_zero_mask.to(out["action_dim_mask"].device))

        return out
