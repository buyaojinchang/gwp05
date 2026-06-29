import json
import os
import shutil
from datetime import timedelta

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from tqdm import tqdm


class DictConfig(dict):
    """Dict that also supports attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


class ModuleDict(nn.ModuleDict):
    """nn.ModuleDict with forward dispatching by key."""

    def forward(self, key, *args, **kwargs):
        return self[key](*args, **kwargs)


def _install_process_group_timeout(timeout_sec) -> None:
    if not timeout_sec:
        return

    timeout = timedelta(seconds=int(timeout_sec))
    os.environ.setdefault("DEEPSPEED_TIMEOUT", str(max(1, int(timeout.total_seconds() // 60))))

    try:
        import torch.distributed as dist
        import torch.distributed.distributed_c10d as c10d

        dist.constants.default_pg_timeout = timeout
        c10d.default_pg_timeout = timeout

        if not hasattr(dist, "_gwp_original_new_group"):
            dist._gwp_original_new_group = dist.new_group

            def _new_group_with_default_timeout(*args, **kwargs):
                if kwargs.get("timeout") is None:
                    kwargs["timeout"] = getattr(dist, "_gwp_new_group_timeout", timeout)
                return dist._gwp_original_new_group(*args, **kwargs)

            dist.new_group = _new_group_with_default_timeout
        dist._gwp_new_group_timeout = timeout
    except Exception as exc:
        print(f"[WARN] failed to install distributed timeout patch: {exc}", flush=True)


class EMA:
    """FP32 EMA for trainable floating point parameters.

    Supports an optional step-adaptive (dynamic) decay schedule. With
    ``dynamic=False`` the behaviour is the classic constant-``decay`` EMA. With
    ``dynamic=True`` the per-step decay ramps up from ``min_decay`` toward
    ``decay`` so early shadow weights track the model quickly instead of being
    anchored to the (noisy) initial weights:

    - default schedule:      ``decay_t = (1 + t) / (10 + t)``
    - inv-gamma (diffusers): ``decay_t = 1 - (1 + t / inv_gamma) ** -power``

    where ``t = max(0, step - update_after_step - 1)`` and the result is clamped
    to ``[min_decay, decay]``. ``step`` is the running EMA update count, so the
    schedule resumes correctly from a restored checkpoint.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.995,
        device: str | torch.device | None = None,
        *,
        dynamic: bool = False,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: float = 1.0,
        power: float = 2.0 / 3.0,
    ):
        self.decay = float(decay)
        self.device = torch.device(device) if device and str(device) != "model" else None
        self.dynamic = bool(dynamic)
        self.min_decay = float(min_decay)
        self.update_after_step = int(update_after_step)
        self.use_ema_warmup = bool(use_ema_warmup)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.updates = 0
        self.last_decay = 0.0 if self.dynamic else self.decay
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        self._init_shadow(model)

    def _target_device(self, param: torch.Tensor) -> torch.device:
        return self.device or param.device

    def _init_shadow(self, model: nn.Module):
        self.shadow.clear()
        for name, param in model.named_parameters():
            if param.requires_grad and param.is_floating_point():
                self.shadow[name] = param.detach().float().to(self._target_device(param)).clone()

    @property
    def tracked_names(self) -> set[str]:
        return set(self.shadow.keys())

    def get_decay(self, step: int) -> float:
        """Step-adaptive decay clamped to ``[min_decay, decay]``."""
        if not self.dynamic:
            return self.decay
        t = max(0, int(step) - self.update_after_step - 1)
        if t <= 0:
            return 0.0
        if self.use_ema_warmup:
            cur = 1.0 - (1.0 + t / self.inv_gamma) ** (-self.power)
        else:
            cur = (1.0 + t) / (10.0 + t)
        return max(self.min_decay, min(cur, self.decay))

    def update(self, model: nn.Module):
        if not self.shadow:
            self._init_shadow(model)
        decay = self.get_decay(self.updates)
        self.last_decay = decay
        with torch.no_grad():
            for name, param in model.named_parameters():
                shadow = self.shadow.get(name)
                if shadow is None:
                    continue
                shadow.mul_(decay).add_(
                    param.detach().float().to(device=shadow.device),
                    alpha=1.0 - decay,
                )
        self.updates += 1

    def state_dict(self, cpu: bool = True):
        shadow = {
            name: tensor.detach().cpu().clone() if cpu else tensor.detach().clone()
            for name, tensor in self.shadow.items()
        }
        return {
            "decay": self.decay,
            "updates": self.updates,
            "dynamic": self.dynamic,
            "min_decay": self.min_decay,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "shadow": shadow,
        }

    def load_state_dict(self, state_dict: dict):
        if "shadow" in state_dict:
            shadow_state = state_dict.get("shadow", {})
            self.decay = float(state_dict.get("decay", self.decay))
            self.updates = int(state_dict.get("updates", self.updates))
            self.dynamic = bool(state_dict.get("dynamic", self.dynamic))
            self.min_decay = float(state_dict.get("min_decay", self.min_decay))
            self.update_after_step = int(state_dict.get("update_after_step", self.update_after_step))
            self.use_ema_warmup = bool(state_dict.get("use_ema_warmup", self.use_ema_warmup))
            self.inv_gamma = float(state_dict.get("inv_gamma", self.inv_gamma))
            self.power = float(state_dict.get("power", self.power))
        else:
            shadow_state = state_dict

        for name, tensor in shadow_state.items():
            if name not in self.shadow or not torch.is_tensor(tensor):
                continue
            self.shadow[name] = tensor.detach().float().to(device=self.shadow[name].device).clone()

    def load_from_model_state_dict(self, model_state: dict[str, torch.Tensor]):
        for name, shadow in list(self.shadow.items()):
            tensor = model_state.get(name)
            if torch.is_tensor(tensor):
                self.shadow[name] = tensor.detach().float().to(device=shadow.device).clone()

    def apply_shadow(self, model: nn.Module):
        self._backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                shadow = self.shadow.get(name)
                if shadow is None:
                    continue
                self._backup[name] = param.detach().clone()
                param.copy_(shadow.to(device=param.device, dtype=param.dtype))

    def restore(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                backup = self._backup.get(name)
                if backup is not None:
                    param.copy_(backup)
        self._backup.clear()


class Trainer:
    """Base trainer using HuggingFace Accelerate + DeepSpeed.

    Subclasses must implement:
        get_models(model_config) -> ModuleDict
        forward_step(batch_dict) -> dict[str, Tensor]
    """

    def __init__(self, config: dict):
        self.config = config
        train_cfg = config.get("train", {})

        mixed_precision = train_cfg.get("mixed_precision", "no")
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
            mixed_precision, torch.float32
        )

        gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)
        timeout_sec = (
            train_cfg.get("process_group_timeout_sec")
            or os.environ.get("TORCH_DISTRIBUTED_TIMEOUT_SEC")
            or os.environ.get("TORCH_NCCL_TIMEOUT_SEC")
        )
        _install_process_group_timeout(timeout_sec)
        accelerator_kwargs = {"gradient_accumulation_steps": gradient_accumulation_steps}
        if timeout_sec:
            accelerator_kwargs["kwargs_handlers"] = [
                InitProcessGroupKwargs(timeout=timedelta(seconds=int(timeout_sec)))
            ]
        self.accelerator = Accelerator(**accelerator_kwargs)

        self.device = self.accelerator.device
        self.process_index = self.accelerator.process_index
        self.cur_step = 0
        self._outputs: list = []
        self.model: ModuleDict | None = None

    def get_models(self, model_config: DictConfig) -> ModuleDict:
        raise NotImplementedError

    def forward_step(self, batch_dict: dict) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint, models, strict=True):
        if checkpoint is None:
            return
        if isinstance(checkpoint, (list, tuple)):
            for ckpt in checkpoint:
                self.load_checkpoint(ckpt, models, strict=strict)
            return

        if self.process_index == 0:
            print(f"Loading checkpoint: {checkpoint}")

        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        for m in models:
            # Try loading as-is first; if all keys are unexpected, strip common prefix and retry
            missing, unexpected = m.load_state_dict(state_dict, strict=False)
            if unexpected and missing:
                # Detect a shared prefix in unexpected keys (e.g. "transformer.")
                prefixes = set(k.split(".")[0] + "." for k in unexpected)
                if len(prefixes) == 1:
                    prefix = prefixes.pop()
                    stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
                    if stripped:
                        missing, unexpected = m.load_state_dict(stripped, strict=strict)
                        if self.process_index == 0:
                            print(f"  Stripped prefix '{prefix}' from checkpoint keys")
            if self.process_index == 0:
                if missing:
                    print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
                if unexpected:
                    print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    def _build_optimizer(self, model: nn.Module):
        opt_cfg = self.config.get("optimizers", {})
        opt_type = opt_cfg.get("type", "AdamW")
        lr = opt_cfg.get("lr", 1e-4)
        weight_decay = opt_cfg.get("weight_decay", 0.01)
        action_lr_mult = opt_cfg.get("action_lr_mult", 1.0)

        # Split params: action branch vs backbone
        action_keywords = ("action_encoder", "action_decoder", "state_encoder", "action_rope")
        action_params = []
        backbone_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name for kw in action_keywords):
                action_params.append(param)
            else:
                backbone_params.append(param)

        if action_lr_mult != 1.0 and action_params:
            param_groups = [
                {"params": backbone_params, "lr": lr},
                {"params": action_params, "lr": lr * action_lr_mult},
            ]
            if self.process_index == 0:
                print(f"Optimizer: backbone lr={lr:.2e}, action lr={lr * action_lr_mult:.2e} "
                      f"(×{action_lr_mult}), {len(backbone_params)} + {len(action_params)} params")
        else:
            param_groups = [{"params": backbone_params + action_params, "lr": lr}]

        if opt_type in ("CAME", "CAME8Bit"):
            try:
                from came_pytorch import CAME
                return CAME(param_groups, lr=lr, weight_decay=weight_decay)
            except ImportError:
                if self.process_index == 0:
                    print("came_pytorch not installed, falling back to AdamW")
                return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

        if opt_type == "Adam8Bit":
            try:
                import bitsandbytes as bnb
                return bnb.optim.Adam8bit(param_groups, lr=lr, weight_decay=weight_decay)
            except ImportError:
                if self.process_index == 0:
                    print("bitsandbytes not installed, falling back to AdamW")
                return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    def _build_scheduler(self, optimizer):
        sch_cfg = self.config.get("schedulers", {})
        sch_type = sch_cfg.get("type", "ConstantScheduler")
        max_steps = self._resolved_max_steps  # use the resolved value (epoch-aware)
        warmup_steps = sch_cfg.get("warmup_steps", 0)
        min_lr_ratio = sch_cfg.get("min_lr_ratio", 0.0)  # cosine decays to lr * min_lr_ratio

        if sch_type == "CosineScheduler":
            base_lr = optimizer.param_groups[0]["lr"]
            # support decay_epochs: convert to steps using the same pre-estimate logic
            decay_epochs = sch_cfg.get("decay_epochs", None)
            if decay_epochs is not None:
                train_cfg = self.config.get("train", {})
                max_epochs = train_cfg.get("max_epochs", 0)
                decay_steps = int(round(max_steps * decay_epochs / max_epochs)) if max_epochs > 0 else None
            else:
                decay_steps = sch_cfg.get("decay_steps", None)
            decay_lr = sch_cfg.get("decay_lr", None)
            # decay_lr takes priority over min_lr_ratio if both are set
            if decay_lr is not None:
                eta_min = decay_lr
            else:
                eta_min = base_lr * min_lr_ratio
            cosine_steps = (decay_steps if decay_steps is not None else max_steps) - warmup_steps

            from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
            schedulers = []
            milestones = []
            if warmup_steps > 0:
                schedulers.append(LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_steps))
                milestones.append(warmup_steps)
            schedulers.append(CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=eta_min))
            # if decay_steps < max_steps, hold at eta_min for the remaining steps
            if decay_steps is not None and decay_steps < max_steps:
                schedulers.append(ConstantLR(optimizer, factor=eta_min / base_lr, total_iters=max_steps - decay_steps))
                milestones.append((decay_steps if warmup_steps == 0 else decay_steps))
            if len(schedulers) == 1:
                return schedulers[0]
            return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)

        # ConstantScheduler (with optional warmup)
        if warmup_steps > 0:
            return torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-2, total_iters=warmup_steps
            )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    def _build_dataset(self):
        import world_action_model as _wam
        from .datasets.lerobot_dataset import LeRobotDataset

        dl_cfg = self.config["dataloaders"]["train"]

        transform_cfg = dict(dl_cfg.get("transform", {}))
        transform_type = transform_cfg.pop("type")
        transform_cls = getattr(_wam, transform_type)
        transform = transform_cls(**transform_cfg)

        data_configs = dl_cfg.get("data_or_config", [])
        is_main = int(os.environ.get("RANK", "0")) == 0

        def _load_one(dc):
            return LeRobotDataset(
                data_path=dc.get("data_path"),
                delta_info=dc.get("delta_info"),
                delta_frames=dc.get("delta_frames"),
                video_backend=dc.get("video_backend", "pyav"),
                transform=transform,
                t5_embed_path=dc.get("t5_embed_path"),
                robotype=dc.get("robotype", "aloha"),
                sample_timeout_sec=dl_cfg.get("sample_timeout_sec"),
                max_sample_retries=dl_cfg.get("max_sample_retries"),
            )

        if len(data_configs) > 5:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            datasets = [None] * len(data_configs)
            with ThreadPoolExecutor(max_workers=32) as pool:
                futures = {pool.submit(_load_one, dc): i for i, dc in enumerate(data_configs)}
                pbar = tqdm(total=len(data_configs), desc="Loading datasets", disable=not is_main)
                for fut in as_completed(futures):
                    idx = futures[fut]
                    datasets[idx] = fut.result()
                    pbar.update(1)
                pbar.close()
        else:
            datasets = [_load_one(dc) for dc in data_configs]

        if len(datasets) == 1:
            return datasets[0]
        return torch.utils.data.ConcatDataset(datasets)

    def _build_dataloader(self, dataset):
        dl_cfg = self.config["dataloaders"]["train"]

        def _as_bool(value) -> bool:
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)

        num_workers = int(dl_cfg.get("num_workers", 4))
        timeout = int(float(dl_cfg.get("timeout", 0) or 0))
        if num_workers <= 0:
            timeout = 0
        kwargs = dict(
            batch_size=int(dl_cfg.get("batch_size_per_gpu", 1)),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            timeout=timeout,
        )
        if num_workers > 0:
            if "prefetch_factor" in dl_cfg:
                kwargs["prefetch_factor"] = int(dl_cfg["prefetch_factor"])
            if "persistent_workers" in dl_cfg:
                kwargs["persistent_workers"] = _as_bool(dl_cfg["persistent_workers"])
        if self.process_index == 0:
            print(f"DataLoader config: {kwargs}", flush=True)
        return DataLoader(dataset, **kwargs)

    def _dump_dataloader_worker_state(self):
        state_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "gwp_dataloader_state")
        if not os.path.isdir(state_dir):
            print(f"[DataLoader debug] no worker state dir found: {state_dir}", flush=True)
            return

        state_files = sorted(
            os.path.join(state_dir, name)
            for name in os.listdir(state_dir)
            if name.endswith(".json")
        )
        if not state_files:
            print(f"[DataLoader debug] no worker state files found in {state_dir}", flush=True)
            return

        latest_by_worker = {}
        for path in state_files:
            try:
                with open(path) as f:
                    state = json.load(f)
                key = (str(state.get("rank")), str(state.get("worker")))
                current = latest_by_worker.get(key)
                if current is None or str(state.get("time", "")) > str(current.get("time", "")):
                    latest_by_worker[key] = state
            except Exception as exc:
                print(f"[DataLoader debug] failed to read {path}: {exc}", flush=True)

        states = sorted(
            latest_by_worker.values(),
            key=lambda state: (str(state.get("rank")), str(state.get("worker"))),
        )
        print(
            f"[DataLoader debug] latest worker states from {state_dir} "
            f"({len(states)} current, {len(state_files)} files):",
            flush=True,
        )
        for state in states:
            try:
                views = state.get("views", {})
                view_text = "; ".join(
                    f"{key}: {value.get('video_path')} frames={value.get('frame_indices')}"
                    for key, value in views.items()
                )
                print(
                    "[DataLoader debug] "
                    f"rank={state.get('rank')} worker={state.get('worker')} pid={state.get('pid')} "
                    f"time={state.get('time')} idx={state.get('idx')} "
                    f"episode={state.get('episode_name')} start={state.get('start')} "
                    f"len={state.get('episode_length')} {view_text}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[DataLoader debug] failed to read {path}: {exc}", flush=True)

    def _find_latest_checkpoint(self, project_dir: str) -> str | None:
        if not os.path.isdir(project_dir):
            return None

        checkpoints = []
        for name in os.listdir(project_dir):
            ckpt_dir = os.path.join(project_dir, name)
            if not name.startswith("checkpoint-") or not os.path.isdir(ckpt_dir):
                continue
            try:
                step_num = int(name.split("-")[1])
            except (IndexError, ValueError):
                continue
            checkpoints.append((step_num, ckpt_dir))
        if not checkpoints:
            return None

        for _, ckpt_dir in sorted(checkpoints, reverse=True):
            model_path = os.path.join(ckpt_dir, "model.pt")
            state_path = os.path.join(ckpt_dir, "training_state.pt")
            full_state_dir = os.path.join(ckpt_dir, "accelerator_state")
            if os.path.exists(state_path) and (os.path.exists(model_path) or os.path.isdir(full_state_dir)):
                return ckpt_dir
            if self.process_index == 0:
                print(f"[WARN] Skipping incomplete checkpoint: {ckpt_dir}", flush=True)
        return None

    def _load_raw_model_checkpoint(self, model_path: str) -> None:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = self._extract_model_state(state_dict)
        target = self.accelerator.unwrap_model(self.model) if hasattr(self.model, "module") else self.model
        missing, unexpected = target.load_state_dict(state_dict, strict=False)
        if self.process_index == 0:
            print(f"Loaded raw model checkpoint from {model_path}")
            if missing:
                print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
            if unexpected:
                print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    def run(self):
        project_dir = self.config.get("project_dir", "./output")
        os.makedirs(project_dir, exist_ok=True)
        train_cfg = self.config.get("train", {})

        model_config = DictConfig(self.config["models"])
        self.model = self.get_models(model_config)

        resume_ckpt_dir = None
        resume_raw_preloaded = False
        if train_cfg.get("resume", False):
            resume_ckpt_dir = self._find_latest_checkpoint(project_dir)
            if resume_ckpt_dir is not None:
                full_state_dir = os.path.join(resume_ckpt_dir, "accelerator_state")
                model_path = os.path.join(resume_ckpt_dir, "model.pt")
                if not os.path.isdir(full_state_dir) and os.path.exists(model_path):
                    # Raw checkpoints must be loaded before DeepSpeed shards the model.
                    self._load_raw_model_checkpoint(model_path)
                    resume_raw_preloaded = True

        dataset = self._build_dataset()
        dataloader = self._build_dataloader(dataset)
        if self.process_index == 0:
            print(f"Dataset size: {len(dataset)}, Dataloader batches: {len(dataloader)}")

        optimizer = self._build_optimizer(self.model)

        if self.process_index == 0:
            print("Preparing model with DeepSpeed (this may take a few minutes)...")
        self.model, optimizer, dataloader = self.accelerator.prepare(
            self.model, optimizer, dataloader,
        )
        if self.process_index == 0:
            print("DeepSpeed ready.")

        # Compute max_steps from actual post-prepare dataloader length
        max_steps = train_cfg.get("max_steps", 100000)
        max_epochs = train_cfg.get("max_epochs", 0)
        if max_epochs > 0:
            steps_per_epoch = len(dataloader)
            max_steps = max_epochs * steps_per_epoch
            if self.process_index == 0:
                print(f"Epoch mode: max_epochs={max_epochs}, steps_per_epoch={steps_per_epoch}, max_steps={max_steps}")
        self._resolved_max_steps = max_steps

        # Build scheduler AFTER prepare so it uses the correct max_steps
        scheduler = self._build_scheduler(optimizer)

        max_steps = self._resolved_max_steps
        checkpoint_interval = train_cfg.get("checkpoint_interval", 5000)
        checkpoint_epoch_interval = int(train_cfg.get("checkpoint_epoch_interval", 0) or 0)
        log_interval = train_cfg.get("log_interval", 1)
        ema_cfg = train_cfg.get("ema", {}) or {}
        with_ema = bool(ema_cfg.get("enabled", train_cfg.get("with_ema", False)))
        ema_decay = float(ema_cfg.get("decay", train_cfg.get("ema_decay", 0.995)))
        ema_device = ema_cfg.get("device", train_cfg.get("ema_device", "model"))

        unwrapped = self.accelerator.unwrap_model(self.model)
        ema = (
            EMA(
                unwrapped,
                decay=ema_decay,
                device=ema_device,
                dynamic=bool(ema_cfg.get("dynamic", False)),
                min_decay=float(ema_cfg.get("min_decay", 0.0)),
                update_after_step=int(ema_cfg.get("update_after_step", 0)),
                use_ema_warmup=bool(ema_cfg.get("use_ema_warmup", False)),
                inv_gamma=float(ema_cfg.get("inv_gamma", 1.0)),
                power=float(ema_cfg.get("power", 2.0 / 3.0)),
            )
            if with_ema
            else None
        )
        if self.process_index == 0 and ema is not None:
            if ema.dynamic:
                sched = "inv_gamma" if ema.use_ema_warmup else "step_ratio"
                print(
                    f"EMA enabled: dynamic decay ({sched}) -> [{ema.min_decay}, {ema.decay}], "
                    f"update_after_step={ema.update_after_step}, device={ema_device}, "
                    f"tracked={len(ema.shadow)} trainable floating tensors"
                )
            else:
                print(
                    f"EMA enabled: decay={ema.decay}, device={ema_device}, "
                    f"tracked={len(ema.shadow)} trainable floating tensors"
                )

        if self.process_index == 0:
            wandb_cfg = self.config.get("wandb", {})
            wandb_settings = wandb.Settings(
                init_timeout=int(wandb_cfg.get("init_timeout", os.environ.get("WANDB_INIT_TIMEOUT", 300)))
            )
            wandb.init(
                project=wandb_cfg.get("project", "gwp-mot"),
                name=wandb_cfg.get("name", os.path.basename(project_dir)),
                config=self.config,
                dir=project_dir,
                resume="allow",
                mode=wandb_cfg.get("mode", "online"),
                settings=wandb_settings,
            )

        if train_cfg.get("resume", False):
            self._resume_restored_scheduler = False
            self.cur_step = self._try_resume(
                project_dir,
                ema,
                optimizer,
                scheduler,
                ckpt_dir=resume_ckpt_dir,
                raw_model_preloaded=resume_raw_preloaded,
            )
            if self.cur_step:
                if not getattr(self, "_resume_restored_scheduler", False):
                    # Raw checkpoints store model/EMA/step but not optimizer state.
                    # Fast-forward the fresh scheduler so LR stays on the original curve.
                    for _ in range(self.cur_step):
                        scheduler.step()
                    if self.process_index == 0:
                        print(f"Advanced scheduler to resumed step {self.cur_step}")

        self.model.train()
        if self.process_index == 0:
            print(f"Starting training from step {self.cur_step}, max_steps={max_steps}")

        pbar = None
        if self.process_index == 0:
            pbar = tqdm(initial=self.cur_step, total=max_steps, desc="Training", unit="step")

        completed_epochs = 0
        while self.cur_step < max_steps:
            dataloader_iter = iter(dataloader)
            while self.cur_step < max_steps:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    break
                except Exception:
                    self._dump_dataloader_worker_state()
                    raise

                if self.cur_step >= max_steps:
                    break

                with self.accelerator.accumulate(self.model):
                    losses = self.forward_step(batch)
                    total_loss = sum(losses.values())
                    self.accelerator.backward(total_loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                self.cur_step += 1

                if ema is not None:
                    ema.update(unwrapped)

                if self.process_index == 0 and self.cur_step % log_interval == 0:
                    log_dict = {}
                    postfix = {}
                    for k, v in losses.items():
                        val = v.item() if hasattr(v, "item") else v
                        postfix[k] = f"{val:.4f}"
                        log_dict[f"train/{k}"] = val
                    log_dict["train/lr"] = optimizer.param_groups[0]["lr"]
                    if ema is not None:
                        log_dict["ema/updates"] = ema.updates
                        log_dict["ema/decay"] = ema.last_decay
                    postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
                    wandb.log(log_dict, step=self.cur_step)
                    if pbar is not None:
                        pbar.set_postfix(postfix)

                if pbar is not None:
                    pbar.update(1)

                if checkpoint_interval > 0 and self.cur_step % checkpoint_interval == 0:
                    self._save_checkpoint(project_dir, ema, optimizer, scheduler)

                self._outputs.clear()

            completed_epochs += 1
            if (
                checkpoint_epoch_interval > 0
                and completed_epochs % checkpoint_epoch_interval == 0
                and self.cur_step < max_steps
            ):
                self._save_checkpoint(project_dir, ema, optimizer, scheduler)

        if pbar is not None:
            pbar.close()

        self._save_checkpoint(project_dir, ema, optimizer, scheduler)
        self.accelerator.end_training()
        if self.process_index == 0:
            wandb.finish()
            print("Training finished.")

    @staticmethod
    def _trainable_floating_param_names(model: nn.Module) -> set[str]:
        return {
            name
            for name, param in model.named_parameters()
            if param.requires_grad and param.is_floating_point()
        }

    @staticmethod
    def _state_dict_for_save(
        model: nn.Module,
        fp32_names: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        state = {}
        for name, tensor in model.state_dict().items():
            if torch.is_tensor(tensor):
                out = tensor.detach().cpu()
                if fp32_names is not None and name in fp32_names and tensor.is_floating_point():
                    out = out.float()
                state[name] = out
            else:
                state[name] = tensor
        return state

    @staticmethod
    def _state_dict_with_ema(
        model: nn.Module,
        ema: EMA,
        fp32_names: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        state = Trainer._state_dict_for_save(model, fp32_names=fp32_names)
        for name, shadow in ema.shadow.items():
            if name in state:
                state[name] = shadow.detach().cpu().float().clone()
        return state

    @staticmethod
    def _extract_model_state(state_dict: dict) -> dict:
        if "state_dict" in state_dict:
            return state_dict["state_dict"]
        if "model_state_dict" in state_dict:
            return state_dict["model_state_dict"]
        return state_dict

    @staticmethod
    def _torch_save_atomic(obj, path: str) -> None:
        tmp_path = f"{path}.tmp.{os.getpid()}"
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _save_checkpoint(
        self,
        project_dir: str,
        ema: EMA | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ):
        self.accelerator.wait_for_everyone()
        save_dir = os.path.join(project_dir, f"checkpoint-{self.cur_step}")

        if self.process_index == 0:
            os.makedirs(save_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()

        train_cfg = self.config.get("train", {})
        save_full_state_value = train_cfg.get("save_full_state", False)
        if isinstance(save_full_state_value, str):
            save_full_state = save_full_state_value.strip().lower() in ("1", "true", "yes", "on")
        else:
            save_full_state = bool(save_full_state_value)
        meta_path = os.path.join(save_dir, "checkpoint_meta.json")

        if self.process_index == 0:
            unwrapped = self.accelerator.unwrap_model(self.model)
            trainable_names = self._trainable_floating_param_names(unwrapped)
            raw_state = self._state_dict_for_save(unwrapped, fp32_names=trainable_names)
            self._torch_save_atomic(raw_state, os.path.join(save_dir, "model.pt"))

            if ema is not None:
                ema_state = self._state_dict_with_ema(unwrapped, ema, fp32_names=ema.tracked_names)
                self._torch_save_atomic(ema_state, os.path.join(save_dir, "model_ema.pt"))
                self._torch_save_atomic(ema.state_dict(cpu=True), os.path.join(save_dir, "ema_state.pt"))

            self._torch_save_atomic(
                {
                    "cur_step": self.cur_step,
                    "ema_updates": ema.updates if ema is not None else 0,
                },
                os.path.join(save_dir, "training_state.pt"),
            )
            if scheduler is not None:
                self._torch_save_atomic(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))

            meta = {
                "format": "world_action_model_full_checkpoint",
                "raw_model_path": "model.pt",
                "raw_fp32_trainable": True,
                "num_trainable_tensors": len(trainable_names),
                "ema_enabled": ema is not None,
                "full_accelerator_state": False,
            }
            if scheduler is not None:
                meta["scheduler_state_path"] = "scheduler.pt"
            if ema is not None:
                meta.update({
                    "ema_model_path": "model_ema.pt",
                    "ema_state_path": "ema_state.pt",
                    "ema_decay": ema.decay,
                    "ema_updates": ema.updates,
                    "ema_fp32_trainable": True,
                    "ema_tracked_tensors": len(ema.tracked_names),
                })
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"Checkpoint saved to {save_dir}")
        self.accelerator.wait_for_everyone()

        full_state_saved = False
        full_state_dir = os.path.join(save_dir, "accelerator_state")
        if save_full_state:
            try:
                self.accelerator.save_state(full_state_dir)
                full_state_saved = True
            except Exception as exc:
                if self.process_index == 0:
                    print(f"[WARN] Failed to save full accelerator state: {exc}", flush=True)
        self.accelerator.wait_for_everyone()

        if save_full_state and not full_state_saved and self.process_index == 0:
            shutil.rmtree(full_state_dir, ignore_errors=True)

        if full_state_saved and self.process_index == 0:
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                meta["full_accelerator_state"] = True
                meta["full_state_path"] = "accelerator_state"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                print(f"Full accelerator state saved to {full_state_dir}")
            except Exception as exc:
                print(f"[WARN] Failed to update full-state checkpoint metadata: {exc}", flush=True)
        self.accelerator.wait_for_everyone()

    def _try_resume(
        self,
        project_dir: str,
        ema: EMA | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        *,
        ckpt_dir: str | None = None,
        raw_model_preloaded: bool = False,
    ) -> int:
        if ckpt_dir is None:
            ckpt_dir = self._find_latest_checkpoint(project_dir)
        if ckpt_dir is None:
            return 0

        full_state_dir = os.path.join(ckpt_dir, "accelerator_state")
        full_state_loaded = False
        if os.path.isdir(full_state_dir):
            if self.process_index == 0:
                print(f"Loading full accelerator state from {full_state_dir}")
            self.accelerator.load_state(full_state_dir)
            full_state_loaded = True
        elif self.process_index == 0:
            print(
                f"[WARN] No full accelerator state found in {ckpt_dir}; "
                "resuming model weights only, with a fresh optimizer state.",
                flush=True,
            )

        model_path = os.path.join(ckpt_dir, "model.pt")
        if not full_state_loaded and not raw_model_preloaded and os.path.exists(model_path):
            if self.process_index == 0:
                print(
                    "[WARN] Raw model checkpoint was not preloaded before DeepSpeed prepare; "
                    "loading it into the prepared model as a fallback.",
                    flush=True,
                )
            self._load_raw_model_checkpoint(model_path)

        scheduler_path = os.path.join(ckpt_dir, "scheduler.pt")
        if scheduler is not None and os.path.exists(scheduler_path) and full_state_loaded:
            scheduler_state = torch.load(scheduler_path, map_location="cpu", weights_only=False)
            scheduler.load_state_dict(scheduler_state)
            self._resume_restored_scheduler = True
            if self.process_index == 0:
                print(f"Loaded scheduler state from {scheduler_path}")
        elif scheduler is not None and os.path.exists(scheduler_path) and self.process_index == 0:
            print(
                f"Found scheduler state at {scheduler_path}, but raw resume has a fresh optimizer; "
                "will fast-forward scheduler instead.",
                flush=True,
            )

        step = 0
        state_path = os.path.join(ckpt_dir, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            step = state.get("cur_step", 0)
            if ema is not None:
                ema.updates = int(state.get("ema_updates", ema.updates))
            if self.process_index == 0:
                print(f"Resumed from {ckpt_dir} at step {step}")

        if ema is not None:
            ema_state_path = os.path.join(ckpt_dir, "ema_state.pt")
            ema_model_path = os.path.join(ckpt_dir, "model_ema.pt")
            if os.path.exists(ema_state_path):
                ema_state = torch.load(ema_state_path, map_location="cpu", weights_only=False)
                ema.load_state_dict(ema_state)
                if self.process_index == 0:
                    print(f"Loaded EMA state from {ema_state_path} (updates={ema.updates})")
            elif os.path.exists(ema_model_path):
                ema_model_state = torch.load(ema_model_path, map_location="cpu", weights_only=False)
                ema.load_from_model_state_dict(self._extract_model_state(ema_model_state))
                if self.process_index == 0:
                    print(f"Seeded EMA shadow from full EMA checkpoint {ema_model_path}")

        if step:
            return step

        return 0
