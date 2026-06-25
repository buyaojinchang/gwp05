"""Hydra-based config composition + training entry.

This is the Hydra front-end for gwp-mot. It composes a layered config
(``train.yaml`` base + ``data``/``model``/``task`` groups) and then maps the
resolved ``OmegaConf`` into the plain ``dict`` schema the existing trainers
already consume. The trainer code itself is unchanged.
"""

import os

from omegaconf import DictConfig, OmegaConf

from .runtime import resolve_runner


def register_default_resolvers() -> None:
    """Register custom OmegaConf resolvers (idempotent)."""
    if not OmegaConf.has_resolver("pow"):
        OmegaConf.register_new_resolver("pow", lambda base, exp: float(base) ** float(exp))


def _scan_lerobot_dirs(data_root: str, categories, task_filter):
    """Reproduce the legacy directory scan: data_root/<category>/<task>/<date>/lerobot."""
    dirs: list[str] = []
    if not os.path.isdir(data_root):
        print(f"[hydra_runtime][WARN] data_root not found, no datasets scanned: {data_root}", flush=True)
        return dirs

    for category in categories:
        cat_dir = os.path.join(data_root, category)
        if not os.path.isdir(cat_dir):
            continue
        for task_name in sorted(os.listdir(cat_dir)):
            task_dir = os.path.join(cat_dir, task_name)
            if not os.path.isdir(task_dir):
                continue
            for date_dir in sorted(os.listdir(task_dir)):
                lerobot_dir = os.path.join(task_dir, date_dir, "lerobot")
                if os.path.isdir(lerobot_dir):
                    dirs.append(lerobot_dir)

    if task_filter in (None, "none", ""):
        return dirs
    if task_filter == "atomic_seen":
        from configs.robocasa_task_sets import is_atomic_seen_data_path

        return [d for d in dirs if is_atomic_seen_data_path(d)]
    raise ValueError(f"Unknown data.task_filter: {task_filter!r}")


def _resolve_data_paths(data_cfg: DictConfig) -> list[str]:
    """Resolve the list of LeRobot dirs according to ``data.layout``.

    - ``scan`` (default): recursively scan ``data_root`` over ``categories``,
      optionally filtered by ``task_filter``.
    - ``task_list``: explicit ``data_root/<task_name>`` per ``task_names``.
    """
    layout = data_cfg.get("layout", "scan")
    if layout == "scan":
        return _scan_lerobot_dirs(
            str(data_cfg.data_root),
            list(OmegaConf.to_container(data_cfg.categories, resolve=True)),
            data_cfg.get("task_filter"),
        )
    if layout == "task_list":
        data_root = str(data_cfg.data_root)
        task_names = list(OmegaConf.to_container(data_cfg.task_names, resolve=True))
        return [os.path.join(data_root, name) for name in task_names]
    raise ValueError(f"Unknown data.layout: {layout!r}")


def _build_data_or_config(data_cfg: DictConfig) -> list[dict]:
    num_frames = int(data_cfg.num_frames)
    view_keys = list(OmegaConf.to_container(data_cfg.view_keys, resolve=True))
    image_frame_offsets = [0, num_frames // 4, num_frames // 2, (3 * num_frames) // 4, num_frames]

    dirs = _resolve_data_paths(data_cfg)

    ds = data_cfg.dataset
    return [
        dict(
            _class_name=str(ds.class_name),
            data_path=p,
            data_size=None,
            delta_info={"action": num_frames},
            delta_frames={k: list(image_frame_offsets) for k in view_keys},
            video_backend=str(ds.video_backend),
            robotype=str(ds.robotype),
        )
        for p in dirs
    ]


def build_trainer_config(cfg: DictConfig) -> dict:
    """Map a composed Hydra config into the legacy trainer dict schema."""
    if cfg.get("data") is None:
        raise ValueError("No `data` group selected. Pass e.g. task=robocasa_all_tshape_mot")
    if cfg.get("model") is None:
        raise ValueError("No `model` group selected. Pass e.g. task=robocasa_all_tshape_mot")

    data_or_config = _build_data_or_config(cfg.data)

    train_dl = {
        "data_or_config": data_or_config,
        "batch_size_per_gpu": int(cfg.data.batch_size_per_gpu),
        "num_workers": int(cfg.data.num_workers),
        "transform": OmegaConf.to_container(cfg.data.transform, resolve=True),
    }
    # optional dataloader knobs (only forwarded when set on the data group)
    for key in ("sample_timeout_sec", "max_sample_retries", "timeout",
                "prefetch_factor", "persistent_workers"):
        val = cfg.data.get(key, None)
        if val is not None:
            train_dl[key] = val

    return {
        "project_dir": str(cfg.project_dir),
        "runners": list(OmegaConf.to_container(cfg.runners, resolve=True)),
        "wandb": OmegaConf.to_container(cfg.wandb, resolve=True),
        "dataloaders": {
            "train": train_dl,
            "test": {},
        },
        "models": OmegaConf.to_container(cfg.model, resolve=True),
        "optimizers": OmegaConf.to_container(cfg.optimizers, resolve=True),
        "schedulers": OmegaConf.to_container(cfg.schedulers, resolve=True),
        "train": OmegaConf.to_container(cfg.train, resolve=True),
        "test": {},
    }


def run_training(cfg: DictConfig) -> None:
    """End-to-end: compose -> map -> build trainer -> train."""
    config = build_trainer_config(cfg)

    runners = config.get("runners", [])
    if not runners:
        raise ValueError("No runners specified in config")

    runner_cls = resolve_runner(runners[0])
    trainer = runner_cls(config)
    trainer.run()
