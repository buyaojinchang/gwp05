#!/usr/bin/env python3
"""Hydra training entry point for gwp-mot.

Usage:
    python scripts/train_hydra.py task=robocasa_all_tshape_mot
    accelerate launch scripts/train_hydra.py task=robocasa_all_tshape_mot

Override anything from the CLI:
    accelerate launch scripts/train_hydra.py task=robocasa_all_tshape_mot \
        train.max_epochs=1 data.batch_size_per_gpu=4

Print the composed config without training:
    python scripts/train_hydra.py task=robocasa_all_tshape_mot --cfg job --resolve
"""

import os
import sys

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_HOME", "/shared_disk/models/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/shared_disk/models/huggingface")

from world_action_model.hydra_runtime import register_default_resolvers, run_training

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
