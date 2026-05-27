#!/usr/bin/env python3
"""Training entry point for gwp-mot.

Usage:
    python scripts/train.py --config configs.robocasa_all_tshape_mot.config
    accelerate launch scripts/train.py --config configs.robocasa_all_tshape_mot.config
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_HOME", "/shared_disk/models/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/shared_disk/models/huggingface")


def main():
    parser = argparse.ArgumentParser(description="Train world action model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs.robocasa_all_tshape_mot.config",
        help="Dotted path to config dict (e.g. configs.robocasa_all_tshape_mot.config)",
    )
    args = parser.parse_args()

    from world_action_model.runtime import run_training
    run_training(args.config)


if __name__ == "__main__":
    main()
