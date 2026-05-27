# gwp-mot

MoT-only RoboCasa branch of GigaWorld-Policy. This repository keeps the Mixture-of-Transformers model, trainer, T-shape RoboCasa inference path, and MoT open-loop evaluator. Legacy GWP/RoboTwin training and open-loop entrypoints are intentionally not part of this branch.

## What Is Kept

- MoT model code: `src/world_action_model/models/transformer_wa_mot.py`, `mot.py`, and `action_state_dit.py`.
- MoT trainer: `src/world_action_model/trainers/mot_casual_trainer_pretrain.py`.
- RoboCasa T-shape configs: `configs/robocasa_all_tshape_mot*.py`.
- RoboCasa T-shape online evaluation server: `experiment/robocasa/inference_server.py`.
- MoT open-loop evaluation: `experiment/openloop/openloop_eval.py` and the two shell wrappers in `experiment/openloop/`.

The MoT implementation still imports the original video expert classes internally, because the MoT video branch is initialized from that backbone. Public run paths in this branch should use the MoT configs and checkpoints.

## Environment

```bash
conda activate /mnt/pfs/users/hengtao.li/conda_envs/gwpmot
pip install -e .
conda install -p /mnt/pfs/users/hengtao.li/conda_envs/gwpmot pytest -y
```

Checkpoint and log output can live under:

```bash
export GWP_MOT_OUTPUT_ROOT=/shared_disk/users/hengtao.li/codex/gwp-mot
```

## Training

Main all-view MoT training:

```bash
bash train_robocasa_all_tshape_mot.sh
```

Front4 two-stage training:

```bash
bash train_robocasa_all_tshape_mot_front4_two_stage.sh
```

Back4 joint training:

```bash
bash train_robocasa_all_tshape_mot_back4_joint.sh
```

All three launch through `scripts/train.py` with DeepSpeed ZeRO-2 via `scripts/accelerate_configs/config_deepspeed_zero2.json`. The default output root is `/shared_disk/users/hengtao.li/codex/gwp-mot` when supported by the launcher/config.

## Open-Loop Evaluation

Open-loop is MoT-only and uses the RoboCasa T-shape image layout by default: `agentview_right` is the head view, with `agentview_left` and `eye_in_hand` concatenated below it.

Pretrain-layout datasets:

```bash
bash experiment/openloop/openloop.sh /path/to/mot/checkpoint/model.pt 0 5 0
```

Collected target-layout datasets:

```bash
bash experiment/openloop/openloop_tshape.sh /path/to/mot/checkpoint/model.pt 0 18 0
```

Useful overrides:

```bash
DATA_ROOT=/path/to/datasets STATS_PATH=/path/to/norm_stats_delta.json NUM_FRAMES=24 ACTION_CHUNK=20 REPLAN_STEPS=20 bash experiment/openloop/openloop_tshape.sh /path/to/mot/checkpoint/model.pt 0 18 0
```

The evaluator rejects non-MoT checkpoints by checking for MoT keys in the checkpoint state dict.

## RoboCasa Online Evaluation

Start T-shape server replicas:

```bash
bash experiment/robocasa/parallel_server_tshape.sh /path/to/mot/checkpoint/model.pt 24
```

Run evaluation clients:

```bash
bash experiment/robocasa/parallel_client.sh atomic_seen
```

Run collection clients:

```bash
bash experiment/robocasa/parallel_client_collect.sh composite_unseen /shared_disk/users/hengtao.li/robocasa_datasets/collected
```

## Data

The data loader expects LeRobot v2 layout:

```text
<data_path>/
├── meta/
├── data/chunk-000/episode_000000.parquet
├── videos/chunk-000/<observation-image-key>/episode_000000.mp4
└── t5_embedding/episode_000000.pt
```

For open-loop, both of these layouts are supported:

```text
pretrain_gwp/{atomic,composite}/<task>/<date>/lerobot
collected/<task>/lerobot
```

## Tests

```bash
conda run -p /mnt/pfs/users/hengtao.li/conda_envs/gwpmot python -m pytest -q   tests/test_mot_smoke.py tests/test_ema_checkpoint.py
```

## Notes

- Use MoT checkpoints for this branch. Legacy single-transformer GWP checkpoints are not supported by the open-loop evaluator or RoboCasa server.
- `ACTION_CHUNK` must be less than or equal to `NUM_FRAMES` for open-loop sampling.
- Logs/checkpoints should be routed to `/shared_disk/users/hengtao.li/codex/gwp-mot` when possible.
