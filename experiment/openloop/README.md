# MoT Open-Loop Evaluation

这里的 open-loop 只服务 `gwp-mot`：读取 RoboCasa LeRobot 数据集中的真实图像和 state，用 MoT checkpoint 预测 action，再和 GT action 画逐维度曲线。它不启动 RoboCasa 仿真，适合快速看 checkpoint 是否跑偏。

旧的 RoboTwin / 单 transformer GWP open-loop 入口已经移除；这里的脚本会拒绝没有 MoT keys 的 checkpoint。

## 入口

训练分布 atomic seen 数据布局（默认 18 个 task，每个 task 取一条）：

```bash
bash experiment/openloop/openloop.sh <MOT_CHECKPOINT> [GPU_ID] [MAX_DATASETS=18] [EPISODE_IDX]
```

目标域 collected 数据布局：

```bash
bash experiment/openloop/openloop_tshape.sh <MOT_CHECKPOINT> [GPU_ID] [MAX_DATASETS] [EPISODE_IDX]
```

示例：

```bash
bash experiment/openloop/openloop_tshape.sh   /shared_disk/users/hengtao.li/codex/gwp-mot/experiments/run/checkpoint-10000/model_ema.pt   0 18 0
```

## 默认设置

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `ATOMIC_SEEN_DATA_ROOT` | `openloop.sh`: pretrain_gwp | atomic seen 对比的数据根目录；普通 `DATA_ROOT` 不影响这个入口 |
| `DATA_ROOT` | `openloop_tshape.sh`: collected/epoch0 | target/collected 诊断入口的数据根目录 |
| `STATS_PATH` | pretrain_gwp/norm_stats_delta.json | state/action 归一化统计 |
| `NUM_FRAMES` | 24 | MoT 一次采样的 action token 数 |
| `ACTION_CHUNK` | 24 或 20 | 每次重规划后落盘/累计的 action 数 |
| `REPLAN_STEPS` | 24 或 20 | 每隔多少帧重新采样 |
| `DST_W`, `DST_H` | 320, 256 | T-shape 输入尺寸 |
| `TSHAPE_HEAD_INDEX` | 2 | raw 三视角模式下，`agentview_right` 作为 head view |
| `TASK_SET` | `openloop.sh`: `atomic_seen` | 选择训练 configs 中的 atomic seen 18 个任务 |
| `ONE_PER_TASK` | `openloop.sh`: `1` | 每个 task 只保留排序后的第一条 LeRobot/date |
| `--input_view_mode` | `auto` | 自动优先使用预拼好的 `observation.images.tshape`，否则使用三路 raw camera 拼 T-shape |

覆盖示例：

```bash
bash experiment/openloop/openloop.sh /path/to/mot/model_ema.pt 0 18 0

# 仅 target/collected 域诊断时才用 openloop_tshape.sh
DATA_ROOT=/path/to/collected STATS_PATH=/path/to/norm_stats_delta.json NUM_FRAMES=24 ACTION_CHUNK=20 REPLAN_STEPS=20 bash experiment/openloop/openloop_tshape.sh /path/to/mot/model.pt 0 18 0
```

## 评估维度

主图和主指标默认按训练里的 `action_dim_mask` 对齐：从 `STATS_PATH` 读取 action std，忽略 zero-std 维度。当前 RoboCasa stats 里通常是 dim 3 (`ee_rot_rx_zero_std`) 被忽略。

需要看全部 12 维时，脚本会额外保存 `ep000_actions_all_dims.png`。如果要让主指标也包含 zero-std 维，可以直接调用 Python 脚本并加 `--include_zero_std_dims`。

## 输出

未指定 `--output_dir` 时，结果写到：

```text
/shared_disk/users/hengtao.li/codex/gwp-mot/openloop/<exp>/<checkpoint>/<model>/
```

每个 task 下会生成：

```text
ep000_actions.png           # active dims 的 GT / Pred 对比图
ep000_actions_all_dims.png  # 全 12 维诊断图
ep000_arrays.npz            # gt, pred, pred_counts, active_dims, ignored_dims
ep000_metrics.json          # mse/mae、view_mode、覆盖帧数、GT/Pred 统计
summary_ep000.json          # 每个 task 和整体平均指标
```

atomic seen 训练数据来自 `/shared_disk/users/hengtao.li/robocasa_datasets/v1.0/pretrain_gwp/atomic/<task>/<date>/lerobot`，默认读取每个 seen task 的第 0 个 episode。

如果数据集中是预拼好的 T-shape 视频，脚本会读取 `observation.images.tshape`；训练 atomic seen 数据通常是三路 raw camera，脚本会自动拼成 T-shape。如果指定的 view 不存在导致没有生成任何预测，脚本会直接报错，不再静默留下全 0 的 pred 用来算指标。
