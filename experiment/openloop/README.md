# MoT Open-Loop Evaluation

这里的 open-loop 只服务 `gwp-mot`：读取 RoboCasa LeRobot 数据集中的真实图像和 state，用 MoT checkpoint 预测 action，再和 GT action 画逐维度曲线。它不启动 RoboCasa 仿真，适合快速看 checkpoint 是否跑偏。

旧的 RoboTwin / 单 transformer GWP open-loop 入口已经移除；这里的脚本会拒绝没有 MoT keys 的 checkpoint。

## 入口

预训练数据布局：

```bash
bash experiment/openloop/openloop.sh <MOT_CHECKPOINT> [GPU_ID] [MAX_DATASETS] [EPISODE_IDX]
```

目标域 collected 数据布局：

```bash
bash experiment/openloop/openloop_tshape.sh <MOT_CHECKPOINT> [GPU_ID] [MAX_DATASETS] [EPISODE_IDX]
```

示例：

```bash
bash experiment/openloop/openloop_tshape.sh   /shared_disk/users/hengtao.li/codex/gwp-mot/experiments/run/checkpoint-10000/model.pt   0 18 0
```

## 默认设置

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `DATA_ROOT` | `openloop.sh`: pretrain_gwp; `openloop_tshape.sh`: collected/epoch0 | 数据根目录 |
| `STATS_PATH` | pretrain_gwp/norm_stats_delta.json | state/action 归一化统计 |
| `NUM_FRAMES` | 24 | MoT 一次采样的 action token 数 |
| `ACTION_CHUNK` | 24 或 20 | 每次重规划后落盘/累计的 action 数 |
| `REPLAN_STEPS` | 24 或 20 | 每隔多少帧重新采样 |
| `DST_W`, `DST_H` | 320, 256 | T-shape head view 尺寸 |
| `TSHAPE_HEAD_INDEX` | 2 | `agentview_right` 作为 head view |

覆盖示例：

```bash
DATA_ROOT=/path/to/collected STATS_PATH=/path/to/norm_stats_delta.json NUM_FRAMES=24 ACTION_CHUNK=20 REPLAN_STEPS=20 bash experiment/openloop/openloop_tshape.sh /path/to/mot/model.pt 0 18 0
```

## 输出

未指定 `--output_dir` 时，结果写到：

```text
/shared_disk/users/hengtao.li/codex/gwp-mot/openloop/<exp>/<checkpoint>/<model>/
```

每个 task 下会生成：

```text
ep000_actions.png
ep000_metrics.json
summary_ep000.json
```

`ep000_actions.png` 是 12 维 action 的 GT / Pred 对比图；`summary_ep000.json` 保存每个 task 和整体平均指标。
