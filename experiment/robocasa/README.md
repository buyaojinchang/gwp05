# RoboCasa MoT T-shape Evaluation

这个目录保留 `gwp-mot` 的在线 RoboCasa 评测与采集路径。Server 使用 MoT checkpoint 和 T-shape 图像布局：`agentview_right` 是 head view，`agentview_left` 与 `eye_in_hand` 半尺寸拼在下方。

旧的单机 debug server/client、非 T-shape server、CloseCabinet 单任务包装已经移除。现在只保留 T-shape 并行 server、评测 client、采集 client。

## 启动 server

```bash
bash experiment/robocasa/parallel_server_tshape.sh <MOT_CHECKPOINT> [ACTION_CHUNK] [SEED]
```

示例：

```bash
bash experiment/robocasa/parallel_server_tshape.sh /path/to/mot/checkpoint/model.pt 24 42
```

常用环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `GWP_MOT_OUTPUT_ROOT` | `/shared_disk/users/hengtao.li/codex/gwp-mot` | server/client 日志和 `.server_tshape_info` 根目录 |
| `NUM_WORKERS` | 4 | server 副本数 |
| `GPU_OFFSET` | 4 | 第一个 server 使用的 GPU id |
| `BASE_PORT` | 19055 | 第一个 websocket 端口 |
| `NUM_FRAMES` | 24 | MoT 采样长度 |
| `ACTION_CHUNK` | 参数 2 或 24 | 每次返回的 action 数，必须 `<= NUM_FRAMES` |

Server 会写入：

```text
$GWP_MOT_OUTPUT_ROOT/robocasa_eval/.server_tshape_info
$GWP_MOT_OUTPUT_ROOT/robocasa_eval/server_tshape/<timestamp>/server_*.log
```

## 评测 client

```bash
bash experiment/robocasa/parallel_client.sh atomic_seen
```

`parallel_client.sh` 会读取 `.server_tshape_info`，自动使用 server 的端口、worker 数和 `ACTION_CHUNK`。

## 采集 client

```bash
bash experiment/robocasa/parallel_client_collect.sh composite_unseen /shared_disk/users/hengtao.li/robocasa_datasets/collected
```

采集数据会落到：

```text
<COLLECT_DIR>/<exp>/<checkpoint>/<model>/<env_name>/lerobot/
```

## 输出

评测日志与 summary 默认写到：

```text
$GWP_MOT_OUTPUT_ROOT/robocasa_eval/eval/<exp>/<checkpoint>/<model>/
```

客户端终端日志写到：

```text
$GWP_MOT_OUTPUT_ROOT/robocasa_eval/client/<timestamp>/client_*.log
```

## 注意

- 必须使用 MoT checkpoint；`inference_server.py` 会拒绝没有 MoT keys 的旧 checkpoint。
- `ACTION_CHUNK` 必须小于等于 `NUM_FRAMES`。
- 如果改了 `GWP_MOT_OUTPUT_ROOT`，server 和 client 两边要使用同一个值。
