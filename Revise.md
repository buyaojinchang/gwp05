# Revise.md — pick_place_g1_sonic_mot 训练改造记录

> 对应需求文件：`/home/s4090/hengtao.li/1_loco_manip_code/gwp05/Review.md`，本文件按 2026-06-26 读取到的新版 Review 更新。  
> 本次未修改 `Review.md`。远端 git worktree 在接手前已有 unrelated dirty/untracked 状态；下面只记录本需求相关改动和验证。

## 总结

已按新版 Review 的 owner 决策把 pick_place 图像分辨率从上一版的 `384x288` 回改为 `320x256`，保留 `resize_mode: crop`，并保留上一版已修好的 train 随机裁剪 / eval 中心裁剪逻辑。`num_frames=56`、T5 embedding、action raw、以及 Design B 的双 state token `[joint_state_token, motion_latent_token]` 均保持有效。样本级校验通过：`images=(5,3,256,320)`、`action=(56,66)`、`state=(2,66)`、`prompt_embeds` 非全 0。标准 full-model 单卡 smoke 仍在 DeepSpeed ZeRO-2 optimizer 初始化阶段 OOM；实际 5-step 省显存 smoke 已跑通并保存 checkpoint。

## 任务 1：图像宽高 / 裁剪

最终选择：`dst_size: [320, 256]`。

原因：新版 Review 中 owner 明确否决 `384x288`，要求回到常见尺寸 `320x256`。该尺寸宽高均可被 32 整除；对 640x480 ego_view，cover 缩放为约 `341x256` 后裁到 `320x256`，保持固定缩放比、不变倍，只产生约 6.2% 水平视野损失，同时提供水平随机裁增强。

实际改动：

- `configs/data/pick_place_g1_sonic.yaml`
  - `dst_size` 从上一版 `[384, 288]` 回改为 `[320, 256]`。
  - 保留 `resize_mode: crop`。
- `src/world_action_model/transforms/wa_transforms_lerobot.py`
  - 保留上一版修复：`is_train=True` 时 random crop，`is_train=False` 时 center crop。
  - 320x256 下 train/eval 的缩放比和窗口大小一致，仅裁剪窗口位置不同，不引入像素尺度不一致。

样本级校验：

```text
images (5, 3, 256, 320)
ref_images (5, 3, 256, 320)
```

## 任务 2：num_frames 改成 56

保持完成状态：

- `configs/data/pick_place_g1_sonic.yaml`
  - `num_frames: 56`。

验证结果：

- 视频输入仍为 5 帧。
- action chunk 为 `(56, 66)`。
- 对应 frame offsets 为 `[0, 14, 28, 42, 56]`。

## 任务 3：T5 embedding

保持完成状态。

代码改动：

- `scripts/generate_t5_embeddings.py`
  - `collect_all_tasks()` 同时匹配 `**/lerobot/meta/tasks.jsonl` 和 `**/meta/tasks.jsonl`。
  - 去重使用 `abspath` 而不是 `realpath`，避免 `pick_place_gwp/meta` 软链被解析回原始 `pick_place/meta`。

实际使用的 Wan 路径：

- 默认 `/shared_disk/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers` 不存在。
- 使用 fallback：`/home/s4090/hengtao.li/2_data_ckpt_cache/locomanip/pretrained_ckpt/Wan2.2-TI2V-5B-Diffusers`。

生成产物：

- `/home/s4090/hengtao.li/2_data_ckpt_cache/locomanip/data/pick_place_gwp/meta/t5_text_embeds.pt`

校验：

```text
t5_text_embeds.pt: dict keys [0]
task 0 embedding shape (13, 4096), abs sum 2539.177...
transform prompt_embeds shape (64, 4096), nonzero=True
```

## 任务 4：action 不做 norm

已确认 action 走 raw、未归一化：

- `configs/data/pick_place_g1_sonic.yaml` 保持 `skip_action_norm: true`。
- `norm_stats_delta.json` 中 `action` 为 identity stats：mean=0、std=1，66 维。
- sonic latent 本身按 raw 值进入 action 分支。

注意：joint state 仍按真实 mean/std 做归一化。

## 任务 5：state = joint + motion latent 双 token

采用 Design B，并保持上一版实现。

### 数据侧

改动文件：`scripts/prepare_pick_place_gwp.py`

生成的 `pick_place_gwp` parquet 包含：

- `action`：66 维 sonic latent，`motion_token[64] + hand_binary[2]`。
- `observation.state`：原始 43 维 joint state。
- `observation.motion_latent`：66 维 sonic latent，作为第二个 state token。

norm stats：

- `observation.state`：整份数据逐维真实 mean/std，43 维。
- `observation.motion_latent`：identity mean=0/std=1，66 维。
- `action`：identity mean=0/std=1，66 维。

数据生成已完成：

```text
Wrote 110 episodes to /home/s4090/hengtao.li/2_data_ckpt_cache/locomanip/data/pick_place_gwp/data
Wrote joint/action norm stats -> /home/s4090/hengtao.li/2_data_ckpt_cache/locomanip/data/pick_place_gwp/norm_stats_delta.json
```

### dataset loader

改动文件：`src/world_action_model/datasets/lerobot_dataset.py`

- `_load_episodes()` 读取可选列 `observation.state` 和 `observation.motion_latent`。
- `_getitem_inner()` 分别输出两路 `[start:start+1]`。
- 对缺列数据集容错：缺失则不放 key，由 transform 置零。

### transform

改动文件：`src/world_action_model/transforms/wa_transforms_lerobot.py`

- 新增 `state_keys`、`state_token_dims`、`state_norm_keys`。
- token0：joint，43 维，按 joint stats z-score 后 pad 到 66。
- token1：motion latent，66 维，identity/raw。
- 缺失模态输出全 0 token，并在 `state_mask` 中标记 false。
- 输出 `state=(2,66)`、`state_mask=(2,)`。
- g1_sonic 的 delta template 仍是全 False；multi-state 模式不把 joint 卷入 `action - state` 逻辑。

### 模型与 trainer

改动文件：

- `configs/data/pick_place_g1_sonic.yaml`
  - `joint_state_dim: 43`、`latent_state_dim: 66`、`state_token_dims: [43, 66]`、`num_state_tokens: 2`。
- `configs/model/mot.yaml`
  - 将 `state_token_dims` 透传给 `action_expert`。
- `src/world_action_model/models/action_state_dit.py`
  - 配置 `state_token_dims=[43,66]` 时使用两个独立 state encoder：`joint` 与 `latent`。
  - 增加可学习 `state_type_embed`。
  - 支持 `state_mask`，缺失 token 编码后置零。
- `src/world_action_model/models/transformer_wa_mot.py`
  - `forward/_forward_full/_forward_action_only` 支持可选 `state_mask`。
  - attention mask 会屏蔽缺失 state token 的 key 列。
- `src/world_action_model/trainers/wa_casual_trainer_pretrain.py`
  - 从 batch 取 `state_mask` 并传给 transformer。

## 本次修改文件清单

本轮新版 Review 直接修改：

- `configs/data/pick_place_g1_sonic.yaml`

上一版任务 2-5 已完成并仍然有效的相关文件：

- `configs/model/mot.yaml`
- `scripts/prepare_pick_place_gwp.py`
- `scripts/generate_t5_embeddings.py`
- `src/world_action_model/datasets/lerobot_dataset.py`
- `src/world_action_model/transforms/wa_transforms_lerobot.py`
- `src/world_action_model/models/action_state_dit.py`
- `src/world_action_model/models/transformer_wa_mot.py`
- `src/world_action_model/trainers/wa_casual_trainer_pretrain.py`

数据产物：

- `/home/s4090/hengtao.li/2_data_ckpt_cache/locomanip/data/pick_place_gwp/data/**/*.parquet`
- `/home/s4090/hengtao.li/2_data_ckpt_cache/locomanip/data/pick_place_gwp/norm_stats_delta.json`
- `/home/s4090/hengtao.li/2_data_ckpt_cache/locomanip/data/pick_place_gwp/meta/t5_text_embeds.pt`

## 校验与 smoke

### 样本级 shape 校验

命令使用远端环境：`/home/s4090/hengtao.li/0_conda_env/gwp05/bin/python`。

关键输出：

```text
images (5, 3, 256, 320) torch.float32
ref_images (5, 3, 256, 320) torch.float32
prompt_embeds (64, 4096) torch.float32, abs sum 2539.177...
action (56, 66) torch.float32
state (2, 66) torch.float32
state_mask (2,) torch.bool
state_mask [True, True]
prompt_nonzero True
```

### Review 标准 smoke 命令

等价运行命令：

```bash
cd /home/s4090/hengtao.li/1_loco_manip_code/gwp05
export PATH=/home/s4090/miniconda3/bin:$PATH
export CONDA_ENV=/home/s4090/hengtao.li/0_conda_env/gwp05
export CUDA_VISIBLE_DEVICES=0
export GWP_DEFAULT_NPROC=1
export WANDB_MODE=offline
export GWP_MOT_OUTPUT_ROOT=/home/s4090/hengtao.li/2_data_ckpt_cache/locomanip/gwp-mot-smoke
bash train_pick_place_g1_sonic_mot.sh train.max_epochs=1 data.batch_size_per_gpu=2 train.max_steps=5
```

结果：数据和模型加载成功，分辨率回改没有引入 shape 错误；但 full-model 单卡 ZeRO-2 仍在 optimizer 初始化阶段 OOM：

```text
Dataset size: 142431, Dataloader batches: 71215
Preparing model with DeepSpeed ...
torch.OutOfMemoryError: Tried to allocate 22.43 GiB. GPU 0 total 47.35 GiB, free 21.80 GiB.
```

说明：这次 OOM 发生在 optimizer 初始化，还没进入第一步；回到 320x256 能降低 activation/VAE 相关显存，但不改变 full-model 参数和 ZeRO optimizer 初始化峰值。

另一个仍存在的训练逻辑点：`trainer.py` 中只要 `train.max_epochs > 0`，就会用 `max_epochs * len(dataloader)` 覆盖 `train.max_steps`，所以 `train.max_epochs=1 train.max_steps=5` 实际不会只跑 5 step。

### 实际 5-step smoke

为验证新版分辨率下前向、反向、optimizer step 和 checkpoint，使用省显存命令：

```bash
cd /home/s4090/hengtao.li/1_loco_manip_code/gwp05
export PATH=/home/s4090/miniconda3/bin:$PATH
export CONDA_ENV=/home/s4090/hengtao.li/0_conda_env/gwp05
export CUDA_VISIBLE_DEVICES=0
export GWP_DEFAULT_NPROC=1
export WANDB_MODE=offline
export GWP_MOT_OUTPUT_ROOT=/home/s4090/hengtao.li/2_data_ckpt_cache/locomanip/gwp-mot-smoke
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash train_pick_place_g1_sonic_mot.sh \
  train.max_epochs=0 train.max_steps=5 \
  data.batch_size_per_gpu=1 \
  train.ema.enabled=false train.with_ema=false train.log_interval=1 \
  +model.freeze_backbone=true
```

结果：成功跑完 5 step。

```text
Freeze video backbone: 825 params frozen, 845 params trainable
Dataset size: 142431, Dataloader batches: 142431
DeepSpeed ready.
Starting training from step 0, max_steps=5
Training: 100%|██████████| 5/5 [... visual_loss=0.4463, action_loss=1.0243, lr=6.45e-07]
Checkpoint saved to /home/s4090/hengtao.li/2_data_ckpt_cache/locomanip/gwp-mot-smoke/experiments/pick_place_g1_sonic_mot_0626_0007/checkpoint-5
Training finished.
```

对应日志：

- `/home/s4090/hengtao.li/2_data_ckpt_cache/locomanip/gwp-mot-smoke/logs/pick_place_g1_sonic_mot/0626_0007_node0.log`

## 需 owner 确认 / 后续建议

1. 是否要改 trainer 的 `max_epochs` / `max_steps` 优先级：当前 `max_epochs>0` 会覆盖 `max_steps`，与 Review 示例命令的“跑 5 step”意图冲突。本轮未改 trainer 语义，只在实际 smoke 中用 `train.max_epochs=0`。
2. full-model 单卡 4090 仍然不够默认 ZeRO-2 + CAME 初始化。可考虑多卡、ZeRO-3/offload、默认冻结 video backbone、关闭 EMA，或另做更轻量的 smoke config。
3. 当前 `launch_lib.sh` 默认 env `/mnt/pfs/users/hengtao.li/conda_envs/gwpmot` 在这台 `sym4090` 上不可用；实际测试用 `CONDA_ENV=/home/s4090/hengtao.li/0_conda_env/gwp05`。
4. 当前默认 `GWP_MOT_OUTPUT_ROOT=/shared_disk/...` 在这台机器无权限；测试改到 `/home/s4090/hengtao.li/2_data_ckpt_cache/locomanip/gwp-mot-smoke`。
