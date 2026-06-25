# 交接 Prompt — pick_place_g1_sonic_mot 训练改造

> 你（接手的 agent）的任务：按本文件的要求修改 gwp05 代码，并把你**做了什么、怎么做的、改了哪些文件、踩了哪些坑、还需人确认的点**写进同目录的 `Revise.md`。本文件（`Review.md`）是只读的需求说明，请不要改它。

## 0. 背景

- 入口脚本：`train_pick_place_g1_sonic_mot.sh` → `gwp_launch pick_place_g1_sonic_mot`
- 任务配置：`configs/task/pick_place_g1_sonic_mot.yaml`（选 `data=pick_place_g1_sonic` + `model=mot`）
- 数据配置：`configs/data/pick_place_g1_sonic.yaml`
- 模型配置：`configs/model/mot.yaml`
- 关键代码：
  - 数据集：`src/world_action_model/datasets/lerobot_dataset.py`
  - transform：`src/world_action_model/transforms/wa_transforms_lerobot.py`
  - 帧/horizon 装配：`src/world_action_model/hydra_runtime.py`（`_build_data_or_config`）
  - MoT 模型：`src/world_action_model/models/transformer_wa_mot.py`、`models/action_state_dit.py`
  - trainer：`src/world_action_model/trainers/mot_casual_trainer_pretrain.py`
  - 数据预处理：`scripts/prepare_pick_place_gwp.py`
  - T5：`scripts/generate_t5_embeddings.py`
- 数据：`pick_place`（110 episodes，50Hz，ego_view 单相机）。已用 `prepare_pick_place_gwp.py` 生成 `pick_place_gwp/`（`action`/`observation.state` 都写成 66 维 sonic latent = motion_token[64]+hand_binary[2]，videos/meta 软链，identity norm）。
- 原始视频分辨率：**640(W) × 480(H)**（来自 `pick_place/meta/info.json`，4:3）。
- 原始 `observation.state`：**43 维关节角**（见 info.json，left/right 各关节）。

---

## 任务 1 —— 图像宽高 / 裁剪（owner 已最终拍板，请按此修订）

> ⚠️ 更新：上一版 `Revise.md` 把 `dst_size` 改成了 `384×288`。**owner 决定否决该方案，改回 `320×256`**。原因见下。请据此回改。

**背景结论（已查证）：**

- VAE = Wan2.2-TI2V-5B：`scale_factor_spatial=16`，transformer `patch_size=(1,2,2)` → **总空间下采样 = 32**。所以 `dst_size` 的宽、高都必须能被 32 整除。
- 源 ego_view = 640×480（精确 4:3）。
- 当前 crop 管线（`wa_transforms_lerobot.py::_process_images`）是**固定缩放比 + 裁剪**：按 cover 比例（由较紧的那条边决定）等比缩放，再裁到 dst。train/eval **用同一缩放比，不变倍**，随机裁剪只是平移裁剪窗口位置，**像素尺度完全一致**。

**owner 的两个硬约束：**

1. **不接受"放大再裁"(oversize/zoom)**：那会让 train 的像素尺度与 eval 不一致。必须保持固定缩放比。
2. **不用 384×288**：尺寸不常见。

**关键权衡（写清楚，避免再次跑偏）：**

> 在"固定缩放比、不变倍"前提下，**随机裁剪的可抖动范围 == 损失的视野**（是同一批像素）。所以"既想大幅 random crop、又想几乎不丢视野"在固定尺度下不可能同时满足。

固定尺度下各候选（源 640×480，均可被 32 整除）：

| dst `W×H` | 宽高比 | 缩放后(cover) | 抖动/损失轴 | 随机裁范围 = 损失 | 评价 |
|---|---|---|---|---|---|
| **320×256** | 5:4 (1.25) | 341×256 | 水平 | **6.2%**（±21px 水平） | **最终选定**，常见、损失小、自带水平随机裁 |
| 320×224 | 10:7 (1.43) | 320×240 | 垂直 | 6.7%（±16px 垂直） | 若想纵向抖动可选 |
| 256×192 | 4:3 (1.333) | 256×192 | — | 0（无随机裁余量） | 精确4:3零损失但无增强 |
| 256×256 | 1:1 | 341×256 | 水平 | 25% | 方形最常见但裁太多 |

`320×256` 在"常见尺寸 + 不变倍 + 仅 6.2% 损失 + 自带水平随机裁(增强鲁棒性)"之间是平衡点。

**需要你做（回改）：**

1. `configs/data/pick_place_g1_sonic.yaml`：把 `dst_size` 从 `[384, 288]` **改回 `[320, 256]`**，保留 `resize_mode: crop`。
2. `wa_transforms_lerobot.py::_process_images`：**保留你上一版已经修好的 eval 中心裁剪 / train 随机裁剪逻辑**（`is_train=True` 随机裁、`is_train=False` 中心裁）。这点不要回退。
3. 确认：320×256 下 train 走固定尺度 + 水平 ±10px 随机裁（不变倍），eval 走固定尺度 + 中心裁，两者像素尺度/窗口大小一致、仅差平移。
4. 注意显存：改回 320×256 比 384×288 更省显存，对单卡 full-model OOM 问题（见冒烟测试）是利好。
5. 在 `Revise.md` 里补一段说明本次回改，并复跑一次样本级 shape 校验（`images` 应为 `(5, 3, 256, 320)`）。

---

## 任务 2 —— num_frames 改成 56

**结论（已分析）：**

- `num_frames` 同时控制两件事（见 `hydra_runtime.py::_build_data_or_config`）：
  1. action chunk 长度 `delta_info={"action": num_frames}`；
  2. 视频采样的 5 个帧偏移 `image_frame_offsets = [0, n//4, n//2, 3n//4, n]`。
- **视频输入永远是 5 帧**，与 num_frames 无关；`MaskGenerator(factor=4)` 作用在这 5 帧上：`(5-1)%4==0` 恒成立，不受 num_frames 影响。
- 56 能被 4 整除 → 偏移 `[0,14,28,42,56]` 干净；episode 平均 ~1350 帧 ≫ 56，样本充足。
- 效果：同样 5 帧，时间跨度从 24/50=0.48s 变成 56/50=1.12s；action chunk 变 56。

**需要你做：** 把 `configs/data/pick_place_g1_sonic.yaml` 的 `num_frames: 24` 改成 `num_frames: 56`。改完确认 transform 里 action 被 pad/截断到 56（`wa_transforms_lerobot.py` 第 267 行附近逻辑），以及 batch shape 正常。

---

## 任务 3 —— 用 `generate_t5_embeddings.py` 生成 T5 embedding

**结论（已分析，有坑）：**

- dataset loader（`lerobot_dataset.py::_load_t5_embeddings`）会找 `<data>/meta/t5_text_embeds.pt`，格式是 dict `{task_index: tensor[L,4096]}`；transform 里 `t5_len=64`，UMT5 输出 4096 维，匹配。
- **坑**：`generate_t5_embeddings.py` 的 glob 是 `**/lerobot/meta/tasks.jsonl`，**要求路径里有 `lerobot/` 这一层**。但 `pick_place_gwp/meta/tasks.jsonl` 没有 `lerobot/` 这层 → 脚本会扫到 0 个数据集，什么都不生成。
- 另外 `pick_place_gwp/meta` 是指向 `pick_place/meta` 的软链；脚本里 `materialize_symlink_dir` 会把软链实体化后再写入，注意别污染共享的源 meta（确认写入的是 `pick_place_gwp/meta` 实体目录）。
- `tasks.jsonl` 内容已确认：`{"task_index": 0, "task": "Walk forward, grab the cola and throw into the trash bin"}`。

**需要你做：**

1. 让 `generate_t5_embeddings.py` 能处理 `pick_place_gwp` 这种**没有 `lerobot/` 层**的结构。建议把 glob 改成同时匹配 `**/lerobot/meta/tasks.jsonl` 和 `**/meta/tasks.jsonl`（去重），或加一个 `--meta_glob`/`--single_dataset` 参数。
2. 跑一遍生成 `pick_place_gwp/meta/t5_text_embeds.pt`，并验证 loader 能读到、shape=[≤64, 4096]、训练时 `prompt_embeds` 不再是全 0。
3. wan_model_path 用默认 `/shared_disk/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers`（若不存在，可用本地 `pretrained_ckpt/Wan2.2-TI2V-5B-Diffusers` 里的 `text_encoder`/`tokenizer`，请在 Revise.md 注明实际用的路径）。

---

## 任务 4 —— action 不再做 norm

**结论：** 当前已经满足——`pick_place_g1_sonic.yaml` 里 `skip_action_norm: true`，且 `prepare_pick_place_gwp.py` 写的是 identity norm（mean=0/std=1）。sonic latent 本身在 [-1,1]。

**需要你做：** 确认现状即可（不用改）。在 Revise.md 里写一句"已确认 action 走 raw、未归一化"。**注意区分**：任务 5 引入的 joint state **需要**归一化（见下），别把它也跳过。

---

## 任务 5 —— state 同时用 joint + motion latent，拼成一个序列（核心改动）

**目标：** state 不再只是单个 token，而是一个**长度为 2 的 state 序列**：`[joint_state_token, motion_latent_token]`。
- 训练时两者都有就都用（拼成序列）；
- 某个数据集缺哪个，就把那个 token **置零**，并用 mask 标记缺失。

**现状（已分析）：**

- `prepare_pick_place_gwp.py` 目前把 `observation.state` 直接写成 66 维 latent（丢掉了原始 43 维关节）。
- transform 输出 `state` 形状 `[1, 66]`（取 `state[start:start+1]`），即 **1 个 state token**。
- 模型**本身已支持多 state token**：`action_state_dit.py::pre_dit` 里 `num_state_tokens = state_tokens.shape[1]`，state/action 的 timestep、1D RoPE 都按整段序列处理；`transformer_wa_mot.py` 全程用 `num_state_tokens = state.shape[1]` 动态切片。**所以把 state 变成 `[B, 2, D]` 在序列维度上是被支持的**，主要工作在"两种模态维度不同 + 编码 + 归一化 + 缺失置零 + mask"。

**需要你做（建议按 Design B；如时间紧可先 Design A 跑通再说，但在 Revise.md 标注）：**

### 数据侧
1. 改 `prepare_pick_place_gwp.py`：在生成 `pick_place_gwp/` 时**同时保留两路 state**：
   - `observation.state` = 原始 43 维关节角（从源 parquet 的 `observation.state` 拷贝）；
   - 新增列 `observation.motion_latent` = 66 维 latent（即现在的 action latent）。
   - `action` 仍 = 66 维 latent（不变）。
2. **norm stats**：joint 关节角不在 [-1,1]，必须算真实 mean/std（按 43 维逐维统计整份数据），写进 `norm_stats_delta.json` 里 joint 的条目；motion latent 维持 identity（raw）。即 norm 要按"每个 state 模态"分别配置。

### dataset loader（`lerobot_dataset.py`）
3. 目前 `_load_episodes` 只读 `observation.state`。改成同时读 `observation.state`(joint) 和 `observation.motion_latent`，分别存入 episode；`_getitem_inner` 里都取 `[start:start+1]` 放进 `data_dict`。对缺列的数据集要容错（缺则不放 key，由 transform 置零）。

### transform（`wa_transforms_lerobot.py`）
4. 构造 2-token state：
   - token0 = joint（按 joint stats 归一化）；token1 = motion latent（raw）。
   - 缺失的模态 → 该 token 全 0。
   - 输出一个 `state_mask`（形如 `[2]` 或 `[B,2]` bool），标记每个 state token 是否有效，传给模型/loss。
   - 注意现有 `delta = action - state`（第 303 行附近）只在 `ad==sd` 时生效；当前 robotype 模板 3 是全 False（no delta），**保持 no-delta**，不要让 joint 卷进 delta 逻辑。请确认改动后 delta 行为不被破坏。

### 模型（`action_state_dit.py` / `transformer_wa_mot.py`）
5. **Design B（推荐）**：给 `ActionStateDiT` 两个独立 state encoder（`joint_encoder: 43→hidden`、`latent_encoder: 66→hidden`），各自编码成一个 token 后在序列维 cat 成 `[B,2,hidden]`；建议再加一个可学习的 **token-type embedding** 区分两种模态。缺失模态的 token 走零向量（可选：仍叠加 type embedding，由 mask 决定是否参与 attention/loss）。
   - **Design A（简化备选）**：单 encoder，把 joint 和 latent 都 pad 到统一维度（如 66）当成 2 个 token 喂同一个 `state_encoder`。实现快，但两模态共享同一线性层、特征轴语义被混淆，效果可能差，仅作兜底。
6. 配置项：在 `configs/data/pick_place_g1_sonic.yaml` / `configs/model/mot.yaml` 增加必要字段（如 `joint_state_dim: 43`、`latent_state_dim: 66`、`num_state_tokens: 2`），并把它们从 data 透传到 transform 与模型构造（参考现有 `state_dim` 的透传链路：data → `transform.model_state_dim` 和 `model.state_dim` via `mot.yaml`）。保持"单处定义、多处引用"。

### 兼容性
7. 改完要保证：state 序列长度变化后，`transformer_wa_mot.py` 里所有 `num_state_tokens` 切片、timestep、RoPE 仍自洽（它已是动态的，重点是验证 shape）。若 state token 参与 loss/timestep，确认缺失 token 被 mask 掉、不产生 NaN。

---

## 验收 / 冒烟测试

单卡跑通（小 batch、1 epoch）：

```bash
CUDA_VISIBLE_DEVICES=0 GWP_DEFAULT_NPROC=1 bash train_pick_place_g1_sonic_mot.sh \
  train.max_epochs=1 data.batch_size_per_gpu=2 train.max_steps=5
```

要求：
- 数据能 load（joint + motion_latent 都进 batch，缺失置零 + mask 正常）；
- 图像分辨率按任务 1 的决定生效，VAE 编码无整除报错；
- num_frames=56 生效，action shape `[B,56,66]`，视频仍 5 帧；
- prompt_embeds 非全 0（T5 生效）；
- 前向 + 反向无 shape / NaN 错误，能跑出几个 step 的 loss。

---

## 你要交付的东西 → 写进 `Revise.md`

请在 `Revise.md` 写明：
1. 每个任务（1~5）实际怎么做的、改了哪些文件（路径 + 关键 diff 说明）。
2. 任务 1 的分辨率取舍最终选了哪个、为什么；eval 随机裁剪是否改成了中心裁剪。
3. 任务 5 用了 Design A 还是 B；state 序列/encoder/type-embedding/mask 的具体设计；joint 的 norm stats 怎么算的。
4. generate_t5_embeddings.py 改了什么、实际用的 wan_model_path、生成产物路径。
5. 冒烟测试命令与结果（贴关键日志/报错）。
6. 还需要 owner 确认或没做完的点（open questions）。
