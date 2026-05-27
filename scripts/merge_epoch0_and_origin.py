"""Merge collected epoch0 dataset with a newly-collected 'origin' rollout set into epoch1.

For each task that exists in both sources:
  - Episodes from `--epoch0` come first (keep original numbering 0..N0-1).
  - Episodes from `--origin` are appended and re-numbered N0..N0+N1-1.
  - `task_index`, `episode_index`, `index` in parquet are remapped.
  - `tasks.jsonl` is merged (epoch0 tasks keep their indices, origin-only texts get appended).
  - `episodes.jsonl` / `episodes_stats.jsonl` are merged & renumbered.
  - `info.json` is regenerated.
  - `stats.json`, `embodiment.json`, `modality.json` are copied from epoch0.
  - `extras/` (ep_meta.json / model.xml.gz / states.npz) are symlinked from epoch0 only
    (origin rollouts have no extras).
  - videos are symlinked.

After merging, optionally regenerate `t5_text_embeds.pt` (recommended) by running
  scripts/generate_t5_embeddings.py on the output root.

Example:
    /mnt/pfs/users/hengtao.li/conda_envs/gwpmot/bin/python \
        scripts/merge_epoch0_and_origin.py \
        --epoch0 /shared_disk/users/hengtao.li/robocasa_datasets/collected/epoch0 \
        --origin /shared_disk/users/hengtao.li/robocasa_datasets/collected/<mot-run>/checkpoint-10000/model \
        --output /shared_disk/users/hengtao.li/robocasa_datasets/collected/<mot-run>/epoch1
"""

import argparse
import json
import os
import shutil

import pandas as pd
import torch
from tqdm import tqdm


VIDEO_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
]


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items, path):
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def find_lerobot_dir(base, task_name):
    task_dir = os.path.join(base, task_name)
    if not os.path.isdir(task_dir):
        return None
    direct = os.path.join(task_dir, "lerobot")
    if os.path.isdir(os.path.join(direct, "meta")):
        return direct
    for sub in sorted(os.listdir(task_dir)):
        candidate = os.path.join(task_dir, sub, "lerobot")
        if os.path.isdir(os.path.join(candidate, "meta")):
            return candidate
    return None


def symlink_safe(src, dst):
    if not os.path.exists(src):
        return
    if os.path.lexists(dst):
        os.unlink(dst)
    os.symlink(os.path.realpath(src), dst)


def merge_one_task(epoch0_lr, origin_lr, out_lr):
    os.makedirs(os.path.join(out_lr, "data", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(out_lr, "meta"), exist_ok=True)

    e0_info = json.load(open(os.path.join(epoch0_lr, "meta", "info.json")))
    e0_episodes = load_jsonl(os.path.join(epoch0_lr, "meta", "episodes.jsonl"))
    og_episodes = load_jsonl(os.path.join(origin_lr, "meta", "episodes.jsonl"))
    e0_tasks = load_jsonl(os.path.join(epoch0_lr, "meta", "tasks.jsonl"))
    og_tasks = load_jsonl(os.path.join(origin_lr, "meta", "tasks.jsonl"))

    e0_stats_path = os.path.join(epoch0_lr, "meta", "episodes_stats.jsonl")
    og_stats_path = os.path.join(origin_lr, "meta", "episodes_stats.jsonl")
    e0_ep_stats = load_jsonl(e0_stats_path) if os.path.exists(e0_stats_path) else []
    og_ep_stats = load_jsonl(og_stats_path) if os.path.exists(og_stats_path) else []
    e0_stats_by_ep = {s["episode_index"]: s for s in e0_ep_stats}
    og_stats_by_ep = {s["episode_index"]: s for s in og_ep_stats}

    n_e0 = len(e0_episodes)

    # Unified task list (epoch0 task indices stay the same; origin-only texts appended).
    task_text_to_idx = {t["task"]: t["task_index"] for t in e0_tasks}
    merged_tasks = list(e0_tasks)
    for t in og_tasks:
        if t["task"] not in task_text_to_idx:
            new_idx = len(merged_tasks)
            task_text_to_idx[t["task"]] = new_idx
            merged_tasks.append({"task_index": new_idx, "task": t["task"]})
    og_task_remap = {t["task_index"]: task_text_to_idx[t["task"]] for t in og_tasks}

    # --- Copy epoch0 parquets (re-index the global `index` only). ---
    total_frames = 0
    for ep in e0_episodes:
        ei = ep["episode_index"]
        src = os.path.join(epoch0_lr, "data", "chunk-000", f"episode_{ei:06d}.parquet")
        dst = os.path.join(out_lr, "data", "chunk-000", f"episode_{ei:06d}.parquet")
        df = pd.read_parquet(src)
        df["index"] = range(total_frames, total_frames + len(df))
        df.to_parquet(dst, index=False)
        total_frames += len(df)

    # --- Copy origin parquets (renumber episode + index + task_index). ---
    merged_episodes = list(e0_episodes)
    merged_ep_stats = []
    # First copy epoch0 stats (update `index` stats since we kept numbering but
    # global index didn't shift for the first n_e0 episodes, so they are fine).
    for ep in e0_episodes:
        ei = ep["episode_index"]
        if ei in e0_stats_by_ep:
            merged_ep_stats.append(e0_stats_by_ep[ei])

    running_frames = total_frames  # = total_frames at end of epoch0
    # epoch0 occupies [0, total_frames). Now origin episodes start from total_frames.

    for ep in og_episodes:
        old_i = ep["episode_index"]
        new_i = n_e0 + old_i
        src = os.path.join(origin_lr, "data", "chunk-000", f"episode_{old_i:06d}.parquet")
        dst = os.path.join(out_lr, "data", "chunk-000", f"episode_{new_i:06d}.parquet")
        df = pd.read_parquet(src)
        df["episode_index"] = new_i
        new_start = running_frames
        running_frames += len(df)
        df["index"] = range(new_start, running_frames)
        df["task_index"] = df["task_index"].map(lambda x: og_task_remap.get(x, x))
        df.to_parquet(dst, index=False)
        merged_episodes.append(
            {"episode_index": new_i, "tasks": ep["tasks"], "length": ep["length"]}
        )
        if old_i in og_stats_by_ep:
            s = og_stats_by_ep[old_i]
            # Update only the index-like stats; pixel / state stats stay valid.
            s = json.loads(json.dumps(s))  # deep copy
            s["episode_index"] = new_i
            if "stats" in s:
                if "episode_index" in s["stats"]:
                    s["stats"]["episode_index"] = {
                        "min": [new_i], "max": [new_i], "mean": [float(new_i)],
                        "std": [0.0], "count": s["stats"]["episode_index"]["count"],
                    }
                if "index" in s["stats"]:
                    n = len(df)
                    idxs = list(range(new_start, new_start + n))
                    s["stats"]["index"] = {
                        "min": [new_start],
                        "max": [new_start + n - 1],
                        "mean": [sum(idxs) / n],
                        "std": s["stats"]["index"].get("std", [0.0]),
                        "count": [n],
                    }
                if "task_index" in s["stats"] and s["stats"]["task_index"].get("min"):
                    old_min = s["stats"]["task_index"]["min"][0]
                    old_max = s["stats"]["task_index"]["max"][0]
                    s["stats"]["task_index"] = {
                        "min": [og_task_remap.get(int(old_min), int(old_min))],
                        "max": [og_task_remap.get(int(old_max), int(old_max))],
                        "mean": [float(og_task_remap.get(int(old_min), int(old_min)))],
                        "std": [0.0],
                        "count": s["stats"]["task_index"]["count"],
                    }
            merged_ep_stats.append(s)
    total_frames = running_frames

    # --- Symlink videos. ---
    for vk in VIDEO_KEYS:
        out_vd = os.path.join(out_lr, "videos", "chunk-000", vk)
        os.makedirs(out_vd, exist_ok=True)
        e0_vd = os.path.join(epoch0_lr, "videos", "chunk-000", vk)
        og_vd = os.path.join(origin_lr, "videos", "chunk-000", vk)
        if os.path.isdir(e0_vd):
            for ep in e0_episodes:
                ei = ep["episode_index"]
                s = os.path.join(e0_vd, f"episode_{ei:06d}.mp4")
                d = os.path.join(out_vd, f"episode_{ei:06d}.mp4")
                symlink_safe(s, d)
        if os.path.isdir(og_vd):
            for ep in og_episodes:
                old_i = ep["episode_index"]
                new_i = n_e0 + old_i
                s = os.path.join(og_vd, f"episode_{old_i:06d}.mp4")
                d = os.path.join(out_vd, f"episode_{new_i:06d}.mp4")
                symlink_safe(s, d)

    # --- extras/ (only epoch0 has it; just symlink episode dirs). ---
    e0_extras = os.path.join(epoch0_lr, "extras")
    if os.path.isdir(e0_extras):
        out_extras = os.path.join(out_lr, "extras")
        os.makedirs(out_extras, exist_ok=True)
        # dataset_meta.json at the root of extras
        dm = os.path.join(e0_extras, "dataset_meta.json")
        if os.path.exists(dm):
            # Copy (not symlink) so we could later modify without touching source.
            dst_dm = os.path.join(out_extras, "dataset_meta.json")
            if not os.path.exists(dst_dm):
                shutil.copy2(dm, dst_dm)
        for ep in e0_episodes:
            ei = ep["episode_index"]
            s = os.path.join(e0_extras, f"episode_{ei:06d}")
            d = os.path.join(out_extras, f"episode_{ei:06d}")
            if os.path.isdir(s):
                symlink_safe(s, d)

    # --- Meta files. ---
    n_total = len(merged_episodes)
    merged_info = dict(e0_info)
    merged_info["total_episodes"] = n_total
    merged_info["total_frames"] = total_frames
    merged_info["total_tasks"] = len(merged_tasks)
    merged_info["total_videos"] = n_total * len(VIDEO_KEYS)
    merged_info["splits"] = {"train": f"0:{n_total}"}
    with open(os.path.join(out_lr, "meta", "info.json"), "w") as f:
        json.dump(merged_info, f, indent=4, ensure_ascii=False)
    save_jsonl(merged_episodes, os.path.join(out_lr, "meta", "episodes.jsonl"))
    save_jsonl(merged_tasks, os.path.join(out_lr, "meta", "tasks.jsonl"))
    if merged_ep_stats:
        save_jsonl(merged_ep_stats, os.path.join(out_lr, "meta", "episodes_stats.jsonl"))

    for fname in ("embodiment.json", "modality.json", "stats.json"):
        s = os.path.join(epoch0_lr, "meta", fname)
        d = os.path.join(out_lr, "meta", fname)
        if os.path.exists(s) and not os.path.exists(d):
            shutil.copy2(s, d)

    # --- t5 embeddings: merge what's available; the caller is expected to
    #     (re)run generate_t5_embeddings.py afterwards to guarantee completeness. ---
    merged_t5 = {}
    t5_e0 = os.path.join(epoch0_lr, "meta", "t5_text_embeds.pt")
    t5_og = os.path.join(origin_lr, "meta", "t5_text_embeds.pt")
    if os.path.exists(t5_e0):
        merged_t5.update(torch.load(t5_e0, map_location="cpu", weights_only=True))
    if os.path.exists(t5_og):
        og_t5 = torch.load(t5_og, map_location="cpu", weights_only=True)
        for old_idx, emb in og_t5.items():
            new_idx = og_task_remap.get(old_idx, old_idx)
            if new_idx not in merged_t5:
                merged_t5[new_idx] = emb
    if merged_t5:
        torch.save(merged_t5, os.path.join(out_lr, "meta", "t5_text_embeds.pt"))

    return n_e0, len(og_episodes), total_frames, len(merged_tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch0", required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tasks = sorted(
        d for d in os.listdir(args.origin)
        if os.path.isdir(os.path.join(args.origin, d))
    )
    print(f"Found {len(tasks)} tasks in origin")

    summary = []
    for task_name in tqdm(tasks, desc="Merging"):
        e0_lr = find_lerobot_dir(args.epoch0, task_name)
        og_lr = find_lerobot_dir(args.origin, task_name)
        if e0_lr is None:
            print(f"  SKIP {task_name}: no epoch0 dataset")
            continue
        if og_lr is None:
            print(f"  SKIP {task_name}: no origin dataset")
            continue
        out_lr = os.path.join(args.output, task_name, "lerobot")
        n0, n1, nf, nt = merge_one_task(e0_lr, og_lr, out_lr)
        summary.append((task_name, n0, n1, nf, nt))
        tqdm.write(
            f"  {task_name}: {n0}(epoch0) + {n1}(origin) = {n0 + n1} eps, "
            f"{nf} frames, {nt} tasks"
        )

    print("\n=== Summary ===")
    for row in summary:
        print(f"  {row[0]:30s}  {row[1]:4d} + {row[2]:4d} = {row[1] + row[2]:4d} eps  "
              f"{row[3]:8d} frames  {row[4]} tasks")
    print(f"\nOutput: {args.output}")
    print("\nNext step (regenerate T5 embeddings to cover all merged tasks):")
    print(
        f"  /mnt/pfs/users/hengtao.li/conda_envs/gwpmot/bin/python "
        f"scripts/generate_t5_embeddings.py --data_root {args.output}"
    )


if __name__ == "__main__":
    main()
