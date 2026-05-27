"""Merge two LeRobot datasets (original atomic + collected) per task.

For each task in collected, find the matching original atomic dataset,
merge episodes (original first, then collected), re-index everything,
and write to output_dir/{task_name}/lerobot/.

Usage:
    python scripts/merge_lerobot_datasets.py \
        --original /shared_disk/users/hengtao.li/robocasa_datasets/v1.0/pretrain_gwp/atomic \
        --collected /shared_disk/users/hengtao.li/robocasa_datasets/collected/<mot-run>/checkpoint-10000/model \
        --output /shared_disk/users/hengtao.li/robocasa_datasets/collected/epoch1
"""

import argparse
import json
import os
import shutil

import pandas as pd
import torch
from tqdm import tqdm


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


VIDEO_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
]


def merge_one_task(orig_lr, coll_lr, out_lr):
    os.makedirs(os.path.join(out_lr, "data", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(out_lr, "meta"), exist_ok=True)

    orig_info = json.load(open(os.path.join(orig_lr, "meta", "info.json")))
    orig_episodes = load_jsonl(os.path.join(orig_lr, "meta", "episodes.jsonl"))
    coll_episodes = load_jsonl(os.path.join(coll_lr, "meta", "episodes.jsonl"))
    orig_tasks = load_jsonl(os.path.join(orig_lr, "meta", "tasks.jsonl"))
    coll_tasks = load_jsonl(os.path.join(coll_lr, "meta", "tasks.jsonl"))

    n_orig = len(orig_episodes)

    # Unified task list
    task_text_to_idx = {t["task"]: t["task_index"] for t in orig_tasks}
    merged_tasks = list(orig_tasks)
    for t in coll_tasks:
        if t["task"] not in task_text_to_idx:
            new_idx = len(merged_tasks)
            task_text_to_idx[t["task"]] = new_idx
            merged_tasks.append({"task_index": new_idx, "task": t["task"]})
    coll_task_remap = {t["task_index"]: task_text_to_idx[t["task"]] for t in coll_tasks}

    # Copy original episodes (re-index global index)
    total_frames = 0
    for ep in orig_episodes:
        ei = ep["episode_index"]
        src = os.path.join(orig_lr, "data", "chunk-000", f"episode_{ei:06d}.parquet")
        dst = os.path.join(out_lr, "data", "chunk-000", f"episode_{ei:06d}.parquet")
        df = pd.read_parquet(src)
        df["index"] = range(total_frames, total_frames + len(df))
        df.to_parquet(dst, index=False)
        total_frames += len(df)

    # Copy collected episodes (renumber)
    merged_episodes = list(orig_episodes)
    for ep in coll_episodes:
        old_i = ep["episode_index"]
        new_i = n_orig + old_i
        src = os.path.join(coll_lr, "data", "chunk-000", f"episode_{old_i:06d}.parquet")
        dst = os.path.join(out_lr, "data", "chunk-000", f"episode_{new_i:06d}.parquet")
        df = pd.read_parquet(src)
        df["episode_index"] = new_i
        df["index"] = range(total_frames, total_frames + len(df))
        df["task_index"] = df["task_index"].map(lambda x: coll_task_remap.get(x, x))
        df.to_parquet(dst, index=False)
        total_frames += len(df)
        merged_episodes.append({"episode_index": new_i, "tasks": ep["tasks"], "length": ep["length"]})

    # Symlink videos
    for vk in VIDEO_KEYS:
        out_vd = os.path.join(out_lr, "videos", "chunk-000", vk)
        os.makedirs(out_vd, exist_ok=True)
        orig_vd = os.path.join(orig_lr, "videos", "chunk-000", vk)
        if os.path.isdir(orig_vd):
            for ep in orig_episodes:
                ei = ep["episode_index"]
                s = os.path.join(orig_vd, f"episode_{ei:06d}.mp4")
                d = os.path.join(out_vd, f"episode_{ei:06d}.mp4")
                if os.path.exists(s) and not os.path.exists(d):
                    os.symlink(os.path.realpath(s), d)
        coll_vd = os.path.join(coll_lr, "videos", "chunk-000", vk)
        if os.path.isdir(coll_vd):
            for ep in coll_episodes:
                old_i = ep["episode_index"]
                new_i = n_orig + old_i
                s = os.path.join(coll_vd, f"episode_{old_i:06d}.mp4")
                d = os.path.join(out_vd, f"episode_{new_i:06d}.mp4")
                if os.path.exists(s) and not os.path.exists(d):
                    os.symlink(os.path.realpath(s), d)

    # Write meta
    n_total = len(merged_episodes)
    merged_info = dict(orig_info)
    merged_info["total_episodes"] = n_total
    merged_info["total_frames"] = total_frames
    merged_info["total_tasks"] = len(merged_tasks)
    merged_info["total_videos"] = n_total * len(VIDEO_KEYS)
    merged_info["splits"] = {"train": f"0:{n_total}"}
    with open(os.path.join(out_lr, "meta", "info.json"), "w") as f:
        json.dump(merged_info, f, indent=4, ensure_ascii=False)
    save_jsonl(merged_episodes, os.path.join(out_lr, "meta", "episodes.jsonl"))
    save_jsonl(merged_tasks, os.path.join(out_lr, "meta", "tasks.jsonl"))

    for fname in ("embodiment.json", "modality.json"):
        s = os.path.join(orig_lr, "meta", fname)
        d = os.path.join(out_lr, "meta", fname)
        if os.path.exists(s) and not os.path.exists(d):
            shutil.copy2(s, d)

    # Merge t5 embeds
    merged_t5 = {}
    t5_orig = os.path.join(orig_lr, "meta", "t5_text_embeds.pt")
    t5_coll = os.path.join(coll_lr, "meta", "t5_text_embeds.pt")
    if os.path.exists(t5_orig):
        merged_t5.update(torch.load(t5_orig, map_location="cpu", weights_only=True))
    if os.path.exists(t5_coll):
        for old_idx, emb in torch.load(t5_coll, map_location="cpu", weights_only=True).items():
            new_idx = coll_task_remap.get(old_idx, old_idx)
            if new_idx not in merged_t5:
                merged_t5[new_idx] = emb
    if merged_t5:
        torch.save(merged_t5, os.path.join(out_lr, "meta", "t5_text_embeds.pt"))

    return n_orig, len(coll_episodes), total_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--collected", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tasks = sorted(d for d in os.listdir(args.collected) if os.path.isdir(os.path.join(args.collected, d)))
    print(f"Found {len(tasks)} tasks in collected")

    for task_name in tqdm(tasks, desc="Merging"):
        orig_lr = find_lerobot_dir(args.original, task_name)
        coll_lr = find_lerobot_dir(args.collected, task_name)
        if orig_lr is None:
            print(f"  SKIP {task_name}: no original dataset")
            continue
        if coll_lr is None:
            print(f"  SKIP {task_name}: no collected dataset")
            continue
        out_lr = os.path.join(args.output, task_name, "lerobot")
        no, nc, nf = merge_one_task(orig_lr, coll_lr, out_lr)
        tqdm.write(f"  {task_name}: {no} + {nc} = {no+nc} episodes, {nf} frames")

    print("Done!")


if __name__ == "__main__":
    main()
