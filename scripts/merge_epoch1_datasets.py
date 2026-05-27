"""Merge two epoch1_truncated datasets that share a common epoch0 prefix.

For each sub-task:
  - Episodes [0, overlap) are shared (epoch0). Keep from D1.
  - Episodes [overlap, D1_total) are D1-only new data.
  - Episodes [overlap, D2_total) are D2-only new data, re-indexed after D1.

The merged dataset has: overlap + D1_new + D2_new episodes.
Parquet files are copied and the episode_index / index columns are re-written
for D2-new episodes. Videos are symlinked. Meta files are regenerated.
"""

import argparse
import json
import os
import shutil

import pandas as pd
from tqdm import tqdm


def find_overlap(d1_episodes_path: str, d2_episodes_path: str) -> int:
    d1, d2 = [], []
    with open(d1_episodes_path) as f:
        for line in f:
            d1.append(json.loads(line.strip()))
    with open(d2_episodes_path) as f:
        for line in f:
            d2.append(json.loads(line.strip()))
    for i in range(min(len(d1), len(d2))):
        if d1[i] != d2[i]:
            return i
    return min(len(d1), len(d2))


def load_episodes(path: str) -> list[dict]:
    eps = []
    with open(path) as f:
        for line in f:
            if line.strip():
                eps.append(json.loads(line.strip()))
    return eps


def merge_task(d1_lerobot: str, d2_lerobot: str, out_lerobot: str):
    d1_ep_path = os.path.join(d1_lerobot, "meta", "episodes.jsonl")
    d2_ep_path = os.path.join(d2_lerobot, "meta", "episodes.jsonl")

    d1_eps = load_episodes(d1_ep_path)
    d2_eps = load_episodes(d2_ep_path)
    overlap = find_overlap(d1_ep_path, d2_ep_path)

    d1_total = len(d1_eps)
    d2_total = len(d2_eps)
    d1_new = d1_total - overlap
    d2_new = d2_total - overlap
    merged_total = overlap + d1_new + d2_new

    os.makedirs(os.path.join(out_lerobot, "data", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(out_lerobot, "meta"), exist_ok=True)

    video_keys = []
    vid_dir = os.path.join(d1_lerobot, "videos", "chunk-000")
    if os.path.isdir(vid_dir):
        video_keys = sorted(os.listdir(vid_dir))
    for vk in video_keys:
        os.makedirs(os.path.join(out_lerobot, "videos", "chunk-000", vk), exist_ok=True)

    merged_eps = []
    total_frames = 0

    # --- Part 1: overlap episodes (from D1) ---
    for i in range(overlap):
        ep = d1_eps[i]
        src_pq = os.path.join(d1_lerobot, "data", "chunk-000", f"episode_{i:06d}.parquet")
        dst_pq = os.path.join(out_lerobot, "data", "chunk-000", f"episode_{i:06d}.parquet")
        os.symlink(os.path.realpath(src_pq), dst_pq)
        for vk in video_keys:
            src_v = os.path.join(d1_lerobot, "videos", "chunk-000", vk, f"episode_{i:06d}.mp4")
            dst_v = os.path.join(out_lerobot, "videos", "chunk-000", vk, f"episode_{i:06d}.mp4")
            os.symlink(os.path.realpath(src_v), dst_v)
        merged_eps.append({"episode_index": i, "tasks": ep["tasks"], "length": ep["length"]})
        total_frames += ep["length"]

    # --- Part 2: D1-only new episodes ---
    for i in range(overlap, d1_total):
        new_idx = i  # same index as D1
        ep = d1_eps[i]
        src_pq = os.path.join(d1_lerobot, "data", "chunk-000", f"episode_{i:06d}.parquet")
        dst_pq = os.path.join(out_lerobot, "data", "chunk-000", f"episode_{new_idx:06d}.parquet")
        os.symlink(os.path.realpath(src_pq), dst_pq)
        for vk in video_keys:
            src_v = os.path.join(d1_lerobot, "videos", "chunk-000", vk, f"episode_{i:06d}.mp4")
            dst_v = os.path.join(out_lerobot, "videos", "chunk-000", vk, f"episode_{new_idx:06d}.mp4")
            os.symlink(os.path.realpath(src_v), dst_v)
        merged_eps.append({"episode_index": new_idx, "tasks": ep["tasks"], "length": ep["length"]})
        total_frames += ep["length"]

    # --- Part 3: D2-only new episodes (re-index) ---
    # Need to rewrite parquet to update episode_index and index columns
    cumulative_index = total_frames  # global frame index continues
    for i in range(overlap, d2_total):
        new_idx = d1_total + (i - overlap)
        ep = d2_eps[i]
        src_pq = os.path.join(d2_lerobot, "data", "chunk-000", f"episode_{i:06d}.parquet")
        dst_pq = os.path.join(out_lerobot, "data", "chunk-000", f"episode_{new_idx:06d}.parquet")

        df = pd.read_parquet(src_pq)
        df["episode_index"] = new_idx
        df["index"] = range(cumulative_index, cumulative_index + len(df))
        df.to_parquet(dst_pq, index=False)
        cumulative_index += len(df)

        for vk in video_keys:
            src_v = os.path.join(d2_lerobot, "videos", "chunk-000", vk, f"episode_{i:06d}.mp4")
            dst_v = os.path.join(out_lerobot, "videos", "chunk-000", vk, f"episode_{new_idx:06d}.mp4")
            os.symlink(os.path.realpath(src_v), dst_v)
        merged_eps.append({"episode_index": new_idx, "tasks": ep["tasks"], "length": ep["length"]})
        total_frames += ep["length"]

    # --- Write meta ---
    # tasks.jsonl: merge from both (deduplicate by task text)
    d1_tasks = load_episodes(os.path.join(d1_lerobot, "meta", "tasks.jsonl"))
    d2_tasks = load_episodes(os.path.join(d2_lerobot, "meta", "tasks.jsonl"))
    seen_tasks = {}
    all_tasks = []
    for t in d1_tasks + d2_tasks:
        if t["task"] not in seen_tasks:
            seen_tasks[t["task"]] = t["task_index"]
            all_tasks.append(t)

    with open(os.path.join(out_lerobot, "meta", "tasks.jsonl"), "w") as f:
        for t in all_tasks:
            f.write(json.dumps(t) + "\n")

    # episodes.jsonl
    with open(os.path.join(out_lerobot, "meta", "episodes.jsonl"), "w") as f:
        for ep in merged_eps:
            f.write(json.dumps(ep) + "\n")

    # info.json: copy from D1 and update counts
    with open(os.path.join(d1_lerobot, "meta", "info.json")) as f:
        info = json.load(f)
    info["total_episodes"] = merged_total
    info["total_frames"] = total_frames
    info["total_videos"] = merged_total * len(video_keys)
    info["total_tasks"] = len(all_tasks)
    info["splits"] = {"train": f"0:{merged_total}"}
    with open(os.path.join(out_lerobot, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)
        f.write("\n")

    # Copy other meta files from D1
    for fname in ["embodiment.json", "modality.json", "stats.json"]:
        src = os.path.join(d1_lerobot, "meta", fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_lerobot, "meta", fname))

    return overlap, d1_new, d2_new, merged_total, total_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1", required=True, help="Path to first epoch1_truncated")
    parser.add_argument("--d2", required=True, help="Path to second epoch1_truncated")
    parser.add_argument("--output", required=True, help="Output merged directory")
    args = parser.parse_args()

    tasks = sorted(os.listdir(args.d1))
    grand_episodes = 0
    grand_frames = 0

    for task_name in tqdm(tasks, desc="Merging tasks"):
        d1_lerobot = os.path.join(args.d1, task_name, "lerobot")
        d2_lerobot = os.path.join(args.d2, task_name, "lerobot")
        out_lerobot = os.path.join(args.output, task_name, "lerobot")

        if not os.path.isdir(d1_lerobot) or not os.path.isdir(d2_lerobot):
            print(f"Skipping {task_name}: missing in one dataset")
            continue

        overlap, d1_new, d2_new, merged_total, total_frames = merge_task(
            d1_lerobot, d2_lerobot, out_lerobot
        )
        grand_episodes += merged_total
        grand_frames += total_frames
        tqdm.write(
            f"  {task_name}: overlap={overlap} + D1_new={d1_new} + D2_new={d2_new} = {merged_total} eps, {total_frames} frames"
        )

    print(f"\nDone! Total: {grand_episodes} episodes, {grand_frames} frames")


if __name__ == "__main__":
    main()
