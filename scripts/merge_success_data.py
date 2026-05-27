"""Merge successful episodes from collected MoT rollout sources with epoch0.

Pass one or more source roots with --sources. Each source root should contain
<task>/lerobot directories. Episodes are kept when max(next.reward) > 0 in
meta/episodes_stats.jsonl. The epoch0 dataset, when provided, is appended in
full because it is assumed to contain successful episodes.

Example:
    python scripts/merge_success_data.py         --sources /shared_disk/users/hengtao.li/robocasa_datasets/collected/<mot-run>/checkpoint-10000/model         --epoch0 /shared_disk/users/hengtao.li/robocasa_datasets/collected/epoch0         --output_root /shared_disk/users/hengtao.li/codex/gwp-mot/data/success_all
"""

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel


COLLECTED = "/shared_disk/users/hengtao.li/robocasa_datasets/collected"
SOURCES: list[str] = []
EPOCH0 = os.path.join(COLLECTED, "epoch0")
OUTPUT_ROOT = "/shared_disk/users/hengtao.li/codex/gwp-mot/data/success_all"


def get_success_episodes(lerobot_dir: str) -> list[int]:
    """Return episode indices where max(next.reward) > 0."""
    stats_path = os.path.join(lerobot_dir, "meta", "episodes_stats.jsonl")
    success = []
    with open(stats_path) as f:
        for line in f:
            ep = json.loads(line.strip())
            if ep["stats"]["next.reward"]["max"][0] > 0:
                success.append(ep["episode_index"])
    return success


def copy_episode_files(src_lerobot: str, episode_idx: int, dst_lerobot: str, new_episode_idx: int):
    """Copy parquet data and video files for a single episode, renaming to new index."""
    src_chunk = episode_idx // 1000
    dst_chunk = new_episode_idx // 1000

    # Copy parquet
    src_parquet = os.path.join(
        src_lerobot, "data", f"chunk-{src_chunk:03d}", f"episode_{episode_idx:06d}.parquet"
    )
    dst_parquet_dir = os.path.join(dst_lerobot, "data", f"chunk-{dst_chunk:03d}")
    os.makedirs(dst_parquet_dir, exist_ok=True)
    dst_parquet = os.path.join(dst_parquet_dir, f"episode_{new_episode_idx:06d}.parquet")
    shutil.copy2(src_parquet, dst_parquet)

    # Copy videos
    src_video_chunk = os.path.join(src_lerobot, "videos", f"chunk-{src_chunk:03d}")
    if os.path.exists(src_video_chunk):
        for video_key in os.listdir(src_video_chunk):
            src_video = os.path.join(
                src_video_chunk, video_key, f"episode_{episode_idx:06d}.mp4"
            )
            if os.path.exists(src_video):
                dst_video_dir = os.path.join(
                    dst_lerobot, "videos", f"chunk-{dst_chunk:03d}", video_key
                )
                os.makedirs(dst_video_dir, exist_ok=True)
                dst_video = os.path.join(dst_video_dir, f"episode_{new_episode_idx:06d}.mp4")
                shutil.copy2(src_video, dst_video)


def reindex_parquet(parquet_path: str, new_episode_idx: int, global_index_offset: int):
    """Update episode_index and global index in a parquet file."""
    import pyarrow.parquet as pq
    import pyarrow as pa

    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    df["episode_index"] = new_episode_idx
    n = len(df)
    df["index"] = list(range(global_index_offset, global_index_offset + n))
    new_table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(new_table, parquet_path)
    return n


def merge_task(task_name: str, source_dirs: list[tuple[str, list[int]]], epoch0_lerobot: str | None):
    """Merge successful episodes for a single task into the output directory."""
    dst_lerobot = os.path.join(OUTPUT_ROOT, task_name, "lerobot")
    os.makedirs(os.path.join(dst_lerobot, "meta"), exist_ok=True)

    new_ep_idx = 0
    global_frame_idx = 0
    all_episodes = []
    all_episodes_stats = []
    task_set = {}

    # Collect from origin sources (only success episodes)
    for src_lerobot, success_eps in source_dirs:
        src_episodes = {}
        with open(os.path.join(src_lerobot, "meta", "episodes.jsonl")) as f:
            for line in f:
                ep = json.loads(line.strip())
                src_episodes[ep["episode_index"]] = ep

        src_stats = {}
        with open(os.path.join(src_lerobot, "meta", "episodes_stats.jsonl")) as f:
            for line in f:
                ep = json.loads(line.strip())
                src_stats[ep["episode_index"]] = ep

        # Read tasks
        with open(os.path.join(src_lerobot, "meta", "tasks.jsonl")) as f:
            for line in f:
                t = json.loads(line.strip())
                if t["task"] not in task_set:
                    task_set[t["task"]] = len(task_set)

        for ep_idx in success_eps:
            copy_episode_files(src_lerobot, ep_idx, dst_lerobot, new_ep_idx)
            n_frames = reindex_parquet(
                os.path.join(
                    dst_lerobot, "data",
                    f"chunk-{new_ep_idx // 1000:03d}",
                    f"episode_{new_ep_idx:06d}.parquet",
                ),
                new_ep_idx, global_frame_idx,
            )

            ep_info = src_episodes[ep_idx]
            ep_info["episode_index"] = new_ep_idx
            ep_info["length"] = n_frames
            all_episodes.append(ep_info)

            stat_info = src_stats[ep_idx]
            stat_info["episode_index"] = new_ep_idx
            all_episodes_stats.append(stat_info)

            global_frame_idx += n_frames
            new_ep_idx += 1

    # Collect from epoch0 (all episodes)
    if epoch0_lerobot and os.path.exists(epoch0_lerobot):
        with open(os.path.join(epoch0_lerobot, "meta", "episodes.jsonl")) as f:
            ep0_episodes = [json.loads(line.strip()) for line in f if line.strip()]
        with open(os.path.join(epoch0_lerobot, "meta", "episodes_stats.jsonl")) as f:
            ep0_stats = {json.loads(line.strip())["episode_index"]: json.loads(line.strip()) for line in f if line.strip()}
        with open(os.path.join(epoch0_lerobot, "meta", "tasks.jsonl")) as f:
            for line in f:
                t = json.loads(line.strip())
                if t["task"] not in task_set:
                    task_set[t["task"]] = len(task_set)

        for ep in ep0_episodes:
            old_idx = ep["episode_index"]
            copy_episode_files(epoch0_lerobot, old_idx, dst_lerobot, new_ep_idx)
            n_frames = reindex_parquet(
                os.path.join(
                    dst_lerobot, "data",
                    f"chunk-{new_ep_idx // 1000:03d}",
                    f"episode_{new_ep_idx:06d}.parquet",
                ),
                new_ep_idx, global_frame_idx,
            )

            ep["episode_index"] = new_ep_idx
            ep["length"] = n_frames
            all_episodes.append(ep)

            if old_idx in ep0_stats:
                stat = ep0_stats[old_idx]
                stat["episode_index"] = new_ep_idx
                all_episodes_stats.append(stat)

            global_frame_idx += n_frames
            new_ep_idx += 1

    # Write meta files
    tasks_list = [{"task_index": idx, "task": text} for text, idx in sorted(task_set.items(), key=lambda x: x[1])]
    with open(os.path.join(dst_lerobot, "meta", "tasks.jsonl"), "w") as f:
        for t in tasks_list:
            f.write(json.dumps(t) + "\n")

    with open(os.path.join(dst_lerobot, "meta", "episodes.jsonl"), "w") as f:
        for ep in all_episodes:
            f.write(json.dumps(ep) + "\n")

    with open(os.path.join(dst_lerobot, "meta", "episodes_stats.jsonl"), "w") as f:
        for s in all_episodes_stats:
            f.write(json.dumps(s) + "\n")

    # Build info.json from first available source
    ref_info_path = None
    for src_lerobot, _ in source_dirs:
        p = os.path.join(src_lerobot, "meta", "info.json")
        if os.path.exists(p):
            ref_info_path = p
            break
    if ref_info_path is None and epoch0_lerobot:
        ref_info_path = os.path.join(epoch0_lerobot, "meta", "info.json")

    with open(ref_info_path) as f:
        info = json.load(f)
    info["total_episodes"] = new_ep_idx
    info["total_frames"] = global_frame_idx
    info["total_tasks"] = len(tasks_list)
    info["total_videos"] = new_ep_idx * 3  # 3 camera views
    info["total_chunks"] = (new_ep_idx - 1) // 1000 + 1 if new_ep_idx > 0 else 0
    info["splits"] = {"train": f"0:{new_ep_idx}"}
    with open(os.path.join(dst_lerobot, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    # Copy other meta files (embodiment.json, modality.json, stats.json)
    for fname in ["embodiment.json", "modality.json", "stats.json"]:
        src_path = None
        for src_lerobot, _ in source_dirs:
            p = os.path.join(src_lerobot, "meta", fname)
            if os.path.exists(p):
                src_path = p
                break
        if src_path is None and epoch0_lerobot:
            p = os.path.join(epoch0_lerobot, "meta", fname)
            if os.path.exists(p):
                src_path = p
        if src_path:
            shutil.copy2(src_path, os.path.join(dst_lerobot, "meta", fname))

    return new_ep_idx, global_frame_idx, task_set


@torch.no_grad()
def encode_texts(texts, tokenizer, model, device, max_length=512):
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
    ).to(device)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.cpu().float()
    results = []
    for i, length in enumerate(inputs.attention_mask.sum(dim=1)):
        results.append(embeddings[i, :length])
    return results


def generate_t5_embeddings(wan_model_path: str, device: str = "cuda", batch_size: int = 32):
    """Generate T5 embeddings for all merged datasets."""
    te_path = os.path.join(wan_model_path, "text_encoder")
    tok_path = os.path.join(wan_model_path, "tokenizer")

    print(f"Loading UMT5 encoder from: {te_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    model = UMT5EncoderModel.from_pretrained(te_path, torch_dtype=torch.float16).to(device)
    model.eval()

    # Collect all unique tasks across all merged datasets
    unique_texts = set()
    task_files = glob.glob(os.path.join(OUTPUT_ROOT, "*/lerobot/meta/tasks.jsonl"))
    for tf in task_files:
        with open(tf) as f:
            for line in f:
                if line.strip():
                    unique_texts.add(json.loads(line.strip())["task"])

    unique_texts = sorted(unique_texts)
    print(f"Total unique task descriptions: {len(unique_texts)}")

    text_to_embed = {}
    for i in tqdm(range(0, len(unique_texts), batch_size), desc="Encoding T5"):
        batch = unique_texts[i : i + batch_size]
        embeds = encode_texts(batch, tokenizer, model, device)
        for text, embed in zip(batch, embeds):
            text_to_embed[text] = embed

    # Save per-task t5_text_embeds.pt
    for tf in task_files:
        meta_dir = os.path.dirname(tf)
        tasks = []
        with open(tf) as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line.strip()))
        embed_dict = {}
        for t in tasks:
            embed_dict[t["task_index"]] = text_to_embed[t["task"]]
        out_path = os.path.join(meta_dir, "t5_text_embeds.pt")
        torch.save(embed_dict, out_path)
        print(f"  Saved {out_path}")

    print("T5 embeddings generation complete.")


def main():
    global SOURCES, EPOCH0, OUTPUT_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="*", default=[],
                        help="Collected rollout roots containing <task>/lerobot directories")
    parser.add_argument("--epoch0", default=EPOCH0,
                        help="Optional epoch0 root containing <task>/lerobot directories")
    parser.add_argument("--output_root", default=OUTPUT_ROOT,
                        help="Merged output root")
    parser.add_argument("--wan_model_path",
        default="/shared_disk/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--skip_merge", action="store_true", help="Skip merge, only generate embeddings")
    args = parser.parse_args()

    SOURCES = args.sources
    EPOCH0 = args.epoch0
    OUTPUT_ROOT = args.output_root
    if not args.skip_merge and not SOURCES and not EPOCH0:
        raise ValueError("Provide --sources and/or --epoch0, or use --skip_merge")

    if not args.skip_merge:
        # Discover all task names
        all_tasks = set()
        for src in SOURCES:
            if os.path.exists(src):
                all_tasks.update(os.listdir(src))
        if os.path.exists(EPOCH0):
            all_tasks.update(os.listdir(EPOCH0))
        all_tasks = sorted(all_tasks)
        print(f"Found {len(all_tasks)} tasks: {all_tasks}")

        total_eps = 0
        total_frames = 0
        for task_name in tqdm(all_tasks, desc="Merging tasks"):
            source_dirs = []
            for src in SOURCES:
                lerobot_dir = os.path.join(src, task_name, "lerobot")
                if os.path.exists(lerobot_dir):
                    success_eps = get_success_episodes(lerobot_dir)
                    if success_eps:
                        source_dirs.append((lerobot_dir, success_eps))
                        print(f"  {src}/{task_name}: {len(success_eps)} success episodes")

            epoch0_lerobot = os.path.join(EPOCH0, task_name, "lerobot")
            if not os.path.exists(epoch0_lerobot):
                epoch0_lerobot = None
            else:
                with open(os.path.join(epoch0_lerobot, "meta", "info.json")) as f:
                    info = json.load(f)
                print(f"  epoch0/{task_name}: {info['total_episodes']} episodes")

            if not source_dirs and epoch0_lerobot is None:
                print(f"  Skipping {task_name}: no data")
                continue

            n_eps, n_frames, _ = merge_task(task_name, source_dirs, epoch0_lerobot)
            total_eps += n_eps
            total_frames += n_frames
            print(f"  => {task_name}: {n_eps} episodes, {n_frames} frames")

        print(f"\nMerge complete: {total_eps} total episodes, {total_frames} total frames")
        print(f"Output: {OUTPUT_ROOT}")

    # Generate T5 embeddings
    print("\nGenerating T5 embeddings...")
    generate_t5_embeddings(args.wan_model_path, args.device, args.batch_size)
    print("All done!")


if __name__ == "__main__":
    main()
