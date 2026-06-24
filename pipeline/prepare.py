"""
Step 1: Build the dataset cache. Run once before training.

Idempotent: safe to re-run; skips steps that are already complete.

Usage:
    python pipeline/prepare.py
    python pipeline/prepare.py --config config.yaml
"""
import argparse
import json
import os
import sys as _sys
import pathlib as _pathlib
import tempfile
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

_tmp = _pathlib.Path(__file__).parent.parent / "tmp"
_tmp.mkdir(exist_ok=True)
tempfile.tempdir = str(_tmp)

from pathlib import Path

from src.cache import load_config, build_cache, recompute_normalization
from src.split import (build_segment_positions, save_segment_positions,
                        build_split, load_segments, load_split, assign_split)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dc  = cfg["data"]

    datasets_dir = Path(dc["datasets_dir"])
    if not datasets_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {datasets_dir}\n"
            f"Set data.datasets_dir in config.yaml to your recordings folder."
        )

    recs = sorted(d for d in os.listdir(datasets_dir) if (datasets_dir / d).is_dir())
    print(f"Found {len(recs)} recordings in {datasets_dir}")
    if not recs:
        raise RuntimeError("No recording subdirectories found (expected rec00/, rec01/, …)")

    # Step 1: CSI + joints + velocity cache
    print("\nStep 1: building CSI / joints / velocity cache")
    cache_dir = build_cache(cfg, recs)

    # Step 2: segment positions
    seg_path = cache_dir / "segment_positions.npz"
    if seg_path.exists():
        print("\nStep 2: segment_positions.npz already exists, skipping")
        segments = load_segments(cache_dir)
    else:
        print("\nStep 2: extracting per-segment mean positions...")
        segments = build_segment_positions(cache_dir, recs, cfg)
        save_segment_positions(cache_dir, segments)
        print(f"  {len(segments)} segments saved to {seg_path}")

    # Step 3: location-based split
    split_path = cache_dir / "split.json"
    if split_path.exists():
        print("\nStep 3: split.json already exists, skipping")
        split_data = load_split(cache_dir)
    else:
        print("\nStep 3: computing k-means position split (seed=42)...")
        split_data = build_split(segments, cfg)
        with open(split_path, "w") as f:
            json.dump(split_data, f, indent=2)
        counts = {}
        for seg in segments:
            s = assign_split(seg["mean_xz"], split_data)
            counts[s] = counts.get(s, 0) + 1
        print("  Split: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        print(f"  Saved: {split_path}")

    # Step 4: normalisation (train+val windows only)
    with open(cache_dir / "meta.json") as f:
        meta = json.load(f)
    if meta.get("normalization") == "train_val_only":
        print("\nStep 4: normalisation already correct, skipping")
    else:
        print("\nStep 4: recomputing normalisation on train+val windows only...")
        recompute_normalization(cache_dir, segments, split_data)

    print(f"\nDone. Cache: {cache_dir}")
    print("Next: python pipeline/train.py")


if __name__ == "__main__":
    main()
