"""Bridge the contrastive model's cache + split to SHARP Doppler windows.

Guarantees the SHARP baseline trains/evaluates on the *exact same* train/val/
holdout partition as the contrastive pose model:

  * Segments and their k-means location split come from the existing
    ``segment_positions.npz`` + ``split.json`` produced by the contrastive
    pipeline. Each segment's split is recovered with ``assign_split`` against
    the same cluster centres -> provably identical membership.
  * A segment's continuous CSI frame range is recovered by replaying the exact
    window-validity loop used to build the cache (markers-only, no CSI load),
    so SHARP windows are drawn from precisely the frames that belong to that
    segment. A hard assertion checks the replayed valid-window count matches
    the cache, so any drift fails loudly before any number is trusted.

Doppler features are amplitude-only (csi_abs), 12 streams (3 rx x 4 ant), and
cached to ``<cache_dir>/sharp_features/<hash>/`` for instant re-runs.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
import torch

import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from src.data import (
    ACTIVITY_LABELS, RECEIVERS, load_markers, fix_marker_flicker, load_synced_csi,
)
from src.split import load_segments, load_split, assign_split

from sharp_baseline.config import SharpConfig
from sharp_baseline.doppler import (
    compute_doppler_profiles_numpy, compute_doppler_profiles_torch,
    create_windows, n_doppler_frames, n_windows,
)

SPLIT_CODES = {"train": 0, "val": 1, "holdout": 2}
SPLIT_NAMES = {v: k for k, v in SPLIT_CODES.items()}


# basics shared with the contrastive pipeline

def list_recordings(cfg: dict) -> list[str]:
    """Same ordering as pipeline/prepare.py so segment ``ri`` maps correctly."""
    import os
    datasets_dir = Path(cfg["data"]["datasets_dir"])
    return sorted(d for d in os.listdir(datasets_dir) if (datasets_dir / d).is_dir())


def build_label_map(segments: list[dict]) -> dict[int, int]:
    """Contiguous 0..C-1 mapping over labels present in segments (matches run_zeroshot)."""
    all_labels = np.unique([s["lbl"] for s in segments])
    return {int(c): i for i, c in enumerate(all_labels)}


def resolve_device(scfg: SharpConfig) -> torch.device:
    if scfg.feature_backend == "torch" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# split-parity: recover per-recording valid-window frame offsets

def recording_valid_starts(rec_dir: Path, cfg: dict) -> tuple[np.ndarray, int]:
    """Replay the cache's window-validity loop using markers only (no CSI load).

    Returns (valid_starts, n_csi) where valid_starts[k] is the starting CSI
    frame of the k-th *valid* window, in the same chronological order the cache
    stored windows. Mirrors src.cache._count_windows exactly.
    """
    dc = cfg["data"]
    raw_window = dc["raw_window"]
    stride = dc["stride"]
    reliable_joints = dc["reliable_joints"]

    meta_row = (
        pl.read_parquet(rec_dir / "meta.parquet")
        .filter(pl.col("receiver_name") == RECEIVERS[0])
        .row(0, named=True)
    )
    n_csi = (
        pl.scan_parquet(rec_dir / "csi.parquet")
        .filter(pl.col("meta_id") == meta_row["meta_id"])
        .select(pl.len())
        .collect()
        .item()
    )
    _, positions, marker_names = load_markers(rec_dir)
    positions = fix_marker_flicker(positions)
    j_idx = [marker_names.index(m) for m in reliable_joints]

    n_frames = len(positions)
    idx_m = (np.arange(n_csi) * n_frames / n_csi).astype(int).clip(0, n_frames - 1)
    markers = positions[idx_m]

    starts = []
    for s in range(0, n_csi - raw_window + 1, stride):
        m_win = markers[s:s + raw_window]
        skip = False
        for mi in j_idx:
            pos = m_win[:, mi, :]
            bad = ((np.abs(pos[:, 0]) < 10) & (np.abs(pos[:, 1]) < 10)) | np.any(np.isnan(pos), axis=1)
            if bad.mean() > 0.5:
                skip = True
                break
        if not skip:
            starts.append(s)
    return np.asarray(starts, dtype=np.int64), n_csi


def segment_frame_range(seg: dict, valid_starts: np.ndarray, raw_window: int) -> tuple[int, int]:
    """[frame_lo, frame_hi) of continuous CSI covered by a segment's windows."""
    lo = int(valid_starts[seg["win_start"]])
    hi = int(valid_starts[seg["win_end"] - 1]) + raw_window
    return lo, hi


# Doppler windowing for a contiguous CSI block (12 streams)

def _doppler(signal: np.ndarray, scfg: SharpConfig, device: torch.device) -> np.ndarray:
    if scfg.feature_backend == "torch":
        return compute_doppler_profiles_torch(
            signal, scfg.doppler_sample_length, scfg.doppler_stride,
            scfg.doppler_bins, scfg.noise_level, device, scfg.feature_batch_size,
        )
    return compute_doppler_profiles_numpy(
        signal, scfg.doppler_sample_length, scfg.doppler_stride,
        scfg.doppler_bins, scfg.noise_level,
    )


def block_windows(
    csi_amp: dict, frame_lo: int, frame_hi: int, scfg: SharpConfig, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """All SHARP windows for one contiguous block, across 12 streams.

    Returns (features (M,1,WL,bins) float16, stream_ids (M,) int8). Stream id =
    receiver_index * antennas_per_receiver + antenna_index.
    """
    feats: list[np.ndarray] = []
    stream_ids: list[int] = []
    for rx_i, rx in enumerate(RECEIVERS):
        block = csi_amp[rx][frame_lo:frame_hi]          # (T, 4, 114)
        for ant in range(scfg.antennas_per_receiver):
            signal = block[:, ant, :]                   # (T, 114)
            doppler = _doppler(signal, scfg, device)    # (n_frames, bins)
            if doppler.shape[0] == 0:
                continue
            doppler = doppler - doppler.mean(axis=0, keepdims=True)
            wins = create_windows(doppler, scfg.window_length, scfg.window_stride)
            if not wins:
                continue
            sid = rx_i * scfg.antennas_per_receiver + ant
            for w in wins:
                feats.append(w.astype(np.float16))
                stream_ids.append(sid)
    if not feats:
        return (np.empty((0, 1, scfg.window_length, scfg.doppler_bins), np.float16),
                np.empty((0,), np.int8))
    arr = np.stack(feats)[:, None, :, :]                # (M,1,WL,bins)
    return arr, np.asarray(stream_ids, dtype=np.int8)


def count_block_windows(n_packets: int, scfg: SharpConfig) -> int:
    """Per-stream window count for a block of n_packets (12x for all streams)."""
    nf = n_doppler_frames(n_packets, scfg.doppler_sample_length, scfg.doppler_stride)
    return n_windows(nf, scfg.window_length, scfg.window_stride)


# dev (train/val/holdout) feature cache

def dev_feature_dir(cache_dir: Path, scfg: SharpConfig) -> Path:
    split_blob = (cache_dir / "split.json").read_bytes()
    key = {"sharp": asdict(scfg), "cache": cache_dir.name,
           "split_md5": hashlib.md5(split_blob).hexdigest()}
    # backend/batch_size don't change feature values -> exclude from identity
    for k in ("feature_backend", "feature_batch_size"):
        key["sharp"].pop(k, None)
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return cache_dir / "sharp_features" / h


def build_dev_features(cfg: dict, scfg: SharpConfig, cache_dir: Path, force: bool = False) -> Path:
    """Compute and cache SHARP Doppler windows for every labelled segment.

    Output dir contains:
        features.npy     (M,1,WL,bins) float16   memmap
        labels.npy       (M,) int64    (mapped 0..C-1)
        splits.npy       (M,) int8     (0 train / 1 val / 2 holdout)
        segment_ids.npy  (M,) int64
        streams.npy      (M,) int8
        meta.json
    """
    out = dev_feature_dir(cache_dir, scfg)
    if out.exists() and (out / "meta.json").exists() and not force:
        print(f"SHARP dev features exist: {out}")
        return out

    device = resolve_device(scfg)
    segments = load_segments(cache_dir)
    split_data = load_split(cache_dir)
    recs = list_recordings(cfg)
    label_map = build_label_map(segments)
    rec_idx_arr = np.load(cache_dir / "recording_idx.npy", mmap_mode="r")
    raw_window = cfg["data"]["raw_window"]

    # segments grouped by recording, preserving a stable global segment id
    by_rec: dict[int, list[tuple[int, dict]]] = {}
    for gid, seg in enumerate(segments):
        by_rec.setdefault(int(seg["ri"]), []).append((gid, seg))

    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "shards").mkdir(parents=True)

    labels_all: list[int] = []
    splits_all: list[int] = []
    segids_all: list[int] = []
    streams_all: list[int] = []
    shard_paths: list[Path] = []
    total = 0
    dropped_short = 0
    split_seg_counts = {"train": 0, "val": 0, "holdout": 0}

    print(f"Building SHARP dev features on {device} -> {out}")
    for ri, rec in enumerate(recs):
        if ri not in by_rec:
            continue
        rec_dir = Path(cfg["data"]["datasets_dir"]) / rec
        try:
            valid_starts, _ = recording_valid_starts(rec_dir, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"  {rec}: skipped: {e}")
            continue

        # parity assertion: replayed valid-window count must match the cache
        cache_n = int(np.sum(rec_idx_arr == ri))
        if len(valid_starts) != cache_n:
            raise RuntimeError(
                f"Split-parity check failed for {rec}: replayed {len(valid_starts)} "
                f"valid windows but cache has {cache_n}. Frame mapping is unsafe, fix "
                f"before trusting any SHARP numbers."
            )

        _, csi_amp = load_synced_csi(rec_dir)        # amplitude per receiver
        rec_feats = []
        for gid, seg in by_rec[ri]:
            split = assign_split(seg["mean_xz"], split_data)
            frame_lo, frame_hi = segment_frame_range(seg, valid_starts, raw_window)
            if count_block_windows(frame_hi - frame_lo, scfg) == 0:
                dropped_short += 1
                continue
            feats, sids = block_windows(csi_amp, frame_lo, frame_hi, scfg, device)
            if feats.shape[0] == 0:
                dropped_short += 1
                continue
            rec_feats.append(feats)
            m = feats.shape[0]
            labels_all.extend([label_map[seg["lbl"]]] * m)
            splits_all.extend([SPLIT_CODES[split]] * m)
            segids_all.extend([gid] * m)
            streams_all.extend(sids.tolist())
            split_seg_counts[split] += 1
            total += m
        if rec_feats:
            shard = tmp / "shards" / f"{ri:04d}.npy"
            np.save(shard, np.concatenate(rec_feats, axis=0))
            shard_paths.append(shard)
        print(f"  {rec}: {sum(s.shape[0] for s in rec_feats)} windows "
              f"({len(by_rec[ri])} segments)", flush=True)

    if total == 0:
        raise RuntimeError("No SHARP dev windows produced.")

    # stitch shards into one memmap
    wl, bins = scfg.window_length, scfg.doppler_bins
    feat_mm = np.lib.format.open_memmap(
        tmp / "features.npy", mode="w+", dtype=np.float16, shape=(total, 1, wl, bins))
    off = 0
    for shard in shard_paths:
        arr = np.load(shard, mmap_mode="r")
        feat_mm[off:off + arr.shape[0]] = arr
        off += arr.shape[0]
    del feat_mm
    assert off == total, (off, total)

    np.save(tmp / "labels.npy", np.asarray(labels_all, dtype=np.int64))
    np.save(tmp / "splits.npy", np.asarray(splits_all, dtype=np.int8))
    np.save(tmp / "segment_ids.npy", np.asarray(segids_all, dtype=np.int64))
    np.save(tmp / "streams.npy", np.asarray(streams_all, dtype=np.int8))
    meta = {
        "sharp": asdict(scfg),
        "cache_dir": str(cache_dir),
        "n_windows": total,
        "n_classes": len(label_map),
        "label_map": {str(k): v for k, v in label_map.items()},
        "split_segment_counts": split_seg_counts,
        "dropped_short_segments": dropped_short,
        "window_split_counts": {
            name: int(np.sum(np.asarray(splits_all) == code))
            for name, code in SPLIT_CODES.items()
        },
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    shutil.rmtree(tmp / "shards")
    if out.exists():
        shutil.rmtree(out)
    tmp.replace(out)
    print(f"\nSHARP dev features: {total} windows, {len(label_map)} classes")
    print(f"  segments per split: {split_seg_counts}")
    print(f"  dropped sub-window segments: {dropped_short}")
    print(f"  saved -> {out}")
    return out


def load_dev_features(feature_dir: Path):
    feats = np.load(feature_dir / "features.npy", mmap_mode="r")
    labels = np.load(feature_dir / "labels.npy")
    splits = np.load(feature_dir / "splits.npy")
    segids = np.load(feature_dir / "segment_ids.npy")
    streams = np.load(feature_dir / "streams.npy")
    meta = json.loads((feature_dir / "meta.json").read_text())
    return feats, labels, splits, segids, streams, meta


# test-session (zero-shot) features

def session_windows(session_dir: Path, scfg: SharpConfig, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """SHARP windows for a whole test session (single activity, no MoCap, no split).

    Returns (features (M,1,WL,bins) float16, stream_ids (M,)).
    """
    _, csi_amp = load_synced_csi(session_dir)
    n = csi_amp[RECEIVERS[0]].shape[0]
    return block_windows(csi_amp, 0, n, scfg, device)


def activity_name(label_int: int) -> str:
    return ACTIVITY_LABELS.get(label_int, str(label_int))
