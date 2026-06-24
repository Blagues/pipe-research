"""
Window cache: build, load, and normalize the sliding-window dataset.

Cache layout (cache/<hash>/):
  csi.npy              (N, 3, T, 4, 114)  float16
  joints.npy           (N, K*3)            float32: hip-centred, rotation-normalised
  joints_vel.npy       (N, K)              float32: mean absolute velocity per joint (mm/frame)
  labels.npy           (N,)                int32
  recording_idx.npy    (N,)                int32
  csi_mean.npy         (3, T, 4, 114)      float32
  csi_std.npy          (3, T, 4, 114)      float32
  segment_positions.npz                    per-segment mean hip XZ
  split.json                               k-means location split
  meta.json                                cache parameters
"""
import copy
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import yaml

from src.data import load_recording, load_markers, fix_marker_flicker, RECEIVERS


# Config helpers

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Return a deep copy of cfg with dotted-key overrides applied."""
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = value
    return cfg


# Cache key / path

def _cache_key(cfg: dict, recs: list[str]) -> dict:
    # Everything that changes the window contents goes in here. Hash it (see
    # cache_path) and you get a different folder per config, so changing the
    # window size or joint set won't silently reuse a stale cache.
    dc = cfg["data"]
    return {
        "RAW_WINDOW":      dc["raw_window"],
        "STRIDE_RAW":      dc["stride"],
        "RELIABLE_JOINTS": dc["reliable_joints"],
        "fix_flicker":     True,
        "rotation_norm":   True,
        "csi_dtype":       "float16",
        "csi_norm":        "zscore_per_subcarrier",
        "recs":            sorted(recs),
    }


def cache_path(cfg: dict, recs: list[str]) -> Path:
    key    = _cache_key(cfg, recs)
    digest = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return Path(cfg["data"]["cache_dir"]) / digest


def find_cache(cfg: dict) -> Optional[Path]:
    """Return the largest valid cache dir whose meta.json matches the current config, or None."""
    cache_dir = Path(cfg["data"]["cache_dir"])
    if not cache_dir.exists():
        return None
    dc = cfg["data"]
    # We don't recompute the hash here: instead we scan every cache folder and
    # match on the fields that matter. If two match (e.g. a half-built one), keep
    # the bigger one, since window count is a decent proxy for "more complete".
    best, best_n_windows = None, -1
    for sub in cache_dir.iterdir():
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        labels_path = sub / "labels.npy"
        meta_path   = sub / "meta.json"
        if not labels_path.exists() or not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if (meta.get("RAW_WINDOW")      != dc["raw_window"] or
            meta.get("STRIDE_RAW")      != dc["stride"] or
            meta.get("RELIABLE_JOINTS") != dc["reliable_joints"]):
            continue
        n_windows = np.load(labels_path, mmap_mode="r").shape[0]
        if n_windows > best_n_windows:
            best, best_n_windows = sub, n_windows
    return best


# Per-recording workers (module-level for pickling)

_WORKER_CFG:       dict = {}
_WORKER_CACHE_DIR: Path = Path()


def _init_worker(cfg: dict, cache_dir: Path):
    global _WORKER_CFG, _WORKER_CACHE_DIR
    _WORKER_CFG       = cfg
    _WORKER_CACHE_DIR = cache_dir


def _count_windows(rec: str) -> tuple[str, int]:
    """Phase 1: count valid windows using markers only: no CSI loaded.

    Uses polars lazy scan to get the CSI row count from parquet metadata,
    then applies the same bad-frame filter as the fill pass.
    """
    cfg = _WORKER_CFG
    dc  = cfg["data"]
    RAW_WINDOW      = dc["raw_window"]
    STRIDE_RAW      = dc["stride"]
    RELIABLE_JOINTS = dc["reliable_joints"]

    rec_dir = Path(dc["datasets_dir"]) / rec
    if not rec_dir.exists():
        return rec, 0
    try:
        meta_row = (
            pl.read_parquet(rec_dir / "meta.parquet")
            .filter(pl.col("receiver_name") == RECEIVERS[0])
            .row(0, named=True)
        )
        # Count CSI rows without loading any data: reads parquet row-group metadata only
        n_csi = (
            pl.scan_parquet(rec_dir / "csi.parquet")
            .filter(pl.col("meta_id") == meta_row["meta_id"])
            .select(pl.len())
            .collect()
            .item()
        )
        _, positions, marker_names = load_markers(rec_dir)
        positions = fix_marker_flicker(positions)
        j_idx = [marker_names.index(m) for m in RELIABLE_JOINTS]
    except Exception:
        return rec, 0

    n_frames = len(positions)
    idx_m    = (np.arange(n_csi) * n_frames / n_csi).astype(int).clip(0, n_frames - 1)
    markers  = positions[idx_m]

    count = 0
    for s in range(0, n_csi - RAW_WINDOW + 1, STRIDE_RAW):
        m_win = markers[s:s + RAW_WINDOW]
        skip  = False
        for mi in j_idx:
            pos = m_win[:, mi, :]
            bad = ((np.abs(pos[:, 0]) < 10) & (np.abs(pos[:, 1]) < 10)) | np.any(np.isnan(pos), axis=1)
            if bad.mean() > 0.5:
                skip = True
                break
        if not skip:
            count += 1
    return rec, count


def _fill_recording(rec: str, ri: int, offset: int) -> tuple[str, int]:
    """Phase 2: process one recording and write directly to the pre-allocated shared mmap.

    Streams window-by-window so peak RAM per worker is around one recording's raw data.
    """
    cfg       = _WORKER_CFG
    cache_dir = _WORKER_CACHE_DIR
    dc        = cfg["data"]
    RAW_WINDOW      = dc["raw_window"]
    STRIDE_RAW      = dc["stride"]
    RELIABLE_JOINTS = dc["reliable_joints"]
    last_frame      = cfg["encoder"]["last_frame_label"]

    rec_dir = Path(dc["datasets_dir"]) / rec
    try:
        data   = load_recording(rec_dir)
        j_idx  = [data.marker_names.index(m) for m in RELIABLE_JOINTS]
        lhip_i = data.marker_names.index("LEFT_HIP_BACK")
        rhip_i = data.marker_names.index("RIGHT_HIP_BACK")
    except Exception as e:
        print(f"  {rec}: skipped: {e}")
        return rec, 0

    N = len(data.labels)

    csi_mm    = np.lib.format.open_memmap(cache_dir / "csi.npy",           mode="r+")
    joints_mm = np.lib.format.open_memmap(cache_dir / "joints.npy",        mode="r+")
    vel_mm    = np.lib.format.open_memmap(cache_dir / "joints_vel.npy",    mode="r+")
    labels_mm = np.lib.format.open_memmap(cache_dir / "labels.npy",        mode="r+")
    rec_mm    = np.lib.format.open_memmap(cache_dir / "recording_idx.npy", mode="r+")

    local_off = 0
    for s in range(0, N - RAW_WINDOW + 1, STRIDE_RAW):
        e      = s + RAW_WINDOW
        m_win  = data.markers[s:e]
        hip_xz = (m_win[:, lhip_i, [0, 2]] + m_win[:, rhip_i, [0, 2]]) / 2

        lh_xz   = m_win[:, lhip_i, [0, 2]]
        rh_xz   = m_win[:, rhip_i, [0, 2]]
        hip_bad = (
            ((np.abs(lh_xz[:, 0]) < 10) & (np.abs(lh_xz[:, 1]) < 10))
            | ((np.abs(rh_xz[:, 0]) < 10) & (np.abs(rh_xz[:, 1]) < 10))
        )
        axis = (rh_xz[~hip_bad] if (~hip_bad).any() else rh_xz).mean(0) - \
               (lh_xz[~hip_bad] if (~hip_bad).any() else lh_xz).mean(0)
        theta        = np.arctan2(axis[1], axis[0])
        cos_t, sin_t = np.cos(-theta), np.sin(-theta)

        joint_vecs, vel_vecs, skip = [], [], False
        for mi in j_idx:
            pos = m_win[:, mi, :]
            bad = ((np.abs(pos[:, 0]) < 10) & (np.abs(pos[:, 1]) < 10)) | np.any(np.isnan(pos), axis=1)
            if bad.mean() > 0.5:
                skip = True
                break
            pos_ok        = pos[~bad].copy()
            pos_ok[:, 0] -= hip_xz[~bad, 0]
            pos_ok[:, 2] -= hip_xz[~bad, 1]
            x = pos_ok[:, 0].copy()
            z = pos_ok[:, 2].copy()
            pos_ok[:, 0] = x * cos_t - z * sin_t
            pos_ok[:, 2] = x * sin_t + z * cos_t
            label_frame = pos_ok[-1] if last_frame else pos_ok.mean(0)
            joint_vecs.append(label_frame)
            speed = np.linalg.norm(np.diff(pos_ok, axis=0), axis=1).mean() if len(pos_ok) > 1 else 0.0
            vel_vecs.append(speed)
        if skip:
            continue

        csi        = np.stack([data.csi_amplitude[rx][s:e] for rx in RECEIVERS], axis=0)
        win_labels = data.labels[s:e]
        valid      = win_labels[win_labels != -1]
        label      = int(np.bincount(valid).argmax()) if len(valid) >= RAW_WINDOW // 2 else -1

        abs_off = offset + local_off
        csi_mm[abs_off]    = csi.astype(np.float16)
        joints_mm[abs_off] = np.concatenate(joint_vecs).astype(np.float32)
        vel_mm[abs_off]    = np.array(vel_vecs, dtype=np.float32)
        labels_mm[abs_off] = label
        rec_mm[abs_off]    = ri
        local_off += 1

    del csi_mm, joints_mm, vel_mm, labels_mm, rec_mm
    return rec, local_off


# Cache build

def build_cache(cfg: dict, recs: list[str]) -> Path:
    """Build (or load) the window cache. Returns the cache directory path.

    Two-phase build: zero temp files, peak disk = final cache size only:

      Phase 1 (parallel, fast): count valid windows per recording using
        markers only (no CSI loaded). Gives exact total for pre-allocation.

      Phase 2 (parallel): workers load CSI + markers, process windows, and
        write directly to their pre-assigned slice of the shared mmap file.
    """
    cache_dir = cache_path(cfg, recs)
    if (cache_dir / "meta.json").exists() and (cache_dir / "labels.npy").exists():
        print(f"Cache exists: {cache_dir.name}  ({np.load(cache_dir/'labels.npy', mmap_mode='r').shape[0]:,} windows)")
        return cache_dir

    dc          = cfg["data"]
    RAW_WINDOW  = dc["raw_window"]
    n_joints    = len(dc["reliable_joints"])
    max_workers = dc["load_workers"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building cache: {len(recs)} recordings, window={RAW_WINDOW}, stride={dc['stride']}")

    # Phase 1: count windows (markers only, no CSI)
    print(f"  Phase 1: counting valid windows across {len(recs)} recordings...")
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(cfg, cache_dir),
    ) as pool:
        for rec, n in pool.map(_count_windows, recs):
            if n > 0:
                counts[rec] = n

    total = sum(counts.values())
    if total == 0:
        raise RuntimeError("No valid windows found in any recording.")
    print(f"  Phase 1 done: {total:,} windows across {len(counts)} recordings.")

    # Now that Phase 1 told us how many windows each recording produces, we can
    # hand every recording its own slice of the big array up front: recording 0
    # gets rows [0, count0), recording 1 gets [count0, count0+count1), and so on.
    # Because the slices never overlap, the Phase 2 workers can all write at the
    # same time without locking. rec_to_ri keeps the original recs ordering so
    # recording_idx stays stable.
    offsets:   dict[str, int] = {}
    rec_to_ri: dict[str, int] = {}
    next_offset = 0
    for ri, rec in enumerate(recs):
        rec_to_ri[rec] = ri
        if rec in counts:
            offsets[rec] = next_offset
            next_offset += counts[rec]

    # Pre-allocate mmap arrays
    print(f"  Pre-allocating {total:,}-window mmap arrays...")
    np.lib.format.open_memmap(cache_dir / "csi.npy",           mode="w+", dtype=np.float16, shape=(total, 3, RAW_WINDOW, 4, 114))
    np.lib.format.open_memmap(cache_dir / "joints.npy",        mode="w+", dtype=np.float32, shape=(total, n_joints * 3))
    np.lib.format.open_memmap(cache_dir / "joints_vel.npy",    mode="w+", dtype=np.float32, shape=(total, n_joints))
    np.lib.format.open_memmap(cache_dir / "labels.npy",        mode="w+", dtype=np.int32,   shape=(total,))
    np.lib.format.open_memmap(cache_dir / "recording_idx.npy", mode="w+", dtype=np.int32,   shape=(total,))

    # Phase 2: fill mmap in parallel
    print(f"  Phase 2: filling cache with {max_workers} workers (direct mmap writes)...")
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(cfg, cache_dir),
    ) as pool:
        futures = {
            pool.submit(_fill_recording, rec, rec_to_ri[rec], offsets[rec]): rec
            for rec in recs if rec in offsets
        }
        for fut in as_completed(futures):
            rec, n_written = fut.result()
            if n_written > 0:
                print(f"  {rec}: {n_written} windows", flush=True)

    # Normalization placeholders: overwritten by recompute_normalization
    csi_shape = (3, RAW_WINDOW, 4, 114)
    np.save(cache_dir / "csi_mean.npy", np.zeros(csi_shape, np.float32))
    np.save(cache_dir / "csi_std.npy",  np.ones(csi_shape,  np.float32))

    with open(cache_dir / "meta.json", "w") as f:
        json.dump(_cache_key(cfg, recs), f, indent=2)
    print(f"Cache saved: {cache_dir}/  ({total:,} windows)")
    return cache_dir


# Normalization

def recompute_normalization(cache_dir: Path, segments: list[dict], split_data: dict) -> None:
    """Recompute csi_mean/csi_std using only train+val windows and overwrite cache files.

    Called once after the split is first computed so that holdout CSI never
    influences the normalisation statistics seen by the encoder.
    """
    from src.split import window_indices_for  # avoid circular import at module level

    rec_idx = np.load(cache_dir / "recording_idx.npy")
    idx     = np.sort(window_indices_for(segments, split_data, {"train", "val"}, rec_idx))

    csi    = np.load(cache_dir / "csi.npy", mmap_mode="r")
    shape  = csi.shape[1:]
    sum_c  = np.zeros(shape, np.float64)
    sum_sq = np.zeros(shape, np.float64)
    n      = 0
    chunk  = 1000
    total  = len(idx)
    for i in range(0, total, chunk):
        batch   = csi[idx[i:i + chunk]].astype(np.float64)
        sum_c  += batch.sum(axis=0)
        sum_sq += (batch ** 2).sum(axis=0)
        n      += len(batch)
        print(f"  normalisation: {min(i + chunk, total):,}/{total:,}", end="\r", flush=True)
    print()

    mean = (sum_c / n).astype(np.float32)
    std  = np.sqrt(np.maximum(sum_sq / n - mean.astype(np.float64) ** 2, 1e-8)).astype(np.float32)
    np.save(cache_dir / "csi_mean.npy", mean)
    np.save(cache_dir / "csi_std.npy",  std)

    meta_path = cache_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta["normalization"] = "train_val_only"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Normalisation recomputed on {n:,} train+val windows.")
