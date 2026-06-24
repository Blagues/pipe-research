"""
Location-based train/val/holdout split and segment utilities.

Segments group consecutive windows that share the same activity label.
The split is computed by k-means clustering on each segment's mean hip XZ
position, so the model cannot exploit room-location cues instead of motion.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from src.data import load_markers


# Segment positions

def build_segment_positions(cache_dir: Path, recs: list[str], cfg: dict) -> list[dict]:
    """Extract per-segment mean hip XZ from the already-built cache."""
    rec_idx      = np.load(cache_dir / "recording_idx.npy")
    labels       = np.load(cache_dir / "labels.npy")
    datasets_dir = Path(cfg["data"]["datasets_dir"])

    segments = []
    for ri, rec in enumerate(recs):
        idxs = np.where(rec_idx == ri)[0]
        if len(idxs) == 0:
            continue
        rec_labels = labels[idxs]

        try:
            _, pos, names = load_markers(datasets_dir / rec)
            lhip = names.index("LEFT_HIP_BACK")
            rhip = names.index("RIGHT_HIP_BACK")
        except Exception:
            pos = None

        n_windows = len(idxs)
        n_frames  = len(pos) if pos is not None else 0

        # Walk the window labels and group each run of consecutive same-label
        # windows into one segment (windows [i, j)). Unlabeled (-1) windows are skipped.
        i = 0
        while i < len(rec_labels):
            lbl = rec_labels[i]
            if lbl == -1:
                i += 1
                continue
            j = i + 1
            while j < len(rec_labels) and rec_labels[j] == lbl:
                j += 1
            if pos is not None and n_frames > 0:
                # Windows and marker frames have different counts, so scale the
                # window span [i, j) to the matching marker-frame span [f0, f1).
                f0 = max(0, round(i * n_frames / n_windows))
                f1 = min(n_frames, max(round(j * n_frames / n_windows), f0 + 1))
                seg_pos = pos[f0:f1]
                # Mean hip XZ over the segment, dropping frames where the hip is near
                # the origin (mocap dropout) so they don't drag the average to (0, 0).
                hip_xz  = (seg_pos[:, lhip, [0, 2]] + seg_pos[:, rhip, [0, 2]]) / 2
                bad     = (np.abs(hip_xz[:, 0]) < 10) & (np.abs(hip_xz[:, 1]) < 10)
                hip_xz  = hip_xz[~bad]
                mean_xz = hip_xz.mean(0).astype(np.float32) if len(hip_xz) > 0 else np.zeros(2, np.float32)
            else:
                mean_xz = np.zeros(2, np.float32)
            segments.append(dict(ri=ri, lbl=int(lbl), win_start=int(i), win_end=int(j), mean_xz=mean_xz))
            i = j
    return segments


def save_segment_positions(cache_dir: Path, segments: list[dict]):
    np.savez(
        cache_dir / "segment_positions.npz",
        rec_i     = np.array([s["ri"]       for s in segments], dtype=np.int32),
        lbl       = np.array([s["lbl"]       for s in segments], dtype=np.int32),
        win_start = np.array([s["win_start"] for s in segments], dtype=np.int32),
        win_end   = np.array([s["win_end"]   for s in segments], dtype=np.int32),
        mean_xz   = np.array([s["mean_xz"]  for s in segments], dtype=np.float32),
    )


def load_segments(cache_dir: Path) -> list[dict]:
    npz = np.load(cache_dir / "segment_positions.npz")
    return [
        dict(ri=int(ri), lbl=int(lbl), win_start=int(ws), win_end=int(we),
             mean_xz=xz.astype(np.float32))
        for ri, lbl, ws, we, xz in zip(npz["rec_i"], npz["lbl"], npz["win_start"], npz["win_end"], npz["mean_xz"])
    ]


# Split computation

def build_split(segments: list[dict], cfg: dict) -> dict:
    sc        = cfg["split"]
    positions = np.array([s["mean_xz"] for s in segments])
    k         = min(sc["n_clusters"], len(segments))
    km        = KMeans(n_clusters=k, random_state=sc["seed"], n_init=10).fit(positions)
    sizes     = np.bincount(km.labels_, minlength=k)
    total     = len(segments)

    # Fill sets with clusters
    order            = np.argsort(-sizes)
    cluster_to_split = ["train"] * k
    holdout_count = 0
    for c in order:
        if holdout_count < total * sc["holdout_frac"]:
            cluster_to_split[c] = "holdout"
            holdout_count += sizes[c]
    val_count = 0
    for c in order:
        if cluster_to_split[c] == "train" and val_count < total * sc["val_frac"]:
            cluster_to_split[c] = "val"
            val_count += sizes[c]

    return {
        "cluster_centers": km.cluster_centers_.tolist(),
        "cluster_split":   cluster_to_split,
        "params": {
            "k":            k,
            "val_frac":     sc["val_frac"],
            "holdout_frac": sc["holdout_frac"],
            "seed":         sc["seed"],
        },
    }


def load_split(cache_dir: Path) -> dict:
    with open(cache_dir / "split.json") as f:
        raw = json.load(f)
    raw["cluster_centers"] = np.array(raw["cluster_centers"], dtype=np.float32)
    return raw


def assign_split(mean_xz: np.ndarray, split_data: dict) -> str:
    # A segment belongs to whichever split its nearest cluster center was assigned to.
    centers = split_data["cluster_centers"]
    dists   = np.sum((centers - mean_xz) ** 2, axis=1)
    return split_data["cluster_split"][int(np.argmin(dists))]


# Window index queries

def window_indices_for(
    segments: list[dict], split_data: dict, targets: set[str], rec_idx: np.ndarray
) -> np.ndarray:
    unique_ri   = np.unique([s["ri"] for s in segments])
    rec_windows = {int(ri): np.where(rec_idx == ri)[0] for ri in unique_ri}
    parts = []
    for seg in segments:
        if assign_split(seg["mean_xz"], split_data) in targets:
            parts.append(rec_windows[seg["ri"]][seg["win_start"]:seg["win_end"]])
    return np.concatenate(parts) if parts else np.array([], dtype=np.int64)


def segments_for(
    segments: list[dict], split_data: dict, targets: set[str], rec_idx: np.ndarray
) -> list[dict]:
    # Like window_indices_for, but returns the segment dicts (each with its window
    # indices attached as "win_idxs") instead of one flat array of indices.
    unique_ri   = np.unique([s["ri"] for s in segments])
    rec_windows = {int(ri): np.where(rec_idx == ri)[0] for ri in unique_ri}
    result = []
    for seg in segments:
        if assign_split(seg["mean_xz"], split_data) in targets:
            entry = dict(seg)
            entry["win_idxs"] = rec_windows[seg["ri"]][seg["win_start"]:seg["win_end"]]
            result.append(entry)
    return result


# Sequence builder

def make_seqs(
    segments: list[dict],
    split_data: dict,
    rec_idx: np.ndarray,
    roles: set[str],
    label_map: dict,
    max_seq_len: int,
    step: int | None = None,
) -> list[tuple[np.ndarray, int]]:
    """Tile each segment into fixed-length sequences of window indices.

    Each contiguous run of same-label windows becomes one or more sequences of
    length max_seq_len. step controls overlap: step < max_seq_len gives overlap,
    step == max_seq_len (default) gives non-overlapping sequences.
    """
    if step is None:
        step = max_seq_len
    seqs = []
    for seg in segments_for(segments, split_data, roles, rec_idx):
        idxs = seg["win_idxs"]
        if len(idxs) == 0:
            continue
        label = label_map[seg["lbl"]]
        for start in range(0, len(idxs), step):
            seqs.append((idxs[start:start + max_seq_len], label))
    return seqs
