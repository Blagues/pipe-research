"""
Raw data loading: CSI amplitude and MoCap markers from parquet files.
This is the only file that knows about the dataset's parquet schema.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


RECEIVERS = ["asus1", "asus2", "asus3"]

ACTIVITY_LABELS = {
    0: "squat",
    1: "jumping_jack",
    2: "jump",
    3: "boxing",
    4: "walk",
    5: "stand",
    6: "run",
    7: "stir_a_pot",
    8: "raising_left_foot",
    9: "raising_left_arm",
}


@dataclass
class Recording:
    csi_amplitude: dict          # {receiver: (N, 4, 114) float32}
    markers: np.ndarray          # (N, n_markers, 3) float32
    marker_names: list[str]
    labels: np.ndarray           # (N,) int32, -1 = unlabeled


def _unwrap_sequence(seq: np.ndarray, max_sequence: int) -> np.ndarray:
    return np.unwrap(seq.astype(np.float64), period=int(max_sequence) + 1).astype(np.int64)


def load_csi_for_receiver(
    rec_dir: Path, receiver: str, session: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (timestamps_us, csi_abs (N,4,114)). Phase is not used by the pipeline."""
    meta = (
        pl.read_parquet(rec_dir / "meta.parquet")
        .filter(pl.col("receiver_name") == receiver)
        .slice(session, 1)
    )
    row = meta.row(0, named=True)
    max_seq        = row["max_sequence"]
    start_epoch    = row["start_epoch"]
    frame_spacing  = row["frame_spacing_ms"]
    meta_id        = row["meta_id"]

    csi_df = (
        pl.scan_parquet(rec_dir / "csi.parquet")
        .filter(pl.col("meta_id") == meta_id)
        .select("sequence_number", "csi_abs")
        .collect()
    )
    seq = _unwrap_sequence(csi_df["sequence_number"].to_numpy(), max_seq)
    timestamps = (
        np.int64(start_epoch * 1_000_000)
        + seq.astype(np.int64) * np.int64(frame_spacing * 1_000)
    )
    # csi_abs stores each antenna as a 1-element list wrapping its 114 subcarriers,
    # so ant[0] peels that wrapper to give a (frames, 4, 114) array.
    csi_abs = np.array([[ant[0] for ant in row] for row in csi_df["csi_abs"].to_list()])
    return timestamps, csi_abs


def load_markers(rec_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (timestamps_us, positions (N, n_markers, 3), marker_names)."""
    df = pl.read_parquet(rec_dir / "markers.parquet")
    timestamps   = df["timestamp"].to_numpy()
    marker_cols  = [c for c in df.columns if c != "timestamp"]
    n_frames, n_markers = len(df), len(marker_cols)
    positions = np.zeros((n_frames, n_markers, 3), dtype=np.float32)
    # Each marker column is a struct {x, y, z} per frame. Frames where the marker
    # was not tracked come back as null (non-dict) and are stored as NaN.
    for i, col in enumerate(marker_cols):
        for j, row in enumerate(df[col].to_list()):
            if row and isinstance(row, dict):
                positions[j, i] = [row.get("x", np.nan), row.get("y", np.nan), row.get("z", np.nan)]
            else:
                positions[j, i] = np.nan
    return timestamps, positions, marker_cols


def fix_marker_flicker(
    positions: np.ndarray, bad_spot_radius: float = 100.0, max_gap: int = 60
) -> np.ndarray:
    """Replace near-origin outlier markers with linear interpolation.

    When the mocap system briefly loses a marker, that frame snaps to near the
    origin (or comes back as NaN). We flag those frames per marker, then fill each
    gap: short gaps with both ends valid are linearly interpolated, while gaps at
    the start/end of the recording just hold the nearest valid value.
    """
    n_frames, n_markers, _ = positions.shape
    fixed = positions.copy()
    # A frame is an outlier if the marker is missing (NaN) or sits within
    # bad_spot_radius of the origin.
    outlier = np.isnan(positions[:, :, 0])
    for m in range(n_markers):
        dist = np.linalg.norm(positions[:, m, :], axis=1)
        outlier[:, m] |= dist < bad_spot_radius

    for m in range(n_markers):
        if not outlier[:, m].any():
            continue
        mask = outlier[:, m]
        changes = np.diff(mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        if mask[0]:
            starts = np.concatenate(([0], starts))
        ends = np.where(changes == -1)[0] + 1
        if mask[-1]:
            ends = np.concatenate((ends, [n_frames]))
        for s, e in zip(starts, ends):
            gap = e - s
            before = next((i for i in range(s - 1, -1, -1) if not outlier[i, m]), None)
            after  = next((i for i in range(e, n_frames)   if not outlier[i, m]), None)
            bv = positions[before, m] if before is not None else None
            av = positions[after,  m] if after  is not None else None
            if bv is not None and av is not None and gap <= max_gap:
                for k, f in enumerate(range(s, e)):
                    t = (k + 1) / (gap + 1)
                    fixed[f, m] = (1 - t) * bv + t * av
            elif bv is not None:
                fixed[s:e, m] = bv
            elif av is not None:
                fixed[s:e, m] = av
    return fixed


def load_labels(rec_dir: Path) -> np.ndarray:
    """Returns (start_us, stop_us, label) rows as (N,3) array."""
    df = pl.read_parquet(rec_dir / "labels.parquet")
    return df.select(
        pl.col("start").cast(pl.Int64) // 1000,
        pl.col("stop").cast(pl.Int64)  // 1000,
        pl.col("label").cast(pl.Int64),
    ).to_numpy()


def load_synced_csi(rec_dir: Path) -> tuple[np.ndarray, dict]:
    """Load every receiver's CSI amplitude and time-align the non-reference
    receivers to the reference receiver's frame timestamps.

    Returns (ts_ref, csi_amp): ts_ref is (N,) microsecond timestamps of the
    reference receiver; csi_amp is a {receiver: (N, 4, 114)} dict aligned to ts_ref.
    """
    ref_rx = RECEIVERS[0]
    ts_ref, amp_ref = load_csi_for_receiver(rec_dir, ref_rx)

    csi_amp = {ref_rx: amp_ref}
    for rx in RECEIVERS[1:]:
        ts_rx, amp_rx = load_csi_for_receiver(rec_dir, rx)
        idx = np.searchsorted(ts_rx, ts_ref).clip(0, len(ts_rx) - 1)
        csi_amp[rx] = amp_rx[idx]
    return ts_ref, csi_amp


def load_recording(rec_dir: Path, fix_flicker: bool = True) -> Recording:
    """Load and synchronize CSI + mocap + labels for one recording."""
    ts_ref, csi_amp = load_synced_csi(rec_dir)
    N = len(ts_ref)

    _, positions, marker_names = load_markers(rec_dir)
    if fix_flicker:
        positions = fix_marker_flicker(positions)

    # Resample marker frames to CSI frame count; MoCap and CSI rates differ
    n_frames = len(positions)
    idx_m = (np.arange(N) * n_frames / N).astype(int).clip(0, n_frames - 1)
    markers_synced = positions[idx_m]

    # Start unlabeled (-1), then stamp each labeled [start, stop) interval onto the
    # frames whose timestamp falls inside it.
    label_rows = load_labels(rec_dir)
    labels = np.full(N, -1, dtype=np.int32)
    for start, stop, lbl in label_rows:
        mask = (ts_ref >= start) & (ts_ref < stop)
        labels[mask] = int(lbl)

    return Recording(
        csi_amplitude=csi_amp,
        markers=markers_synced,
        marker_names=marker_names,
        labels=labels,
    )
