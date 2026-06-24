"""
Section 6.3: Window Size
==========================================
Sweeps window sizes {10, 50, 100, 200} ms with fixed body-state loss
(σp=150, σv=1, whitening), stride fixed at 10 ms.

Important: each cache requires between 50 - 280 GB, depending on window size!
The script checks free space before building and skips window sizes that would exceed available disk.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

_tmp = Path(__file__).parent / "tmp"
_tmp.mkdir(exist_ok=True)
tempfile.tempdir = str(_tmp)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from src.data import ACTIVITY_LABELS, RECEIVERS, load_synced_csi
from src.cache import load_config, apply_overrides, build_cache, find_cache, recompute_normalization
from src.split import (build_segment_positions, save_segment_positions,
                        build_split, load_segments, load_split, segments_for)
from src.training import build_encoder, embed_all, train_har, SequenceDataset


# global params
SEED      = 42
WINDOWS   = [10, 50, 100, 200]
LOCATIONS = ["middle", "edge", "outside"]

_NAME_FIX  = {"raise_left_arm": "raising_left_arm",
               "raise_left_foot": "raising_left_foot",
               "stir_pot": "stir_a_pot"}
_LABEL_INV = {v: k for k, v in ACTIVITY_LABELS.items()}

SEP  = "═" * 62
SEP2 = "─" * 62


def _parse_session(name: str) -> tuple[int, str]:
    parts = name.split("_")
    act   = _NAME_FIX.get("_".join(parts[2:-1]), "_".join(parts[2:-1]))
    return _LABEL_INV[act], parts[-1]


def _load_test_windows(session_dir: Path, cfg: dict) -> np.ndarray:
    dc = cfg["data"]
    rw, stride = dc["raw_window"], dc["stride"]
    _, csi_amp = load_synced_csi(session_dir)
    stacked = np.stack([csi_amp[rx] for rx in RECEIVERS], axis=0)
    N = stacked.shape[1]
    wins = [stacked[:, s:s + rw] for s in range(0, N - rw + 1, stride)]
    return np.stack(wins).astype(np.float16) if wins else np.zeros((0, 3, rw, 4, 114), np.float16)


@torch.no_grad()
def _eval_on_seqs(clf, emb, seqs, msl, device) -> tuple[int, int]:
    clf.eval()
    loader  = DataLoader(SequenceDataset(emb, seqs, msl), batch_size=32)
    correct = total = 0
    for x, mask, y in loader:
        preds   = clf(x.to(device), mask.to(device)).argmax(1).cpu()
        correct += (preds == y).sum().item()
        total   += len(y)
    return correct, total


def _make_seqs(segments, split_data, role, rec_idx, label_map, msl, step=None):
    if step is None:
        step = msl
    seqs = []
    for seg in segments_for(segments, split_data, role, rec_idx):
        idxs = seg["win_idxs"]
        if len(idxs) == 0:
            continue
        lbl = label_map[seg["lbl"]]
        for start in range(0, len(idxs), step):
            seqs.append((idxs[start:start + msl], lbl))
    return seqs


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg_base = load_config("config.yaml")
    dc       = cfg_base["data"]
    datasets_dir = Path(dc["datasets_dir"])
    test_dir     = Path(dc["test_datasets_dir"])
    recs = sorted(d for d in os.listdir(datasets_dir) if (datasets_dir / d).is_dir())

    out = Path("runs/window_sweep/results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    results = {int(k): v for k, v in json.loads(out.read_text()).items()} if out.exists() else {}

    for win in WINDOWS:
        if win in results and results[win].get("zeroshot"):
            print(f"\n  WIN={win} already complete, skipping.")
            continue

        batch = max(32, 128 * 100 // win)
        cfg = apply_overrides(cfg_base, {
            "data.raw_window":       win,
            "encoder.similarity":    "pose_vel",
            "encoder.sigma":         150,
            "encoder.sigma_v":       1.0,
            "encoder.epochs":        2,
            "encoder.har_interval":  2,
            "encoder.tsne_interval": 0,
            "encoder.batch_size":    batch,
        })

        print(f"\n{SEP}")
        print(f"  WIN={win} ms  (batch={batch})")
        print(SEP)

        # build or reuse cache
        existing = find_cache(cfg)
        if existing is not None:
            cache_dir, cache_is_new = existing, False
            print(f"  Reusing cache: {cache_dir}")
        else:
            cache_root = Path(dc.get("cache_dir", "../cache"))
            free_gb    = shutil.disk_usage(cache_root if cache_root.exists() else cache_root.parent).free / 1e9
            need_gb    = max(40, int(win / 100 * 270))
            if free_gb < need_gb:
                print(f"  SKIP WIN={win}: {free_gb:.0f} GB free, need ~{need_gb} GB. "
                      f"Free up disk and restart.")
                continue
            print(f"  Disk: {free_gb:.0f} GB free (need ~{need_gb} GB)")
            cache_dir, cache_is_new = build_cache(cfg, recs), True

        seg_path = cache_dir / "segment_positions.npz"
        if seg_path.exists():
            segments = load_segments(cache_dir)
        else:
            segments = build_segment_positions(cache_dir, recs, cfg)
            save_segment_positions(cache_dir, segments)

        split_path = cache_dir / "split.json"
        if split_path.exists():
            split_data = load_split(cache_dir)
        else:
            split_data = build_split(segments, cfg)
            with open(split_path, "w") as f:
                json.dump(split_data, f, indent=2)

        with open(cache_dir / "meta.json") as f:
            meta = json.load(f)
        if meta.get("normalization") != "train_val_only":
            print("  Recomputing normalisation (train+val only)...")
            recompute_normalization(cache_dir, segments, split_data)

        csi_mean = np.load(cache_dir / "csi_mean.npy")
        csi_std  = np.load(cache_dir / "csi_std.npy")
        rec_idx  = np.load(cache_dir / "recording_idx.npy")

        all_labels = sorted({s["lbl"] for s in segments})
        label_map  = {int(c): i for i, c in enumerate(all_labels)}
        n_classes  = len(all_labels)
        msl        = cfg["har"]["max_seq_len"]

        train_seqs = _make_seqs(segments, split_data, {"train"},   rec_idx, label_map, msl, step=max(1, msl // 2))
        val_seqs   = _make_seqs(segments, split_data, {"val"},     rec_idx, label_map, msl)
        hld_seqs   = _make_seqs(segments, split_data, {"holdout"}, rec_idx, label_map, msl)

        # train encoder
        run_dir = Path(f"runs/window_sweep/win{win}")
        from src.training import train_encoder
        train_encoder(cfg, run_dir)

        # load trained encoder
        enc_cfg = json.loads((run_dir / "config.json").read_text())
        encoder = build_encoder(enc_cfg).to(device)
        encoder.load_state_dict(torch.load(run_dir / "best_model.pt",
                                            map_location=device, weights_only=True))
        encoder.eval()

        gn       = enc_cfg["encoder"].get("global_norm", True)
        eff_mean = csi_mean if gn else np.zeros_like(csi_mean)
        eff_std  = csi_std  if gn else np.ones_like(csi_std)

        # embed all cache windows
        csi_mm    = np.load(cache_dir / "csi.npy", mmap_mode="r")
        emb_batch = max(256, min(4096, int(4096 * (100 / win) ** 2)))
        print(f"  Embedding {len(csi_mm):,} windows (batch={emb_batch})...")
        emb = embed_all(encoder, csi_mm, eff_mean, eff_std, enc_cfg, device, batch=emb_batch)

        # evaluate on holdout using HAR classifier
        har_cfg = dict(enc_cfg)
        har_cfg["har"] = dict(enc_cfg["har"])
        har_cfg["har"]["epochs"] = 40
        har_cfg["har"]["patience"] = 7
        print("  Training HAR probe (40 epochs)...")
        val_acc, hld_acc, clf = train_har(
            emb, train_seqs, val_seqs, hld_seqs,
            n_classes, har_cfg, device, return_clf=True,
        )
        print(f"  Table 4: val={val_acc:.3f}  holdout={hld_acc:.3f}")

        # evaluate on zero-shot
        loc_correct = {loc: 0 for loc in LOCATIONS}
        loc_total   = {loc: 0 for loc in LOCATIONS}
        for sess in sorted(test_dir.iterdir()):
            try:
                session_label, loc = _parse_session(sess.name)
            except (KeyError, IndexError):
                print(f"  SKIP (parse): {sess.name}")
                continue
            mapped  = label_map[session_label]
            windows = _load_test_windows(sess, enc_cfg)
            if len(windows) == 0:
                continue
            test_emb = embed_all(encoder, windows, eff_mean, eff_std, enc_cfg, device)
            seqs = [(np.arange(len(windows))[s:s + msl], mapped)
                    for s in range(0, len(windows), msl)]
            c, t = _eval_on_seqs(clf, test_emb, seqs, msl, device)
            loc_correct[loc] += c
            loc_total[loc]   += t
            print(f"    {sess.name:<46} {c}/{t}")

        per_loc   = {loc: round(loc_correct[loc] / loc_total[loc], 4) if loc_total[loc] else None
                     for loc in LOCATIONS}
        overall_c = sum(loc_correct.values())
        overall_t = sum(loc_total.values())
        overall   = round(overall_c / overall_t, 4) if overall_t else None
        print(f"  Table 5: middle={per_loc['middle']}  edge={per_loc['edge']}  "
              f"outside={per_loc['outside']}  overall={overall}")

        results[win] = {
            "win": win, "val_acc": round(val_acc, 4), "hld_acc": round(hld_acc, 4),
            "zeroshot": {"per_loc": per_loc, "overall": overall},
        }
        out.write_text(json.dumps(results, indent=2, default=str))

        # free disk. we keep the 100 cache for other experiments
        if cache_is_new and win != 100:
            shutil.rmtree(cache_dir)
            print("  Cache removed.")

    # print table summary
    done = [w for w in WINDOWS if w in results]
    if done:
        def fmt(v): return f"{v:.3f}" if v is not None else "   -  "

        print(f"\n\n{SEP}")
        print("  TABLE 4: In-distribution HAR accuracy (body-state σv=1, whitening, stride=10ms)")
        print(SEP)
        print(f"  {'Window':>8} {'Val':>8} {'Holdout':>8}")
        print(f"  {SEP2}")
        for win in WINDOWS:
            r = results.get(win, {})
            print(f"  {win:>5} ms {fmt(r.get('val_acc')):>8} {fmt(r.get('hld_acc')):>8}")

        print(f"\n\n{SEP}")
        print("  TABLE 5: Zero-shot accuracy per location (%)")
        print(SEP)
        print(f"  {'Window':>8} {'Middle':>8} {'Edge':>8} {'Outside':>8} {'Overall':>8}")
        print(f"  {SEP2}")
        for win in WINDOWS:
            r  = results.get(win, {})
            zs = r.get("zeroshot", {})
            pl = zs.get("per_loc", {})
            print(f"  {win:>5} ms {fmt(pl.get('middle')):>8} {fmt(pl.get('edge')):>8}"
                  f" {fmt(pl.get('outside')):>8} {fmt(zs.get('overall')):>8}")

        out.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n  Saved: {out}")
        print("\n  Next: python scripts/exp_position_invariance.py")


if __name__ == "__main__":
    main()
