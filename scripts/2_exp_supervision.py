"""
Section 6.2: Supervision Objectives
======================================================
Bandwidth sweep across three loss variants:
  pose: σ_p ∈ {75, 150, 300} mm
  vel: σ_v ∈ {0.1, 0.25, 0.5, 2} mm/frame
  body-state: σ_v ∈ {0.25, 0.5, 1, 2} mm/frame  (σ_p fixed at 150)

SupCon comparison: best body-state (σv=1) vs. supervised
  contrastive learning with direct activity labels (SupCon).
  Reports in-distribution holdout and zero-shot accuracy.
"""
import csv
import fcntl
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from src.data import ACTIVITY_LABELS, RECEIVERS, load_synced_csi
from src.cache import load_config, apply_overrides, find_cache
from src.split import load_segments, load_split, segments_for
from src.training import build_encoder, embed_all, train_har, SequenceDataset


# global params

SEED      = 42
LOCATIONS = ["middle", "edge", "outside"]

_NAME_FIX  = {"raise_left_arm": "raising_left_arm",
               "raise_left_foot": "raising_left_foot",
               "stir_pot": "stir_a_pot"}
_LABEL_INV = {v: k for k, v in ACTIVITY_LABELS.items()}

SIGMA_SWEEP = [
    ("pose σ=75",          "pose_s75",   {"encoder.similarity": "pose",     "encoder.sigma": 75}),
    ("pose σ=150",         "pose_s150",  {"encoder.similarity": "pose",     "encoder.sigma": 150}),
    ("pose σ=300",         "pose_s300",  {"encoder.similarity": "pose",     "encoder.sigma": 300}),
    ("vel σv=0.1",         "vel_sv0_1",  {"encoder.similarity": "vel",      "encoder.sigma_v": 0.1}),
    ("vel σv=0.25",        "vel_sv0_25", {"encoder.similarity": "vel",      "encoder.sigma_v": 0.25}),
    ("vel σv=0.5",         "vel_sv0_5",  {"encoder.similarity": "vel",      "encoder.sigma_v": 0.5}),
    ("vel σv=2",           "vel_sv2",    {"encoder.similarity": "vel",      "encoder.sigma_v": 2}),
    ("body-state σv=0.25", "pv_sv0_25",  {"encoder.similarity": "pose_vel", "encoder.sigma_v": 0.25}),
    ("body-state σv=0.5",  "pv_sv0_5",  {"encoder.similarity": "pose_vel", "encoder.sigma_v": 0.5}),
    ("body-state σv=1",    "pv_sv1",    {"encoder.similarity": "pose_vel", "encoder.sigma_v": 1.0}),
    ("body-state σv=2",    "pv_sv2",    {"encoder.similarity": "pose_vel", "encoder.sigma_v": 2.0}),
]
SWEEP_NAMES = [r for _, r, _ in SIGMA_SWEEP]

SEP  = "═" * 56
SEP2 = "─" * 56


# helper functions

def _locked_save(path: Path, run_name: str, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        content = f.read()
        data = json.loads(content) if content.strip() else {}
        data[run_name] = value
        f.seek(0)
        f.truncate()
        f.write(json.dumps(data, indent=2))
        fcntl.flock(f, fcntl.LOCK_UN)


def _read_ep2_from_log(run_dir: Path) -> tuple[float, float]:
    rows = list(csv.DictReader(open(run_dir / "activity_log.csv")))
    ep2  = rows[0] if rows else {}
    return float(ep2.get("val", 0)), float(ep2.get("holdout", 0))


def _parse_session(name: str) -> tuple[int, str]:
    parts = name.split("_")
    act   = _NAME_FIX.get("_".join(parts[2:-1]), "_".join(parts[2:-1]))
    return _LABEL_INV[act], parts[-1]


def _load_test_windows(session_dir: Path, cfg: dict) -> np.ndarray:
    dc = cfg["data"]
    _, csi_amp = load_synced_csi(session_dir)
    stacked = np.stack([csi_amp[rx] for rx in RECEIVERS], axis=0)
    N = stacked.shape[1]
    rw, st = dc["raw_window"], dc["stride"]
    wins = [stacked[:, s:s + rw] for s in range(0, N - rw + 1, st)]
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


def _make_cache_seqs(segments, split_data, rec_idx, label_map, role, msl, step=None):
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


def _run_zeroshot(encoder, clf, label_map, enc_cfg, eff_mean, eff_std, msl, device, test_dir: Path) -> dict:
    loc_correct = {loc: 0 for loc in LOCATIONS}
    loc_total   = {loc: 0 for loc in LOCATIONS}
    for sess in sorted(test_dir.iterdir()):
        try:
            session_label, loc = _parse_session(sess.name)
        except (KeyError, IndexError):
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
    overall_c = sum(loc_correct.values())
    overall_t = sum(loc_total.values())
    return {
        "per_loc": {loc: round(loc_correct[loc] / loc_total[loc], 4) if loc_total[loc] else None
                    for loc in LOCATIONS},
        "overall": round(overall_c / overall_t, 4) if overall_t else None,
    }


# sigma sweep

def run_sigma_sweep(cfg_base: dict, best_prep: str):
    out = Path("runs/supervision/sigma_sweep/results.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(SEP)
    print("  TABLE 2: SUPERVISION BANDWIDTH SWEEP")
    print(f"  preprocess={best_prep}  2 epochs  L=2")
    print(SEP)

    for label, run_name, overrides in SIGMA_SWEEP:
        results = json.loads(out.read_text()) if out.exists() else {}
        if run_name in results:
            print(f"  [{run_name}] already done: ep2_hld={results[run_name]['ep2_hld']:.4f}  (skipping)")
            continue

        print(f"\n{SEP2}\n  {label}\n{SEP2}")
        run_dir = Path(f"runs/supervision/sigma_sweep/{run_name}")
        cfg = apply_overrides(cfg_base, {
            "encoder.preprocess":   best_prep,
            "encoder.epochs":       2,
            "encoder.har_interval": 2,
            **overrides,
        })
        from src.training import train_encoder
        train_encoder(cfg, run_dir)

        val, hld = _read_ep2_from_log(run_dir)
        _locked_save(out, run_name, {"label": label, "ep2_val": val, "ep2_hld": hld})
        print(f"  ep2 val={val:.4f}  hld={hld:.4f}")

    results = json.loads(out.read_text()) if out.exists() else {}
    if set(SWEEP_NAMES) == set(results.keys()):
        print(f"\n\n{SEP}")
        print("  TABLE 2: HAR accuracy per bandwidth hyperparameter  (ep2 holdout)")
        print(f"  {'Config':<24} {'Val':>8} {'Holdout':>8}")
        print(f"  {SEP2}")
        for label, run_name, _ in SIGMA_SWEEP:
            r = results.get(run_name, {})
            print(f"  {label:<24} {r.get('ep2_val', 0):>8.3f} {r.get('ep2_hld', 0):>8.3f}")
        out.write_text(json.dumps(results, indent=2))
        print(f"\n  Saved: {out}")
    else:
        missing = [r for r in SWEEP_NAMES if r not in results]
        print(f"\n  Done. Waiting for: {', '.join(missing)}")

    return results


# SupCon comparison

def run_supcon_compare(cfg_base: dict, best_prep: str):
    """Train SupCon (label) encoder, then run holdout + zero-shot for both
    body-state σv=1 (already in sigma sweep) and SupCon. Produces Table 3."""
    from src.training import train_encoder

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train supcon
    supcon_dir = Path("runs/supervision/label")
    if not (supcon_dir / "best_model.pt").exists():
        print(f"\n{SEP2}\n  SupCon (direct labels)\n{SEP2}")
        cfg = apply_overrides(cfg_base, {
            "encoder.preprocess": best_prep,
            "encoder.similarity": "label",
        })
        train_encoder(cfg, supcon_dir)
    else:
        print("\n  [label] already done, skipping training")

    # load main cache
    cfg      = load_config("config.yaml")
    cache_dir = find_cache(cfg)
    if cache_dir is None:
        raise RuntimeError("No main cache: run prepare.py first.")

    csi_mm   = np.load(cache_dir / "csi.npy", mmap_mode="r")
    csi_mean = np.load(cache_dir / "csi_mean.npy")
    csi_std  = np.load(cache_dir / "csi_std.npy")
    rec_idx  = np.load(cache_dir / "recording_idx.npy", mmap_mode="r")
    segments  = load_segments(cache_dir)
    split_data = load_split(cache_dir)
    all_labels = sorted({s["lbl"] for s in segments})
    label_map  = {int(c): i for i, c in enumerate(all_labels)}
    n_classes  = len(all_labels)
    msl        = cfg["har"]["max_seq_len"]

    train_seqs = _make_cache_seqs(segments, split_data, rec_idx, label_map, {"train"}, msl, step=max(1, msl // 2))
    val_seqs   = _make_cache_seqs(segments, split_data, rec_idx, label_map, {"val"},   msl)
    hld_seqs   = _make_cache_seqs(segments, split_data, rec_idx, label_map, {"holdout"}, msl)

    har_cfg = dict(cfg)
    har_cfg["har"] = dict(cfg["har"])
    har_cfg["har"]["epochs"] = 40
    har_cfg["har"]["patience"] = 7

    test_dir = Path(cfg_base["data"]["test_datasets_dir"])
    out     = Path("runs/supervision/supcon_compare.json")
    compare = json.loads(out.read_text()) if out.exists() else {}

    configs = [
        ("Body-State (σv=1)", Path("runs/supervision/sigma_sweep/pv_sv1")),
        ("Direct labels (SupCon)", supcon_dir),
    ]

    for name, run_dir in configs:
        if name in compare:
            print(f"\n  [{name}] already done, skipping eval")
            continue
        if not (run_dir / "best_model.pt").exists():
            print(f"\n  [skip] {name}: no checkpoint at {run_dir}")
            continue

        enc_cfg = json.loads((run_dir / "config.json").read_text())
        encoder = build_encoder(enc_cfg).to(device)
        encoder.load_state_dict(torch.load(run_dir / "best_model.pt",
                                            map_location=device, weights_only=True))
        encoder.eval()

        gn       = enc_cfg["encoder"].get("global_norm", True)
        eff_mean = csi_mean if gn else np.zeros_like(csi_mean)
        eff_std  = csi_std  if gn else np.ones_like(csi_std)

        print(f"\n{SEP2}\n  {name}: embedding + HAR probe\n{SEP2}")
        emb = embed_all(encoder, csi_mm, eff_mean, eff_std, enc_cfg, device)
        val_acc, hld_acc, clf = train_har(
            emb, train_seqs, val_seqs, hld_seqs,
            n_classes, har_cfg, device, return_clf=True,
        )
        print(f"  HAR val={val_acc:.3f}  holdout={hld_acc:.3f}")

        print(f"  Running zero-shot eval on {test_dir}...")
        zs = _run_zeroshot(encoder, clf, label_map, enc_cfg, eff_mean, eff_std, msl, device, test_dir)

        compare[name] = {
            "val_acc": round(val_acc, 4),
            "hld_acc": round(hld_acc, 4),
            "zeroshot": zs,
        }
        out.write_text(json.dumps(compare, indent=2))

    if len(compare) == len(configs):
        print(f"\n\n{SEP}")
        print("  TABLE 3: SupCon vs Body-State: holdout and zero-shot accuracy")
        print(SEP)
        print(f"  {'Supervision':<28} {'Holdout':>8} {'Zero-shot':>10}")
        print(f"  {SEP2}")
        for name, _ in configs:
            r  = compare.get(name, {})
            zs = r.get("zeroshot", {}).get("overall")
            print(f"  {name:<28} {r.get('hld_acc', 0):>8.3f}"
                  f" {zs:>10.3f}" if zs is not None else f"  {name:<28}  -")
        print(f"\n  Saved: {out}")

def main():
    p1 = Path("runs/normalization/results.json")
    best_prep = json.loads(p1.read_text()).get("best", "whiten") if p1.exists() else "whiten"
    print(f"  Using preprocessing: {best_prep}  (from runs/normalization/results.json)")

    cfg_base = load_config("config.yaml")

    if find_cache(cfg_base) is None:
        raise RuntimeError("No cache found: run  python prepare.py  first.")

    # Part 1: sigma sweep, Table 2
    run_sigma_sweep(cfg_base, best_prep)

    # Part 2: SupCon comparison, Table 3
    run_supcon_compare(cfg_base, best_prep)

    print("\n  Next: python scripts/exp_window_size.py")


if __name__ == "__main__":
    main()
