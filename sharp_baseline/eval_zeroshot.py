"""Zero-shot transfer of the trained SHARP baseline."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from src.cache import load_config
from src.data import ACTIVITY_LABELS

from sharp_baseline.config import SharpConfig
from sharp_baseline.adapter import session_windows, resolve_device
from sharp_baseline.model import SharpCsiNetwork

LOCATIONS = ["middle", "edge", "outside"]
# test-set activity name -> canonical name (matches scripts/run_zeroshot.py)
_NAME_FIX = {
    "raise_left_arm": "raising_left_arm",
    "raise_left_foot": "raising_left_foot",
    "stir_pot": "stir_a_pot",
}
_LABEL_INV = {v: k for k, v in ACTIVITY_LABELS.items()}


def parse_session(name: str) -> tuple[int, str]:
    parts = name.split("_")
    loc = parts[-1]
    act = "_".join(parts[2:-1])
    act = _NAME_FIX.get(act, act)
    return _LABEL_INV[act], loc


def latest_run() -> Path:
    runs = sorted(Path("runs").glob("*_sharp_baseline"))
    if not runs:
        raise RuntimeError("No runs/*_sharp_baseline found: train first.")
    return runs[-1]


@torch.no_grad()
def session_prediction(model, feats, streams, offset, scale, device, n_classes) -> int | None:
    """Summed-softmax over all windows/streams -> one label for the session."""
    if feats.shape[0] == 0:
        return None
    acc = np.zeros(n_classes)
    for s in range(0, feats.shape[0], 256):
        batch = np.asarray(feats[s:s + 256], dtype=np.float32)
        batch = (batch - offset) / scale
        probs = model(torch.from_numpy(batch).to(device)).softmax(1).cpu().numpy()
        acc += probs.sum(0)
    return int(acc.argmax())


def main(run=None, config: str = "config.yaml"):
    run_dir  = Path(run) if run else latest_run()
    cfg      = json.loads((run_dir / "config.json").read_text())
    main_cfg = load_config(config)
    test_dir = Path(main_cfg["data"]["test_datasets_dir"])
    scfg = SharpConfig(**cfg["sharp"])
    n_classes = cfg["n_classes"]
    label_map = {int(k): v for k, v in cfg["label_map"].items()}      # activity_int -> mapped
    offset = cfg["feature_standardization"]["offset"]
    scale = cfg["feature_standardization"]["scale"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_device = resolve_device(scfg)
    print(f"Run: {run_dir}\nDevice: {device}\n")

    model = SharpCsiNetwork(num_classes=n_classes).to(device)
    # materialise LazyLinear with a dummy forward before loading weights
    dummy = torch.zeros(1, 1, scfg.window_length, scfg.doppler_bins, device=device)
    model(dummy)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    model.eval()

    inv_map = {v: k for k, v in label_map.items()}
    sessions = sorted(test_dir.iterdir())
    print(f"Found {len(sessions)} test sessions\n")

    loc_cm = {loc: [0, 0] for loc in LOCATIONS}                       # [correct, total]
    per_class = {loc: [np.zeros(n_classes, int), np.zeros(n_classes, int)] for loc in LOCATIONS}

    for sdir in sessions:
        try:
            act_int, loc = parse_session(sdir.name)
        except KeyError:
            print(f"  {sdir.name}: unknown activity, skipped")
            continue
        if act_int not in label_map:
            print(f"  {sdir.name}: activity absent from training classes, skipped")
            continue
        mapped = label_map[act_int]
        feats, streams = session_windows(sdir, scfg, feat_device)
        pred = session_prediction(model, feats, streams, offset, scale, device, n_classes)
        if pred is None:
            print(f"  {sdir.name:<45} no windows (too short)")
            continue
        correct = int(pred == mapped)
        loc_cm[loc][0] += correct
        loc_cm[loc][1] += 1
        per_class[loc][1][mapped] += 1
        per_class[loc][0][mapped] += correct
        print(f"  {sdir.name:<45} pred={ACTIVITY_LABELS[inv_map[pred]]:<18} "
              f"{'OK' if correct else 'X'}")

    sep = "=" * 60
    print(f"\n{sep}\n  SHARP ZERO-SHOT (per-session) ACCURACY\n{sep}")
    print(f"  {'Location':<12}{'acc':>8}{'n':>6}")
    for loc in LOCATIONS:
        c, t = loc_cm[loc]
        print(f"  {loc:<12}{(c/t if t else 0):>8.3f}{t:>6}")
    tot_c = sum(loc_cm[loc][0] for loc in LOCATIONS)
    tot_t = sum(loc_cm[loc][1] for loc in LOCATIONS)
    print(f"  {'overall':<12}{(tot_c/tot_t if tot_t else 0):>8.3f}{tot_t:>6}")

    print("\n  Per-class accuracy")
    print(f"  {'activity':<22}{'middle':>8}{'edge':>8}{'outside':>8}")
    for mapped in range(n_classes):
        name = ACTIVITY_LABELS[inv_map[mapped]]
        row = []
        for loc in LOCATIONS:
            c, t = per_class[loc][0][mapped], per_class[loc][1][mapped]
            row.append(f"{c/t:.2f}" if t else "-")
        print(f"  {name:<22}{row[0]:>8}{row[1]:>8}{row[2]:>8}")

    out = Path("runs/sharp_zeroshot/results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    save = {"run": str(run_dir), "overall": {}, "per_class": {}}
    for loc in LOCATIONS:
        c, t = loc_cm[loc]
        save["overall"][loc] = {"acc": round(c / t, 4) if t else 0, "correct": c, "total": t}
    for mapped in range(n_classes):
        name = ACTIVITY_LABELS[inv_map[mapped]]
        save["per_class"][name] = {}
        for loc in LOCATIONS:
            c, t = int(per_class[loc][0][mapped]), int(per_class[loc][1][mapped])
            save["per_class"][name][loc] = round(c / t, 4) if t else 0
    out.write_text(json.dumps(save, indent=2))
    print(f"\n  Saved -> {out}")


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--run",    default=None, help="runs/<ts>_sharp_baseline dir")
    _p.add_argument("--config", default="config.yaml", help="main config for dataset paths")
    _a = _p.parse_args()
    main(run=_a.run, config=_a.config)

