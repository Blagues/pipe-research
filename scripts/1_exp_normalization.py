"""
6.1 Evaluates six within-window CSI preprocessing strategies.
==================================================

  none: global z-score only (train-set mean/std)
  diff: frame-to-frame temporal difference
  demean: subtract per-window temporal mean
  l1_frame: ℓ₁ per-frame gain normalisation (Portner et al. 2026)
  l1_whiten: ℓ₁ per-frame + temporal whitening
  whiten: demean + divide by per-window std
"""
import argparse
import csv
import json
from pathlib import Path

import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from src.cache import load_config, apply_overrides, find_cache
from src.training import train_encoder

MODES = ["none", "diff", "demean", "whiten", "l1_frame", "l1_whiten"]

# Hardcoded parameters as used by this experiment in the paper
EPOCHS = 2
FRAME = 100

SEP  = "═" * 56
SEP2 = "─" * 56


def read_activity_log(run_dir: Path) -> dict[int, dict]:
    log = run_dir / "activity_log.csv"
    if not log.exists():
        return {}
    return {int(r["epoch"]): r for r in csv.DictReader(log.open())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, default=None,
                        help="Run a single mode (omit to run all sequentially)")
    args = parser.parse_args()
    modes = [args.mode] if args.mode else MODES

    cfg_base     = load_config("config.yaml")
    results_path = Path("runs/normalization/results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    if find_cache(cfg_base) is None:
        raise RuntimeError("No cache found: run  python prepare.py  first.")

    print(SEP)
    print("  NORMALIZATION SWEEP: Table 1")
    print(SEP)
    print(f"  Modes : {', '.join(modes)}")
    print("  Fixed : L=2, d=256, pose-only loss σ=150, 2 epochs")
    print(SEP)

    for mode in modes:
        run_dir = Path(f"runs/normalization/{mode}")
        if mode in results:
            print(f"\n  [{mode}] already done: ep2_hld={results[mode]['ep2_hld']:.3f}  (skipping)")
            continue

        print(f"\n{SEP2}")
        print(f"  Running: preprocess = {mode}")
        print(SEP2)

        # Override base config with our settings
        cfg = apply_overrides(cfg_base, {"encoder.preprocess": mode, "encoder.epochs": EPOCHS, "data.raw_window": FRAME})

        train_encoder(cfg, run_dir)

        by_epoch = read_activity_log(run_dir)
        ep2 = by_epoch.get(2, {})
        results[mode] = {
            "ep2_val": float(ep2.get("val", 0)),
            "ep2_hld": float(ep2.get("holdout", 0)),
        }
        results_path.write_text(json.dumps(results, indent=2))

    completed = [m for m in MODES if m in results]

    # all modes in results -> print results
    if set(MODES) == set(completed):
        print(f"\n\n{SEP}")
        print("  TABLE 1: Effect of CSI preprocessing on HAR accuracy")
        print("  (pose-only loss, L=2, 2 epochs)")
        print(SEP)
        print(f"  {'Preprocessing':<12} {'Val acc':>8} {'Holdout acc':>12}")
        print(f"  {SEP2}")
        best_mode, best_hld = None, -1.0
        for mode in MODES:
            r   = results[mode]
            tag = " ← best" if r["ep2_hld"] == max(results[m]["ep2_hld"] for m in MODES) else ""
            print(f"  {mode:<12} {r['ep2_val']:>8.3f}  {r['ep2_hld']:>11.3f}{tag}")
            if r["ep2_hld"] > best_hld:
                best_hld  = r["ep2_hld"]
                best_mode = mode
        results["best"] = best_mode
        results_path.write_text(json.dumps(results, indent=2))
        print(f"\n  Best preprocessing: {best_mode}  (holdout={best_hld:.3f})")
        print(f"  Saved: {results_path}")
    else:
        missing = [m for m in MODES if m not in results]
        print(f"\n  Done. Waiting for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
