"""
Step 3: Evaluate the trained encoder. Produces a results table.

Runs in order:
  1. Contrastive encoder HAR  (loads best_model.pt from --ckpt or latest run)
  2. Summary table + HAR curve + t-SNE comparison

Usage:
    python pipeline/evaluate.py
    python pipeline/evaluate.py --config config.yaml --ckpt runs/my_run/best_model.pt
"""
import argparse
import json
import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch

from src.cache import load_config, find_cache
from src.split import load_segments, load_split, make_seqs
from src.training import build_encoder, embed_all, train_har


SEED = 42


def _find_latest_ckpt(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None
    candidates = [d / "best_model.pt" for d in runs_dir.iterdir()
                  if d.is_dir() and (d / "best_model.pt").exists()]
    return max(candidates, key=lambda p: p.parent.name) if candidates else None


def _setup(cfg: dict, device):
    """Load cache, segments, and split; build sequence sets for all splits."""
    cache_dir = find_cache(cfg)
    if cache_dir is None:
        raise RuntimeError("No cache found: run pipeline/prepare.py first.")

    csi      = np.load(cache_dir / "csi.npy",           mmap_mode="r")
    csi_mean = np.load(cache_dir / "csi_mean.npy")
    csi_std  = np.load(cache_dir / "csi_std.npy")
    rec_idx  = np.load(cache_dir / "recording_idx.npy", mmap_mode="r")

    segments   = load_segments(cache_dir)
    split_data = load_split(cache_dir)
    all_labels = np.unique([s["lbl"] for s in segments])
    label_map  = {int(c): i for i, c in enumerate(all_labels)}
    n_classes  = len(all_labels)
    msl        = cfg["har"]["max_seq_len"]

    return dict(
        cache_dir=cache_dir, csi=csi, csi_mean=csi_mean, csi_std=csi_std,
        rec_idx=rec_idx, segments=segments, split_data=split_data,
        label_map=label_map, n_classes=n_classes,
        train_seqs=make_seqs(segments, split_data, rec_idx, {"train"},   label_map, msl, step=max(1, msl // 2)),
        val_seqs  =make_seqs(segments, split_data, rec_idx, {"val"},     label_map, msl),
        hld_seqs  =make_seqs(segments, split_data, rec_idx, {"holdout"}, label_map, msl),
    )


def eval_contrastive(cfg: dict, ckpt_path: Path, ctx: dict, device) -> tuple[float, float]:
    print(f"\nContrastive encoder ({ckpt_path})")
    encoder = build_encoder(cfg).to(device)
    encoder.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    emb = embed_all(encoder, ctx["csi"], ctx["csi_mean"], ctx["csi_std"], cfg, device)
    return train_har(emb, ctx["train_seqs"], ctx["val_seqs"], ctx["hld_seqs"], ctx["n_classes"], cfg, device)


def plot_har_curve(ckpt_path: Path, out_path: Path, results: dict):
    log = ckpt_path.parent / "activity_log.csv"
    if not log.exists():
        return
    epochs, vals, hlds = [], [], []
    with open(log) as f:
        next(f)
        for line in f:
            e, v, h = line.strip().split(",")
            epochs.append(int(e))
            vals.append(float(v))
            hlds.append(float(h))
    if not epochs:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, vals,  "o-", color="#4C72B0", label="val")
    ax.plot(epochs, hlds,  "o-", color="#C44E52", label="holdout")
    peak   = max(hlds)
    peak_e = epochs[hlds.index(peak)]
    ax.axvline(peak_e, color="#C44E52", ls="--", alpha=0.5, label=f"peak holdout epoch {peak_e} ({peak:.3f})")
    ax.set_xlabel("Contrastive epoch")
    ax.set_ylabel("HAR accuracy")
    ax.set_title("HAR accuracy vs contrastive training epoch")
    ax.set_ylim(0.1, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_tsne_comparison(ckpt_path: Path, out_path: Path):
    run_dir    = ckpt_path.parent
    tsne_first = run_dir / "tsne_001.png"
    tsnes      = sorted(run_dir.glob("tsne_*.png"))
    if len(tsnes) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Embedding structure: epoch 1 vs final epoch", fontsize=12, fontweight="bold")
    axes[0].imshow(mpimg.imread(tsne_first))
    axes[0].axis("off")
    axes[0].set_title("Epoch 1", fontsize=12)
    axes[1].imshow(mpimg.imread(tsnes[-1]))
    axes[1].axis("off")
    axes[1].set_title(tsnes[-1].stem.replace("tsne_", "Epoch "), fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ckpt",   default=None, help="Path to contrastive encoder best_model.pt")
    parser.add_argument("--out",    default=None, help="Output directory for results (default: ckpt parent)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(args.ckpt) if args.ckpt else _find_latest_ckpt(Path(cfg["output"]["runs_dir"]))
    if ckpt_path is None:
        raise RuntimeError("No checkpoint found: run pipeline/train.py first or pass --ckpt.")
    print(f"Encoder: {ckpt_path}")

    out_dir = Path(args.out) if args.out else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx     = _setup(cfg, device)
    results = {}

    val, hld = eval_contrastive(cfg, ckpt_path, ctx, device)
    results["contrastive"] = {"val": val, "holdout": hld}

    plot_har_curve(ckpt_path, out_dir / "har_curve.png", results)
    plot_tsne_comparison(ckpt_path, out_dir / "tsne_comparison.png")

    print(f"\n{'Method':<45} {'val':>6}  {'holdout':>8}")
    print("─" * 62)
    for name, r in results.items():
        val_s = f"{r['val']:.3f}"     if not np.isnan(r['val'])     else "  N/A"
        hld_s = f"{r['holdout']:.3f}" if not np.isnan(r['holdout']) else "     N/A"
        print(f"  {name:<43} {val_s:>6}  {hld_s:>8}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_dir}/results.json")


if __name__ == "__main__":
    main()
