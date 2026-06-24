"""Train the SHARP amplitude-Doppler CNN on the contrastive model's split.

Trains on the train split (per-stream windows), early-stops on val stream-window
accuracy, and evaluates on the holdout split with per-segment late fusion.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Determinism: set before torch initialises CUDA (matches pipeline/train.py)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from src.cache import load_config, find_cache

from sharp_baseline.config import SharpConfig
from sharp_baseline.adapter import (
    build_dev_features, load_dev_features, activity_name, SPLIT_CODES,
)
from sharp_baseline.model import SharpCsiNetwork, initialize_lazy_modules

class WindowDataset(Dataset):
    def __init__(self, features, indices, labels, offset: float, scale: float):
        self.features = features
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels
        self.offset = float(offset)
        self.scale = float(scale) or 1.0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = np.asarray(self.features[idx], dtype=np.float32)
        x = (x - self.offset) / self.scale
        return torch.from_numpy(x), torch.tensor(int(self.labels[idx]), dtype=torch.long)


def compute_stats(features, indices, scfg: SharpConfig) -> tuple[float, float]:
    """Mean/std over a sample of train+val windows."""
    rng = np.random.default_rng(scfg.seed)
    idx = np.asarray(indices, dtype=np.int64)
    if len(idx) > scfg.stats_max_windows:
        idx = rng.choice(idx, size=scfg.stats_max_windows, replace=False)
    total = total_sq = 0.0
    count = 0
    for s in range(0, len(idx), 128):
        batch = np.asarray(features[np.sort(idx[s:s + 128])], dtype=np.float64)
        total += float(batch.sum())
        total_sq += float(np.square(batch).sum())
        count += batch.size
    mean = total / count
    var = max(0.0, total_sq / count - mean * mean)
    return float(mean), float(np.sqrt(var)) or 1.0


def set_determinism(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def stream_accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        preds = model(x.to(device)).argmax(1).cpu()
        correct += (preds == y).sum().item()
        total += len(y)
    return correct / total if total else 0.0


@torch.no_grad()
def segment_eval(model, features, indices, labels, segids, offset, scale, device,
                 n_classes: int) -> dict:
    """Per-segment late fusion: sum softmax over each segment's windows/streams."""
    model.eval()
    ds = WindowDataset(features, indices, labels, offset, scale)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    seg_logits: dict[int, np.ndarray] = {}
    seg_label: dict[int, int] = {}
    pos = 0
    idx_arr = np.asarray(indices, dtype=np.int64)
    for x, _ in loader:
        probs = model(x.to(device)).softmax(1).cpu().numpy()
        for p in probs:
            gidx = int(idx_arr[pos])
            sid = int(segids[gidx])
            seg_logits[sid] = seg_logits.get(sid, np.zeros(n_classes)) + p
            seg_label[sid] = int(labels[gidx])
            pos += 1
    y_true = np.array([seg_label[s] for s in seg_logits])
    y_pred = np.array([int(seg_logits[s].argmax()) for s in seg_logits])
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    pc_c = np.zeros(n_classes, int)
    pc_t = np.zeros(n_classes, int)
    for t, p in zip(y_true, y_pred):
        pc_t[t] += 1
        pc_c[t] += int(p == t)
    per_class = {activity_name_for(i, labels): (int(pc_c[i]), int(pc_t[i])) for i in range(n_classes)}
    return {"accuracy": acc, "n_segments": int(len(y_true)), "per_class": per_class}


# label-int recovery for naming (inverse of contiguous label_map)
_INV_MAP: dict[int, int] = {}


def activity_name_for(mapped_idx: int, _labels) -> str:
    return activity_name(_INV_MAP.get(mapped_idx, mapped_idx))


def main(config: str = "config.yaml", force_features: bool = False):
    scfg = SharpConfig()
    set_determinism(scfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    cfg = load_config(config)
    cache_dir = find_cache(cfg)
    if cache_dir is None:
        raise RuntimeError("No cache found: run pipeline/prepare.py first.")

    feature_dir = build_dev_features(cfg, scfg, cache_dir, force=force_features)
    features, labels, splits, segids, streams, meta = load_dev_features(feature_dir)
    n_classes = meta["n_classes"]
    global _INV_MAP
    _INV_MAP = {v: int(k) for k, v in meta["label_map"].items()}

    train_idx = np.where(splits == SPLIT_CODES["train"])[0]
    val_idx = np.where(splits == SPLIT_CODES["val"])[0]
    hld_idx = np.where(splits == SPLIT_CODES["holdout"])[0]
    print(f"Windows: train {len(train_idx)}  val {len(val_idx)}  holdout {len(hld_idx)}")

    offset, scale = compute_stats(features, np.concatenate([train_idx, val_idx]), scfg)
    print(f"Feature standardisation (train+val): offset={offset:.4f} scale={scale:.4f}\n")

    g = torch.Generator().manual_seed(scfg.seed)
    train_loader = DataLoader(
        WindowDataset(features, train_idx, labels, offset, scale),
        batch_size=scfg.batch_size, shuffle=True, generator=g,
        num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(
        WindowDataset(features, val_idx, labels, offset, scale),
        batch_size=scfg.batch_size, shuffle=False)

    model = SharpCsiNetwork(num_classes=n_classes).to(device)
    initialize_lazy_modules(model, train_loader, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=scfg.learning_rate,
                                 weight_decay=scfg.weight_decay)

    run_dir = Path(cfg["output"]["runs_dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}_sharp_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val = -1.0
    best_epoch = 0
    no_improve = 0
    history = []
    t0 = time.monotonic()
    for epoch in range(1, scfg.epochs + 1):
        model.train()
        tl = tc = tn = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tl += loss.item() * y.size(0)
            tc += (logits.argmax(1) == y).sum().item()
            tn += y.size(0)
        val_acc = stream_accuracy(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": tl / tn,
                        "train_acc": tc / tn, "val_acc": val_acc})
        print(f"epoch {epoch:03d} | train loss {tl/tn:.4f} acc {tc/tn:.4f} | "
              f"val acc {val_acc:.4f} | {time.monotonic()-t0:.0f}s", flush=True)
        if val_acc > best_val:
            best_val, best_epoch, no_improve = val_acc, epoch, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= scfg.patience:
                print(f"Early stop after {scfg.patience} epochs without val improvement.")
                break

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))

    # per-segment late-fused evaluation on every split
    results = {}
    for name, idx in (("train", train_idx), ("val", val_idx), ("holdout", hld_idx)):
        results[name] = segment_eval(model, features, idx, labels, segids,
                                     offset, scale, device, n_classes)
        print(f"  {name:8s} per-segment acc = {results[name]['accuracy']:.4f} "
              f"({results[name]['n_segments']} segments)")

    config_out = {
        "sharp": meta["sharp"], "label_map": meta["label_map"], "n_classes": n_classes,
        "feature_standardization": {"offset": offset, "scale": scale},
        "feature_dir": str(feature_dir), "best_epoch": best_epoch, "best_val_stream_acc": best_val,
    }
    (run_dir / "config.json").write_text(json.dumps(config_out, indent=2))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps({
        "best_epoch": best_epoch, "best_val_stream_acc": best_val,
        "per_segment": {k: {"accuracy": v["accuracy"], "n_segments": v["n_segments"],
                            "per_class": v["per_class"]} for k, v in results.items()},
        "window_split_counts": meta["window_split_counts"],
        "split_segment_counts": meta["split_segment_counts"],
        "dropped_short_segments": meta["dropped_short_segments"],
    }, indent=2))
    print(f"\nHoldout per-segment accuracy: {results['holdout']['accuracy']:.4f}")
    print(f"Saved -> {run_dir}")


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--config", default="config.yaml")
    _p.add_argument("--force-features", action="store_true")
    _a = _p.parse_args()
    main(config=_a.config, force_features=_a.force_features)
