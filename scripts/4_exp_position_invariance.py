"""
Section 6.4: Position Invariance
==================================================
Requires 3_exp_window_size.py to have been run! (uses runs/window_sweep/win50/).

Table 6: Per-activity difference between in-distribution holdout accuracy

Table 7: Cross-location hip-position regression probe.

Reads:  config.yaml,  runs/window_sweep/win50/  (encoder from 3_exp_window_size.py)
"""
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from src.data import ACTIVITY_LABELS, RECEIVERS, load_synced_csi, load_markers, fix_marker_flicker, load_labels
from src.cache import load_config, apply_overrides, find_cache, recompute_normalization, build_cache
from src.split import (build_segment_positions, save_segment_positions,
                        build_split, load_segments, load_split, assign_split, segments_for)
from src.training import build_encoder, apply_preprocess, embed_all, train_har, SequenceDataset

from sklearn.decomposition import IncrementalPCA

SEED      = 42
WIN       = 50
LOCATIONS = ["middle", "edge", "outside"]
WIN50_DIR = Path("runs/window_sweep/win50")

_NAME_FIX  = {"raise_left_arm": "raising_left_arm",
               "raise_left_foot": "raising_left_foot",
               "stir_pot": "stir_a_pot"}
_LABEL_INV = {v: k for k, v in ACTIVITY_LABELS.items()}

SEP  = "═" * 78
SEP2 = "─" * 78

# Hip regression probe config
STRIDE      = 25         # window hop in frames for the regression probe (coarser than 10)
PCA_DIM     = 4096
PCA_BATCH   = 4096
MLP_WIDTH   = 1024
MLP_DEPTH   = 8
MLP_DROPOUT = 0.1
CSI_MMAP    = Path("runs/probe_hip/full_csi.f16.mmap")
CSI_EPOCHS  = 80
CSI_BS      = 1024

SPLIT_ID = {"train": 0, "val": 1, "holdout": 2}


# Helpers shared by both tables

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
def _eval_per_class(clf, emb, seqs, msl, device, n_classes):
    clf.eval()
    loader = DataLoader(SequenceDataset(emb, seqs, msl), batch_size=32)
    pc_c = np.zeros(n_classes, int)
    pc_t = np.zeros(n_classes, int)
    for x, mask, y in loader:
        preds = clf(x.to(device), mask.to(device)).argmax(1).cpu().numpy()
        for gt, pr in zip(y.cpu().numpy(), preds):
            pc_t[gt] += 1
            pc_c[gt] += int(pr == gt)
    return pc_c, pc_t


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


def _setup_win50_cache(cfg_base):
    """Build or reuse win=50 cache; returns (cache_dir, is_new)."""
    batch = max(32, 128 * 100 // WIN)
    cfg = apply_overrides(cfg_base, {
        "data.raw_window": WIN, "encoder.similarity": "pose_vel",
        "encoder.sigma": 150, "encoder.sigma_v": 1.0, "encoder.batch_size": batch,
    })
    dc = cfg["data"]
    datasets_dir = Path(dc["datasets_dir"])
    recs = sorted(d for d in os.listdir(datasets_dir) if (datasets_dir / d).is_dir())

    existing = find_cache(cfg)
    if existing is not None:
        return existing, False, recs, cfg

    cache_root = Path(dc.get("cache_dir", "../cache"))
    free_gb  = shutil.disk_usage(cache_root if cache_root.exists() else cache_root.parent).free / 1e9
    need_gb  = max(40, int(WIN / 100 * 270))
    if free_gb < need_gb:
        raise RuntimeError(f"Not enough disk for win=50 cache: {free_gb:.0f} < {need_gb} GB")
    cache_dir = build_cache(cfg, recs)

    seg_path = cache_dir / "segment_positions.npz"
    if not seg_path.exists():
        segments = build_segment_positions(cache_dir, recs, cfg)
        save_segment_positions(cache_dir, segments)

    split_path = cache_dir / "split.json"
    if not split_path.exists():
        segments = load_segments(cache_dir)
        split_data = build_split(segments, cfg)
        split_path.write_text(json.dumps({
            "cluster_centers": np.array(split_data["cluster_centers"]).tolist(),
            "cluster_split": split_data["cluster_split"], "params": split_data["params"],
        }, indent=2))

    with open(cache_dir / "meta.json") as f:
        meta = json.load(f)
    if meta.get("normalization") != "train_val_only":
        segments   = load_segments(cache_dir)
        split_data = load_split(cache_dir)
        recompute_normalization(cache_dir, segments, split_data)

    return cache_dir, True, recs, cfg


# Table 6: per-activity holdout vs zone delta

def run_per_activity_delta(device):
    out = Path("runs/position_invariance/per_activity_delta.json")
    if out.exists():
        print(f"\n  Table 6 already computed, skipping  ({out})")
        return

    print(f"\n{SEP}")
    print("  TABLE 6: Per-activity holdout vs zero-shot delta  (WIN=50)")
    print(SEP)

    if not (WIN50_DIR / "best_model.pt").exists():
        raise RuntimeError(f"No encoder at {WIN50_DIR}: run 3_exp_window_size.py first.")

    cfg_base   = load_config("config.yaml")
    test_dir   = Path(cfg_base["data"]["test_datasets_dir"])
    cache_dir, cache_is_new, recs, cfg = _setup_win50_cache(cfg_base)
    segments   = load_segments(cache_dir)
    split_data = load_split(cache_dir)
    rec_idx    = np.load(cache_dir / "recording_idx.npy")
    csi_mean   = np.load(cache_dir / "csi_mean.npy")
    csi_std    = np.load(cache_dir / "csi_std.npy")
    csi_mm     = np.load(cache_dir / "csi.npy", mmap_mode="r")

    all_labels = sorted({s["lbl"] for s in segments})
    label_map  = {int(c): i for i, c in enumerate(all_labels)}
    n_classes  = len(all_labels)
    msl        = cfg["har"]["max_seq_len"]

    train_seqs = _make_seqs(segments, split_data, {"train"},   rec_idx, label_map, msl, step=max(1, msl // 2))
    val_seqs   = _make_seqs(segments, split_data, {"val"},     rec_idx, label_map, msl)
    hld_seqs   = _make_seqs(segments, split_data, {"holdout"}, rec_idx, label_map, msl)

    enc_cfg = json.loads((WIN50_DIR / "config.json").read_text())
    encoder = build_encoder(enc_cfg).to(device)
    encoder.load_state_dict(torch.load(WIN50_DIR / "best_model.pt",
                                        map_location=device, weights_only=True))
    encoder.eval()

    gn       = enc_cfg["encoder"].get("global_norm", True)
    eff_mean = csi_mean if gn else np.zeros_like(csi_mean)
    eff_std  = csi_std  if gn else np.ones_like(csi_std)

    emb_batch = max(256, min(4096, int(4096 * (100 / WIN) ** 2)))
    print(f"  Embedding {len(csi_mm):,} windows (batch={emb_batch})...")
    emb = embed_all(encoder, csi_mm, eff_mean, eff_std, enc_cfg, device, batch=emb_batch)

    har_cfg = dict(enc_cfg)
    har_cfg["har"] = dict(enc_cfg["har"])
    har_cfg["har"]["epochs"] = 40
    har_cfg["har"]["patience"] = 7
    print("  Training HAR probe (40 epochs)...")
    val_acc, hld_acc, clf = train_har(
        emb, train_seqs, val_seqs, hld_seqs, n_classes, har_cfg, device, return_clf=True,
    )
    print(f"  HAR val={val_acc:.3f}  holdout={hld_acc:.3f}")

    # Per-activity holdout accuracy
    hc, ht = _eval_per_class(clf, emb, hld_seqs, msl, device, n_classes)

    # Per-activity zero-shot accuracy by zone
    grid = {i: {loc: [0, 0] for loc in LOCATIONS} for i in range(n_classes)}
    for sess in sorted(test_dir.iterdir()):
        try:
            session_label, loc = _parse_session(sess.name)
        except (KeyError, IndexError):
            print(f"  SKIP (parse): {sess.name}")
            continue
        mapped   = label_map[session_label]
        windows  = _load_test_windows(sess, enc_cfg)
        if len(windows) == 0:
            continue
        test_emb = embed_all(encoder, windows, eff_mean, eff_std, enc_cfg, device)
        seqs     = [(np.arange(len(windows))[s:s + msl], mapped)
                    for s in range(0, len(windows), msl)]
        pc_c, pc_t = _eval_per_class(clf, test_emb, seqs, msl, device, n_classes)
        grid[mapped][loc][0] += int(pc_c[mapped])
        grid[mapped][loc][1] += int(pc_t[mapped])

    # Print Table 6
    names = [ACTIVITY_LABELS[i] for i in range(n_classes)]
    print(f"\n{SEP}")
    print("  TABLE 6: Holdout acc, zone acc, and DIFFERENCE  (zone − holdout, win=50)")
    print(SEP)
    print(f"  {'Activity':<18}{'Hold':>7}{'Mid':>7}{'Edge':>7}{'Out':>7}   |  {'ΔMid':>7}{'ΔEdge':>7}{'ΔOut':>7}")
    print("  " + "─" * 74)

    out_rows = {}
    for i, name in enumerate(names):
        h    = hc[i] / ht[i] if ht[i] else None
        accs = {loc: (grid[i][loc][0] / grid[i][loc][1] if grid[i][loc][1] else None)
                for loc in LOCATIONS}
        deltas = {loc: (accs[loc] - h if (accs[loc] is not None and h is not None) else None)
                  for loc in LOCATIONS}
        def f(v): return f"{v:+.3f}" if v is not None else "   -  "
        def g(v): return f"{v:.3f}"  if v is not None else "  -  "
        print(f"  {name:<18}{g(h):>7}{g(accs['middle']):>7}{g(accs['edge']):>7}"
              f"{g(accs['outside']):>7}   |  {f(deltas['middle']):>7}{f(deltas['edge']):>7}"
              f"{f(deltas['outside']):>7}")
        out_rows[name] = {
            "holdout": round(h, 4) if h is not None else None,
            "acc":     {loc: round(accs[loc], 4) if accs[loc] is not None else None for loc in LOCATIONS},
            "delta":   {loc: round(deltas[loc], 4) if deltas[loc] is not None else None for loc in LOCATIONS},
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"win": WIN, "val_acc": round(val_acc, 4),
                               "hld_acc": round(hld_acc, 4), "per_activity": out_rows}, indent=2))
    print(f"\n  Saved: {out}")

    if cache_is_new:
        shutil.rmtree(cache_dir)
        print("  Win=50 cache removed.")


# Table 7: hip-position regression probe

class _MLP(nn.Module):
    def __init__(self, din, width=MLP_WIDTH, depth=MLP_DEPTH, dout=2, p=MLP_DROPOUT):
        super().__init__()
        self.inp = nn.Linear(din, width)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Dropout(p))
            for _ in range(depth)
        ])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dout))

    def forward(self, x):
        x = self.inp(x)
        for blk in self.blocks:
            x = x + blk(x)
        return self.head(x)


def _predict(model, X, device, bs=2048):
    model.eval()
    out = []
    with torch.no_grad():
        for j in range(0, len(X), bs):
            out.append(model(torch.from_numpy(np.ascontiguousarray(X[j:j + bs])).to(device)).cpu().numpy())
    return np.concatenate(out)


def _fit_eval(Xtr, ytr, Xte, yte, device, scaler=True, epochs=200, bs=512, patience=12):
    Xtr = np.asarray(Xtr, np.float32)
    Xte = np.asarray(Xte, np.float32)
    if scaler:
        mu = Xtr.mean(0, keepdims=True)
        sd = Xtr.std(0, keepdims=True) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
    ym = ytr.mean(0)
    ysd = ytr.std(0) + 1e-6
    Ytr = ((ytr - ym) / ysd).astype(np.float32)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(Xtr))
    nval = max(1, int(0.1 * len(idx)))
    vi, ti = idx[:nval], idx[nval:]
    torch.manual_seed(SEED)
    model = _MLP(Xtr.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    Xv = torch.from_numpy(np.ascontiguousarray(Xtr[vi])).to(device)
    Yv = torch.from_numpy(Ytr[vi]).to(device)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = ti[rng.permutation(len(ti))]
        for j in range(0, len(perm), bs):
            b  = perm[j:j + bs]
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[b])).to(device)
            yb = torch.from_numpy(Ytr[b]).to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = lossf(model(Xv), Yv).item()
        if vloss < best - 1e-5:
            best = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    pred = _predict(model, Xte, device) * ysd + ym
    err  = float(np.median(np.linalg.norm(pred - yte, axis=1)))
    ss_res = ((pred - yte) ** 2).sum()
    ss_tot = ((yte - ytr.mean(0)) ** 2).sum()
    return err, float(1 - ss_res / max(ss_tot, 1e-9))


def _fit_eval_memmap(mm, idx_fit, yfit, idx_test, ytest, device,
                     epochs=CSI_EPOCHS, bs=CSI_BS, patience=8):
    def gather(rows):
        o = np.argsort(rows)
        x = np.asarray(mm[rows[o]], np.float32)
        return x, o
    ym = yfit.mean(0)
    ysd = yfit.std(0) + 1e-6
    Yf = ((yfit - ym) / ysd).astype(np.float32)
    rng   = np.random.RandomState(SEED)
    perm  = rng.permutation(len(idx_fit))
    nval = max(1, int(0.1 * len(perm)))
    vloc, tloc = perm[:nval], perm[nval:]
    Xv   = np.asarray(mm[np.sort(idx_fit[vloc])], np.float32)
    vord = np.argsort(idx_fit[vloc])
    Yv   = torch.from_numpy(Yf[vloc][vord]).to(device)

    def val_loss(model, lossf):
        model.eval()
        tot = n = 0
        with torch.no_grad():
            for j in range(0, len(Xv), bs):
                xb = torch.from_numpy(np.ascontiguousarray(Xv[j:j + bs])).to(device)
                tot += lossf(model(xb), Yv[j:j + bs]).item() * len(xb)
                n += len(xb)
        return tot / n

    torch.manual_seed(SEED)
    model = _MLP(mm.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        ep_perm = tloc[rng.permutation(len(tloc))]
        for j in range(0, len(ep_perm), bs):
            loc = ep_perm[j:j + bs]
            xb_np, o = gather(idx_fit[loc])
            xb = torch.from_numpy(xb_np).to(device)
            yb = torch.from_numpy(Yf[loc][o]).to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        vloss = val_loss(model, lossf)
        if vloss < best - 1e-5:
            best = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for j in range(0, len(idx_test), bs):
            xb = torch.from_numpy(np.asarray(mm[idx_test[j:j + bs]], np.float32)).to(device)
            preds.append(model(xb).cpu().numpy())
    pred = np.concatenate(preds) * ysd + ym
    err  = float(np.median(np.linalg.norm(pred - ytest, axis=1)))
    ss_res = ((pred - ytest) ** 2).sum()
    ss_tot = ((ytest - yfit.mean(0)) ** 2).sum()
    return err, float(1 - ss_res / max(ss_tot, 1e-9))


def _reconstruct_split(cfg):
    dd   = Path(cfg["data"]["datasets_dir"])
    recs = sorted(d for d in os.listdir(dd) if (dd / d).is_dir())
    segments, seg_meta = [], []
    for ri, rec in enumerate(recs):
        try:
            m_ts, pos, names = load_markers(dd / rec)
            pos = fix_marker_flicker(pos)
            lab = load_labels(dd / rec)
            lh, rh = names.index("LEFT_HIP_BACK"), names.index("RIGHT_HIP_BACK")
        except Exception:
            continue
        m_ts = np.asarray(m_ts, np.float64)
        if lab.size:
            scale = np.median(np.abs(lab[:, 0])) / max(np.median(np.abs(m_ts)), 1)
            if scale > 100:
                m_ts *= 1000.0
            elif scale < 0.01:
                m_ts /= 1000.0
        for start, stop, lbl in lab:
            if lbl == -1:
                continue
            sel = (m_ts >= start) & (m_ts < stop)
            if sel.sum() < 5:
                continue
            hip = (pos[sel][:, lh][:, [0, 2]] + pos[sel][:, rh][:, [0, 2]]) / 2
            bad = (np.abs(hip[:, 0]) < 10) & (np.abs(hip[:, 1]) < 10)
            hip = hip[~bad]
            if len(hip) == 0:
                continue
            segments.append(dict(ri=ri, lbl=int(lbl), mean_xz=hip.mean(0).astype(np.float32)))
            seg_meta.append((rec, int(start), int(stop)))
    split = build_split(segments, cfg)
    split["cluster_centers"] = np.array(split["cluster_centers"], np.float32)
    keep = {}
    for seg, (rec, start, stop) in zip(segments, seg_meta):
        sp = assign_split(seg["mean_xz"], split)
        keep.setdefault(rec, []).append((start, stop, sp))
    return keep


def _build_windows(dd, rec, intervals, rw):
    try:
        ts, csi_amp = load_synced_csi(dd / rec)
        stacked = np.stack([csi_amp[rx] for rx in RECEIVERS], axis=0)
        _, pos, names = load_markers(dd / rec)
        pos = fix_marker_flicker(pos)
        lh, rh = names.index("LEFT_HIP_BACK"), names.index("RIGHT_HIP_BACK")
    except Exception as e:
        print(f"  {rec}: skip ({e})")
        return None
    N = stacked.shape[1]
    idx_m = (np.arange(N) * len(pos) / N).astype(int).clip(0, len(pos) - 1)
    hip = (pos[idx_m][:, lh][:, [0, 2]] + pos[idx_m][:, rh][:, [0, 2]]) / 2
    split_of = np.full(N, -1, np.int8)
    for (a, b, sp) in intervals:
        split_of[(ts >= a) & (ts < b)] = SPLIT_ID[sp]
    wins, targs, sids = [], [], []
    for s in range(0, N - rw + 1, STRIDE):
        seg   = split_of[s:s + rw]
        valid = seg >= 0
        if valid.mean() <= 0.5:
            continue
        sid = int(np.bincount(seg[valid].astype(int), minlength=3).argmax())
        h   = hip[s:s + rw]
        good = ~((np.abs(h[:, 0]) < 10) & (np.abs(h[:, 1]) < 10))
        if good.sum() < rw // 2:
            continue
        wins.append(stacked[:, s:s + rw])
        targs.append(h[good].mean(0))
        sids.append(sid)
    if not wins:
        return None
    return (np.stack(wins).astype(np.float16), np.array(targs, np.float32), np.array(sids, np.int8))


def run_hip_regression(device):
    out = Path("runs/probe_hip/results.json")
    if out.exists():
        print(f"\n  Table 7 already computed, skipping  ({out})")
        data = json.loads(out.read_text())
        print(f"  Naive={data['naive_err_mm']:.0f} mm  "
              f"Embedding={data['embedding']['err_mm']:.0f} mm  "
              f"PCA-CSI={data['whitened_csi_pca']['err_mm']:.0f} mm")
        return

    print(f"\n{SEP}")
    print("  TABLE 7: Cross-location hip-position regression probe  (WIN=50)")
    print(SEP)

    if not (WIN50_DIR / "best_model.pt").exists():
        raise RuntimeError(f"No encoder at {WIN50_DIR}: run 3_exp_window_size.py first.")

    torch.manual_seed(SEED)
    cfg     = load_config("config.yaml")
    enc_cfg = json.loads((WIN50_DIR / "config.json").read_text())
    rw      = enc_cfg["data"]["raw_window"]
    mode    = enc_cfg["encoder"]["preprocess"]
    assert enc_cfg["encoder"].get("global_norm", True) is False

    print("  Reconstructing split from MoCap (all intervals)...")
    keep = _reconstruct_split(cfg)
    recs = sorted(keep.items())
    print(f"  {len(recs)} recordings")

    encoder = build_encoder(enc_cfg).to(device)
    encoder.load_state_dict(torch.load(WIN50_DIR / "best_model.pt",
                                        map_location=device, weights_only=True))
    encoder.eval()
    eff_mean = np.zeros((3, rw, 4, 114), np.float32)
    eff_std  = np.ones((3, rw, 4, 114), np.float32)
    dd = Path(cfg["data"]["datasets_dir"])

    def whiten_flat(w):
        return apply_preprocess(torch.from_numpy(w.astype(np.float32)),
                                mode, time_axis=2).reshape(len(w), -1).numpy()

    # Pass 1: embed + IncrementalPCA fit
    print("\n  Pass 1/2: embedding + IncrementalPCA fit...")
    ipca = IncrementalPCA(n_components=PCA_DIM)
    Emb, Y, S = [], [], []
    buf, buf_n = [], 0
    used = []
    for rec, intervals in recs:
        res = _build_windows(dd, rec, intervals, rw)
        if res is None:
            continue
        w, targs, sids = res
        Emb.append(embed_all(encoder, w, eff_mean, eff_std, enc_cfg, device))
        Y.append(targs)
        S.append(sids)
        used.append((rec, intervals))
        fitmask = sids != SPLIT_ID["holdout"]
        if fitmask.any():
            buf.append(whiten_flat(w[fitmask]))
            buf_n += int(fitmask.sum())
            if buf_n >= PCA_BATCH:
                ipca.partial_fit(np.concatenate(buf))
                buf, buf_n = [], 0
        c = {0: "tr", 1: "vl", 2: "ho"}
        cnt = {c[k]: int((sids == k).sum()) for k in (0, 1, 2)}
        print(f"  {rec:<8} win={len(w):>5}  {cnt}")
    if buf_n >= PCA_DIM:
        ipca.partial_fit(np.concatenate(buf))

    Emb = np.concatenate(Emb)
    Y = np.concatenate(Y)
    S = np.concatenate(S)
    fit = S != SPLIT_ID["holdout"]
    test = S == SPLIT_ID["holdout"]
    print(f"\n  Total: {len(Y):,} windows   train+val={fit.sum():,}  holdout={test.sum():,}")
    print(f"  Holdout hip-XZ spread (std): {Y[test].std(0).round(0)} mm")

    # Pass 2: PCA transform + stream full CSI to disk memmap
    D = 3 * rw * 4 * 114
    print(f"\n  Pass 2/2: PCA transform + writing full CSI to {CSI_MMAP} "
          f"({len(Y)}×{D} f16 = {len(Y)*D*2/1e9:.0f} GB)...")
    CSI_MMAP.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(CSI_MMAP, dtype=np.float16, mode="w+", shape=(len(Y), D))
    Cp = []
    off = 0
    for rec, intervals in used:
        res = _build_windows(dd, rec, intervals, rw)
        if res is None:
            continue
        w, _, _ = res
        Cwt = whiten_flat(w)
        Cp.append(ipca.transform(Cwt).astype(np.float32))
        mm[off:off + len(w)] = Cwt.astype(np.float16)
        off += len(w)
    mm.flush()
    Cp = np.concatenate(Cp)
    idx_fit = np.where(fit)[0]
    idx_test = np.where(test)[0]

    # Fit and evaluate three representations
    naive = float(np.median(np.linalg.norm(Y[fit].mean(0) - Y[test], axis=1)))
    print("\n  Fitting MLPs (embedding, PCA-CSI, full-CSI)...")
    e_err, e_r2 = _fit_eval(Emb[fit], Y[fit], Emb[test], Y[test], device)
    print(f"  embedding done  (err={e_err:.0f} mm)")
    p_err, p_r2 = _fit_eval(Cp[fit], Y[fit], Cp[test], Y[test], device)
    print(f"  pca done        (err={p_err:.0f} mm)")
    f_err, f_r2 = _fit_eval_memmap(mm, idx_fit, Y[fit], idx_test, Y[test], device)
    print(f"  full-csi done   (err={f_err:.0f} mm)")

    print(f"\n{'='*60}")
    print("  TABLE 7: Cross-location hip-position regression")
    print(f"{'='*60}")
    print(f"  {'Representation':<30} {'Median err':>10}  {'R²':>6}")
    print(f"  {'─'*50}")
    print(f"  {'Naive (mean train position)':<30} {naive:>8.0f} mm   {0.0:>+.2f}")
    print(f"  {'Whitened CSI (PCA-4096)':<30} {p_err:>8.0f} mm   {p_r2:>+.2f}")
    print(f"  {'Embedding (256)':<30} {e_err:>8.0f} mm   {e_r2:>+.2f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_windows": int(len(Y)), "n_holdout": int(test.sum()),
        "naive_err_mm": naive,
        "embedding":        {"err_mm": e_err, "r2": e_r2},
        "whitened_csi_pca": {"err_mm": p_err, "r2": p_r2},
        "whitened_csi_full": {"err_mm": f_err, "r2": f_r2},
    }, indent=2))
    print(f"\n  Saved: {out}")
    del mm
    CSI_MMAP.unlink(missing_ok=True)


# Entry point

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Table 6: per-activity delta
    run_per_activity_delta(device)

    # Table 7: hip regression probe
    run_hip_regression(device)

    print("\n  Next: python scripts/exp_sharp.py")


if __name__ == "__main__":
    main()
