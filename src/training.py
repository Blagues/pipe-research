"""
Training and evaluation utilities: preprocessing, encoder factory, HAR probe,
contrastive dataset and training loop.
"""
import csv
import json
import random
import warnings
from pathlib import Path

# sklearn / torch / matplotlib emit chatty UserWarnings during training; quiet them.
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader, Subset

from src.cache import find_cache
from src.split import load_segments, load_split, window_indices_for, make_seqs
from src.encoder import DualHeadCSIEncoder
from src.classifier import ActivityTransformer
from src.losses import (soft_similarity_matrix, soft_similarity_matrix_vel,
                         soft_similarity_matrix_dual, soft_info_nce)
from src.data import ACTIVITY_LABELS


SEED = 42


# Within-window preprocessing

def apply_preprocess(x, mode: str, time_axis: int):
    """Apply within-window preprocessing. Accepts a numpy array or torch tensor
    and returns the same type."""
    if mode == "none":
        return x
    was_numpy = not isinstance(x, torch.Tensor)
    if was_numpy:
        x = torch.from_numpy(np.ascontiguousarray(x))

    if mode == "diff":
        d   = torch.diff(x, dim=time_axis)
        out = torch.cat([x.narrow(time_axis, 0, 1), d], dim=time_axis)
    elif mode == "demean":
        out = x - x.mean(dim=time_axis, keepdim=True)
    elif mode == "stdev":
        out = x / x.std(dim=time_axis, keepdim=True).clamp(min=1e-3)
    elif mode == "whiten":
        x   = x - x.mean(dim=time_axis, keepdim=True)
        out = x / x.std(dim=time_axis, keepdim=True).clamp(min=1e-3)
    elif mode in ("l1_frame", "l1_whiten"):
        # ℓ₁ per-frame: divide each time step by mean amplitude across antenna+subcarrier dims
        # removes AGC-induced global gain (Portner et al, 2026)
        x = x / x.abs().mean(dim=(-2, -1), keepdim=True).clamp(min=1e-3)
        if mode == "l1_whiten":
            x = x - x.mean(dim=time_axis, keepdim=True)
            x = x / x.std(dim=time_axis, keepdim=True).clamp(min=1e-3)
        out = x
    else:
        raise ValueError(f"Unknown preprocess mode: {mode!r}")

    return out.numpy() if was_numpy else out


# Encoder factory

def build_encoder(cfg: dict) -> DualHeadCSIEncoder:
    ec = cfg["encoder"]
    return DualHeadCSIEncoder(
        T=cfg["data"]["raw_window"],
        d=ec["d_model"], N=ec["n_self_attn"], M=ec["n_cross_attn"],
        mlp_dim=ec["mlp_dim"], n_heads=ec["n_heads"], ffn_dim=ec["ffn_dim"],
        mlp_dropout=ec["dropout_mlp"], self_attn_dropout=ec["dropout_attn"],
        cross_attn_dropout=ec["dropout_attn"], proj_dropout=ec["dropout_proj"],
    )


# HAR probe utilities

class SequenceDataset(Dataset):
    """One segment = one sequence, padded to max_seq_len with attention mask."""
    def __init__(self, embeddings: np.ndarray, sequences: list[tuple[np.ndarray, int]], max_seq_len: int):
        self.embeddings  = embeddings
        self.seqs        = sequences
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        idxs, label = self.seqs[i]
        # Gather the segment's window embeddings, pad up to max_seq_len, and build a
        # mask that is False on real frames and True on padding.
        real = min(len(idxs), self.max_seq_len)
        d_model = self.embeddings.shape[1]
        x = torch.zeros(self.max_seq_len, d_model)
        x[:real] = torch.from_numpy(self.embeddings[np.array(idxs[:real])])
        mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        mask[:real] = False
        return x, mask, label


@torch.no_grad()
def embed_all(encoder, csi: np.ndarray, csi_mean: np.ndarray, csi_std: np.ndarray,
              cfg: dict, device, batch: int = 256) -> np.ndarray:
    """Embed all windows with the encoder; returns (N, d) float32."""
    encoder.eval()
    # Apply the same global normalization + within-window preprocessing the encoder
    # saw during training, then embed in batches to keep memory bounded.
    parts  = []
    mode   = cfg["encoder"]["preprocess"]
    mean_t = torch.from_numpy(csi_mean).to(device)
    std_t  = torch.from_numpy(csi_std).to(device)
    for i in range(0, len(csi), batch):
        x = torch.from_numpy(csi[i:i + batch].astype(np.float32)).to(device)
        x = (x - mean_t) / std_t
        x = apply_preprocess(x, mode, time_axis=2)
        z, _ = encoder(x)
        parts.append(z.cpu().numpy())
        print(f"  {min(i + batch, len(csi))}/{len(csi)}", end="\r", flush=True)
    print()
    return np.concatenate(parts)


def train_har(embeddings: np.ndarray, train_seqs, val_seqs, hld_seqs,
              n_classes: int, cfg: dict, device,
              step_log_path=None, epoch_log_path=None,
              encoder_epoch=None, return_clf=False):
    """Train ActivityTransformer on embeddings; returns (val_acc, holdout_acc)."""
    torch.manual_seed(42)
    np.random.seed(42)
    hc  = cfg["har"]
    msl = hc["max_seq_len"]
    d_model = embeddings.shape[1]
    clf   = ActivityTransformer(d_model, n_classes, n_heads=hc["n_heads"], n_layers=hc["n_layers"],
                                dropout=hc["dropout"], ffn_mult=hc["ffn_mult"],
                                max_seq_len=msl).to(device)
    opt   = torch.optim.Adam(clf.parameters(), lr=hc["lr"], weight_decay=hc["weight_decay"])
    # Linear warmup for warmup_epochs, then cosine decay down to lr_min.
    warmup = torch.optim.lr_scheduler.LinearLR(opt, 1/hc["warmup_epochs"], 1.0, hc["warmup_epochs"])
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hc["epochs"] - hc["warmup_epochs"], eta_min=hc["lr_min"])
    sched  = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[hc["warmup_epochs"]])

    t_loader = DataLoader(SequenceDataset(embeddings, train_seqs, msl), batch_size=hc["batch_size"], shuffle=True)
    v_loader = DataLoader(SequenceDataset(embeddings, val_seqs,   msl), batch_size=hc["batch_size"])
    h_loader = DataLoader(SequenceDataset(embeddings, hld_seqs,   msl), batch_size=hc["batch_size"])

    def accuracy(loader):
        clf.eval()
        correct = total = 0
        with torch.no_grad():
            for x, mask, y in loader:
                correct += (clf(x.to(device), mask.to(device)).argmax(1) == y.to(device)).sum().item()
                total   += len(y)
        return correct / total if total else 0.0

    def val_loss():
        clf.eval()
        total = 0.0
        with torch.no_grad():
            for x, mask, y in v_loader:
                total += nn.functional.cross_entropy(
                    clf(x.to(device), mask.to(device)), y.to(device)).item()
        return total / max(len(v_loader), 1)

    step_f  = open(step_log_path,  "a", newline="") if step_log_path  else None
    epoch_f = open(epoch_log_path, "a", newline="") if epoch_log_path else None
    step_w  = csv.writer(step_f)  if step_f  else None
    epoch_w = csv.writer(epoch_f) if epoch_f else None

    # Track the best-val weights in memory and restore them at the end; stop early
    # if val accuracy goes `patience` epochs without improving.
    global_step = 0
    best_val, patience_left = 0.0, hc["patience"]
    best_state = {k: v.clone() for k, v in clf.state_dict().items()}
    try:
        for epoch in range(1, hc["epochs"] + 1):
            clf.train()
            for x, mask, y in t_loader:
                loss = nn.functional.cross_entropy(
                    clf(x.to(device), mask.to(device)), y.to(device),
                    label_smoothing=hc["label_smoothing"])
                opt.zero_grad()
                loss.backward()
                opt.step()
                global_step += 1
                if step_w:
                    step_w.writerow([encoder_epoch, epoch, global_step, f"{loss.item():.6f}"])
            sched.step()
            v_acc  = accuracy(v_loader)
            v_loss = val_loss()
            if epoch_w:
                epoch_w.writerow([encoder_epoch, epoch, f"{v_loss:.6f}", f"{v_acc:.4f}"])
            print(f"    HAR epoch {epoch:3d}/{hc['epochs']}  val={v_acc:.3f}  val_loss={v_loss:.4f}", flush=True)
            if v_acc > best_val:
                best_val = v_acc
                patience_left = hc["patience"]
                best_state = {k: v.clone() for k, v in clf.state_dict().items()}
            else:
                patience_left -= 1
                if patience_left == 0:
                    print(f"    Early stop at epoch {epoch}")
                    break
    finally:
        if step_f:
            step_f.close()
        if epoch_f:
            epoch_f.close()

    clf.load_state_dict(best_state)
    val_acc, hld_acc = accuracy(v_loader), accuracy(h_loader)
    if return_clf:
        return val_acc, hld_acc, clf
    return val_acc, hld_acc


# Contrastive dataset

class _ContrastiveDataset(Dataset):
    def __init__(self, cache_dir: Path, cfg: dict):
        self.csi         = np.load(cache_dir / "csi.npy",           mmap_mode="r")
        self.joints      = np.load(cache_dir / "joints.npy",        mmap_mode="r")
        self.labels      = np.load(cache_dir / "labels.npy",        mmap_mode="r")
        self.rec_idx     = np.load(cache_dir / "recording_idx.npy", mmap_mode="r")
        self.csi_mean    = np.load(cache_dir / "csi_mean.npy")
        self.csi_std     = np.load(cache_dir / "csi_std.npy")
        self.preprocess  = cfg["encoder"]["preprocess"]
        self.global_norm = cfg["encoder"].get("global_norm", True)
        vel_path         = cache_dir / "joints_vel.npy"
        self.vel         = np.load(vel_path, mmap_mode="r") if vel_path.exists() else None

    def __len__(self):
        return len(self.csi)

    def __getitem__(self, idx):
        # Optional global z-score, then within-window preprocessing (same as embed_all).
        csi = self.csi[idx].astype(np.float32)
        if self.global_norm:
            csi = (csi - self.csi_mean) / self.csi_std
        csi = apply_preprocess(csi, self.preprocess, time_axis=1)
        # vel is zeros when the cache has no velocity labels (older cache builds).
        vel = self.vel[idx].copy() if self.vel is not None else np.zeros(self.joints.shape[1] // 3, dtype=np.float32)
        return (torch.from_numpy(csi.copy()), torch.from_numpy(self.joints[idx].copy()),
                torch.from_numpy(vel), idx)


def _worker_init(worker_id):
    np.random.seed(SEED + worker_id)


# t-SNE snapshot

@torch.no_grad()
def _save_tsne(model, tsne_csi: np.ndarray, tsne_labels: np.ndarray,
               epoch: int, run_dir: Path, device, cfg: dict):
    model.eval()
    parts = []
    for i in range(0, len(tsne_csi), 128):
        z, _ = model(torch.from_numpy(tsne_csi[i:i+128]).to(device))
        parts.append(z.cpu().numpy())
    emb = np.concatenate(parts)
    xy  = TSNE(n_components=2, perplexity=cfg["encoder"]["tsne_perplexity"],
               init="pca", random_state=SEED, max_iter=1000).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls in np.unique(tsne_labels):
        m = tsne_labels == cls
        ax.scatter(xy[m, 0], xy[m, 1], label=ACTIVITY_LABELS.get(int(cls), str(cls)), s=8, alpha=0.6)
    ax.legend(markerscale=2, fontsize=8)
    ax.set_title(f"Epoch {epoch}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(run_dir / f"tsne_{epoch:03d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    model.train()


# Main training function

def train_encoder(cfg: dict, run_dir: Path, resume: bool = False) -> Path:
    """Train contrastive encoder; returns path to best_model.pt."""
    # Pin every RNG and force deterministic kernels so a run is reproducible.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cache_dir = find_cache(cfg)
    if cache_dir is None:
        raise RuntimeError("No cache found: run prepare.py first.")
    if not (cache_dir / "split.json").exists():
        raise RuntimeError("split.json missing: run prepare.py first.")

    dataset    = _ContrastiveDataset(cache_dir, cfg)
    segments   = load_segments(cache_dir)
    split_data = load_split(cache_dir)

    dev_idx = window_indices_for(segments, split_data, {"train", "val"}, dataset.rec_idx)
    val_idx = window_indices_for(segments, split_data, {"val"},          dataset.rec_idx)
    hld_idx = window_indices_for(segments, split_data, {"holdout"},      dataset.rec_idx)

    ec       = cfg["encoder"]
    sim_mode = ec.get("similarity", "pose")

    loader = DataLoader(
        Subset(dataset, dev_idx), batch_size=ec["batch_size"], shuffle=True,
        drop_last=True, num_workers=4, pin_memory=True,
        prefetch_factor=4, persistent_workers=True,
        worker_init_fn=_worker_init, generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=ec["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    hld_loader = DataLoader(Subset(dataset, hld_idx), batch_size=ec["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # Re-seed right before building the model so weight init does not depend on how
    # many random draws the DataLoader setup above consumed.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = build_encoder(cfg).to(device)
    print(f"Encoder params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=ec["lr"])
    if ec["warmup_epochs"] > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(opt, 1/ec["warmup_epochs"], 1.0, ec["warmup_epochs"])
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(ec["epochs"] - ec["warmup_epochs"], 1), eta_min=ec["lr_min"])
        sched  = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[ec["warmup_epochs"]])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(ec["epochs"], 1), eta_min=ec["lr_min"])

    start_epoch = 1
    best_loss   = float("inf")

    if resume and (run_dir / "checkpoint.pt").exists():
        ckpt = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss   = ckpt["best_loss"]
        print(f"Resumed from epoch {ckpt['epoch']}, best_loss={best_loss:.4f}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        for log_name, header in [
            ("training_log.csv",  ["epoch", "train_loss", "val_loss", "hld_loss", "emb_std", "lr"]),
            ("activity_log.csv",  ["epoch", "val", "holdout"]),
            ("har_step_log.csv",  ["encoder_epoch", "har_epoch", "step", "train_loss"]),
            ("har_epoch_log.csv", ["encoder_epoch", "har_epoch", "val_loss", "val_acc"]),
        ]:
            with open(run_dir / log_name, "w", newline="") as f:
                csv.writer(f).writerow(header)

    labeled_dev = dev_idx[dataset.labels[dev_idx] != -1]
    tsne_idx    = np.random.default_rng(SEED).choice(labeled_dev, min(ec["tsne_samples"], len(labeled_dev)), replace=False)
    tsne_csi    = dataset.csi[tsne_idx].astype(np.float32)
    if dataset.global_norm:
        tsne_csi = (tsne_csi - dataset.csi_mean) / dataset.csi_std
    tsne_csi    = apply_preprocess(tsne_csi, ec["preprocess"], time_axis=2)
    tsne_labels = dataset.labels[tsne_idx]

    all_labels = np.unique([s["lbl"] for s in segments])
    label_map  = {int(c): i for i, c in enumerate(all_labels)}
    n_classes  = len(all_labels)
    msl        = cfg["har"]["max_seq_len"]

    def eval_har_epoch(epoch):
        # Quick HAR probe used as a mid-training signal: re-embed all windows with the
        # current encoder and train a short classifier (40 epochs, patience 7), cheaper
        # than the full run in evaluate.py.
        csi      = np.load(cache_dir / "csi.npy",      mmap_mode="r")
        csi_mean = np.load(cache_dir / "csi_mean.npy")
        csi_std  = np.load(cache_dir / "csi_std.npy")
        if not cfg["encoder"].get("global_norm", True):
            csi_mean = np.zeros_like(csi_mean)
            csi_std  = np.ones_like(csi_std)
        emb     = embed_all(model, csi, csi_mean, csi_std, cfg, device)
        har_cfg = dict(cfg)
        har_cfg["har"] = {**cfg["har"], "epochs": 40, "patience": 7}
        return train_har(
            emb,
            make_seqs(segments, split_data, dataset.rec_idx, {"train"},   label_map, msl, step=max(1, msl // 2)),
            make_seqs(segments, split_data, dataset.rec_idx, {"val"},     label_map, msl),
            make_seqs(segments, split_data, dataset.rec_idx, {"holdout"}, label_map, msl),
            n_classes, har_cfg, device,
            step_log_path=run_dir / "har_step_log.csv",
            epoch_log_path=run_dir / "har_epoch_log.csv",
            encoder_epoch=epoch,
        )

    labels_t = torch.as_tensor(np.asarray(dataset.labels), dtype=torch.long, device=device)

    def _row_norm(S):
        S.fill_diagonal_(0.0)
        return S / S.sum(dim=1, keepdim=True).clamp(min=1e-8)

    def _similarity(joints, vel, idx):
        # Build the soft-target matrix for the configured supervision mode:
        # pose_vel = body-state (pose x velocity), vel = velocity only,
        # label = SupCon on activity labels, default = pose only.
        if sim_mode == "pose_vel":
            return soft_similarity_matrix_dual(joints, vel, sigma_p=ec["sigma"], sigma_v=ec["sigma_v"])
        if sim_mode == "vel":
            return soft_similarity_matrix_vel(vel, sigma_v=ec["sigma_v"])
        if sim_mode == "label":
            lbl   = labels_t[idx]
            valid = lbl >= 0
            S = (lbl[:, None] == lbl[None, :]).float() * (valid[:, None] & valid[None, :]).float()
            return _row_norm(S)
        return soft_similarity_matrix(joints, sigma=ec["sigma"])

    har_interval  = ec.get("har_interval", 2)
    ckpt_interval = ec.get("checkpoint_interval", 0)

    for epoch in range(start_epoch, ec["epochs"] + 1):
        model.train()
        total, last_z = 0.0, None
        for csi_batch, joints, vel, idx in loader:
            csi_batch, joints, vel, idx = csi_batch.to(device), joints.to(device), vel.to(device), idx.to(device)
            z, p = model(csi_batch)
            S    = _similarity(joints, vel, idx)
            loss = soft_info_nce(p, S, tau=ec["tau"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total  += loss.item()
            last_z  = z.detach()
        sched.step()

        avg_loss = total / len(loader)
        emb_std  = last_z.std(dim=0).mean().item()
        lr_now   = sched.get_last_lr()[0]

        model.eval()
        val_total = hld_total = 0.0
        with torch.no_grad():
            for csi_b, joints, vel, idx in val_loader:
                csi_b, joints, vel, idx = csi_b.to(device), joints.to(device), vel.to(device), idx.to(device)
                _, p = model(csi_b)
                val_total += soft_info_nce(p, _similarity(joints, vel, idx), tau=ec["tau"]).item()
            for csi_b, joints, vel, idx in hld_loader:
                csi_b, joints, vel, idx = csi_b.to(device), joints.to(device), vel.to(device), idx.to(device)
                _, p = model(csi_b)
                hld_total += soft_info_nce(p, _similarity(joints, vel, idx), tau=ec["tau"]).item()
        enc_val_loss = val_total / max(len(val_loader), 1)
        enc_hld_loss = hld_total / max(len(hld_loader), 1)
        model.train()

        print(f"Epoch {epoch:3d}/{ec['epochs']}  loss={avg_loss:.4f}  val_loss={enc_val_loss:.4f}  hld_loss={enc_hld_loss:.4f}  emb_std={emb_std:.3f}  lr={lr_now:.2e}")

        with open(run_dir / "training_log.csv", "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{avg_loss:.6f}", f"{enc_val_loss:.6f}", f"{enc_hld_loss:.6f}", f"{emb_std:.6f}", f"{lr_now:.2e}"])

        # best_model.pt keeps the weights with the lowest *training* contrastive loss;
        # checkpoint.pt is the latest epoch with optimizer/scheduler state for resuming.
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "best_loss": best_loss}, run_dir / "checkpoint.pt")

        if ckpt_interval > 0 and epoch % ckpt_interval == 0:
            torch.save(model.state_dict(), run_dir / f"epoch_{epoch:03d}.pt")

        if ec["tsne_interval"] > 0 and epoch % ec["tsne_interval"] == 0:
            _save_tsne(model, tsne_csi, tsne_labels, epoch, run_dir, device, cfg)

        if har_interval > 0 and (epoch % har_interval == 0 or epoch == ec["epochs"]):
            print("  evaluating HAR...", flush=True)
            val_acc, hld_acc = eval_har_epoch(epoch)
            print(f"  HAR  val={val_acc:.3f}  holdout={hld_acc:.3f}", flush=True)
            with open(run_dir / "activity_log.csv", "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{val_acc:.4f}", f"{hld_acc:.4f}"])

    return run_dir / "best_model.pt"
