"""
Step 2: Train the contrastive CSI encoder.

Saves to runs/<run_name>/
  best_model.pt: lowest contrastive loss
  checkpoint.pt: latest epoch (for resuming)
  training_log.csv: loss / emb_std / lr per epoch
  activity_log.csv: val / holdout HAR accuracy per epoch
  tsne_NNN.png: t-SNE of dev embeddings every tsne_interval epochs

Usage:
    python pipeline/train.py
    python pipeline/train.py --config config.yaml --run-name my_run
    python pipeline/train.py --resume runs/my_run
"""
import argparse
import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

from datetime import datetime
from pathlib import Path

from src.cache import load_config
from src.training import train_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume",   default=None, metavar="RUN_DIR")
    parser.add_argument("--epochs",   type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["encoder"]["epochs"] = args.epochs

    if args.resume:
        run_dir = Path(args.resume)
    else:
        name    = args.run_name or datetime.now().strftime("%m-%d_%H-%M")
        run_dir = Path(cfg["output"]["runs_dir"]) / name

    train_encoder(cfg, run_dir, resume=bool(args.resume))
    print(f"\nDone. Run: {run_dir}")
    print("Next: python pipeline/evaluate.py")


if __name__ == "__main__":
    main()
