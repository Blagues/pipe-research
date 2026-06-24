"""
Section 6.5: Comparison with SHARP
=====================================================
Trains the SHARP amplitude-Doppler CNN and evaluates it on the same
train/val/holdout split as PIPE, then zero-shot transfers to unseen locations.
"""
import argparse
import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

from sharp_baseline.train import main as sharp_train
from sharp_baseline.eval_zeroshot import main as sharp_eval

SEP = "═" * 60


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    print(SEP)
    print("  SHARP BASELINE: Table 8: Train / Val / Holdout")
    print(SEP)
    sharp_train(config=args.config)

    print(f"\n{SEP}")
    print("  SHARP BASELINE: Table 9: Zero-shot transfer")
    print(SEP)
    sharp_eval(config=args.config)

    print("\n  Results saved to runs/*_sharp_baseline/  and  runs/sharp_zeroshot/")


if __name__ == "__main__":
    main()
