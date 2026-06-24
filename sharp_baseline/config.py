"""Locked configuration for the SHARP amplitude-Doppler baseline.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SharpConfig:
    doppler_sample_length: int = 64      # STFT length in packets (~64 ms @ 1 kHz)
    doppler_stride: int = 10             # hop in packets -> ~100 Hz Doppler-frame rate
    doppler_bins: int = 100              # FFT bins (+-500 Hz spectrum)
    noise_level: float = -1.2            # power floor = 10**noise_level (upstream default)

    # Classifier window
    window_length: int = 200             # 2.0 s
    window_stride: int = 50              # 0.5 s hop

    # 3 receivers x 4 antennas = 12 late-fused streams (summed softmax at eval).
    antennas_per_receiver: int = 4

    feature_backend: str = "torch"       # "torch" (GPU) or "numpy"
    feature_batch_size: int = 1024

    epochs: int = 75
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    patience: int = 10                   # early stop on val (stream-window) accuracy
    seed: int = 42
    stats_max_windows: int = 4096
