"""Amplitude-only SHARP Doppler transform + windowing."""
from __future__ import annotations

import numpy as np
import torch


def compute_doppler_profiles_numpy(
    csi_abs: np.ndarray,
    sample_length: int,
    stride: int,
    doppler_bins: int,
    noise_level: float,
) -> np.ndarray:
    csi_abs = np.nan_to_num(csi_abs.astype(np.float32, copy=False))
    packet_means = np.mean(csi_abs, axis=1, keepdims=True)
    csi_abs = csi_abs / np.where(packet_means == 0.0, 1.0, packet_means)

    profiles = []
    window = np.hanning(sample_length).astype(np.float32)[:, None]
    for start in range(0, csi_abs.shape[0] - sample_length + 1, stride):
        cut = csi_abs[start : start + sample_length] * window
        doppler_fft = np.fft.fft(cut, n=doppler_bins, axis=0)
        doppler_fft = np.fft.fftshift(doppler_fft, axes=0)
        power = np.abs(doppler_fft * np.conj(doppler_fft))
        profiles.append(np.sum(power, axis=1).real)

    if not profiles:
        return np.empty((0, doppler_bins), dtype=np.float32)

    profile_array = np.asarray(profiles, dtype=np.float32)
    profile_max = np.max(profile_array, axis=1, keepdims=True)
    profile_array = profile_array / np.where(profile_max == 0.0, 1.0, profile_max)
    floor = np.float32(10.0**noise_level)
    profile_array[profile_array < floor] = floor
    return profile_array


def compute_doppler_profiles_torch(
    csi_abs: np.ndarray,
    sample_length: int,
    stride: int,
    doppler_bins: int,
    noise_level: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if csi_abs.shape[0] < sample_length:
        return np.empty((0, doppler_bins), dtype=np.float32)

    batch_size = max(1, batch_size)
    with torch.no_grad():
        csi = torch.from_numpy(np.nan_to_num(csi_abs.astype(np.float32, copy=False))).to(device=device)
        packet_means = csi.mean(dim=1, keepdim=True)
        csi = csi / torch.where(packet_means == 0.0, torch.ones_like(packet_means), packet_means)
        windows = csi.unfold(0, sample_length, stride).transpose(1, 2)
        hann = torch.hann_window(
            sample_length, periodic=False, dtype=torch.float32, device=device
        ).view(1, sample_length, 1)
        floor = float(10.0**noise_level)
        profile_chunks = []
        for start in range(0, windows.shape[0], batch_size):
            batch = windows[start : start + batch_size] * hann
            doppler_fft = torch.fft.fft(batch, n=doppler_bins, dim=1)
            doppler_fft = torch.fft.fftshift(doppler_fft, dim=1)
            power = doppler_fft.abs().square().sum(dim=2)
            profile_max = power.max(dim=1, keepdim=True).values
            power = power / torch.where(profile_max == 0.0, torch.ones_like(profile_max), profile_max)
            power = torch.clamp_min(power, floor)
            profile_chunks.append(power.cpu())

    return torch.cat(profile_chunks, dim=0).numpy().astype(np.float32, copy=False)


def n_doppler_frames(n_packets: int, sample_length: int, stride: int) -> int:
    """Frame count both backends produce (used for pre-counting windows)."""
    if n_packets < sample_length:
        return 0
    return (n_packets - sample_length) // stride + 1


def create_windows(doppler: np.ndarray, window_length: int, stride: int) -> list[np.ndarray]:
    """Slice a (n_frames, doppler_bins) trace into fixed-length windows."""
    if doppler.shape[0] < window_length:
        return []
    return [
        doppler[start : start + window_length]
        for start in range(0, doppler.shape[0] - window_length + 1, stride)
    ]


def n_windows(n_frames: int, window_length: int, stride: int) -> int:
    """Window count create_windows() yields (used for pre-counting)."""
    if n_frames < window_length:
        return 0
    return (n_frames - window_length) // stride + 1
