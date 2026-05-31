#!/usr/bin/env python3
"""
classical_features.py

Feature extraction for a non-neural RF signal portrait.

The module extracts interpretable features from complex IQ samples:
  - energy and amplitude statistics
  - instantaneous phase/frequency statistics
  - spectral features from a Welch PSD estimate
  - complex moments and fourth-order cumulant descriptors
  - cyclic-autocorrelation features on a small alpha/lag grid

The cyclic features are deliberately lightweight. They are not a full high-
resolution SCF estimator; they are compact descriptors for classical ML
baselines and signal-portrait construction.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample, welch
from scipy.stats import kurtosis, skew


EPS = 1e-12


@dataclass(frozen=True)
class FeatureConfig:
    fs: float = 100e6
    num_samples: int = 1024
    welch_nperseg: int = 256
    welch_nfft: int = 512
    cyclo_enabled: bool = True
    cyclo_alpha_bins: int = 33
    cyclo_max_alpha: float = 0.5
    cyclo_lags: Tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)


def normalize_power(x: NDArray[np.complex128]) -> NDArray[np.complex128]:
    x = np.asarray(x, dtype=np.complex128)
    p = float(np.mean(np.abs(x) ** 2))
    if not np.isfinite(p) or p <= 0.0:
        return x
    return (x / math.sqrt(p)).astype(np.complex128)


def prepare_iq(x: NDArray[np.complex128], num_samples: int) -> NDArray[np.complex128]:
    x = np.asarray(x, dtype=np.complex128).ravel()
    if x.size == 0:
        raise ValueError("empty IQ vector")
    x = x - np.mean(x)
    if num_samples > 0 and x.size != num_samples:
        x = resample(x, num_samples).astype(np.complex128)
    return normalize_power(x)


def safe_float(v: float) -> float:
    if not np.isfinite(v):
        return 0.0
    return float(v)


def real_stats(prefix: str, a: NDArray[np.float64]) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64).ravel()
    if a.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_skew": 0.0,
            f"{prefix}_kurtosis": 0.0,
        }
    return {
        f"{prefix}_mean": safe_float(np.mean(a)),
        f"{prefix}_std": safe_float(np.std(a)),
        f"{prefix}_min": safe_float(np.min(a)),
        f"{prefix}_max": safe_float(np.max(a)),
        f"{prefix}_skew": safe_float(skew(a, bias=False)) if a.size > 2 else 0.0,
        f"{prefix}_kurtosis": safe_float(kurtosis(a, fisher=False, bias=False)) if a.size > 3 else 0.0,
    }


def spectral_features(x: NDArray[np.complex128], cfg: FeatureConfig) -> Dict[str, float]:
    nperseg = min(cfg.welch_nperseg, x.size)
    nfft = max(cfg.welch_nfft, nperseg)
    freqs, psd = welch(
        x,
        fs=cfg.fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=nfft,
        return_onesided=False,
        scaling="density",
    )
    freqs = np.fft.fftshift(freqs.astype(np.float64))
    psd = np.fft.fftshift(np.asarray(psd, dtype=np.float64))
    psd = np.maximum(psd, 0.0)
    total = float(np.sum(psd) + EPS)
    p = psd / total

    centroid = float(np.sum(freqs * p))
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * p)))
    entropy = float(-np.sum(p * np.log2(p + EPS)) / math.log2(len(p))) if len(p) > 1 else 0.0
    flatness = float(np.exp(np.mean(np.log(psd + EPS))) / (np.mean(psd) + EPS))

    cdf = np.cumsum(p)
    lo_idx = int(np.searchsorted(cdf, 0.005))
    hi_idx = int(np.searchsorted(cdf, 0.995))
    lo_idx = min(max(lo_idx, 0), len(freqs) - 1)
    hi_idx = min(max(hi_idx, 0), len(freqs) - 1)
    occupied_bw = float(abs(freqs[hi_idx] - freqs[lo_idx]))

    peak_idx = int(np.argmax(psd))
    peak_freq = float(freqs[peak_idx])
    peak_to_mean = float(np.max(psd) / (np.mean(psd) + EPS))

    # Band energy ratios in four normalized frequency quadrants.
    abs_f = np.abs(freqs)
    nyq = cfg.fs / 2.0
    low = float(np.sum(psd[abs_f < 0.125 * cfg.fs]) / total)
    mid = float(np.sum(psd[(abs_f >= 0.125 * cfg.fs) & (abs_f < 0.25 * cfg.fs)]) / total)
    high = float(np.sum(psd[abs_f >= 0.25 * cfg.fs]) / total)

    return {
        "spec_centroid_hz": safe_float(centroid),
        "spec_spread_hz": safe_float(spread),
        "spec_entropy": safe_float(entropy),
        "spec_flatness": safe_float(flatness),
        "spec_occupied_bw_99_hz": safe_float(occupied_bw),
        "spec_peak_freq_hz": safe_float(peak_freq),
        "spec_peak_to_mean": safe_float(peak_to_mean),
        "spec_low_energy_ratio": safe_float(low),
        "spec_mid_energy_ratio": safe_float(mid),
        "spec_high_energy_ratio": safe_float(high),
        "spec_nyquist_hz": safe_float(nyq),
    }


def moment_pq(x: NDArray[np.complex128], p: int, q: int) -> complex:
    # Common AMC convention: M_pq = E[x^(p-q) * conj(x)^q].
    return complex(np.mean((x ** (p - q)) * (np.conj(x) ** q)))


def complex_feature(prefix: str, z: complex) -> Dict[str, float]:
    return {
        f"{prefix}_real": safe_float(float(np.real(z))),
        f"{prefix}_imag": safe_float(float(np.imag(z))),
        f"{prefix}_abs": safe_float(float(abs(z))),
        f"{prefix}_angle": safe_float(float(np.angle(z))),
    }


def cumulant_features(x: NDArray[np.complex128]) -> Dict[str, float]:
    m20 = moment_pq(x, 2, 0)
    m21 = moment_pq(x, 2, 1)
    m40 = moment_pq(x, 4, 0)
    m41 = moment_pq(x, 4, 1)
    m42 = moment_pq(x, 4, 2)

    c20 = m20
    c21 = m21
    c40 = m40 - 3.0 * (m20 ** 2)
    c41 = m41 - 3.0 * m20 * m21
    c42 = m42 - abs(m20) ** 2 - 2.0 * (m21 ** 2)

    denom2 = abs(c21) ** 2 + EPS
    denom1 = abs(c21) + EPS

    feats: Dict[str, float] = {}
    for name, val in [("M20", m20), ("M21", m21), ("M40", m40), ("M41", m41), ("M42", m42), ("C20", c20), ("C21", c21), ("C40", c40), ("C41", c41), ("C42", c42)]:
        feats.update(complex_feature(name, val))

    feats.update({
        "C20_abs_norm": safe_float(abs(c20) / denom1),
        "C40_abs_norm": safe_float(abs(c40) / denom2),
        "C41_abs_norm": safe_float(abs(c41) / denom2),
        "C42_abs_norm": safe_float(abs(c42) / denom2),
    })
    return feats


def instantaneous_features(x: NDArray[np.complex128], fs: float) -> Dict[str, float]:
    amp = np.abs(x).astype(np.float64)
    power = amp ** 2
    phase = np.unwrap(np.angle(x)).astype(np.float64)
    dphi = np.diff(phase)
    inst_freq = (fs / (2.0 * np.pi)) * dphi if dphi.size else np.zeros(1, dtype=np.float64)
    # Remove a robust center so features describe modulation spread more than residual CFO.
    inst_freq_centered = inst_freq - np.median(inst_freq)

    feats: Dict[str, float] = {}
    feats.update(real_stats("amp", amp))
    feats.update(real_stats("power", power))
    feats.update(real_stats("phase_unwrapped", phase))
    feats.update(real_stats("inst_freq_hz", inst_freq_centered.astype(np.float64)))

    mean_power = float(np.mean(power) + EPS)
    papr = float(np.max(power) / mean_power)
    feats.update({
        "mean_power": safe_float(mean_power),
        "papr_linear": safe_float(papr),
        "papr_db": safe_float(10.0 * math.log10(papr + EPS)),
        "rms_amp": safe_float(math.sqrt(mean_power)),
        "iq_real_mean": safe_float(np.mean(np.real(x))),
        "iq_imag_mean": safe_float(np.mean(np.imag(x))),
        "iq_real_std": safe_float(np.std(np.real(x))),
        "iq_imag_std": safe_float(np.std(np.imag(x))),
        "iq_corr_real_imag": safe_float(np.corrcoef(np.real(x), np.imag(x))[0, 1]) if x.size > 1 else 0.0,
    })
    return feats


def cyclic_autocorr_summary(x: NDArray[np.complex128], cfg: FeatureConfig) -> Dict[str, float]:
    if not cfg.cyclo_enabled:
        return {}

    n = x.size
    alphas = np.linspace(0.0, cfg.cyclo_max_alpha, cfg.cyclo_alpha_bins, dtype=np.float64)
    lags = tuple(int(lag) for lag in cfg.cyclo_lags if int(lag) >= 0 and int(lag) < n // 2)
    if not lags:
        lags = (0,)

    power = float(np.mean(np.abs(x) ** 2) + EPS)
    nc_vals = []
    cj_vals = []
    nc_meta = []
    cj_meta = []

    for lag in lags:
        if lag == 0:
            a = x
            b = x
        else:
            a = x[lag: n - lag]
            b = x[: n - 2 * lag]
        m = min(a.size, b.size)
        if m <= 2:
            continue
        n_idx = np.arange(m, dtype=np.float64)
        expo = np.exp(-1j * 2.0 * np.pi * alphas[:, None] * n_idx[None, :])
        z_nc = a[:m] * np.conj(b[:m])
        z_cj = a[:m] * b[:m]
        caf_nc = np.abs(expo @ z_nc / m) / power
        caf_cj = np.abs(expo @ z_cj / m) / power
        nc_vals.append(caf_nc)
        cj_vals.append(caf_cj)
        for alpha in alphas:
            nc_meta.append((float(alpha), int(lag)))
            cj_meta.append((float(alpha), int(lag)))

    if not nc_vals:
        return {}

    nc = np.concatenate(nc_vals)
    cj = np.concatenate(cj_vals)

    def summarize(prefix: str, vals: NDArray[np.float64], meta: List[Tuple[float, int]]) -> Dict[str, float]:
        vals = np.asarray(vals, dtype=np.float64)
        # Exclude alpha = 0 for peak features so plain energy does not dominate.
        nonzero_mask = np.asarray([abs(a) > 1e-12 for a, _ in meta], dtype=bool)
        nz = vals[nonzero_mask]
        nz_meta = [m for m, keep in zip(meta, nonzero_mask) if keep]
        if nz.size == 0:
            nz = vals
            nz_meta = meta
        idx = int(np.argmax(nz))
        peak_alpha, peak_lag = nz_meta[idx]
        prob = nz / (np.sum(nz) + EPS)
        entropy = float(-np.sum(prob * np.log2(prob + EPS)) / math.log2(len(prob))) if len(prob) > 1 else 0.0
        sorted_vals = np.sort(nz)[::-1]
        second = float(sorted_vals[1]) if sorted_vals.size > 1 else 0.0
        return {
            f"{prefix}_max": safe_float(float(np.max(nz))),
            f"{prefix}_mean": safe_float(float(np.mean(nz))),
            f"{prefix}_std": safe_float(float(np.std(nz))),
            f"{prefix}_entropy": safe_float(entropy),
            f"{prefix}_peak_alpha_norm": safe_float(peak_alpha),
            f"{prefix}_peak_lag": safe_float(float(peak_lag)),
            f"{prefix}_peak_to_second": safe_float(float(np.max(nz) / (second + EPS))),
        }

    out: Dict[str, float] = {}
    out.update(summarize("caf_nonconj", nc, nc_meta))
    out.update(summarize("caf_conj", cj, cj_meta))
    return out


def extract_features(x: NDArray[np.complex128], cfg: FeatureConfig) -> Dict[str, float]:
    x = prepare_iq(x, cfg.num_samples)
    feats: Dict[str, float] = {}
    feats.update(instantaneous_features(x, cfg.fs))
    feats.update(spectral_features(x, cfg))
    feats.update(cumulant_features(x))
    feats.update(cyclic_autocorr_summary(x, cfg))
    # Final cleanup to keep sklearn happy.
    for key, value in list(feats.items()):
        feats[key] = safe_float(float(value))
    return feats


def load_npz_iq(path: Path) -> Tuple[NDArray[np.complex128], float]:
    with np.load(path, allow_pickle=False) as data:
        if "iq" not in data:
            raise KeyError(f"{path} does not contain 'iq'")
        iq = data["iq"].astype(np.complex128)
        fs = float(data["fs"]) if "fs" in data else 1.0
    return iq, fs


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract statistical and cyclic features from one IQ .npz file.")
    p.add_argument("input", type=str, help="Input .npz file with an 'iq' array")
    p.add_argument("--fs", type=float, default=None, help="Override sample rate")
    p.add_argument("--num-samples", type=int, default=1024)
    p.add_argument("--no-cyclo", action="store_true")
    p.add_argument("--json", action="store_true", help="Print JSON instead of key=value lines")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    iq, fs_from_file = load_npz_iq(Path(args.input))
    cfg = FeatureConfig(fs=args.fs or fs_from_file, num_samples=args.num_samples, cyclo_enabled=not args.no_cyclo)
    feats = extract_features(iq, cfg)
    if args.json:
        print(json.dumps({"config": asdict(cfg), "features": feats}, indent=2, sort_keys=True))
    else:
        for key in sorted(feats):
            print(f"{key}={feats[key]}")


if __name__ == "__main__":
    main()
