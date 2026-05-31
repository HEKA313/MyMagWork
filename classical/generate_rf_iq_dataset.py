#!/usr/bin/env python3
"""
generate_rf_iq_dataset.py

Synthetic IQ dataset generator for classical RF-signal identification methods.
It reuses the waveform models from generate_rf_tfa_dataset.py but saves raw
complex baseband IQ samples instead of CWD/STFT images.

Output layout:
  dataset/
    iq/train/<label>/*.npz
    iq/val/<label>/*.npz
    iq/test/<label>/*.npz
    metadata.csv
    generation_config.json

Each .npz contains:
  iq: complex64 vector
  fs: sample rate, float64
  snr_db: SNR used for AWGN, float64
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.append("D:\\Pycharm Projects\\VKR\\next_gen")
print(sys.path)
from generate_rf_tfa_dataset import (
    CLASS_INFO,
    MIXED_RF_CLASSES,
    PROFILE_CLASSES,
    GeneratorConfig,
    add_awgn,
    classes_from_args,
    make_waveform,
    random_int,
    safe_snr_name,
    stratified_split_name,
    split_name,
)


def generate_iq_dataset(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    gen_cfg = GeneratorConfig(
        fs=args.fs,
        min_n=args.num_samples,
        max_n=args.num_samples,
        bw_frac_min=args.bw_frac_min,
        bw_frac_max=args.bw_frac_max,
        freq_offset_frac=args.freq_offset_frac,
        polytime_phase_states=args.polytime_phase_states,
        multipath=args.multipath,
        phase_noise=args.phase_noise,
        iq_imbalance=args.iq_imbalance,
        amplitude_jitter=args.amplitude_jitter,
    )

    classes = classes_from_args(args)
    if args.train_fraction + args.val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1.0")

    total = len(classes) * len(args.snrs) * args.samples_per_class_snr
    written = 0
    metadata_path = out_dir / "metadata.csv"
    config_path = out_dir / "generation_config.json"

    with config_path.open("w", encoding="utf-8") as f:
        json.dump({**vars(args), "classes_resolved": list(classes)}, f, ensure_ascii=False, indent=2)

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "relative_path", "split", "label", "domain", "family", "snr_db",
            "sample_index", "fs_hz", "num_samples", "profile", "params_json",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for label in classes:
            info = CLASS_INFO[label]
            for snr_db in args.snrs:
                snr_tag = safe_snr_name(snr_db)
                for sample_idx in range(args.samples_per_class_snr):
                    n = int(args.num_samples)
                    x, params = make_waveform(label, n, args.fs, gen_cfg, rng)
                    y = add_awgn(x, float(snr_db), rng)

                    if args.split_mode == "stratified":
                        split = stratified_split_name(sample_idx, args.samples_per_class_snr, args.train_fraction, args.val_fraction)
                    else:
                        split = split_name(rng, args.train_fraction, args.val_fraction)

                    class_dir = out_dir / "iq" / split / label
                    class_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"{label}_snr{snr_tag}_{sample_idx:06d}.npz"
                    path = class_dir / filename

                    np.savez_compressed(
                        path,
                        iq=y.astype(np.complex64),
                        fs=np.asarray(args.fs, dtype=np.float64),
                        snr_db=np.asarray(snr_db, dtype=np.float64),
                    )
                    rel = path.relative_to(out_dir).as_posix()
                    writer.writerow({
                        "relative_path": rel,
                        "split": split,
                        "label": label,
                        "domain": info["domain"],
                        "family": info["family"],
                        "snr_db": snr_db,
                        "sample_index": sample_idx,
                        "fs_hz": args.fs,
                        "num_samples": n,
                        "profile": args.profile,
                        "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    })
                    written += 1
                    if args.progress and (written % args.progress == 0 or written == total):
                        print(f"[{written:>7}/{total}] saved {rel}")

    print(f"Done. IQ files: {written}. Metadata: {metadata_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate synthetic RF IQ datasets for classical ML baselines.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", type=str, default="rf_iq_dataset", help="Output dataset directory")
    p.add_argument("--profile", choices=sorted(PROFILE_CLASSES), default="mixed_rf", help="Class profile")
    p.add_argument("--classes", nargs="+", default=None, help="Optional explicit class subset")
    p.add_argument("--samples-per-class-snr", type=int, default=200, help="IQ signals per class per SNR")
    p.add_argument("--snrs", nargs="+", type=float, default=[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], help="SNR values in dB")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--fs", type=float, default=100e6)
    p.add_argument("--num-samples", type=int, default=1024, help="Fixed IQ length saved per sample")
    p.add_argument("--bw-frac-min", type=float, default=0.05)
    p.add_argument("--bw-frac-max", type=float, default=0.30)
    p.add_argument("--freq-offset-frac", type=float, default=0.06)
    p.add_argument("--polytime-phase-states", type=int, default=2)
    p.add_argument("--multipath", action="store_true")
    p.add_argument("--phase-noise", action="store_true")
    p.add_argument("--iq-imbalance", action="store_true")
    p.add_argument("--amplitude-jitter", action="store_true")
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--split-mode", choices=["stratified", "random"], default="stratified")
    p.add_argument("--progress", type=int, default=500, help="Print every N files, 0 disables")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    generate_iq_dataset(args)


if __name__ == "__main__":
    main()
