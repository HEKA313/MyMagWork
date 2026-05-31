#!/usr/bin/env python3
"""Инференс математического классификатора для отдельных IQ-файлов или папки."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np

from math_template_classifier import (
    TemplateBankConfig,
    build_template_bank,
    load_iq,
    predict_from_scores,
    resolve_classes,
    score_batch,
)


def collect_inputs(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.npz"))


def main() -> None:
    p = argparse.ArgumentParser(description="Классификация IQ-файлов математическим банком эталонов.")
    p.add_argument("--input", required=True, help=".npz файл или папка с .npz")
    p.add_argument("--out", required=True, help="CSV с результатами")
    p.add_argument("--profile", default="core_lpi", choices=["core_lpi", "math_radar", "mixed_rf"])
    p.add_argument("--classes", nargs="+", default=None)
    p.add_argument("--include-communications", action="store_true")
    p.add_argument("--fs", type=float, default=100e6)
    p.add_argument("--num-samples", type=int, default=1024)
    p.add_argument("--templates-per-class", type=int, default=16)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--frequency-invariant", dest="frequency_invariant", action="store_true", default=True)
    p.add_argument("--no-frequency-invariant", dest="frequency_invariant", action="store_false")
    p.add_argument("--frequency-nfft", type=int, default=2048)
    p.add_argument("--template-chunk-size", type=int, default=32)
    p.add_argument("--noise-threshold", type=float, default=None)
    args = p.parse_args()

    classes = resolve_classes(args.profile, args.classes, args.include_communications)
    bank_cfg = TemplateBankConfig(
        fs=args.fs,
        num_samples=args.num_samples,
        templates_per_class=args.templates_per_class,
        seed=args.seed,
        include_communications=args.include_communications,
    )
    bank = build_template_bank(classes, bank_cfg)
    files = collect_inputs(Path(args.input))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {args.input}")

    rows = []
    for start in range(0, len(files), args.batch_size):
        chunk = files[start:start + args.batch_size]
        xs = []
        for path in chunk:
            x, _, _ = load_iq(path, args.num_samples)
            xs.append(x)
        scores, _ = score_batch(
            np.stack(xs, axis=0),
            bank,
            frequency_invariant=args.frequency_invariant,
            frequency_nfft=args.frequency_nfft,
            template_chunk_size=args.template_chunk_size,
        )
        preds, best_scores, second_scores, best_idx = predict_from_scores(scores, bank.classes, args.noise_threshold)
        order = np.argsort(scores, axis=1)[:, ::-1]
        for i, path in enumerate(chunk):
            second_label = bank.classes[int(order[i, 1])] if len(bank.classes) > 1 else bank.classes[int(order[i, 0])]
            rows.append({
                "path": str(path),
                "predicted_label": preds[i],
                "best_template_class": bank.classes[int(order[i, 0])],
                "second_template_class": second_label,
                "best_score": float(best_scores[i]),
                "second_score": float(second_scores[i]),
                "score_margin": float(best_scores[i] - second_scores[i]),
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} predictions to {out}")


if __name__ == "__main__":
    main()
