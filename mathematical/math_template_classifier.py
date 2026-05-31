#!/usr/bin/env python3
"""
math_template_classifier.py

Математическая ветка идентификации сигналов без обучения.

Идея метода:
  1. Для каждого класса задается математическая модель сигнала.
  2. По моделям строится банк эталонных сигналов.
  3. Для принятого IQ-сигнала считается нормированная корреляция с каждым эталоном.
  4. Выбирается класс с максимальным значением критерия.

Критерий:
  score(x, s) = |<x, s>|^2 / (||x||^2 ||s||^2)

Это практический вариант банка согласованных фильтров / обобщенного критерия
отношения правдоподобия для случая неизвестных параметров, когда максимум берется
по сетке эталонов. В коде нет обучения параметров классификатора: качество зависит
от полноты банка эталонов и близости математических моделей к сигналам в датасете.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

sys.path.append("D:\\Pycharm Projects\\VKR\\next_gen")

from generate_rf_tfa_dataset import (
    CLASS_INFO,
    CORE_LPI_CLASSES,
    EXTENDED_RADAR_CLASSES,
    MIXED_RF_CLASSES,
    GeneratorConfig,
    make_waveform,
    normalize_power,
)


# Классы, для которых математическая модель наиболее естественна.
# Коммуникационные классы со случайной информационной последовательностью требуют
# маргинализации по неизвестным символам; поэтому в банке эталонов они отключены по умолчанию.
MATH_CORE_CLASSES: Tuple[str, ...] = CORE_LPI_CLASSES
MATH_RADAR_CLASSES: Tuple[str, ...] = tuple(
    c for c in EXTENDED_RADAR_CLASSES
    if c not in {"OFDM_Radar", "NoiseOnly"}
)
MATH_MIXED_CLASSES: Tuple[str, ...] = MIXED_RF_CLASSES

MATH_PROFILES: Dict[str, Tuple[str, ...]] = {
    "core_lpi": MATH_CORE_CLASSES,
    "math_radar": MATH_RADAR_CLASSES,
    "mixed_rf": MATH_MIXED_CLASSES,
}


@dataclass(frozen=True)
class TemplateBankConfig:
    fs: float = 100e6
    num_samples: int = 1024
    templates_per_class: int = 16
    seed: int = 2026
    bw_frac_min: float = 0.05
    bw_frac_max: float = 0.30
    freq_offset_frac: float = 0.06
    polytime_phase_states: int = 2
    include_communications: bool = False
    include_noise_template: bool = False


@dataclass
class TemplateBank:
    templates: np.ndarray              # shape: [num_templates, num_samples], normalized complex64
    labels: List[str]                   # label for each template
    template_info: List[Dict[str, object]]
    classes: List[str]
    config: TemplateBankConfig


COMMUNICATION_CLASSES = {
    "BPSK_Random", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK",
    "B_FM", "DSB_AM", "SSB_AM", "OFDM_Radar",
}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex64)
    nrm = float(np.linalg.norm(x))
    if not np.isfinite(nrm) or nrm <= 0.0:
        return np.zeros_like(x, dtype=np.complex64)
    return (x / nrm).astype(np.complex64)


def fix_length(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex64).reshape(-1)
    if x.size == n:
        return x
    if x.size > 0 and x.size != n:
        return resample(x.astype(np.complex128), n).astype(np.complex64)
    return np.zeros(n, dtype=np.complex64)


def load_iq(path: Path, n: int) -> Tuple[np.ndarray, float, Optional[float]]:
    with np.load(path) as data:
        x = data["iq"].astype(np.complex64)
        fs = float(data["fs"]) if "fs" in data else float("nan")
        snr = float(data["snr_db"]) if "snr_db" in data else None
    x = fix_length(x, n)
    x = l2_normalize(x)
    return x, fs, snr


def resolve_classes(profile: str, explicit_classes: Optional[Sequence[str]], include_communications: bool) -> List[str]:
    if explicit_classes:
        classes = list(explicit_classes)
    else:
        if profile not in MATH_PROFILES:
            raise ValueError(f"Unknown profile {profile}. Available: {sorted(MATH_PROFILES)}")
        classes = list(MATH_PROFILES[profile])

    out: List[str] = []
    for label in classes:
        if label not in CLASS_INFO:
            raise ValueError(f"Unknown class: {label}")
        if label in COMMUNICATION_CLASSES and not include_communications:
            # По умолчанию случайные коммуникационные сигналы не включаем в эталонный банк.
            continue
        if label == "NoiseOnly":
            continue
        out.append(label)
    return out


def build_template_bank(classes: Sequence[str], cfg: TemplateBankConfig) -> TemplateBank:
    rng_master = np.random.default_rng(cfg.seed)
    gen_cfg = GeneratorConfig(
        fs=cfg.fs,
        min_n=cfg.num_samples,
        max_n=cfg.num_samples,
        bw_frac_min=cfg.bw_frac_min,
        bw_frac_max=cfg.bw_frac_max,
        freq_offset_frac=cfg.freq_offset_frac,
        polytime_phase_states=cfg.polytime_phase_states,
        multipath=False,
        phase_noise=False,
        iq_imbalance=False,
        amplitude_jitter=False,
    )

    templates: List[np.ndarray] = []
    labels: List[str] = []
    info: List[Dict[str, object]] = []

    for label in classes:
        for idx in range(cfg.templates_per_class):
            seed = int(rng_master.integers(0, 2**32 - 1))
            rng = np.random.default_rng(seed)
            x, params = make_waveform(label, cfg.num_samples, cfg.fs, gen_cfg, rng)
            x = l2_normalize(x.astype(np.complex64))
            templates.append(x)
            labels.append(label)
            info.append({"label": label, "template_index": idx, "seed": seed, "params": params})

    if not templates:
        raise ValueError("Template bank is empty. Check selected classes/profile.")

    arr = np.stack(templates, axis=0).astype(np.complex64)
    return TemplateBank(
        templates=arr,
        labels=labels,
        template_info=info,
        classes=sorted(set(labels)),
        config=cfg,
    )


def raw_template_scores(
    x_batch: np.ndarray,
    templates: np.ndarray,
    frequency_invariant: bool = True,
    frequency_nfft: int = 2048,
    template_chunk_size: int = 32,
) -> np.ndarray:
    """Return raw scores for every template.

    If frequency_invariant=True, the criterion maximizes the matched-filter output
    over an additional linear phase/frequency shift:

        max_f |sum_n x[n] conj(s[n]) exp(-j 2 pi f n)|^2

    This is important for our synthetic data because every non-noise signal is
    additionally shifted in frequency.
    """
    x_batch = np.asarray(x_batch, dtype=np.complex64)
    templates = np.asarray(templates, dtype=np.complex64)
    b = x_batch.shape[0]
    t_total = templates.shape[0]

    if not frequency_invariant:
        return (np.abs(x_batch @ np.conj(templates).T) ** 2).astype(np.float32)

    nfft = max(int(frequency_nfft), int(x_batch.shape[1]))
    chunk = max(1, int(template_chunk_size))
    out = np.zeros((b, t_total), dtype=np.float32)
    for start in range(0, t_total, chunk):
        stop = min(start + chunk, t_total)
        # z[b, t, n] = x_b[n] conj(s_t[n])
        z = x_batch[:, None, :] * np.conj(templates[start:stop][None, :, :])
        spec = np.fft.fft(z, n=nfft, axis=2)
        out[:, start:stop] = (np.max(np.abs(spec) ** 2, axis=2)).astype(np.float32)
    return out


def score_batch(
    x_batch: np.ndarray,
    bank: TemplateBank,
    frequency_invariant: bool = True,
    frequency_nfft: int = 2048,
    template_chunk_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return class scores and best template indices for each class."""
    raw_scores = raw_template_scores(
        x_batch,
        bank.templates,
        frequency_invariant=frequency_invariant,
        frequency_nfft=frequency_nfft,
        template_chunk_size=template_chunk_size,
    )

    class_to_indices: Dict[str, List[int]] = {}
    for i, label in enumerate(bank.labels):
        class_to_indices.setdefault(label, []).append(i)

    class_labels = bank.classes
    scores = np.zeros((x_batch.shape[0], len(class_labels)), dtype=np.float32)
    best_template_indices = np.zeros((x_batch.shape[0], len(class_labels)), dtype=np.int32)

    for j, label in enumerate(class_labels):
        idx = np.asarray(class_to_indices[label], dtype=np.int64)
        local = raw_scores[:, idx]
        arg = np.argmax(local, axis=1)
        scores[:, j] = local[np.arange(local.shape[0]), arg]
        best_template_indices[:, j] = idx[arg]

    return scores, best_template_indices


def predict_from_scores(
    scores: np.ndarray,
    class_labels: Sequence[str],
    noise_threshold: Optional[float] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(scores, axis=1)[:, ::-1]
    best_idx = order[:, 0]
    second_idx = order[:, 1] if scores.shape[1] > 1 else order[:, 0]
    best_scores = scores[np.arange(scores.shape[0]), best_idx]
    second_scores = scores[np.arange(scores.shape[0]), second_idx]

    preds = [class_labels[i] for i in best_idx]
    if noise_threshold is not None:
        preds = ["NoiseOnly" if float(s) < noise_threshold else p for p, s in zip(preds, best_scores)]
    return preds, best_scores, second_scores, best_idx


def metrics_dict(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, object]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=list(labels), zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0
    )
    per_class = []
    for label, p, r, f, s in zip(labels, precision, recall, f1, support):
        per_class.append({
            "label": label,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        })
    return {"accuracy": acc, "macro_f1": macro_f1, "per_class": per_class}


def save_confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str], out_dir: Path, prefix: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(out_dir / f"{prefix}.confusion_counts.csv")

    row_sum = cm_df.sum(axis=1).replace(0, np.nan)
    cm_norm = cm_df.div(row_sum, axis=0).fillna(0.0)
    cm_norm.to_csv(out_dir / f"{prefix}.confusion_norm.csv")

    fig_w = max(8.0, 0.35 * len(labels) + 4.0)
    fig_h = max(6.0, 0.35 * len(labels) + 3.0)
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(cm_norm.values, aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Нормированная доля")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title(prefix)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}.confusion_norm.png", dpi=160)
    plt.close()


def evaluate_dataset(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = data_root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")
    meta = pd.read_csv(metadata_path)

    if args.split:
        meta = meta[meta["split"].isin(args.split)].copy()
    classes = resolve_classes(args.profile, args.classes, args.include_communications)
    if args.profile != "mixed_rf" or args.classes or not args.include_communications:
        # Фильтруем датасет по классам математического профиля.
        meta = meta[meta["label"].isin(classes + (["NoiseOnly"] if args.noise_threshold is not None else []))].copy()

    if args.max_samples and args.max_samples > 0 and len(meta) > args.max_samples:
        meta = meta.sample(n=args.max_samples, random_state=args.seed).sort_index().copy()

    bank_cfg = TemplateBankConfig(
        fs=args.fs,
        num_samples=args.num_samples,
        templates_per_class=args.templates_per_class,
        seed=args.seed,
        bw_frac_min=args.bw_frac_min,
        bw_frac_max=args.bw_frac_max,
        freq_offset_frac=args.freq_offset_frac,
        polytime_phase_states=args.polytime_phase_states,
        include_communications=args.include_communications,
        include_noise_template=False,
    )
    bank = build_template_bank(classes, bank_cfg)
    class_labels = list(bank.classes)
    eval_labels = sorted(set(meta["label"].astype(str)).union(class_labels).union(["NoiseOnly"] if args.noise_threshold is not None else []))

    with (out_dir / "template_bank_config.json").open("w", encoding="utf-8") as f:
        json.dump({
            "config": bank_cfg.__dict__,
            "classes": bank.classes,
            "num_templates": int(bank.templates.shape[0]),
            "template_info": bank.template_info,
            "noise_threshold": args.noise_threshold,
        }, f, ensure_ascii=False, indent=2)

    print(f"Датасет: {data_root}")
    print(f"Образцов для оценки: {len(meta)}")
    print(f"Классов в банке эталонов: {len(bank.classes)}")
    print(f"Эталонов: {bank.templates.shape[0]}")

    rows: List[Dict[str, object]] = []
    t0 = time.time()
    idxs = list(range(len(meta)))
    for start in range(0, len(idxs), args.batch_size):
        batch_indices = idxs[start:start + args.batch_size]
        xs = []
        batch_meta = []
        for i in batch_indices:
            row = meta.iloc[i]
            path = data_root / str(row["relative_path"])
            x, fs, snr_from_file = load_iq(path, args.num_samples)
            xs.append(x)
            batch_meta.append(row)
        x_batch = np.stack(xs, axis=0).astype(np.complex64)
        scores, best_tidx = score_batch(
            x_batch,
            bank,
            frequency_invariant=args.frequency_invariant,
            frequency_nfft=args.frequency_nfft,
            template_chunk_size=args.template_chunk_size,
        )
        preds, best_scores, second_scores, best_idx = predict_from_scores(scores, class_labels, args.noise_threshold)
        order = np.argsort(scores, axis=1)[:, ::-1]

        for local_i, row in enumerate(batch_meta):
            pred = preds[local_i]
            best_class = class_labels[int(order[local_i, 0])]
            second_class = class_labels[int(order[local_i, 1])] if len(class_labels) > 1 else best_class
            snr_val = row.get("snr_db", np.nan)
            rows.append({
                "relative_path": row["relative_path"],
                "split": row.get("split", ""),
                "true_label": row["label"],
                "predicted_label": pred,
                "best_template_class": best_class,
                "second_template_class": second_class,
                "best_score": float(best_scores[local_i]),
                "second_score": float(second_scores[local_i]),
                "score_margin": float(best_scores[local_i] - second_scores[local_i]),
                "snr_db": snr_val,
                "domain": row.get("domain", ""),
                "family": row.get("family", ""),
            })
        if args.progress and (start + len(batch_indices)) % args.progress < args.batch_size:
            done = min(start + len(batch_indices), len(meta))
            rate = done / max(time.time() - t0, 1e-9)
            print(f"Оценено {done}/{len(meta)} ({rate:.1f} сигнал/с)")

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    y_true = pred_df["true_label"].astype(str).tolist()
    y_pred = pred_df["predicted_label"].astype(str).tolist()
    metrics = metrics_dict(y_true, y_pred, eval_labels)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame(metrics["per_class"]).to_csv(out_dir / "per_class_metrics.csv", index=False)
    save_confusion_matrix(y_true, y_pred, eval_labels, out_dir, "math_template")

    if "snr_db" in pred_df.columns:
        snr_rows = []
        for snr, g in pred_df.dropna(subset=["snr_db"]).groupby("snr_db"):
            snr_rows.append({
                "snr_db": float(snr),
                "accuracy": float(accuracy_score(g["true_label"], g["predicted_label"])),
                "num_samples": int(len(g)),
            })
        if snr_rows:
            pd.DataFrame(snr_rows).sort_values("snr_db").to_csv(out_dir / "per_snr_accuracy.csv", index=False)

    print(f"Готово. accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    print(f"Результаты сохранены: {out_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Математическая классификация IQ-сигналов через банк эталонов и нормированную корреляцию.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=str, required=True, help="Корень IQ-датасета с metadata.csv")
    p.add_argument("--out-dir", type=str, required=True, help="Папка для результатов")
    p.add_argument("--profile", choices=sorted(MATH_PROFILES), default="core_lpi", help="Математический профиль классов")
    p.add_argument("--classes", nargs="+", default=None, help="Явный список классов")
    p.add_argument("--split", nargs="+", default=["test"], help="Какие split оценивать")
    p.add_argument("--max-samples", type=int, default=0, help="Ограничить число образцов для быстрого теста, 0 = без ограничения")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--fs", type=float, default=100e6)
    p.add_argument("--num-samples", type=int, default=1024)
    p.add_argument("--templates-per-class", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--bw-frac-min", type=float, default=0.05)
    p.add_argument("--bw-frac-max", type=float, default=0.30)
    p.add_argument("--freq-offset-frac", type=float, default=0.06)
    p.add_argument("--polytime-phase-states", type=int, default=2)
    p.add_argument("--include-communications", action="store_true", help="Включить коммуникационные классы в банк эталонов; для случайных символов это приближенный режим")
    p.add_argument("--frequency-invariant", dest="frequency_invariant", action="store_true", default=True, help="Максимизировать корреляцию по частотному сдвигу")
    p.add_argument("--no-frequency-invariant", dest="frequency_invariant", action="store_false", help="Отключить максимум по частотному сдвигу")
    p.add_argument("--frequency-nfft", type=int, default=2048, help="Размер БПФ для поиска частотного сдвига")
    p.add_argument("--template-chunk-size", type=int, default=32, help="Сколько эталонов обрабатывать за раз при частотном поиске")
    p.add_argument("--noise-threshold", type=float, default=None, help="Если max score ниже порога, предсказывать NoiseOnly")
    p.add_argument("--progress", type=int, default=1000, help="Печатать прогресс каждые N сигналов; 0 отключает")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    evaluate_dataset(args)


if __name__ == "__main__":
    main()
