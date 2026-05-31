"""
Этот файл применяет сохранённую классическую модель к одному .npz файлу или к папке .npz файлов.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import joblib
import numpy as np

from classical_features import FeatureConfig, extract_features, load_npz_iq
from train_classical_rf import get_confidence


# Возвращает список входных .npz-файлов: один файл или все файлы внутри папки
def collect_inputs(path: Path) -> List[Path]:
	if path.is_file():
		return [path]
	if path.is_dir():
		return sorted(path.rglob("*.npz"))
	raise FileNotFoundError(path)


# Возвращает список входных .npz-файлов: один файл или все файлы внутри папки
def build_feature_config(args: argparse.Namespace, fs: float) -> FeatureConfig:
	return FeatureConfig(
		fs=fs,
		num_samples=args.feature_num_samples,
		welch_nperseg=args.welch_nperseg,
		welch_nfft=args.welch_nfft,
		cyclo_enabled=not args.no_cyclo,
		cyclo_alpha_bins=args.cyclo_alpha_bins,
		cyclo_max_alpha=args.cyclo_max_alpha,
		cyclo_lags=tuple(args.cyclo_lags),
	)


# Загружает обученную модель, извлекает признаки из новых IQ-файлов и сохраняет предсказания
def predict(args: argparse.Namespace) -> None:
	bundle = joblib.load(args.model)
	model = bundle["model"]
	le = bundle["label_encoder"]
	feature_columns = bundle["feature_columns"]
	target = bundle.get("target", "label")

	paths = collect_inputs(Path(args.input))
	if not paths:
		raise FileNotFoundError(f"No .npz files found under {args.input}")

	rows = []
	for path in paths:
		iq, fs_from_file = load_npz_iq(path)
		fs = float(args.fs) if args.fs is not None else fs_from_file
		cfg = build_feature_config(args, fs)
		feats = extract_features(iq, cfg)
		x = np.asarray([[feats.get(name, 0.0) for name in feature_columns]], dtype=np.float64)
		pred_idx = int(model.predict(x)[0])
		conf = float(get_confidence(model, x)[0])
		pred_name = str(le.inverse_transform([pred_idx])[0])
		rows.append({
			"path": str(path),
			"target": target,
			"prediction": pred_name,
			"confidence": conf,
		})

	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=["path", "target", "prediction", "confidence"])
		writer.writeheader()
		writer.writerows(rows)
	print(f"Saved predictions: {out_path}")


# Описывает CLI-параметры для применения обученной классической модели
def build_argparser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Predict RF signal class/domain/family with a classical model.",
	                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	p.add_argument("--model", type=str, required=True, help="Path to model.joblib")
	p.add_argument("--input", type=str, required=True, help="Input .npz file or directory")
	p.add_argument("--out", type=str, default="classical_predictions.csv")
	p.add_argument("--fs", type=float, default=None)
	p.add_argument("--feature-num-samples", type=int, default=1024)
	p.add_argument("--welch-nperseg", type=int, default=256)
	p.add_argument("--welch-nfft", type=int, default=512)
	p.add_argument("--no-cyclo", action="store_true")
	p.add_argument("--cyclo-alpha-bins", type=int, default=33)
	p.add_argument("--cyclo-max-alpha", type=float, default=0.5)
	p.add_argument("--cyclo-lags", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16, 32])
	return p


# Точка входа: запускает предсказание классической модели
def main() -> None:
	args = build_argparser().parse_args()
	predict(args)


if __name__ == "__main__":
	main()
