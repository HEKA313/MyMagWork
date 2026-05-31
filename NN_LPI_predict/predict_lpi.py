#!/usr/bin/env python3
"""
Этот файл применяет уже обученную модель к одному изображению или папке изображений.
— загружает checkpoint;
— восстанавливают архитектуру;
— подготавливает изображений;
— считает вероятности классов;
— сохраняет результат в CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from lpi_dataset import IMAGE_EXTS, pil_resize_bicubic
from lpi_model import build_model


# Возвращает список изображений: один файл или все изображения внутри папки
def iter_images(path: Path):
	if path.is_file():
		return [path]
	return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


# Выбирает устройство для вычислений: auto, cpu или cuda
def select_device(device_arg: str) -> torch.device:
	if device_arg == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return torch.device(device_arg)


# Загружает checkpoint с учётом различий между версиями PyTorch
def safe_torch_load(path: str | Path, map_location):
	try:
		return torch.load(path, map_location=map_location, weights_only=False)
	except TypeError:
		return torch.load(path, map_location=map_location)


# Выполняет ту же предобработку изображения, что и при обучении
def preprocess(path: Path, image_size: int, mean: float, std: float) -> torch.Tensor:
	img = Image.open(path).convert("L")
	if img.size != (image_size, image_size):
		img = pil_resize_bicubic(img, (image_size, image_size))
	arr = np.asarray(img, dtype=np.float32) / 255.0
	arr = (arr - mean) / max(std, 1e-8)
	return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


# Точка входа: загружает модель, выполняет инференс и сохраняет predictions.csv
def main() -> None:
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--checkpoint", required=True)
	parser.add_argument("--input", required=True, help="Image file or folder")
	parser.add_argument("--out", default="predictions.csv")
	parser.add_argument("--device", default="auto")
	args = parser.parse_args()

	device = select_device(args.device)
	ckpt = safe_torch_load(args.checkpoint, map_location=device)
	cfg = ckpt.get("config", {})
	arch = ckpt.get("arch", cfg.get("arch", "improved_b0"))
	class_to_idx = ckpt["class_to_idx"]
	idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}
	image_size = int(ckpt.get("image_size", cfg.get("image_size", 224)))
	mean = float(ckpt.get("mean", cfg.get("mean", 0.5)))
	std = float(ckpt.get("std", cfg.get("std", 0.5)))

	model = build_model(
		arch,
		num_classes=len(class_to_idx),
		in_channels=1,
		dropout=float(cfg.get("dropout", 0.2)),
		stochastic_depth_prob=float(cfg.get("stochastic_depth", 0.2)),
		simam_lambda=float(cfg.get("simam_lambda", 1e-4)),
		block_dropout=float(cfg.get("block_dropout", 0.0)),
	).to(device)
	model.load_state_dict(ckpt["model_state"])
	model.eval()

	rows = []
	for img_path in iter_images(Path(args.input).expanduser()):
		x = preprocess(img_path, image_size, mean, std).to(device)
		with torch.no_grad():
			probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().numpy()
		pred_idx = int(np.argmax(probs))
		row = {
			"path": str(img_path),
			"predicted_label": idx_to_class[pred_idx],
			"confidence": float(probs[pred_idx]),
		}
		for idx in range(len(idx_to_class)):
			row["p_{}".format(idx_to_class[idx])] = float(probs[idx])
		rows.append(row)

	out_path = Path(args.out)
	with out_path.open("w", newline="", encoding="utf-8") as f:
		fieldnames = list(rows[0].keys()) if rows else ["path", "predicted_label", "confidence"]
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)
	print("Saved {} predictions to {}".format(len(rows), out_path))


if __name__ == "__main__":
	main()
