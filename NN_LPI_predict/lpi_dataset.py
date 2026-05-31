"""
Этот файл отвечате за работу с датасетами и может выполнять:
— поиск изображений спектрограмм;
— определение классов;
— загрузку изображений;
— нормировку
— аугментацию;
— извлечение ОСШ из metadata.csv или имени файла.
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset

# Допустимые форматы изображений частотно-временных портретов
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Фиксированный порядок классов LPI-сигналов для воспроизводимого кодирования меток
DEFAULT_CLASSES = [
    "Rect",
    "LFM",
    "Barker",
    "Costas",
    "Frank",
    "P1",
    "P2",
    "P3",
    "P4",
    "T1",
    "T2",
    "T3",
    "T4",
]


# Определяет корневую папку разбиений датасета: либо data_root, либо data_root/images
def resolve_dataset_root(data_root: str) -> Path:
    root = Path(data_root).expanduser().resolve()
    if (root / "train").is_dir():
        return root
    if (root / "images" / "train").is_dir():
        return root / "images"
    raise FileNotFoundError(
        "Expected train split at {}/train or {}/images/train".format(root, root)
    )


# Ищет metadata.csv в нескольких возможных местах структуры датасета
def find_metadata_path(original_root: str, split_root: Path) -> Optional[Path]:
    root = Path(original_root).expanduser().resolve()
    candidates = [
        root / "metadata.csv",
        split_root / "metadata.csv",
        split_root.parent / "metadata.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


# Формирует порядок классов по папкам train с учётом заданного или стандартного порядка
def ordered_classes_from_train(train_dir: Path, class_order: Optional[Sequence[str]] = None) -> List[str]:
    found = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if class_order is not None:
        ordered = [c for c in class_order if c in found]
        extras = [c for c in found if c not in ordered]
        ordered.extend(sorted(extras))
        return ordered
    ordered = [c for c in DEFAULT_CLASSES if c in found]
    extras = [c for c in found if c not in ordered]
    ordered.extend(sorted(extras))
    return ordered


# Изменяет размер изображения бикубической интерполяцией
def pil_resize_bicubic(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    resampling = getattr(Image, "Resampling", None)
    bicubic = resampling.BICUBIC if resampling is not None else Image.BICUBIC
    return img.resize(size, bicubic)


# Инициализирует датасет одного разбиения train/val/test и сканирует изображения
class LPISpectrogramDataset(Dataset):
    def __init__(
            self,
            split_root: Path,
            split: str,
            classes: Sequence[str],
            image_size: int = 224,
            augment: bool = False,
            mean: float = 0.5,
            std: float = 0.5,
    ) -> None:
        self.split_root = Path(split_root)
        self.split = str(split)
        self.root = self.split_root / self.split
        self.classes = list(classes)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.mean = float(mean)
        self.std = float(std)
        self.samples = self._scan()
        if not self.samples:
            raise FileNotFoundError("No images found in {}".format(self.root))

    # Рекурсивно собирает пути к изображениям и соответствующие числовые метки
    def _scan(self) -> List[Tuple[str, int]]:
        samples = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            label = self.class_to_idx[class_name]
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                    samples.append((str(path), label))
        return samples

    # Возвращает число объектов в датасете
    def __len__(self) -> int:
        return len(self.samples)

    # Выполняет аугментацию спектрограмм без поворотов и отражений
    def _augment_image(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.35:
            arr = np.asarray(img)
            max_shift = max(1, int(round(0.02 * self.image_size)))
            dy = random.randint(-max_shift, max_shift)
            dx = random.randint(-max_shift, max_shift)
            arr = np.roll(arr, shift=dy, axis=0)
            arr = np.roll(arr, shift=dx, axis=1)
            img = Image.fromarray(arr)
        if random.random() < 0.20:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.6)))
        if random.random() < 0.35:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.90, 1.10))
        if random.random() < 0.35:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.90, 1.10))
        if random.random() < 0.20:
            arr = np.asarray(img, dtype=np.float32)
            arr += np.random.normal(loc=0.0, scale=random.uniform(1.0, 4.0), size=arr.shape)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        return img

    # Загружает изображение, нормирует его и возвращает тензор, метку и путь
    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = Image.open(path).convert("L")
        if img.size != (self.image_size, self.image_size):
            img = pil_resize_bicubic(img, (self.image_size, self.image_size))
        if self.augment:
            img = self._augment_image(img)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / max(self.std, 1e-8)
        x = torch.from_numpy(arr).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y, path


# Создаёт датасеты train/val/test с единым порядком классов
def make_datasets(
        data_root: str,
        image_size: int,
        augment: bool,
        mean: float,
        std: float,
        class_order: Optional[Sequence[str]] = None,
):
    split_root = resolve_dataset_root(data_root)
    train_dir = split_root / "train"
    val_dir = split_root / "val"
    test_dir = split_root / "test"
    if not val_dir.is_dir():
        raise FileNotFoundError("Validation split is required: {}".format(val_dir))
    classes = ordered_classes_from_train(train_dir, class_order=class_order)
    train_ds = LPISpectrogramDataset(split_root, "train", classes, image_size, augment, mean, std)
    val_ds = LPISpectrogramDataset(split_root, "val", classes, image_size, False, mean, std)
    test_ds = None
    if test_dir.is_dir():
        test_ds = LPISpectrogramDataset(split_root, "test", classes, image_size, False, mean, std)
    return split_root, train_ds, val_ds, test_ds


# Загружает из metadata.csv соответствие путь_к_файлу -> ОСШ
def load_metadata_snr_map(metadata_path: Optional[Path]) -> Dict[str, float]:
    if metadata_path is None or not Path(metadata_path).is_file():
        return {}
    snr_map = {}
    with Path(metadata_path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row.get("relative_path") or row.get("path") or row.get("file") or ""
            snr = row.get("snr_db") or row.get("snr") or ""
            if not rel or snr == "":
                continue
            try:
                snr_value = float(snr)
            except ValueError:
                continue
            rel_norm = rel.replace("\\", "/")
            snr_map[rel_norm] = snr_value
            if rel_norm.startswith("images/"):
                snr_map[rel_norm[len("images/"):]] = snr_value
    return snr_map


# Загружает из metadata.csv соответствие путь_к_файлу -> ОСШ
def snr_from_filename(path: str) -> Optional[float]:
    name = Path(path).name
    patterns = [
        r"snr[_-]?([mp][0-9]+(?:p[0-9]+)?)(?:db|dB)?",
        r"snr[_-]?([+-]?[0-9]+(?:\.[0-9]+)?)(?:db|dB)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, name, flags=re.IGNORECASE)
        if not match:
            continue
        token = match.group(1)
        if token[0].lower() == "m":
            return -float(token[1:].replace("p", "."))
        if token[0].lower() == "p":
            return float(token[1:].replace("p", "."))
        return float(token)
    return None


# Возвращает ОСШ для файла: сначала из metadata.csv, затем из имени файла
def snr_for_path(path: str, split_root: Path, snr_map: Dict[str, float]) -> Optional[float]:
    p = Path(path).resolve()
    try:
        rel = p.relative_to(split_root).as_posix()
        if rel in snr_map:
            return snr_map[rel]
    except ValueError:
        pass
    return snr_from_filename(path)
