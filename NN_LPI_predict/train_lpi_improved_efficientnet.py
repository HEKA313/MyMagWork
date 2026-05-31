#!/usr/bin/env python3
"""Train Improved EfficientNet-B0 for LPI radar spectrogram classification.

Input dataset layouts accepted:
  dataset/train/<class>/*.png
  dataset/val/<class>/*.png
  dataset/test/<class>/*.png
or:
  dataset/images/train/<class>/*.png
  dataset/images/val/<class>/*.png
  dataset/images/test/<class>/*.png

The script saves checkpoints, per-class metrics, a confusion matrix, and per-SNR
accuracy when SNR is available in metadata.csv or filenames.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lpi_dataset import DEFAULT_CLASSES, find_metadata_path, load_metadata_snr_map, make_datasets, snr_for_path
from lpi_model import build_model, count_parameters


def make_progress(iterable, enabled: bool, desc: str, total: Optional[int] = None):
    """Return tqdm progress iterator when tqdm is installed; otherwise return iterable."""
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, leave=False)
    except Exception:
        return iterable


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def autocast_context(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")
    return torch.cuda.amp.autocast(enabled=enabled and device.type == "cuda")


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def safe_torch_load(path: str | Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class FocalLoss(nn.Module):
    """Multi-class focal loss for softmax classification."""

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = str(reduction)
        if torch.is_tensor(alpha):
            self.register_buffer("alpha_tensor", alpha.float())
            self.alpha_scalar = None
        elif alpha is None:
            self.register_buffer("alpha_tensor", torch.empty(0))
            self.alpha_scalar = None
        else:
            self.register_buffer("alpha_tensor", torch.empty(0))
            self.alpha_scalar = float(alpha)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -torch.pow(1.0 - target_probs, self.gamma) * target_log_probs
        if self.alpha_tensor.numel() > 0:
            loss = loss * self.alpha_tensor.gather(0, targets)
        elif self.alpha_scalar is not None:
            loss = loss * self.alpha_scalar
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def make_loss(args: argparse.Namespace, train_ds) -> nn.Module:
    if args.loss == "ce":
        return nn.CrossEntropyLoss()
    if args.focal_alpha == "none":
        alpha = None
    elif args.focal_alpha == "balanced":
        labels = [int(y) for _, y in train_ds.samples]
        counts = np.bincount(labels, minlength=len(train_ds.classes)).astype(np.float64)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
        alpha = torch.tensor(weights, dtype=torch.float32)
    else:
        alpha = float(args.focal_alpha)
    return FocalLoss(gamma=args.focal_gamma, alpha=alpha)


def make_loader(dataset, batch_size: int, workers: int, train: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(train),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def accuracy_from_logits(logits: Tensor, targets: Tensor) -> Tuple[int, int]:
    preds = torch.argmax(logits, dim=1)
    return int((preds == targets).sum().item()), int(targets.numel())


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp: bool, grad_clip: float, show_progress: bool = False, epoch: int = 0):
    model.train()
    loss_sum = 0.0
    correct_sum = 0
    n_sum = 0
    iterator = make_progress(loader, show_progress, desc="train epoch {}".format(epoch) if epoch else "train", total=len(loader))
    for batch_idx, (images, targets, _paths) in enumerate(iterator, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(targets.numel())
        loss_sum += float(loss.detach().item()) * batch_size
        correct, count = accuracy_from_logits(logits.detach(), targets)
        correct_sum += correct
        n_sum += count
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                loss="{:.4f}".format(loss_sum / max(1, n_sum)),
                acc="{:.3f}".format(correct_sum / max(1, n_sum)),
            )
    return loss_sum / max(1, n_sum), correct_sum / max(1, n_sum)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp: bool, show_progress: bool = False, desc: str = "eval") -> Dict:
    model.eval()
    loss_sum = 0.0
    correct_sum = 0
    n_sum = 0
    targets_all = []
    preds_all = []
    paths_all = []
    iterator = make_progress(loader, show_progress, desc=desc, total=len(loader))
    for images, targets, paths in iterator:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(device, amp):
            logits = model(images)
            loss = criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        batch_size = int(targets.numel())
        loss_sum += float(loss.detach().item()) * batch_size
        correct_sum += int((preds == targets).sum().item())
        n_sum += batch_size
        targets_all.extend(targets.cpu().numpy().tolist())
        preds_all.extend(preds.cpu().numpy().tolist())
        paths_all.extend(list(paths))
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                loss="{:.4f}".format(loss_sum / max(1, n_sum)),
                acc="{:.3f}".format(correct_sum / max(1, n_sum)),
            )
    return {
        "loss": loss_sum / max(1, n_sum),
        "accuracy": correct_sum / max(1, n_sum),
        "targets": np.asarray(targets_all, dtype=np.int64),
        "preds": np.asarray(preds_all, dtype=np.int64),
        "paths": paths_all,
    }


def confusion_matrix_np(targets: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(targets, preds):
        cm[int(target), int(pred)] += 1
    return cm


def per_class_metrics(cm: np.ndarray, class_names: Sequence[str]) -> List[Dict]:
    rows = []
    for idx, name in enumerate(class_names):
        tp = float(cm[idx, idx])
        support = float(cm[idx, :].sum())
        pred_count = float(cm[:, idx].sum())
        recall = tp / support if support else 0.0
        precision = tp / pred_count if pred_count else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "class": name,
                "support": int(support),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def write_dict_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_history(path: Path, rows: Sequence[Dict]) -> None:
    write_dict_csv(path, rows)


def save_confusion_plot(cm: np.ndarray, class_names: Sequence[str], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("matplotlib unavailable, skipping confusion plot: {}".format(exc))
        return
    norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, "{:.1f}".format(norm[i, j] * 100.0), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_curves(history: Sequence[Dict], path: Path) -> None:
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("matplotlib unavailable, skipping curves: {}".format(exc))
        return
    epochs = [int(row["epoch"]) for row in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [float(row["train_acc"]) for row in history], label="train_acc")
    ax.plot(epochs, [float(row["val_acc"]) for row in history], label="val_acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def per_snr_accuracy(paths: Sequence[str], targets: np.ndarray, preds: np.ndarray, split_root: Path, snr_map: Dict[str, float]) -> List[Dict]:
    buckets = {}
    for path, target, pred in zip(paths, targets, preds):
        snr = snr_for_path(path, split_root, snr_map)
        if snr is None:
            continue
        buckets.setdefault(float(snr), []).append(1 if int(target) == int(pred) else 0)
    rows = []
    for snr in sorted(buckets):
        values = buckets[snr]
        rows.append({"snr_db": snr, "n": len(values), "accuracy": float(np.mean(values))})
    return rows


def save_reports(result: Dict, out_dir: Path, split_root: Path, snr_map: Dict[str, float], class_names: Sequence[str], prefix: str) -> Dict:
    cm = confusion_matrix_np(result["targets"], result["preds"], len(class_names))
    np.savetxt(out_dir / "{}_confusion_matrix.csv".format(prefix), cm, fmt="%d", delimiter=",")
    save_confusion_plot(cm, class_names, out_dir / "{}_confusion_matrix.png".format(prefix))
    class_rows = per_class_metrics(cm, class_names)
    write_dict_csv(out_dir / "{}_per_class_metrics.csv".format(prefix), class_rows)
    snr_rows = per_snr_accuracy(result["paths"], result["targets"], result["preds"], split_root, snr_map)
    if snr_rows:
        write_dict_csv(out_dir / "{}_per_snr_accuracy.csv".format(prefix), snr_rows)
    summary = {
        "loss": float(result["loss"]),
        "accuracy": float(result["accuracy"]),
        "macro_precision": float(np.mean([row["precision"] for row in class_rows])) if class_rows else 0.0,
        "macro_recall": float(np.mean([row["recall"] for row in class_rows])) if class_rows else 0.0,
        "macro_f1": float(np.mean([row["f1"] for row in class_rows])) if class_rows else 0.0,
        "num_samples": int(len(result["targets"])),
    }
    with (out_dir / "{}_metrics.json".format(prefix)).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def build_optimizer(args: argparse.Namespace, model: nn.Module):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def parse_class_order(value: str) -> Optional[List[str]]:
    if value.strip().lower() == "auto":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", required=True, help="Dataset root")
    parser.add_argument("--out-dir", default="runs_old/lpi_improved_effnet_b0", help="Output directory")
    parser.add_argument("--arch", default="improved_b0", choices=["improved_b0", "baseline_b0"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", default="focal", choices=["focal", "ce"])
    parser.add_argument("--focal-alpha", default="0.25", help="Float, 'balanced', or 'none'")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--mean", type=float, default=0.5)
    parser.add_argument("--std", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--stochastic-depth", type=float, default=0.2)
    parser.add_argument("--block-dropout", type=float, default=0.0)
    parser.add_argument("--simam-lambda", type=float, default=1e-4)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping", type=int, default=12, help="Patience by val accuracy; 0 disables")
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--resume", default="", help="Checkpoint to resume from")
    parser.add_argument("--eval-only", default="", help="Checkpoint path for evaluation only")
    parser.add_argument("--progress", action="store_true", help="Show live batch progress bars with tqdm")
    parser.add_argument("--no-progress", action="store_true", help="Disable live progress bars")
    parser.add_argument("--class-order", default=",".join(DEFAULT_CLASSES), help="Comma-separated class order, or 'auto'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    class_order = parse_class_order(args.class_order)
    split_root, train_ds, val_ds, test_ds = make_datasets(
        args.data_root,
        image_size=args.image_size,
        augment=args.augment,
        mean=args.mean,
        std=args.std,
        class_order=class_order,
    )
    metadata_path = find_metadata_path(args.data_root, split_root)
    snr_map = load_metadata_snr_map(metadata_path)
    class_names = list(train_ds.classes)

    device = select_device(args.device)
    amp = bool(args.amp and device.type == "cuda")
    train_loader = make_loader(train_ds, args.batch_size, args.workers, train=True, device=device)
    val_loader = make_loader(val_ds, args.batch_size, args.workers, train=False, device=device)
    test_loader = make_loader(test_ds, args.batch_size, args.workers, train=False, device=device) if test_ds is not None else None

    model = build_model(
        args.arch,
        num_classes=len(class_names),
        in_channels=1,
        dropout=args.dropout,
        stochastic_depth_prob=args.stochastic_depth,
        simam_lambda=args.simam_lambda,
        block_dropout=args.block_dropout,
    ).to(device)
    total_params, trainable_params = count_parameters(model)
    criterion = make_loss(args, train_ds).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = make_grad_scaler(amp)

    start_epoch = 1
    best_val_acc = -1.0
    best_epoch = 0
    history = []
    if args.resume:
        ckpt = safe_torch_load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_acc = float(ckpt.get("best_val_acc", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        history = list(ckpt.get("history", []))

    config = vars(args).copy()
    config.update(
        {
            "split_root": str(split_root),
            "metadata_path": str(metadata_path) if metadata_path else None,
            "class_to_idx": train_ds.class_to_idx,
            "class_names": class_names,
            "num_classes": len(class_names),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(device),
            "amp_enabled": amp,
            "num_train": len(train_ds),
            "num_val": len(val_ds),
            "num_test": len(test_ds) if test_ds is not None else 0,
        }
    )
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("Dataset root: {}".format(split_root))
    print("Samples: train={}, val={}, test={}".format(len(train_ds), len(val_ds), len(test_ds) if test_ds is not None else 0))
    print("Classes: {}".format(class_names))
    print("Model: {}, parameters={:,}, trainable={:,}".format(args.arch, total_params, trainable_params))
    show_progress = bool(args.progress and not args.no_progress)
    print("Device: {}, amp={}".format(device, amp))
    print("Progress bars: {}".format("enabled" if show_progress else "disabled"))

    if args.eval_only:
        ckpt = safe_torch_load(args.eval_only, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        loader = test_loader if test_loader is not None else val_loader
        prefix = "test" if test_loader is not None else "val"
        result = evaluate(model, loader, criterion, device, amp, show_progress=show_progress, desc=prefix)
        summary = save_reports(result, out_dir, split_root, snr_map, class_names, prefix=prefix)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    bad_epochs = 0
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, amp, args.grad_clip, show_progress=show_progress, epoch=epoch)
        val_result = evaluate(model, val_loader, criterion, device, amp, show_progress=show_progress, desc="val epoch {}".format(epoch))
        val_loss = float(val_result["loss"])
        val_acc = float(val_result["accuracy"])
        scheduler.step()
        seconds = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": seconds,
        }
        history.append(row)
        save_history(out_dir / "history.csv", history)
        save_curves(history, out_dir / "training_curves.png")
        print(
            "epoch {}/{} train_loss={:.5f} train_acc={:.4f} val_loss={:.5f} val_acc={:.4f} lr={:.6g} time={:.1f}s".format(
                epoch, args.epochs, train_loss, train_acc, val_loss, val_acc, row["lr"], seconds
            )
        )

        last_payload = {
            "epoch": epoch,
            "arch": args.arch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "class_to_idx": train_ds.class_to_idx,
            "idx_to_class": {int(v): k for k, v in train_ds.class_to_idx.items()},
            "class_names": class_names,
            "image_size": args.image_size,
            "mean": args.mean,
            "std": args.std,
            "config": config,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "history": history,
        }
        torch.save(last_payload, out_dir / "last_model.pt")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            bad_epochs = 0
            last_payload["best_val_acc"] = best_val_acc
            last_payload["best_epoch"] = best_epoch
            torch.save(last_payload, out_dir / "best_model.pt")
        else:
            bad_epochs += 1
            if args.early_stopping and bad_epochs >= args.early_stopping:
                print("Early stopping: no validation improvement for {} epochs".format(bad_epochs))
                break

    print("Best validation accuracy: {:.4f} at epoch {}".format(best_val_acc, best_epoch))
    ckpt = safe_torch_load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    final_loader = test_loader if test_loader is not None else val_loader
    final_prefix = "test" if test_loader is not None else "val"
    final_result = evaluate(model, final_loader, criterion, device, amp, show_progress=show_progress, desc=final_prefix)
    summary = save_reports(final_result, out_dir, split_root, snr_map, class_names, prefix=final_prefix)
    summary["evaluated_split"] = final_prefix
    summary["best_val_acc"] = float(best_val_acc)
    summary["best_epoch"] = int(best_epoch)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Final metrics:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
