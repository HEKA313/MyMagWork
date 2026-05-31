#!/usr/bin/env python3
"""
train_classical_rf.py

Classical RF-signal identification branch:
  IQ -> statistical + cyclic features -> SVM / Random Forest / Gradient Boosting

It can train separate classifiers for label/domain/family targets and saves the
feature table, models, metrics, confusion matrices, and per-SNR accuracy.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from classical_features import FeatureConfig, extract_features, load_npz_iq


META_COLUMNS = {"relative_path", "split", "label", "domain", "family", "snr_db", "sample_index", "fs_hz", "num_samples", "profile", "params_json"}


def maybe_tqdm(iterable, enabled: bool, **kwargs):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def load_metadata(data_root: Path) -> pd.DataFrame:
    metadata_path =  data_root / "metadata.csv"
    print(data_root)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")
    df = pd.read_csv(metadata_path)
    required = {"relative_path", "split", "label", "domain", "family", "snr_db"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing columns: {sorted(missing)}")
    return df


def build_feature_config(args: argparse.Namespace, fs: float) -> FeatureConfig:
    lags = tuple(int(v) for v in args.cyclo_lags)
    return FeatureConfig(
        fs=fs,
        num_samples=args.feature_num_samples,
        welch_nperseg=args.welch_nperseg,
        welch_nfft=args.welch_nfft,
        cyclo_enabled=not args.no_cyclo,
        cyclo_alpha_bins=args.cyclo_alpha_bins,
        cyclo_max_alpha=args.cyclo_max_alpha,
        cyclo_lags=lags,
    )


def extract_feature_table(args: argparse.Namespace) -> pd.DataFrame:
    data_root = Path(args.data_root)
    metadata = load_metadata(data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_csv = Path(args.features_csv) if args.features_csv else out_dir / "features.csv"
    config_json = out_dir / "feature_config.json"

    if feature_csv.exists() and not args.recompute_features:
        print(f"Loading cached features: {feature_csv}")
        return pd.read_csv(feature_csv)

    rows: List[Dict[str, object]] = []
    iterator = maybe_tqdm(metadata.iterrows(), args.progress, total=len(metadata), desc="extract features")

    for _, row in iterator:
        rel = str(row["relative_path"])
        path = data_root / rel
        iq, fs_from_file = load_npz_iq(path)
        fs = float(args.fs) if args.fs is not None else float(row.get("fs_hz", fs_from_file))
        cfg = build_feature_config(args, fs)
        feats = extract_features(iq, cfg)
        rows.append({
            "relative_path": rel,
            "split": row["split"],
            "label": row["label"],
            "domain": row["domain"],
            "family": row["family"],
            "snr_db": float(row["snr_db"]),
            **feats,
        })

    feature_df = pd.DataFrame(rows)
    feature_df.to_csv(feature_csv, index=False)
    with config_json.open("w", encoding="utf-8") as f:
        sample_fs = float(args.fs) if args.fs is not None else float(metadata["fs_hz"].iloc[0]) if "fs_hz" in metadata.columns else 1.0
        json.dump(asdict(build_feature_config(args, sample_fs)), f, indent=2)
    print(f"Saved features: {feature_csv}")
    return feature_df


def feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in META_COLUMNS]
    numeric = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
    return numeric


def make_model(model_name: str, args: argparse.Namespace, seed: int):
    if model_name == "svm":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(C=args.svm_c, kernel=args.svm_kernel, gamma=args.svm_gamma, probability=args.svm_probability, class_weight="balanced", random_state=seed)),
        ])
    if model_name == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=args.rf_estimators, max_depth=args.rf_max_depth, min_samples_leaf=args.rf_min_samples_leaf, class_weight="balanced_subsample", n_jobs=args.jobs, random_state=seed)),
        ])
    if model_name in {"gradient_boosting", "hist_gradient_boosting", "hgb"}:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(max_iter=args.gb_max_iter, learning_rate=args.gb_learning_rate, l2_regularization=args.gb_l2, max_leaf_nodes=args.gb_max_leaf_nodes, random_state=seed)),
        ])
    raise ValueError(f"unknown model: {model_name}")


def subsample_for_svm(X: np.ndarray, y: np.ndarray, args: argparse.Namespace, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    max_n = int(args.svm_max_train)
    if max_n <= 0 or X.shape[0] <= max_n:
        return X, y
    try:
        X_sub, _, y_sub, _ = train_test_split(X, y, train_size=max_n, random_state=seed, stratify=y)
    except ValueError:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_n, replace=False)
        X_sub, y_sub = X[idx], y[idx]
    print(f"SVM train subset: {X_sub.shape[0]} / {X.shape[0]} samples")
    return X_sub, y_sub


def get_confidence(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            return np.max(proba, axis=1)
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X)
            scores = np.asarray(scores)
            if scores.ndim == 1:
                return 1.0 / (1.0 + np.exp(-np.abs(scores)))
            # Softmax-like confidence for comparability, not calibrated probability.
            scores = scores - np.max(scores, axis=1, keepdims=True)
            exp = np.exp(scores)
            prob = exp / np.sum(exp, axis=1, keepdims=True)
            return np.max(prob, axis=1)
        except Exception:
            pass
    return np.ones(X.shape[0], dtype=np.float64)


def save_confusion_matrix(cm: np.ndarray, labels: Sequence[str], csv_path: Path, png_path: Path, title: str) -> None:
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(csv_path)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.45), max(6, len(labels) * 0.40)))
        im = ax.imshow(cm, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"Could not save confusion matrix PNG: {exc}")


def evaluate_split(model, X: np.ndarray, y_true: np.ndarray, labels: Sequence[str], split_df: pd.DataFrame, out_prefix: Path, title: str) -> Dict[str, object]:
    y_pred = model.predict(X)
    conf = get_confidence(model, X)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "num_samples": int(len(y_true)),
        "mean_confidence": float(np.mean(conf)) if len(conf) else 0.0,
        "classification_report": classification_report(y_true, y_pred, labels=np.arange(len(labels)), target_names=list(labels), zero_division=0, output_dict=True),
    }

    save_confusion_matrix(cm, labels, out_prefix.with_suffix(".cm_counts.csv"), out_prefix.with_suffix(".cm_counts.png"), title + " counts")
    save_confusion_matrix(cm_norm, labels, out_prefix.with_suffix(".cm_norm.csv"), out_prefix.with_suffix(".cm_norm.png"), title + " normalized")

    pred_df = split_df[["relative_path", "split", "label", "domain", "family", "snr_db"]].copy()
    pred_df["true_encoded"] = y_true
    pred_df["pred_encoded"] = y_pred
    pred_df["true_name"] = [labels[i] for i in y_true]
    pred_df["pred_name"] = [labels[i] for i in y_pred]
    pred_df["confidence"] = conf
    pred_df.to_csv(out_prefix.with_suffix(".predictions.csv"), index=False)

    per_snr = []
    for snr, idx in pred_df.groupby("snr_db").groups.items():
        ids = np.asarray(list(idx), dtype=int)
        # ids are original dataframe indices; convert through local array positions.
    local = pred_df.reset_index(drop=True)
    for snr_db, group in local.groupby("snr_db"):
        per_snr.append({
            "snr_db": float(snr_db),
            "accuracy": float(np.mean(group["true_encoded"].to_numpy() == group["pred_encoded"].to_numpy())),
            "num_samples": int(len(group)),
        })
    pd.DataFrame(per_snr).sort_values("snr_db").to_csv(out_prefix.with_suffix(".per_snr_accuracy.csv"), index=False)
    metrics["per_snr_accuracy"] = per_snr
    return metrics


def save_feature_importance(model, feature_names: Sequence[str], path: Path, top_k: int = 40) -> None:
    try:
        clf = model.named_steps.get("clf") if hasattr(model, "named_steps") else model
        importances = getattr(clf, "feature_importances_", None)
        if importances is None:
            return
        order = np.argsort(importances)[::-1][:top_k]
        pd.DataFrame({
            "feature": [feature_names[i] for i in order],
            "importance": [float(importances[i]) for i in order],
        }).to_csv(path, index=False)
    except Exception as exc:
        print(f"Could not save feature importance: {exc}")


def train_all(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_df = extract_feature_table(args)
    fcols = feature_columns(feature_df)
    if not fcols:
        raise ValueError("No numeric feature columns found")

    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    with (out_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(fcols, f, indent=2)

    summary: Dict[str, object] = {"targets": {}, "feature_count": len(fcols), "num_rows": len(feature_df)}

    for target in args.targets:
        if target not in feature_df.columns:
            raise ValueError(f"target column not found in features: {target}")
        target_dir = out_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)

        le = LabelEncoder()
        le.fit(feature_df[target].astype(str))
        labels = list(le.classes_)
        joblib.dump(le, target_dir / "label_encoder.joblib")

        train_df = feature_df[feature_df["split"] == "train"].copy()
        val_df = feature_df[feature_df["split"] == "val"].copy()
        test_df = feature_df[feature_df["split"] == "test"].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError("train and test splits must be non-empty")

        X_train = train_df[fcols].to_numpy(dtype=np.float64)
        y_train = le.transform(train_df[target].astype(str))
        X_val = val_df[fcols].to_numpy(dtype=np.float64) if len(val_df) else None
        y_val = le.transform(val_df[target].astype(str)) if len(val_df) else None
        X_test = test_df[fcols].to_numpy(dtype=np.float64)
        y_test = le.transform(test_df[target].astype(str))

        summary["targets"][target] = {"labels": labels, "models": {}}

        for model_name in args.models:
            canonical_name = "gradient_boosting" if model_name in {"hgb", "hist_gradient_boosting"} else model_name
            model_dir = target_dir / canonical_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model = make_model(model_name, args, args.seed)
            X_fit, y_fit = X_train, y_train
            if len(np.unique(y_fit)) < 2:
                print(f"Target={target} has one training class; using DummyClassifier for model={canonical_name}")
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", DummyClassifier(strategy="most_frequent")),
                ])
            elif model_name == "svm":
                X_fit, y_fit = subsample_for_svm(X_train, y_train, args, args.seed)

            print(f"Training target={target} model={canonical_name} features={len(fcols)} train={len(y_fit)}")
            t0 = time.time()
            model.fit(X_fit, y_fit)
            train_seconds = time.time() - t0
            joblib.dump({"model": model, "label_encoder": le, "feature_columns": fcols, "target": target}, model_dir / "model.joblib")

            model_metrics: Dict[str, object] = {"train_seconds": float(train_seconds), "num_train_samples_used": int(len(y_fit))}
            if X_val is not None and y_val is not None and len(y_val):
                model_metrics["val"] = evaluate_split(model, X_val, y_val, labels, val_df.reset_index(drop=True), model_dir / "val", f"{target} {canonical_name} val")
            model_metrics["test"] = evaluate_split(model, X_test, y_test, labels, test_df.reset_index(drop=True), model_dir / "test", f"{target} {canonical_name} test")
            save_feature_importance(model, fcols, model_dir / "feature_importance_top.csv")

            with (model_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(to_jsonable(model_metrics), f, indent=2, ensure_ascii=False)
            summary["targets"][target]["models"][canonical_name] = {
                "test_accuracy": model_metrics["test"]["accuracy"],
                "test_macro_f1": model_metrics["test"]["macro_f1"],
                "train_seconds": train_seconds,
            }
            print(f"Done target={target} model={canonical_name}: test_acc={model_metrics['test']['accuracy']:.4f}, test_f1={model_metrics['test']['macro_f1']:.4f}")

    with (out_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2, ensure_ascii=False)
    print(f"Saved summary: {out_dir / 'summary_metrics.json'}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train classical RF classifiers on statistical and cyclic features.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", type=str, required=True, help="Dataset directory from generate_rf_iq_dataset.py")
    p.add_argument("--out-dir", type=str, default="runs_old/classical_rf")
    p.add_argument("--features-csv", type=str, default=None, help="Optional cached feature CSV path")
    p.add_argument("--recompute-features", action="store_true")
    p.add_argument("--targets", nargs="+", default=["label", "domain", "family"], choices=["label", "domain", "family"], help="Targets to train")
    p.add_argument("--models", nargs="+", default=["svm", "random_forest", "gradient_boosting"], choices=["svm", "random_forest", "gradient_boosting", "hist_gradient_boosting", "hgb"], help="Classical ML models")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--progress", action="store_true")

    p.add_argument("--fs", type=float, default=None, help="Override sample rate")
    p.add_argument("--feature-num-samples", type=int, default=1024)
    p.add_argument("--welch-nperseg", type=int, default=256)
    p.add_argument("--welch-nfft", type=int, default=512)
    p.add_argument("--no-cyclo", action="store_true", help="Disable cyclic autocorrelation features")
    p.add_argument("--cyclo-alpha-bins", type=int, default=33)
    p.add_argument("--cyclo-max-alpha", type=float, default=0.5)
    p.add_argument("--cyclo-lags", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16, 32])

    p.add_argument("--svm-kernel", choices=["rbf", "linear", "poly", "sigmoid"], default="rbf")
    p.add_argument("--svm-c", type=float, default=10.0)
    p.add_argument("--svm-gamma", default="scale")
    p.add_argument("--svm-probability", action="store_true", help="Enable calibrated probabilities for SVC; slower")
    p.add_argument("--svm-max-train", type=int, default=15000, help="Subsample training data for SVM; 0 disables cap")

    p.add_argument("--rf-estimators", type=int, default=500)
    p.add_argument("--rf-max-depth", type=int, default=None)
    p.add_argument("--rf-min-samples-leaf", type=int, default=1)

    p.add_argument("--gb-max-iter", type=int, default=300)
    p.add_argument("--gb-learning-rate", type=float, default=0.06)
    p.add_argument("--gb-l2", type=float, default=0.0)
    p.add_argument("--gb-max-leaf-nodes", type=int, default=31)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    train_all(args)


if __name__ == "__main__":
    main()
