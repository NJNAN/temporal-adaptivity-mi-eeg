"""Check whether LOSO Riemann-TSLR is limited by cross-subject alignment.

This diagnostic compares the existing train-only standardized Riemann-TSLR LOSO
pipeline with an unsupervised Euclidean Alignment (EA) variant. EA estimates one
reference covariance per subject from that subject's unlabeled trials and applies
the inverse square root before fitting the same covariance -> tangent-space ->
logistic-regression pipeline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class AlignmentConfig:
    subjects: List[int]
    seed: int
    downsample_factor: int
    data_dir: str
    output_dir: str


def load_core(repo_root: Path):
    script_path = repo_root / "scripts" / "run_mi_experiments.py"
    spec = importlib.util.spec_from_file_location("mi_exp_core_loso_alignment", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def seed_cache(repo_root: Path, target_cache: Path) -> None:
    source_cache = repo_root / "outputs" / "revision_loso" / "cache"
    if not source_cache.exists() or source_cache.resolve() == target_cache.resolve():
        return
    target_cache.mkdir(parents=True, exist_ok=True)
    for source_file in source_cache.glob("subject_*.npz"):
        target_file = target_cache / source_file.name
        if not target_file.exists():
            shutil.copy2(source_file, target_file)


def stack_subjects(module, subjects: List[int], data_dir: Path, cache_dir: Path, downsample_factor: int):
    rows = []
    labels = []
    subject_ids = []
    for subject in subjects:
        x_subject, y_subject = module.load_subject_data(subject, data_dir, cache_dir)
        x_subject = module.downsample_trials(x_subject, downsample_factor)
        rows.append(x_subject)
        labels.append(y_subject)
        subject_ids.extend([subject] * len(y_subject))
    return np.concatenate(rows), np.concatenate(labels), np.asarray(subject_ids, dtype=np.int64)


def inverse_sqrt(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, eps)
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def mean_covariance(trials: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    covs = []
    for trial in trials:
        centered = trial - trial.mean(axis=1, keepdims=True)
        cov = centered @ centered.T / max(centered.shape[1] - 1, 1)
        covs.append(cov)
    mean_cov = np.mean(covs, axis=0)
    mean_cov = 0.5 * (mean_cov + mean_cov.T)
    mean_cov += eps * np.eye(mean_cov.shape[0])
    return mean_cov


def euclidean_align_by_subject(x: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
    aligned = np.empty_like(x)
    for subject in np.unique(subject_ids):
        mask = subject_ids == subject
        transform = inverse_sqrt(mean_covariance(x[mask]))
        aligned[mask] = np.einsum("ij,tjk->tik", transform, x[mask])
    return aligned


def summarize(rows: List[Dict[str, object]], module, output_dir: Path) -> dict:
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "riemann_alignment_loso_metrics.csv", index=False)
    summary_rows = []
    summary = {}
    for variant, group in metrics_df.groupby("variant"):
        acc = group["accuracy"].to_numpy(dtype=float)
        f1 = group["f1"].to_numpy(dtype=float)
        summary[variant] = {
            "accuracy_mean": float(acc.mean()),
            "accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
            "f1_mean": float(f1.mean()),
            "f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
        }
        summary_rows.append({"variant": variant, **summary[variant]})
    pd.DataFrame(summary_rows).to_csv(output_dir / "riemann_alignment_loso_summary.csv", index=False)
    stats = {}
    if {"standard", "euclidean_alignment"}.issubset(set(metrics_df["variant"])):
        paired_df = metrics_df[["subject", "variant", "accuracy"]].rename(columns={"variant": "model"})
        stats["standard_vs_euclidean_alignment"] = module.paired_test(
            paired_df,
            "standard",
            "euclidean_alignment",
        )
    pd.DataFrame([{"comparison": key, **value} for key, value in stats.items()]).to_csv(
        output_dir / "riemann_alignment_loso_stats.csv",
        index=False,
    )
    return {"summary": summary, "stats": stats}


def run(config: AlignmentConfig) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    module = load_core(repo_root)
    output_dir = repo_root / config.output_dir
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_cache(repo_root, cache_dir)

    x_all, y_all, subject_ids = stack_subjects(
        module,
        config.subjects,
        repo_root / config.data_dir,
        cache_dir,
        config.downsample_factor,
    )
    rows = []
    assignments = []
    for test_subject in config.subjects:
        train_subjects = [subject for subject in config.subjects if subject != test_subject]
        val_subject = train_subjects[(test_subject + config.seed) % len(train_subjects)]
        train_mask = np.isin(subject_ids, [subject for subject in train_subjects if subject != val_subject])
        val_mask = subject_ids == val_subject
        test_mask = subject_ids == test_subject
        mean, std = module.compute_standardizer(x_all[train_mask])
        x_train = module.apply_standardizer(x_all[train_mask], mean, std)
        x_val = module.apply_standardizer(x_all[val_mask], mean, std)
        x_test = module.apply_standardizer(x_all[test_mask], mean, std)
        subject_train = subject_ids[train_mask]
        subject_val = subject_ids[val_mask]
        subject_test = subject_ids[test_mask]
        assignments.append(
            {
                "heldout_subject": test_subject,
                "validation_subject": int(val_subject),
                "train_subjects": " ".join(str(subject) for subject in train_subjects if subject != val_subject),
                "n_train": int(train_mask.sum()),
                "n_val": int(val_mask.sum()),
                "n_test": int(test_mask.sum()),
            }
        )

        variants = {
            "standard": (x_train, x_val, x_test),
            "euclidean_alignment": (
                euclidean_align_by_subject(x_train, subject_train),
                euclidean_align_by_subject(x_val, subject_val),
                euclidean_align_by_subject(x_test, subject_test),
            ),
        }
        for variant, (x_train_variant, x_val_variant, x_test_variant) in variants.items():
            fit_info = module.fit_riemann_tslr(x_train_variant, y_all[train_mask], x_val_variant, y_all[val_mask])
            metrics = module.evaluate_classical_model(fit_info["model"], x_test_variant, y_all[test_mask])
            rows.append(
                {
                    "heldout_subject": test_subject,
                    "subject": test_subject,
                    "validation_subject": int(val_subject),
                    "variant": variant,
                    "model": "riemann_tslr",
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                    "best_c": float(fit_info["best_c"]),
                    "best_val_accuracy": fit_info["best_val_accuracy"],
                }
            )
            print(f"heldout_subject={test_subject} variant={variant} acc={metrics['accuracy']:.3f}", flush=True)

    pd.DataFrame(assignments).to_csv(output_dir / "riemann_alignment_loso_assignments.csv", index=False)
    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": asdict(config),
        "note": (
            "euclidean_alignment uses unlabeled trials from each subject, including the held-out subject, "
            "as a test-time unsupervised alignment diagnostic."
        ),
        **summarize(rows, module, output_dir),
    }
    (output_dir / "riemann_alignment_loso_results_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def parse_args() -> AlignmentConfig:
    parser = argparse.ArgumentParser(description="Run LOSO Riemann alignment diagnostic.")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/revision_loso_riemann_alignment")
    return AlignmentConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
