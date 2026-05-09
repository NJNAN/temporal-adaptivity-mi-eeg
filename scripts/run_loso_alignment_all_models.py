"""Aligned LOSO comparison for all model families.

This script extends the Riemann-only Euclidean-alignment diagnostic to the neural
baselines. The aligned condition uses unlabeled trials from each subject to
estimate that subject's Euclidean-alignment transform, so standard and aligned
LOSO rankings are reported separately.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class Config:
    subjects: List[int]
    models: List[str]
    variants: List[str]
    epochs: int
    patience: int
    min_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    cfc_hidden_size: int
    lstm_hidden_size: int
    cfc_dt: float
    cfc_tau_init: float
    downsample_factor: int
    device: str
    data_dir: str
    output_dir: str
    standard_loso_dir: str


def load_core(repo_root: Path):
    script_path = repo_root / "scripts" / "run_mi_experiments.py"
    spec = importlib.util.spec_from_file_location("mi_exp_core_loso_alignment_all", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def seed_cache(repo_root: Path, target_cache: Path, fallback_name: str = "revision_loso") -> None:
    source_cache = repo_root / "outputs" / fallback_name / "cache"
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
    return aligned.astype(np.float32)


def load_standard_rows(repo_root: Path, config: Config, module) -> List[Dict[str, object]]:
    path = repo_root / config.standard_loso_dir / "loso_metrics.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for row in df.to_dict(orient="records"):
        model = str(row["model"])
        subject = int(row.get("heldout_subject", row.get("subject")))
        if model not in config.models or subject not in config.subjects:
            continue
        rows.append(
            {
                "variant": "standard",
                "heldout_subject": subject,
                "subject": subject,
                "validation_subject": int(row["validation_subject"]),
                "model": model,
                "model_display": module.get_model_display_name(model),
                "accuracy": float(row["accuracy"]),
                "f1": float(row["f1"]),
                "best_epoch": float(row.get("best_epoch", float("nan"))),
                "best_val_accuracy": float(row.get("best_val_accuracy", float("nan"))),
                "best_val_loss": float(row.get("best_val_loss", float("nan"))),
                "best_c": float(row.get("best_c", float("nan"))),
            }
        )
    return rows


def completed_keys(rows: List[Dict[str, object]]) -> set[tuple[str, int, str]]:
    keys = set()
    for row in rows:
        if pd.isna(row.get("accuracy", np.nan)):
            continue
        keys.add((str(row["variant"]), int(row["heldout_subject"]), str(row["model"])))
    return keys


def summarize(rows: List[Dict[str, object]], module, output_dir: Path) -> tuple[dict, dict, dict]:
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "aligned_loso_metrics.csv", index=False)
    summary = {}
    summary_rows = []
    for (variant, model), group in df.groupby(["variant", "model"]):
        acc = group["accuracy"].to_numpy(dtype=float)
        f1 = group["f1"].to_numpy(dtype=float)
        item = {
            "variant": variant,
            "model": model,
            "model_display": module.get_model_display_name(model),
            "accuracy_mean": float(acc.mean()),
            "accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
            "f1_mean": float(f1.mean()),
            "f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
        }
        summary[f"{variant}:{model}"] = item
        summary_rows.append(item)
    pd.DataFrame(summary_rows).sort_values(["variant", "accuracy_mean"], ascending=[True, False]).to_csv(
        output_dir / "aligned_loso_summary.csv",
        index=False,
    )

    within_variant_tests = {}
    for variant, group in df.groupby("variant"):
        for model_a, model_b in combinations(sorted(group["model"].unique()), 2):
            within_variant_tests[f"{variant}:{model_a}_vs_{model_b}"] = module.paired_test(
                group[["subject", "model", "accuracy"]],
                model_a,
                model_b,
            )
    module.apply_holm_correction(within_variant_tests, p_value_key="p_value", output_key="holm_p_value")

    alignment_tests = {}
    for model in sorted(df["model"].unique()):
        subset = df.loc[df["model"] == model, ["subject", "variant", "accuracy"]].rename(columns={"variant": "model"})
        if {"standard", "euclidean_alignment"}.issubset(set(subset["model"])):
            alignment_tests[f"{model}:standard_vs_euclidean_alignment"] = module.paired_test(
                subset,
                "standard",
                "euclidean_alignment",
            )
    module.apply_holm_correction(alignment_tests, p_value_key="p_value", output_key="holm_p_value")

    pd.DataFrame([{"comparison": key, **value} for key, value in within_variant_tests.items()]).to_csv(
        output_dir / "aligned_loso_within_variant_stats.csv",
        index=False,
    )
    pd.DataFrame([{"comparison": key, **value} for key, value in alignment_tests.items()]).to_csv(
        output_dir / "aligned_loso_alignment_stats.csv",
        index=False,
    )
    return summary, within_variant_tests, alignment_tests


def run(config: Config) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    module = load_core(repo_root)
    module.seed_everything(config.seed)
    device = module.get_device(config.device)
    output_dir = repo_root / config.output_dir
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_cache(repo_root, cache_dir)
    metrics_path = output_dir / "aligned_loso_metrics.csv"

    rows = pd.read_csv(metrics_path).to_dict(orient="records") if metrics_path.exists() else []
    if "standard" in config.variants:
        existing_standard = completed_keys(rows)
        for row in load_standard_rows(repo_root, config, module):
            key = ("standard", int(row["heldout_subject"]), str(row["model"]))
            if key not in existing_standard:
                rows.append(row)
                existing_standard.add(key)
        if rows:
            pd.DataFrame(rows).to_csv(metrics_path, index=False)

    x_all, y_all, subject_ids = stack_subjects(module, config.subjects, repo_root / config.data_dir, cache_dir, config.downsample_factor)
    n_channels = x_all.shape[1]
    n_samples = x_all.shape[2]
    parameter_counts = {
        model: module.get_parameter_count(model, n_channels, n_samples, len(module.LABEL_ORDER), config.cfc_hidden_size, config.lstm_hidden_size, config.cfc_dt, config.cfc_tau_init)
        for model in config.models
    }
    completed = completed_keys(rows)
    variants_to_train = [variant for variant in config.variants if variant != "standard"]
    total_runs = len(config.subjects) * len(config.models) * len(variants_to_train)
    run_index = sum(1 for key in completed if key[0] != "standard")

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
        y_train = y_all[train_mask]
        y_val = y_all[val_mask]
        y_test = y_all[test_mask]
        subject_train = subject_ids[train_mask]
        subject_val = subject_ids[val_mask]
        subject_test = subject_ids[test_mask]
        variant_data = {
            "euclidean_alignment": (
                euclidean_align_by_subject(x_train, subject_train),
                euclidean_align_by_subject(x_val, subject_val),
                euclidean_align_by_subject(x_test, subject_test),
            )
        }
        for variant in variants_to_train:
            x_train_v, x_val_v, x_test_v = variant_data[variant]
            train_loader = module.build_loader(x_train_v, y_train, config.batch_size, True, device)
            val_loader = module.build_loader(x_val_v, y_val, config.batch_size, False, device)
            test_loader = module.build_loader(x_test_v, y_test, config.batch_size, False, device)
            for model_name in config.models:
                if (variant, test_subject, model_name) in completed:
                    continue
                run_index += 1
                module.seed_everything(config.seed + test_subject * 100 + config.models.index(model_name))
                print(f"[{run_index}/{total_runs}] variant={variant} heldout_subject={test_subject} model={model_name}", flush=True)
                if module.is_classical_model(model_name):
                    fit_info = module.fit_riemann_tslr(x_train_v, y_train, x_val_v, y_val)
                    metrics = module.evaluate_classical_model(fit_info["model"], x_test_v, y_test)
                else:
                    model = module.build_model(
                        model_name,
                        n_channels,
                        n_samples,
                        len(module.LABEL_ORDER),
                        config.cfc_hidden_size,
                        config.lstm_hidden_size,
                        config.cfc_dt,
                        config.cfc_tau_init,
                    )
                    fit_info = module.train_one_model(model, train_loader, val_loader, device, config.epochs, config.patience, config.min_epochs, config.learning_rate, config.weight_decay)
                    metrics = module.evaluate_model(model, test_loader, device)
                rows.append(
                    {
                        "variant": variant,
                        "heldout_subject": test_subject,
                        "subject": test_subject,
                        "validation_subject": int(val_subject),
                        "model": model_name,
                        "model_display": module.get_model_display_name(model_name),
                        "accuracy": metrics["accuracy"],
                        "f1": metrics["f1"],
                        "best_epoch": fit_info["best_epoch"],
                        "best_val_accuracy": fit_info["best_val_accuracy"],
                        "best_val_loss": fit_info["best_val_loss"],
                        "best_c": float(fit_info.get("best_c", float("nan"))),
                    }
                )
                completed.add((variant, test_subject, model_name))
                pd.DataFrame(rows).to_csv(metrics_path, index=False)

    summary, within_tests, alignment_tests = summarize(rows, module, output_dir)
    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "leave-one-subject-out standard vs unsupervised Euclidean alignment",
        "config": asdict(config),
        "parameter_counts": parameter_counts,
        "summary": summary,
        "within_variant_tests": within_tests,
        "alignment_tests": alignment_tests,
        "note": "Euclidean alignment uses unlabeled trials from each subject, including the held-out subject, and should be interpreted as a test-time unsupervised alignment condition.",
    }
    (output_dir / "aligned_loso_results_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Run all-model LOSO with Euclidean alignment.")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--models", nargs="*", default=["shallow_convnet", "riemann_tslr", "eegnet", "mi_mamba", "tiny_transformer", "cfc", "lstm"])
    parser.add_argument("--variants", nargs="*", default=["standard", "euclidean_alignment"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--cfc-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--cfc-dt", type=float, default=1.0)
    parser.add_argument("--cfc-tau-init", type=float, default=1.0)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/revision_loso_alignment_all")
    parser.add_argument("--standard-loso-dir", default="outputs/revision_loso")
    return Config(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
