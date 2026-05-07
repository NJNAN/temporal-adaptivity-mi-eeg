"""Leave-one-subject-out cross-subject MI-EEG benchmark.

This revision script answers the reviewer request for a cross-subject protocol.
It reuses the core preprocessing, model definitions, training loop, and paired
statistics from `run_mi_experiments.py`.
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
import torch


@dataclass
class LOSOConfig:
    subjects: List[int]
    models: List[str]
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


def load_core(repo_root: Path):
    script_path = repo_root / "scripts" / "run_mi_experiments.py"
    spec = importlib.util.spec_from_file_location("mi_exp_core_loso", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def seed_cache(repo_root: Path, target_cache: Path, fallback_name: str) -> None:
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
    return (
        np.concatenate(rows, axis=0),
        np.concatenate(labels, axis=0),
        np.asarray(subject_ids, dtype=np.int64),
    )


def summarize(rows: List[Dict[str, object]], module, output_dir: Path, models: List[str]):
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "loso_metrics.csv", index=False)
    summary_rows = []
    summary = {}
    for model_name, group in metrics_df.groupby("model"):
        acc = group["accuracy"].to_numpy(dtype=float)
        f1 = group["f1"].to_numpy(dtype=float)
        summary[model_name] = {
            "accuracy_mean": float(acc.mean()),
            "accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
            "f1_mean": float(f1.mean()),
            "f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
        }
        summary_rows.append(
            {
                "model": model_name,
                "model_display": module.get_model_display_name(model_name),
                **summary[model_name],
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "loso_subject_summary.csv", index=False)

    stat_tests = {}
    for model_a, model_b in combinations(models, 2):
        if {model_a, model_b}.issubset(set(metrics_df["model"])):
            stat_tests[f"{model_a}_vs_{model_b}"] = module.paired_test(metrics_df, model_a, model_b)
    module.apply_holm_correction(stat_tests, p_value_key="p_value", output_key="holm_p_value")
    module.apply_holm_correction(stat_tests, p_value_key="wilcoxon_p_value", output_key="wilcoxon_holm_p_value")
    pd.DataFrame([{"comparison": key, **value} for key, value in stat_tests.items()]).to_csv(
        output_dir / "loso_stats.csv",
        index=False,
    )
    return summary, stat_tests


def run_loso(config: LOSOConfig) -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    module = load_core(repo_root)
    module.seed_everything(config.seed)
    device = module.get_device(config.device)
    data_dir = repo_root / config.data_dir
    output_dir = repo_root / config.output_dir
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_cache(repo_root, cache_dir, "bspc_pooled")

    x_all, y_all, subject_ids = stack_subjects(
        module,
        config.subjects,
        data_dir,
        cache_dir,
        config.downsample_factor,
    )
    n_channels = x_all.shape[1]
    n_samples = x_all.shape[2]
    parameter_counts = {
        model_name: module.get_parameter_count(
            model_name=model_name,
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=len(module.LABEL_ORDER),
            cfc_hidden_size=config.cfc_hidden_size,
            lstm_hidden_size=config.lstm_hidden_size,
            cfc_dt=config.cfc_dt,
            cfc_tau_init=config.cfc_tau_init,
        )
        for model_name in config.models
    }

    rows: List[Dict[str, object]] = []
    assignments = []
    total_runs = len(config.subjects) * len(config.models)
    run_index = 0
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
        assignments.append(
            {
                "test_subject": test_subject,
                "validation_subject": int(val_subject),
                "train_subjects": " ".join(str(subject) for subject in train_subjects if subject != val_subject),
                "n_train": int(train_mask.sum()),
                "n_val": int(val_mask.sum()),
                "n_test": int(test_mask.sum()),
            }
        )
        train_loader = module.build_loader(x_train, y_train, config.batch_size, True, device)
        val_loader = module.build_loader(x_val, y_val, config.batch_size, False, device)
        test_loader = module.build_loader(x_test, y_test, config.batch_size, False, device)
        for model_name in config.models:
            run_index += 1
            model_seed = config.seed + test_subject * 100 + config.models.index(model_name)
            module.seed_everything(model_seed)
            print(f"[{run_index}/{total_runs}] heldout_subject={test_subject} model={model_name}", flush=True)
            if module.is_classical_model(model_name):
                fit_info = module.fit_riemann_tslr(x_train, y_train, x_val, y_val)
                metrics = module.evaluate_classical_model(fit_info["model"], x_test, y_test)
            else:
                model = module.build_model(
                    model_name=model_name,
                    n_channels=n_channels,
                    n_samples=n_samples,
                    n_classes=len(module.LABEL_ORDER),
                    cfc_hidden_size=config.cfc_hidden_size,
                    lstm_hidden_size=config.lstm_hidden_size,
                    cfc_dt=config.cfc_dt,
                    cfc_tau_init=config.cfc_tau_init,
                )
                fit_info = module.train_one_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    epochs=config.epochs,
                    patience=config.patience,
                    min_epochs=config.min_epochs,
                    learning_rate=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
                metrics = module.evaluate_model(model, test_loader, device)
            rows.append(
                {
                    "test_subject": test_subject,
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
            pd.DataFrame(rows).to_csv(output_dir / "loso_metrics.csv", index=False)

    pd.DataFrame(assignments).to_csv(output_dir / "loso_assignments.csv", index=False)
    summary, stat_tests = summarize(rows, module, output_dir, config.models)
    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "leave-one-subject-out",
        "config": asdict(config),
        "parameter_counts": parameter_counts,
        "summary": summary,
        "stat_tests": stat_tests,
    }
    (output_dir / "loso_results_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> LOSOConfig:
    parser = argparse.ArgumentParser(description="Run leave-one-subject-out MI-EEG benchmark.")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument(
        "--models",
        nargs="*",
        default=["shallow_convnet", "riemann_tslr", "eegnet", "mi_mamba", "tiny_transformer", "cfc", "lstm"],
    )
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
    parser.add_argument("--output-dir", default="outputs/revision_loso")
    args = parser.parse_args()
    return LOSOConfig(**vars(args))


if __name__ == "__main__":
    run_loso(parse_args())
