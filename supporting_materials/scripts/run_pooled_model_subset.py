from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def load_core_module(repo_root: Path):
    script_path = repo_root / "scripts" / "run_mi_experiments.py"
    spec = importlib.util.spec_from_file_location("mi_exp_core", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class PooledSubsetConfig:
    subjects: List[int]
    models: List[str]
    num_folds: int
    epochs: int
    patience: int
    min_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    val_fraction: float
    seed: int
    cfc_hidden_size: int
    lstm_hidden_size: int
    downsample_factor: int
    device: str
    data_dir: str
    output_dir: str


def run_pooled_subset(config: PooledSubsetConfig) -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    module = load_core_module(repo_root)
    module.seed_everything(config.seed)
    device = module.get_device(config.device)

    data_dir = repo_root / config.data_dir
    output_dir = repo_root / config.output_dir
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    module.prepare_subject_cache(config.subjects, data_dir, cache_dir)

    fold_rows = []
    parameter_counts: Dict[str, int] = {}

    total_runs = len(config.subjects) * config.num_folds * len(config.models)
    run_index = 0

    for subject in config.subjects:
        X_raw, y = module.load_subject_data(subject=subject, data_dir=data_dir, cache_dir=cache_dir)
        X_raw = module.downsample_trials(X_raw, config.downsample_factor)
        splitter = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed + subject)
        for fold_idx, (train_val_idx, test_idx) in enumerate(splitter.split(X_raw, y), start=1):
            X_train_val = X_raw[train_val_idx]
            y_train_val = y[train_val_idx]
            inner_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=config.val_fraction,
                random_state=config.seed + subject * 10 + fold_idx,
            )
            train_inner_idx, val_inner_idx = next(inner_splitter.split(X_train_val, y_train_val))
            X_train = X_train_val[train_inner_idx]
            y_train = y_train_val[train_inner_idx]
            X_val = X_train_val[val_inner_idx]
            y_val = y_train_val[val_inner_idx]
            X_test = X_raw[test_idx]
            y_test = y[test_idx]

            mean, std = module.compute_standardizer(X_train)
            X_train = module.apply_standardizer(X_train, mean, std)
            X_val = module.apply_standardizer(X_val, mean, std)
            X_test = module.apply_standardizer(X_test, mean, std)

            n_channels = X_train.shape[1]
            n_samples = X_train.shape[2]
            if not parameter_counts:
                for model_name in config.models:
                    probe_model = module.build_model(
                        model_name=model_name,
                        n_channels=n_channels,
                        n_samples=n_samples,
                        n_classes=len(module.LABEL_ORDER),
                        cfc_hidden_size=config.cfc_hidden_size,
                        lstm_hidden_size=config.lstm_hidden_size,
                    )
                    parameter_counts[model_name] = module.count_parameters(probe_model)

            train_loader = module.build_loader(X_train, y_train, config.batch_size, True, device)
            val_loader = module.build_loader(X_val, y_val, config.batch_size, False, device)
            test_loader = module.build_loader(X_test, y_test, config.batch_size, False, device)

            for model_name in config.models:
                run_index += 1
                model_seed = config.seed + subject * 100 + fold_idx * 10 + config.models.index(model_name)
                module.seed_everything(model_seed)
                model = module.build_model(
                    model_name=model_name,
                    n_channels=n_channels,
                    n_samples=n_samples,
                    n_classes=len(module.LABEL_ORDER),
                    cfc_hidden_size=config.cfc_hidden_size,
                    lstm_hidden_size=config.lstm_hidden_size,
                )
                print(
                    f"[{run_index}/{total_runs}] subject={subject} fold={fold_idx}/{config.num_folds} "
                    f"model={model_name} device={device.type}",
                    flush=True,
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
                fold_rows.append(
                    {
                        "subject": subject,
                        "fold": fold_idx,
                        "model": model_name,
                        "accuracy": metrics["accuracy"],
                        "f1": metrics["f1"],
                        "best_epoch": fit_info["best_epoch"],
                        "best_val_accuracy": fit_info["best_val_accuracy"],
                        "best_val_loss": fit_info["best_val_loss"],
                    }
                )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    subject_summary_df, summary = module.summarize_subject_metrics(fold_rows)
    subject_summary_df.to_csv(output_dir / "subject_summary.csv", index=False)
    subject_stability = module.save_subject_accuracy_artifacts(subject_summary_df, output_dir)

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "config": asdict(config),
        "parameter_counts": parameter_counts,
        "summary": summary,
        "subject_stability": subject_stability,
    }
    (output_dir / "results_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> PooledSubsetConfig:
    parser = argparse.ArgumentParser(description="Run pooled subject-dependent CV for a subset of MI-EEG models.")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--models", nargs="*", default=["shallow_convnet"])
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260318)
    parser.add_argument("--cfc-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/pool_extra")
    args = parser.parse_args()
    return PooledSubsetConfig(
        subjects=list(args.subjects),
        models=list(args.models),
        num_folds=args.num_folds,
        epochs=args.epochs,
        patience=args.patience,
        min_epochs=args.min_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        seed=args.seed,
        cfc_hidden_size=args.cfc_hidden_size,
        lstm_hidden_size=args.lstm_hidden_size,
        downsample_factor=args.downsample_factor,
        device=args.device,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    run_pooled_subset(parse_args())
