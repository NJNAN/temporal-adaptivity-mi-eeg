"""grouped-pooled 控制实验。

对应论文：
1. 主结果之外的 grouped pooled control。
2. 用 session+run 分组来验证 pooled trial-level 结果是否被组内相似性夸大。

这是论文“boundary study”里用来加固结论的重要对照，而不是新的主协议。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def load_core_module(repo_root: Path):
    """加载主实验脚本，确保 grouped control 与正文模型实现完全一致。"""
    script_path = repo_root / "scripts" / "run_mi_experiments.py"
    spec = importlib.util.spec_from_file_location("mi_exp_core_grouped", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class GroupedControlConfig:
    """grouped-pooled 控制实验配置。"""
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


def load_subject_with_groups(module, subject: int, data_dir: Path, cache_dir: Path):
    """载入带 session 与 run 标记的数据。

    对应论文中“session-plus-run grouping”这一控制协议。
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"subject_{subject:02d}.npz"
    if cache_file.exists():
        cached = np.load(cache_file, allow_pickle=True)
        return (
            cached["X"].astype(np.float32),
            cached["y"].astype(np.int64),
            cached["session"].astype(str),
            cached["run"].astype(str),
        )

    module.ensure_mne_path(data_dir)
    dataset = module.BNCI2014Dataset()
    paradigm = module.MotorImagery(n_classes=4, fmin=8, fmax=30, tmin=0.0, tmax=4.0)
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
    y_int = np.asarray([module.LABEL_TO_INDEX[str(label)] for label in y], dtype=np.int64)
    sessions = metadata["session"].astype(str).to_numpy()
    runs = metadata["run"].astype(str).to_numpy()
    X = X.astype(np.float32)
    np.savez_compressed(cache_file, X=X, y=y_int, session=sessions, run=runs)
    return X, y_int, sessions, runs


def run_grouped_control(config: GroupedControlConfig) -> Dict[str, object]:
    """运行 grouped-pooled control。

    该实验直接支撑论文里“pooled 结果不是由 trial shuffle 泄漏式乐观估计造成”的说法。
    """
    repo_root = Path(__file__).resolve().parents[1]
    module = load_core_module(repo_root)
    module.seed_everything(config.seed)
    device = module.get_device(config.device)

    data_dir = repo_root / config.data_dir
    output_dir = repo_root / config.output_dir
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics_path = output_dir / "fold_metrics.csv"
    prediction_path = output_dir / "predictions.csv"
    assignment_path = output_dir / "grouped_fold_assignments.csv"

    fold_rows: List[Dict[str, object]] = []
    if fold_metrics_path.exists():
        fold_rows = pd.read_csv(fold_metrics_path).to_dict(orient="records")
    prediction_rows: List[Dict[str, object]] = []
    if prediction_path.exists():
        prediction_rows = pd.read_csv(prediction_path).to_dict(orient="records")
    parameter_counts: Dict[str, int] = {}
    assignment_rows: List[Dict[str, object]] = []
    if assignment_path.exists():
        assignment_rows = pd.read_csv(assignment_path).to_dict(orient="records")
    assignment_keys = {
        (int(row["subject"]), int(row["fold"]), int(row["trial_index"]))
        for row in assignment_rows
    }

    completed_runs = {
        (int(row["subject"]), int(row["fold"]), str(row["model"]))
        for row in fold_rows
    }

    total_runs = len(config.subjects) * config.num_folds * len(config.models)
    run_index = 0

    for subject in config.subjects:
        X_raw, y, sessions, runs = load_subject_with_groups(module, subject, data_dir, cache_dir)
        X_raw = module.downsample_trials(X_raw, config.downsample_factor)
        groups = np.asarray([f"{session}_run{run}" for session, run in zip(sessions, runs)], dtype=object)

        n_trials, n_channels, n_samples = X_raw.shape
        if not parameter_counts:
            for model_name in config.models:
                parameter_counts[model_name] = module.get_parameter_count(
                    model_name=model_name,
                    n_channels=n_channels,
                    n_samples=n_samples,
                    n_classes=len(module.LABEL_ORDER),
                    cfc_hidden_size=config.cfc_hidden_size,
                    lstm_hidden_size=config.lstm_hidden_size,
                )

        splitter = GroupKFold(n_splits=config.num_folds)
        for fold_idx, (train_val_idx, test_idx) in enumerate(splitter.split(X_raw, y, groups=groups), start=1):
            inner_splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=config.val_fraction,
                random_state=config.seed + subject * 10 + fold_idx,
            )
            inner_train_idx, val_idx = next(
                inner_splitter.split(X_raw[train_val_idx], y[train_val_idx], groups=groups[train_val_idx])
            )
            train_idx = train_val_idx[inner_train_idx]
            val_idx = train_val_idx[val_idx]

            mean, std = module.compute_standardizer(X_raw[train_idx])
            X_train = module.apply_standardizer(X_raw[train_idx], mean, std)
            X_val = module.apply_standardizer(X_raw[val_idx], mean, std)
            X_test = module.apply_standardizer(X_raw[test_idx], mean, std)

            # 记录每个 trial 在 grouped control 中的去向，用于论文复现材料里的 fold assignment。
            split_map = {}
            for idx in train_idx:
                split_map[int(idx)] = "train"
            for idx in val_idx:
                split_map[int(idx)] = "val"
            for idx in test_idx:
                split_map[int(idx)] = "test"
            for trial_index in range(len(y)):
                assignment_key = (subject, fold_idx, trial_index)
                if assignment_key not in assignment_keys:
                    assignment_rows.append(
                        {
                            "subject": subject,
                            "fold": fold_idx,
                            "trial_index": trial_index,
                            "session": sessions[trial_index],
                            "run": runs[trial_index],
                            "group": groups[trial_index],
                            "label_index": int(y[trial_index]),
                            "label_name": module.INDEX_TO_LABEL[int(y[trial_index])],
                            "split": split_map[trial_index],
                        }
                    )
                    assignment_keys.add(assignment_key)

            train_loader = module.build_loader(X_train, y[train_idx], config.batch_size, True, device)
            val_loader = module.build_loader(X_val, y[val_idx], config.batch_size, False, device)
            test_loader = module.build_loader(X_test, y[test_idx], config.batch_size, False, device)

            for model_name in config.models:
                run_index += 1
                # 这里的种子对应 grouped control 单次运行，不是论文里额外的 multi-seed 稳定性分析。
                model_seed = config.seed + subject * 100 + fold_idx * 10 + config.models.index(model_name)
                module.seed_everything(model_seed)
                print(
                    f"[{run_index}/{total_runs}] subject={subject} fold={fold_idx}/{config.num_folds} "
                    f"model={model_name} device={device.type}",
                    flush=True,
                )
                run_key = (subject, fold_idx, model_name)
                if run_key in completed_runs:
                    print(
                        f"[{run_index}/{total_runs}] subject={subject} fold={fold_idx}/{config.num_folds} "
                        f"model={model_name} already completed, skipping",
                        flush=True,
                    )
                    continue
                if module.is_classical_model(model_name):
                    fit_info = module.fit_riemann_tslr(
                        x_train=X_train,
                        y_train=y[train_idx],
                        x_val=X_val,
                        y_val=y[val_idx],
                    )
                    metrics = module.evaluate_classical_model(
                        model=fit_info["model"],
                        x=X_test,
                        y=y[test_idx],
                        return_predictions=True,
                    )
                else:
                    model = module.build_model(
                        model_name=model_name,
                        n_channels=n_channels,
                        n_samples=n_samples,
                        n_classes=len(module.LABEL_ORDER),
                        cfc_hidden_size=config.cfc_hidden_size,
                        lstm_hidden_size=config.lstm_hidden_size,
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
                    metrics = module.evaluate_model(model, test_loader, device, return_predictions=True)

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
                        "best_c": float(fit_info.get("best_c", float("nan"))),
                    }
                )
                module.add_prediction_rows(
                    prediction_rows,
                    subject=subject,
                    model=model_name,
                    y_true=metrics["y_true"],
                    y_pred=metrics["y_pred"],
                    fold=fold_idx,
                    protocol="grouped_pooled",
                )
                completed_runs.add(run_key)
                pd.DataFrame(fold_rows).to_csv(fold_metrics_path, index=False)
                pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)
                pd.DataFrame(assignment_rows).to_csv(assignment_path, index=False)

    fold_df = pd.DataFrame(fold_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    assignment_df = pd.DataFrame(assignment_rows)
    fold_df.to_csv(fold_metrics_path, index=False)
    prediction_df.to_csv(prediction_path, index=False)
    assignment_df.to_csv(assignment_path, index=False)

    subject_summary_df, summary = module.summarize_subject_metrics(fold_rows)
    subject_summary_df.to_csv(output_dir / "subject_summary.csv", index=False)
    subject_stability = module.save_subject_accuracy_artifacts(subject_summary_df, output_dir)
    module.save_prediction_artifacts(prediction_df, output_dir)

    stat_tests = {
        f"{model_a}_vs_{model_b}": module.paired_test(subject_summary_df, model_a, model_b)
        for model_a, model_b in combinations(config.models, 2)
    }
    module.apply_holm_correction(stat_tests, p_value_key="p_value", output_key="holm_p_value")
    module.apply_holm_correction(stat_tests, p_value_key="wilcoxon_p_value", output_key="wilcoxon_holm_p_value")
    pd.DataFrame([{"comparison": key, **value} for key, value in stat_tests.items()]).to_csv(
        output_dir / "stat_tests.csv",
        index=False,
    )

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "config": asdict(config),
        "parameter_counts": parameter_counts,
        "summary": summary,
        "subject_stability": subject_stability,
        "stat_tests": stat_tests,
        "group_definition": "session+run",
    }
    (output_dir / "results_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> GroupedControlConfig:
    """解析命令行参数，对应论文 grouped-pooled 补充控制的可复现实验入口。"""
    parser = argparse.ArgumentParser(description="Run grouped pooled MI-EEG control grouped by session+run.")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument(
        "--models",
        nargs="*",
        default=["shallow_convnet", "riemann_tslr", "eegnet", "tiny_transformer", "hybrid_cfc", "cfc", "lstm"],
    )
    parser.add_argument("--num-folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260321)
    parser.add_argument("--cfc-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/bspc_grouped_cv")
    args = parser.parse_args()
    return GroupedControlConfig(
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
    run_grouped_control(parse_args())
