"""BNCI2014-004 辅助 sanity-check 实验。

对应论文：
1. Discussion/limitations 中“主结论严格限定在 IV-2a”的辅助背景说明。
2. Supporting materials 中的小数据集补充结果。

该脚本不改变主论文结论，只用于回答“结论是否只依赖单一数据集”这一审稿疑问。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from moabb.paradigms import MotorImagery
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from moabb.datasets import BNCI2014_004 as BNCI2014_004_Dataset
except ImportError:
    from moabb.datasets import BNCI2014004 as BNCI2014_004_Dataset


def load_module(path: Path, module_name: str):
    """动态加载主实验脚本，复用与论文主文一致的模型和训练逻辑。"""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class AuxConfig:
    """辅助数据集实验配置。

    对应论文补充 sanity check 的训练协议定义。
    """
    subjects: List[int]
    models: List[str]
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


def derive_seed(base_seed: int, *tokens: int) -> int:
    """派生可复现随机种子，保证辅助实验与主实验一样可追踪。"""
    seed_sequence = np.random.SeedSequence([base_seed, *tokens])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def clear_subject_downloads(subject: int, data_dir: Path) -> None:
    """清理下载失败的缓存，避免 MOABB 在辅助数据集上留下损坏文件。"""
    download_dir = data_dir / "MNE-bnci-data" / "database" / "data-sets" / "004-2014"
    if not download_dir.exists():
        return
    prefix = f"B{subject:02d}"
    patterns = [
        f"{prefix}T.mat",
        f"{prefix}T.mat.*",
        f"{prefix}E.mat",
        f"{prefix}E.mat.*",
    ]
    for pattern in patterns:
        for path in download_dir.glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                continue


def load_subject_data(subject: int, data_dir: Path, cache_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """载入 BNCI2014-004 单被试数据。

    对应论文中“外部小数据集仅作为补充验证”的数据准备步骤。
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"subject_{subject:02d}.npz"
    if cache_file.exists():
        cached = np.load(cache_file, allow_pickle=True)
        return (
            cached["X"].astype(np.float32),
            cached["y"].astype(np.int64),
            cached["sessions"].astype(str),
            cached["label_order"].astype(str).tolist(),
        )

    os_env_path = str(data_dir.resolve())
    Path(os_env_path).mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = os_env_path
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    dataset = BNCI2014_004_Dataset()
    paradigm = MotorImagery(n_classes=2, fmin=8, fmax=30, tmin=0.0, tmax=4.5)
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
            break
        except Exception as exc:
            last_error = exc
            print(f"subject={subject} BNCI2014-004 preparation failed on attempt {attempt}/4: {exc}", flush=True)
            clear_subject_downloads(subject, data_dir)
            time.sleep(3 * attempt)
    else:
        raise last_error if last_error is not None else RuntimeError(f"Unable to prepare BNCI2014-004 subject {subject}.")

    label_order = sorted({str(label) for label in y})
    label_to_index = {label: index for index, label in enumerate(label_order)}
    y_int = np.asarray([label_to_index[str(label)] for label in y], dtype=np.int64)
    sessions = metadata["session"].astype(str).to_numpy()
    X = X.astype(np.float32)
    np.savez_compressed(cache_file, X=X, y=y_int, sessions=sessions, label_order=np.asarray(label_order))
    return X, y_int, sessions, label_order


def run_aux(config: AuxConfig) -> Dict[str, object]:
    """运行辅助二分类 MI sanity check，并导出补充材料结果。"""
    repo_root = Path(__file__).resolve().parents[1]
    core = load_module(repo_root / "scripts" / "run_mi_experiments.py", "mi_core_bnci004_aux")
    core.seed_everything(config.seed)
    device = core.get_device(config.device)
    data_dir = repo_root / config.data_dir
    output_dir = repo_root / config.output_dir
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    parameter_counts: Dict[str, int] = {}
    label_order_global: List[str] | None = None
    total_runs = len(config.subjects) * len(config.models)
    run_index = 0

    for subject in config.subjects:
        X_raw, y, sessions, label_order = load_subject_data(subject, data_dir, cache_dir)
        label_order_global = label_order
        X_raw = core.downsample_trials(X_raw, config.downsample_factor)
        train_mask = np.isin(sessions, np.asarray(["0train", "1train", "2train"]))
        test_mask = np.isin(sessions, np.asarray(["3test", "4test"]))
        X_train_full = X_raw[train_mask]
        y_train_full = y[train_mask]
        X_test = X_raw[test_mask]
        y_test = y[test_mask]

        # 辅助实验同样区分划分随机性与训练随机性，便于和论文主协议保持一致。
        split_seed = derive_seed(config.seed, subject, 0)
        train_seed = derive_seed(config.seed, subject, 1)
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=config.val_fraction,
            random_state=split_seed,
        )
        train_idx, val_idx = next(splitter.split(np.zeros(len(X_train_full)), y_train_full))

        mean, std = core.compute_standardizer(X_train_full[train_idx])
        X_train = core.apply_standardizer(X_train_full[train_idx], mean, std)
        X_val = core.apply_standardizer(X_train_full[val_idx], mean, std)
        X_test_std = core.apply_standardizer(X_test, mean, std)

        n_channels = X_train.shape[1]
        n_samples = X_train.shape[2]
        n_classes = len(label_order)
        if not parameter_counts:
            for model_name in config.models:
                parameter_counts[model_name] = core.get_parameter_count(
                    model_name=model_name,
                    n_channels=n_channels,
                    n_samples=n_samples,
                    n_classes=n_classes,
                    cfc_hidden_size=config.cfc_hidden_size,
                    lstm_hidden_size=config.lstm_hidden_size,
                )

        train_loader = core.build_loader(X_train, y_train_full[train_idx], config.batch_size, True, device)
        val_loader = core.build_loader(X_val, y_train_full[val_idx], config.batch_size, False, device)
        test_loader = core.build_loader(X_test_std, y_test, config.batch_size, False, device)

        for model_name in config.models:
            run_index += 1
            core.seed_everything(train_seed)
            print(
                f"[{run_index}/{total_runs}] repeat_seed={config.seed} split_seed={split_seed} "
                f"train_seed={train_seed} subject={subject} model={model_name} device={device.type}",
                flush=True,
            )
            if core.is_classical_model(model_name):
                fit_info = core.fit_riemann_tslr(
                    x_train=X_train,
                    y_train=y_train_full[train_idx],
                    x_val=X_val,
                    y_val=y_train_full[val_idx],
                )
                metrics = core.evaluate_classical_model(
                    model=fit_info["model"],
                    x=X_test_std,
                    y=y_test,
                    return_predictions=False,
                )
            else:
                model = core.build_model(
                    model_name=model_name,
                    n_channels=n_channels,
                    n_samples=n_samples,
                    n_classes=n_classes,
                    cfc_hidden_size=config.cfc_hidden_size,
                    lstm_hidden_size=config.lstm_hidden_size,
                )
                fit_info = core.train_one_model(
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
                metrics = core.evaluate_model(model, test_loader, device, return_predictions=False)

            rows.append(
                {
                    "seed": config.seed,
                    "split_seed": split_seed,
                    "train_seed": train_seed,
                    "subject": subject,
                    "model": model_name,
                    "model_display": core.get_model_display_name(model_name),
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                    "best_epoch": fit_info["best_epoch"],
                    "best_val_accuracy": fit_info["best_val_accuracy"],
                    "best_val_loss": fit_info["best_val_loss"],
                    "best_c": float(fit_info.get("best_c", float("nan"))),
                }
            )

    subject_df = pd.DataFrame(rows).sort_values(["subject", "model"]).reset_index(drop=True)
    summary: Dict[str, Dict[str, float]] = {}
    for model_name, group in subject_df.groupby("model"):
        summary[model_name] = {
            "accuracy_mean": float(group["accuracy"].mean()),
            "accuracy_std": float(group["accuracy"].std(ddof=1)) if len(group) > 1 else 0.0,
            "f1_mean": float(group["f1"].mean()),
            "f1_std": float(group["f1"].std(ddof=1)) if len(group) > 1 else 0.0,
        }

    stat_tests: Dict[str, Dict[str, float]] = {}
    for model_a, model_b in combinations(config.models, 2):
        stat_tests[f"{model_a}_vs_{model_b}"] = core.paired_test(subject_df, model_a, model_b)
    core.apply_holm_correction(stat_tests, p_value_key="p_value", output_key="holm_p_value")
    core.apply_holm_correction(stat_tests, p_value_key="wilcoxon_p_value", output_key="wilcoxon_holm_p_value")

    subject_df.to_csv(output_dir / "aux_metrics.csv", index=False)
    pd.DataFrame([{"comparison": key, **value} for key, value in stat_tests.items()]).to_csv(
        output_dir / "stat_tests.csv",
        index=False,
    )
    results = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "BNCI2014-004",
        "protocol": "train sessions 0/1/2, test sessions 3/4",
        "device": device.type,
        "label_order": label_order_global,
        "config": asdict(config),
        "parameter_counts": parameter_counts,
        "summary": summary,
        "stat_tests": stat_tests,
    }
    (output_dir / "results_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    """命令行入口，对应论文补充 sanity check 的单独运行脚本。"""
    parser = argparse.ArgumentParser(description="Auxiliary BNCI2014-004 binary MI sanity check.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["shallow_convnet", "riemann_tslr", "eegnet", "tiny_transformer", "cfc", "lstm"],
    )
    parser.add_argument("--subjects", nargs="*", type=int, default=list(range(1, 10)))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfc-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    config = AuxConfig(
        subjects=args.subjects,
        models=args.models,
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
    run_aux(config)


if __name__ == "__main__":
    main()
