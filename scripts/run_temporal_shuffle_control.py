"""时间顺序打乱控制实验。

对应论文：
1. 机制性反证实验：检验 temporal order 本身是否是类别判别瓶颈。
2. 结果与结论部分关于“temporal order itself is not the limiting factor”的证据。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit


def load_module(path: Path, module_name: str):
    """动态加载主实验脚本，保证 shuffle control 与正文主协议共享实现。"""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class TemporalShuffleConfig:
    """时间打乱控制实验配置。"""
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
    shuffle_repeats: int
    device: str
    data_dir: str
    output_dir: str


def apply_temporal_shuffle(trials: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """对单 trial 内的时间维做随机置换。

    这一步对应论文的 temporal shuffle control：保留空间/频谱统计，破坏时序顺序。
    """
    shuffled = np.empty_like(trials)
    num_timepoints = trials.shape[-1]
    for trial_index in range(trials.shape[0]):
        permutation = rng.permutation(num_timepoints)
        shuffled[trial_index] = trials[trial_index][:, permutation]
    return shuffled


def summarize(rows: List[Dict[str, object]], core) -> Dict[str, object]:
    """汇总 clean 与 temporal-shuffle 的落差，并形成论文统计表。"""
    metrics_df = pd.DataFrame(rows)
    clean_df = (
        metrics_df.loc[metrics_df["condition"] == "clean", ["subject", "model", "accuracy", "f1"]]
        .drop_duplicates()
        .sort_values(["model", "subject"])
        .reset_index(drop=True)
    )
    shuffle_df = (
        metrics_df.loc[metrics_df["condition"] == "temporal_shuffle"]
        .groupby(["subject", "model"], as_index=False)[["accuracy", "f1"]]
        .mean()
        .rename(columns={"accuracy": "shuffle_accuracy", "f1": "shuffle_f1"})
        .sort_values(["model", "subject"])
        .reset_index(drop=True)
    )
    merged = clean_df.merge(shuffle_df, on=["subject", "model"], how="inner")
    merged["accuracy_drop"] = merged["accuracy"] - merged["shuffle_accuracy"]
    merged["f1_drop"] = merged["f1"] - merged["shuffle_f1"]

    summary_rows: List[Dict[str, object]] = []
    paired_tests: Dict[str, Dict[str, float]] = {}
    for model_name, group in merged.groupby("model"):
        ci_low, ci_high = core.confidence_interval_95(group["accuracy_drop"].to_numpy(dtype=float))
        paired_df = pd.DataFrame(
            {
                "subject": group["subject"].to_numpy(dtype=int),
                "clean": group["accuracy"].to_numpy(dtype=float),
                "temporal_shuffle": group["shuffle_accuracy"].to_numpy(dtype=float),
            }
        ).melt(id_vars="subject", var_name="model", value_name="accuracy")
        paired_tests[f"clean_vs_shuffle_{model_name}"] = core.paired_test(
            paired_df,
            "clean",
            "temporal_shuffle",
        )
        summary_rows.append(
            {
                "model": model_name,
                "model_display": core.get_model_display_name(model_name),
                "clean_accuracy_mean": float(group["accuracy"].mean()),
                "clean_accuracy_std": float(group["accuracy"].std(ddof=1)) if len(group) > 1 else 0.0,
                "shuffle_accuracy_mean": float(group["shuffle_accuracy"].mean()),
                "shuffle_accuracy_std": float(group["shuffle_accuracy"].std(ddof=1)) if len(group) > 1 else 0.0,
                "accuracy_drop_mean": float(group["accuracy_drop"].mean()),
                "accuracy_drop_std": float(group["accuracy_drop"].std(ddof=1)) if len(group) > 1 else 0.0,
                "accuracy_drop_ci95_low": float(ci_low),
                "accuracy_drop_ci95_high": float(ci_high),
                "clean_f1_mean": float(group["f1"].mean()),
                "shuffle_f1_mean": float(group["shuffle_f1"].mean()),
                "f1_drop_mean": float(group["f1_drop"].mean()),
            }
        )

    core.apply_holm_correction(paired_tests, p_value_key="p_value", output_key="holm_p_value")
    core.apply_holm_correction(paired_tests, p_value_key="wilcoxon_p_value", output_key="wilcoxon_holm_p_value")
    return {
        "metrics_df": metrics_df,
        "subject_summary_df": merged,
        "summary_df": pd.DataFrame(summary_rows).sort_values(
            by="model_display",
            key=lambda series: series.map({name: index for index, name in enumerate(core.MODEL_ORDER)}).fillna(999),
        ),
        "paired_tests": paired_tests,
    }


def save_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    """绘制 temporal shuffle 后的性能下降图，对应补充材料图。"""
    if summary_df.empty:
        return
    sns.set_theme(style="whitegrid")
    plot_df = summary_df.copy()
    plot_df["model_display"] = pd.Categorical(plot_df["model_display"], categories=plot_df["model_display"], ordered=True)
    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.errorbar(
        x=np.arange(len(plot_df)),
        y=plot_df["accuracy_drop_mean"].to_numpy(dtype=float),
        yerr=np.vstack(
            [
                plot_df["accuracy_drop_mean"].to_numpy(dtype=float) - plot_df["accuracy_drop_ci95_low"].to_numpy(dtype=float),
                plot_df["accuracy_drop_ci95_high"].to_numpy(dtype=float) - plot_df["accuracy_drop_mean"].to_numpy(dtype=float),
            ]
        ),
        fmt="o",
        capsize=4,
        linewidth=1.5,
        color="#2f4f4f",
    )
    axis.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
    axis.set_xticks(np.arange(len(plot_df)))
    axis.set_xticklabels(plot_df["model_display"].tolist(), rotation=20, ha="right")
    axis.set_ylabel("Accuracy Drop After Temporal Shuffle (%)")
    axis.set_title("Session-wise Temporal Shuffle Control")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_temporal_shuffle_control(config: TemporalShuffleConfig) -> Dict[str, object]:
    """运行 session-wise temporal shuffle control。"""
    repo_root = Path(__file__).resolve().parents[1]
    core = load_module(repo_root / "scripts" / "run_mi_experiments.py", "mi_exp_core_temporal_shuffle")
    sessionwise = load_module(repo_root / "scripts" / "run_sessionwise_mi_comparison.py", "mi_exp_sessionwise_temporal_shuffle")

    core.seed_everything(config.seed)
    device = core.get_device(config.device)
    data_dir = repo_root / config.data_dir
    output_dir = repo_root / config.output_dir
    cache_dir = repo_root / "outputs" / "bspc_sessionwise" / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    total_runs = len(config.subjects) * len(config.models)
    run_index = 0

    for subject in config.subjects:
        X_raw, y, sessions = sessionwise.load_subject_session_data(core, subject, data_dir, cache_dir)
        X_raw = core.downsample_trials(X_raw, config.downsample_factor)
        train_mask = sessions == "0train"
        test_mask = sessions == "1test"
        X_train_full = X_raw[train_mask]
        y_train_full = y[train_mask]
        X_test = X_raw[test_mask]
        y_test = y[test_mask]

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=config.val_fraction,
            random_state=config.seed + subject,
        )
        train_idx, val_idx = next(splitter.split(X_train_full, y_train_full))
        mean, std = core.compute_standardizer(X_train_full[train_idx])
        X_train = core.apply_standardizer(X_train_full[train_idx], mean, std)
        X_val = core.apply_standardizer(X_train_full[val_idx], mean, std)
        X_test_std = core.apply_standardizer(X_test, mean, std)

        n_channels = X_train.shape[1]
        n_samples = X_train.shape[2]
        train_loader = core.build_loader(X_train, y_train_full[train_idx], config.batch_size, True, device)
        val_loader = core.build_loader(X_val, y_train_full[val_idx], config.batch_size, False, device)
        test_loader = core.build_loader(X_test_std, y_test, config.batch_size, False, device)

        for model_index, model_name in enumerate(config.models):
            run_index += 1
            model_seed = config.seed + subject * 100 + model_index
            core.seed_everything(model_seed)
            print(
                f"[{run_index}/{total_runs}] subject={subject} model={model_name} base_seed={config.seed} model_seed={model_seed} device={device.type}",
                flush=True,
            )

            if core.is_classical_model(model_name):
                fit_info = core.fit_riemann_tslr(
                    x_train=X_train,
                    y_train=y_train_full[train_idx],
                    x_val=X_val,
                    y_val=y_train_full[val_idx],
                )
                runtime_model = fit_info["model"]
                clean_metrics = core.evaluate_classical_model(runtime_model, X_test_std, y_test)
            else:
                runtime_model = core.build_model(
                    model_name=model_name,
                    n_channels=n_channels,
                    n_samples=n_samples,
                    n_classes=len(core.LABEL_ORDER),
                    cfc_hidden_size=config.cfc_hidden_size,
                    lstm_hidden_size=config.lstm_hidden_size,
                )
                core.train_one_model(
                    model=runtime_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    epochs=config.epochs,
                    patience=config.patience,
                    min_epochs=config.min_epochs,
                    learning_rate=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
                clean_metrics = core.evaluate_model(runtime_model, test_loader, device)

            rows.append(
                {
                    "subject": subject,
                    "model": model_name,
                    "condition": "clean",
                    "shuffle_seed": -1,
                    "accuracy": clean_metrics["accuracy"],
                    "f1": clean_metrics["f1"],
                }
            )

            # 多次 shuffle 是为了避免单个置换恰好对某个模型有利或不利。
            for shuffle_seed in range(config.shuffle_repeats):
                rng = np.random.default_rng(config.seed + subject * 10_000 + model_index * 1_000 + shuffle_seed)
                shuffled_raw = apply_temporal_shuffle(X_test, rng)
                shuffled_std = core.apply_standardizer(shuffled_raw, mean, std)
                if core.is_classical_model(model_name):
                    shuffle_metrics = core.evaluate_classical_model(runtime_model, shuffled_std, y_test)
                else:
                    shuffled_loader = core.build_loader(shuffled_std, y_test, config.batch_size, False, device)
                    shuffle_metrics = core.evaluate_model(runtime_model, shuffled_loader, device)
                rows.append(
                    {
                        "subject": subject,
                        "model": model_name,
                        "condition": "temporal_shuffle",
                        "shuffle_seed": shuffle_seed,
                        "accuracy": shuffle_metrics["accuracy"],
                        "f1": shuffle_metrics["f1"],
                    }
                )

    result = summarize(rows, core)
    metrics_df = result["metrics_df"]
    subject_summary_df = result["subject_summary_df"]
    summary_df = result["summary_df"]
    paired_tests = result["paired_tests"]

    metrics_df.to_csv(output_dir / "temporal_shuffle_metrics.csv", index=False)
    subject_summary_df.to_csv(output_dir / "temporal_shuffle_subject_summary.csv", index=False)
    summary_df.to_csv(output_dir / "temporal_shuffle_summary.csv", index=False)
    pd.DataFrame([{"comparison": key, **value} for key, value in paired_tests.items()]).to_csv(
        output_dir / "temporal_shuffle_stats.csv",
        index=False,
    )
    save_plot(summary_df, output_dir / "temporal_shuffle_drop.pdf")

    output = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "config": asdict(config),
        "summary": {
            row["model"]: {
                "clean_accuracy_mean": row["clean_accuracy_mean"],
                "shuffle_accuracy_mean": row["shuffle_accuracy_mean"],
                "accuracy_drop_mean": row["accuracy_drop_mean"],
                "accuracy_drop_ci95_low": row["accuracy_drop_ci95_low"],
                "accuracy_drop_ci95_high": row["accuracy_drop_ci95_high"],
            }
            for row in summary_df.to_dict(orient="records")
        },
        "paired_tests": paired_tests,
    }
    (output_dir / "temporal_shuffle_results_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)
    return output


def parse_args() -> TemporalShuffleConfig:
    """命令行入口，对应论文 temporal shuffle 机制验证。"""
    parser = argparse.ArgumentParser(description="Run a session-wise temporal-shuffle control on representative MI-EEG models.")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument(
        "--models",
        nargs="*",
        default=["shallow_convnet", "riemann_tslr", "eegnet", "tiny_transformer", "cfc", "lstm"],
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260320)
    parser.add_argument("--cfc-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--shuffle-repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/bspc_temporal_shuffle")
    args = parser.parse_args()
    return TemporalShuffleConfig(
        subjects=list(args.subjects),
        models=list(args.models),
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
        shuffle_repeats=args.shuffle_repeats,
        device=args.device,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    run_temporal_shuffle_control(parse_args())
