"""结构化扰动 sweep 实验。

对应论文：
1. robustness analysis 中的带限噪声与通道丢失 sweep。
2. 补充材料里的准确率曲线与统计检验。

该脚本回答的是“模型在结构化破坏下的行为边界”，不是主 clean accuracy 结论。
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_module(path: Path, module_name: str):
    """动态加载主实验与 session-wise 工具脚本。"""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class SweepConfig:
    """结构化扰动 sweep 配置。"""
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
    repeats: int
    snr_levels: List[float]
    dropout_levels: List[float]
    device: str
    data_dir: str
    output_dir: str


def plot_sweep(summary_df: pd.DataFrame, output_path: Path, perturbation_type: str, module) -> None:
    """绘制论文补充材料中的 sweep 曲线图。"""
    subset = summary_df.loc[summary_df["perturbation_type"] == perturbation_type].copy()
    if subset.empty:
        return
    subset["model_display"] = subset["model"].map(module.get_model_display_name)
    order = [label for label in module.MODEL_ORDER if label in subset["model_display"].unique()]
    subset["model_display"] = pd.Categorical(subset["model_display"], categories=order, ordered=True)
    subset = subset.sort_values(["model_display", "level"])

    sns.set_theme(style="whitegrid")
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for model_display, group in subset.groupby("model_display", observed=False):
        if group.empty:
            continue
        axis.plot(
            group["level"],
            group["accuracy_mean"],
            marker="o",
            linewidth=2,
            label=model_display,
            color=module.MODEL_PALETTE.get(str(model_display), None),
        )
        axis.fill_between(
            group["level"].to_numpy(dtype=float),
            group["accuracy_ci95_low"].to_numpy(dtype=float),
            group["accuracy_ci95_high"].to_numpy(dtype=float),
            alpha=0.15,
            color=module.MODEL_PALETTE.get(str(model_display), None),
        )
    if perturbation_type == "band_limited_noise":
        axis.set_xlabel("SNR (dB)")
        axis.invert_xaxis()
        axis.set_title("Band-limited Noise Sweep")
    else:
        axis.set_xlabel("Dropped Channel Fraction")
        axis.set_title("Channel Dropout Sweep")
    axis.set_ylabel("Accuracy (%)")
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_sweep(config: SweepConfig) -> Dict[str, object]:
    """运行多强度 band-limited noise 与 channel dropout sweep。"""
    repo_root = Path(__file__).resolve().parents[1]
    core = load_module(repo_root / "scripts" / "run_mi_experiments.py", "mi_exp_core_sweep")
    sessionwise = load_module(repo_root / "scripts" / "run_sessionwise_mi_comparison.py", "mi_exp_sessionwise_sweep")

    core.seed_everything(config.seed)
    device = core.get_device(config.device)
    data_dir = repo_root / config.data_dir
    output_dir = repo_root / config.output_dir
    cache_dir = repo_root / "outputs" / "bspc_sessionwise" / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    clean_rows: List[Dict[str, object]] = []

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
        # 这里沿用 session-wise 主协议，只在测试阶段替换为不同强度的结构化扰动。
        splitter = sessionwise.StratifiedShuffleSplit(
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
        sfreq = 250.0 / config.downsample_factor

        for model_name in config.models:
            run_index += 1
            model_seed = config.seed + subject * 100 + config.models.index(model_name)
            core.seed_everything(model_seed)
            print(f"[{run_index}/{total_runs}] subject={subject} model={model_name} device={device.type}", flush=True)

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
                fit_info = core.train_one_model(
                    model=runtime_model,
                    train_loader=core.build_loader(X_train, y_train_full[train_idx], config.batch_size, True, device),
                    val_loader=core.build_loader(X_val, y_train_full[val_idx], config.batch_size, False, device),
                    device=device,
                    epochs=config.epochs,
                    patience=config.patience,
                    min_epochs=config.min_epochs,
                    learning_rate=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
                clean_metrics = core.evaluate_model(
                    runtime_model,
                    core.build_loader(X_test_std, y_test, config.batch_size, False, device),
                    device,
                )

            clean_rows.append(
                {
                    "subject": subject,
                    "model": model_name,
                    "accuracy": clean_metrics["accuracy"],
                    "f1": clean_metrics["f1"],
                }
            )

            for repeat in range(config.repeats):
                for snr_db in config.snr_levels:
                    perturbed_raw = sessionwise.add_band_limited_noise(
                        X_test,
                        sfreq=sfreq,
                        snr_db=snr_db,
                        rng=np.random.default_rng(
                            config.seed + subject * 10_000 + config.models.index(model_name) * 1_000 + repeat * 100 + int(snr_db)
                        ),
                    )
                    perturbed_std = core.apply_standardizer(perturbed_raw, mean, std)
                    if core.is_classical_model(model_name):
                        metrics = core.evaluate_classical_model(runtime_model, perturbed_std, y_test)
                    else:
                        metrics = core.evaluate_model(
                            runtime_model,
                            core.build_loader(perturbed_std, y_test, config.batch_size, False, device),
                            device,
                        )
                    rows.append(
                        {
                            "subject": subject,
                            "model": model_name,
                            "perturbation_type": "band_limited_noise",
                            "level": float(snr_db),
                            "seed": repeat,
                            "accuracy": metrics["accuracy"],
                            "f1": metrics["f1"],
                        }
                    )

                for drop_fraction in config.dropout_levels:
                    perturbed_raw = sessionwise.apply_channel_dropout(
                        X_test,
                        drop_fraction=drop_fraction,
                        rng=np.random.default_rng(
                            config.seed + subject * 20_000 + config.models.index(model_name) * 1_000 + repeat * 100 + int(drop_fraction * 100)
                        ),
                    )
                    perturbed_std = core.apply_standardizer(perturbed_raw, mean, std)
                    if core.is_classical_model(model_name):
                        metrics = core.evaluate_classical_model(runtime_model, perturbed_std, y_test)
                    else:
                        metrics = core.evaluate_model(
                            runtime_model,
                            core.build_loader(perturbed_std, y_test, config.batch_size, False, device),
                            device,
                        )
                    rows.append(
                        {
                            "subject": subject,
                            "model": model_name,
                            "perturbation_type": "channel_dropout",
                            "level": float(drop_fraction),
                            "seed": repeat,
                            "accuracy": metrics["accuracy"],
                            "f1": metrics["f1"],
                        }
                    )

    raw_df = pd.DataFrame(rows)
    clean_df = pd.DataFrame(clean_rows)
    raw_df.to_csv(output_dir / "sweep_metrics.csv", index=False)
    clean_df.to_csv(output_dir / "clean_subject_metrics.csv", index=False)

    subject_df = (
        raw_df.groupby(["subject", "model", "perturbation_type", "level"], as_index=False)[["accuracy", "f1"]]
        .mean()
        .sort_values(["perturbation_type", "level", "model", "subject"])
        .reset_index(drop=True)
    )
    subject_df.to_csv(output_dir / "sweep_subject_summary.csv", index=False)

    summary_rows = []
    for (model_name, perturbation_type, level), group in subject_df.groupby(["model", "perturbation_type", "level"]):
        ci_low, ci_high = core.confidence_interval_95(group["accuracy"].to_numpy())
        summary_rows.append(
            {
                "model": model_name,
                "model_display": core.get_model_display_name(model_name),
                "perturbation_type": perturbation_type,
                "level": float(level),
                "accuracy_mean": float(group["accuracy"].mean()),
                "accuracy_std": float(group["accuracy"].std(ddof=1)) if len(group) > 1 else 0.0,
                "accuracy_ci95_low": float(ci_low),
                "accuracy_ci95_high": float(ci_high),
                "f1_mean": float(group["f1"].mean()),
                "f1_std": float(group["f1"].std(ddof=1)) if len(group) > 1 else 0.0,
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["perturbation_type", "level", "model_display"])
    summary_df.to_csv(output_dir / "sweep_summary.csv", index=False)

    comparison_pairs = [
        ("cfc", "gru"),
        ("cfc", "eegnet"),
        ("cfc", "shallow_convnet"),
        ("cfc", "riemann_tslr"),
    ]
    stats_rows = []
    for perturbation_type, perturb_group in subject_df.groupby("perturbation_type"):
        for level, level_group in perturb_group.groupby("level"):
            for model_a, model_b in comparison_pairs:
                if not {model_a, model_b}.issubset(set(level_group["model"].unique())):
                    continue
                stats = core.paired_test(level_group[["subject", "model", "accuracy"]], model_a, model_b)
                stats_rows.append(
                    {
                        "perturbation_type": perturbation_type,
                        "level": float(level),
                        "comparison": f"{model_a}_vs_{model_b}",
                        **stats,
                    }
                )
    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty:
        stats_df["holm_p_value"] = np.nan
        stats_df["wilcoxon_holm_p_value"] = np.nan
        for perturbation_type, group in stats_df.groupby("perturbation_type"):
            adjusted_t = core.holm_adjust(group["p_value"].tolist())
            adjusted_w = core.holm_adjust(group["wilcoxon_p_value"].tolist())
            stats_df.loc[group.index, "holm_p_value"] = adjusted_t
            stats_df.loc[group.index, "wilcoxon_holm_p_value"] = adjusted_w
    stats_df.to_csv(output_dir / "sweep_stats.csv", index=False)

    plot_sweep(summary_df, output_dir / "band_noise_accuracy_sweep.pdf", "band_limited_noise", core)
    plot_sweep(summary_df, output_dir / "channel_dropout_accuracy_sweep.pdf", "channel_dropout", core)

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": asdict(config),
        "clean_summary": clean_df.groupby("model")[["accuracy", "f1"]].mean().to_dict(orient="index"),
        "sweep_summary": {
            f"{row['model']}|{row['perturbation_type']}|{row['level']}": {
                "accuracy_mean": row["accuracy_mean"],
                "accuracy_std": row["accuracy_std"],
                "accuracy_ci95_low": row["accuracy_ci95_low"],
                "accuracy_ci95_high": row["accuracy_ci95_high"],
            }
            for _, row in summary_df.iterrows()
        },
    }
    (output_dir / "results_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> SweepConfig:
    """命令行入口，对应论文 robustness sweep 的可复现脚本。"""
    parser = argparse.ArgumentParser(description="Run a session-wise structured perturbation sweep.")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument(
        "--models",
        nargs="*",
        default=["shallow_convnet", "riemann_tslr", "eegnet", "tiny_transformer", "cfc", "gru", "lstm"],
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260322)
    parser.add_argument("--cfc-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--snr-levels", type=float, nargs="*", default=[20.0, 10.0, 5.0, 0.0])
    parser.add_argument("--dropout-levels", type=float, nargs="*", default=[0.1, 0.3, 0.5])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/bspc_perturbation_sweep")
    args = parser.parse_args()
    return SweepConfig(
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
        repeats=args.repeats,
        snr_levels=list(args.snr_levels),
        dropout_levels=list(args.dropout_levels),
        device=args.device,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    run_sweep(parse_args())
