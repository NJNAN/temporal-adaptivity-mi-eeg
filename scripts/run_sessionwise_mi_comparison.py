"""session-wise 主对照脚本。

对应论文：
1. 严格的 `session 1 train -> session 2 test` 主泛化检查。
2. tau 的 trial-level、time-resolved 和频带相关性分析。
3. 结构化扰动实验（带限噪声、通道丢失）及其统计检验。

这份脚本是论文“Temporal Adaptivity Is Not Enough”结论最关键的证据来源。
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
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.signal import butter, filtfilt, welch
from scipy.stats import friedmanchisquare, pearsonr, ttest_rel
from sklearn.model_selection import StratifiedShuffleSplit


# BCI IV-2a 的 22 通道固定顺序，用于补充“运动相关通道局部 tau”分析。
BCI_IV_2A_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]
MOTOR_CHANNEL_NAMES = ["C3", "C4", "CP3", "CP4"]
TAU_WINDOW_SPECS = [
    ("early", 2.0, 2.5),
    ("mid", 2.5, 3.5),
    ("late", 3.5, 4.0),
]


def load_core_module(repo_root: Path):
    """加载 pooled 主脚本，复用相同模型定义、统计函数与显示名。"""
    script_path = repo_root / "scripts" / "run_mi_experiments.py"
    spec = importlib.util.spec_from_file_location("mi_exp_core", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_records(path: Path) -> List[Dict[str, object]]:
    """读取断点 CSV；文件不存在时返回空表。"""
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def completed_model_subjects(rows: List[Dict[str, object]]) -> set[tuple[int, str]]:
    """从 session-wise metrics 中恢复已完成的 (subject, model)。"""
    completed: set[tuple[int, str]] = set()
    for row in rows:
        subject = row.get("subject")
        model = row.get("model")
        accuracy = row.get("accuracy")
        if subject is None or model is None or pd.isna(subject) or pd.isna(accuracy):
            continue
        completed.add((int(float(subject)), str(model)))
    return completed


def load_dataframe_if_exists(path: Path) -> pd.DataFrame | None:
    """读取可选中间表，用于恢复 tau 分析。"""
    if not path.exists():
        return None
    return pd.read_csv(path)


@dataclass
class SessionwiseConfig:
    """session-wise 主实验配置。

    对应论文严格协议下的统一训练参数与扰动设置。
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
    cfc_dt: float
    cfc_tau_init: float
    downsample_factor: int
    structured_repeats: int
    device: str
    data_dir: str
    output_dir: str


def load_subject_session_data(module, subject: int, data_dir: Path, cache_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """载入带 session 标签的 IV-2a 数据。

    对应论文主泛化协议：session 1 训练，session 2 测试。
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"subject_{subject:02d}.npz"
    if cache_file.exists():
        cached = np.load(cache_file, allow_pickle=True)
        return (
            cached["X"].astype(np.float32),
            cached["y"].astype(np.int64),
            cached["session"].astype(str),
        )

    module.ensure_mne_path(data_dir)
    dataset = module.BNCI2014Dataset()
    paradigm = module.MotorImagery(n_classes=4, fmin=8, fmax=30, tmin=0.0, tmax=4.0)
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
    y_int = np.asarray([module.LABEL_TO_INDEX[str(label)] for label in y], dtype=np.int64)
    sessions = metadata["session"].astype(str).to_numpy()
    X = X.astype(np.float32)
    np.savez_compressed(cache_file, X=X, y=y_int, session=sessions)
    return X, y_int, sessions


def summarize_subject_metrics(rows: List[Dict[str, object]]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """把 session-wise 单被试结果汇总成论文主表可用的均值和标准差。"""
    df = pd.DataFrame(rows)
    summary: Dict[str, Dict[str, float]] = {}
    for model_name, group in df.groupby("model"):
        accuracy_values = group["accuracy"].to_numpy()
        f1_values = group["f1"].to_numpy()
        summary[model_name] = {
            "accuracy_mean": float(np.mean(accuracy_values)),
            "accuracy_std": float(np.std(accuracy_values, ddof=1)) if len(accuracy_values) > 1 else 0.0,
            "f1_mean": float(np.mean(f1_values)),
            "f1_std": float(np.std(f1_values, ddof=1)) if len(f1_values) > 1 else 0.0,
        }
    return df, summary


def compute_band_power(trials: np.ndarray, sfreq: float, band: Tuple[float, float]) -> np.ndarray:
    """计算频带功率。

    对应论文里 tau 与 μ/β 功率相关性分析。
    """
    if trials.size == 0:
        return np.zeros(0, dtype=np.float32)
    nperseg = min(256, trials.shape[-1])
    freqs, psd = welch(trials, fs=sfreq, axis=-1, nperseg=nperseg)
    band_mask = (freqs >= band[0]) & (freqs < band[1])
    if not np.any(band_mask):
        return np.zeros(trials.shape[0], dtype=np.float32)
    return psd[:, :, band_mask].mean(axis=(1, 2)).astype(np.float32)


def add_band_limited_noise(
    trials: np.ndarray,
    sfreq: float,
    snr_db: float,
    rng: np.random.Generator,
    band: Tuple[float, float] = (8.0, 30.0),
) -> np.ndarray:
    """向 test trial 加入带限噪声。

    对应论文结构化扰动实验中的 in-band noise 条件。
    """
    white = rng.normal(0.0, 1.0, size=trials.shape).astype(np.float32)
    b, a = butter(4, band, btype="bandpass", fs=sfreq)
    filtered_noise = filtfilt(b, a, white, axis=-1).astype(np.float32)
    signal_power = np.mean(np.square(trials), axis=(1, 2), keepdims=True)
    noise_power = np.mean(np.square(filtered_noise), axis=(1, 2), keepdims=True)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / np.clip(noise_power, 1e-8, None)).astype(np.float32)
    return trials + filtered_noise * scale


def apply_channel_dropout(
    trials: np.ndarray,
    drop_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """随机屏蔽部分通道。

    对应论文结构化扰动中的 channel dropout 条件。
    """
    dropped = trials.copy()
    num_channels = dropped.shape[1]
    num_drop = max(1, int(round(num_channels * drop_fraction)))
    for trial_idx in range(dropped.shape[0]):
        drop_indices = rng.choice(num_channels, size=num_drop, replace=False)
        dropped[trial_idx, drop_indices, :] = 0.0
    return dropped


def channel_indices(channel_names: List[str], selected_names: List[str]) -> List[int]:
    """返回指定通道在固定通道顺序中的索引。"""
    index_map = {name: index for index, name in enumerate(channel_names)}
    return [index_map[name] for name in selected_names if name in index_map]


def keep_only_channels(trials: np.ndarray, keep_indices: List[int]) -> np.ndarray:
    """仅保留指定通道，其余通道置零。

    对应补强实验里的“局部空间 tau 分析”：不重训模型，只在推理时保留运动区通道，
    检查原来的全局平均结论是否被非运动区输入掩盖。
    """
    masked = np.zeros_like(trials)
    masked[:, keep_indices, :] = trials[:, keep_indices, :]
    return masked


def collect_cfc_trial_analysis(
    module,
    model,
    loader,
    raw_trials: np.ndarray,
    subject: int,
    device,
    sfreq: float,
    analysis_scope: str = "global",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """收集 CfC-style 的 trial-level tau 统计。

    对应论文 `Temporal Adaptivity versus Class Discriminability` 的第一手数据表。
    """
    rows: List[Dict[str, object]] = []
    timecourse_rows: List[Dict[str, object]] = []
    cursor = 0
    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            batch_size = targets.size(0)
            raw_batch = raw_trials[cursor:cursor + batch_size]
            trial_indices = np.arange(cursor, cursor + batch_size, dtype=np.int64)
            cursor += batch_size
            target_np = targets.numpy()
            logits, aux = model(features.to(device, non_blocking=True), return_aux=True)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            tau_tensor = aux["tau"].detach().cpu().numpy()
            # 论文主文采用 trial 级 tau：先对 hidden units 与 time 求均值。
            tau_mean = tau_tensor.mean(axis=(1, 2))
            tau_timecourse = tau_tensor.mean(axis=2)
            mu_power = compute_band_power(raw_batch, sfreq=sfreq, band=(8.0, 13.0))
            beta_power = compute_band_power(raw_batch, sfreq=sfreq, band=(13.0, 30.0))
            for index in range(batch_size):
                label_index = int(target_np[index])
                rows.append(
                    {
                        "subject": subject,
                        "trial_index": int(trial_indices[index]),
                        "analysis_scope": analysis_scope,
                        "class_index": label_index,
                        "class_name": module.INDEX_TO_LABEL[label_index],
                        "tau_mean": float(tau_mean[index]),
                        "mu_power": float(mu_power[index]),
                        "beta_power": float(beta_power[index]),
                        "correct": int(predictions[index] == label_index),
                    }
                )
                for time_index, tau_value in enumerate(tau_timecourse[index]):
                    timecourse_rows.append(
                        {
                            "subject": subject,
                            "trial_index": int(trial_indices[index]),
                            "analysis_scope": analysis_scope,
                            "class_index": label_index,
                            "class_name": module.INDEX_TO_LABEL[label_index],
                            "time_index": time_index,
                            "time_seconds": float(2.0 + time_index / sfreq),
                            "tau_mean": float(tau_value),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(timecourse_rows)


def summarize_tau_trial_analysis(
    module,
    tau_trial_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """汇总 tau 的 trial-level 统计。

    对应论文中三段证据链中的前两段：
    1. 类间分布/统计检验；
    2. 与 μ/β 功率的相关性。
    """
    subject_class_df = (
        tau_trial_df.groupby(["subject", "class_name"], as_index=False)[["tau_mean", "mu_power", "beta_power", "correct"]]
        .mean()
        .sort_values(["subject", "class_name"])
        .reset_index(drop=True)
    )
    tau_pivot = subject_class_df.pivot(index="subject", columns="class_name", values="tau_mean")
    class_order = module.LABEL_ORDER
    friedman = friedmanchisquare(*[tau_pivot[label].to_numpy() for label in class_order])
    pairwise: Dict[str, Dict[str, float]] = {}
    for class_a, class_b in combinations(class_order, 2):
        a_values = tau_pivot[class_a].to_numpy()
        b_values = tau_pivot[class_b].to_numpy()
        test = ttest_rel(a_values, b_values)
        diff = a_values - b_values
        pairwise[f"{class_a}_vs_{class_b}"] = {
            "t_statistic": float(test.statistic),
            "p_value": float(test.pvalue),
            "cohen_d": float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)),
            "mean_diff": float(np.mean(diff)),
        }

    tau_summary: Dict[str, Dict[str, float]] = {}
    for label in class_order:
        values = tau_trial_df.loc[tau_trial_df["class_name"] == label, "tau_mean"].to_numpy()
        tau_summary[label] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
            "count": int(values.size),
        }

    mu_corr = pearsonr(subject_class_df["tau_mean"].to_numpy(), subject_class_df["mu_power"].to_numpy())
    beta_corr = pearsonr(subject_class_df["tau_mean"].to_numpy(), subject_class_df["beta_power"].to_numpy())
    summary = {
        "subject_level_friedman": {
            "statistic": float(friedman.statistic),
            "p_value": float(friedman.pvalue),
        },
        "pairwise_subject_level": pairwise,
        "class_trial_summary": tau_summary,
        "subject_class_correlation": {
            "mu_power": {
                "pearson_r": float(mu_corr.statistic),
                "p_value": float(mu_corr.pvalue),
            },
            "beta_power": {
                "pearson_r": float(beta_corr.statistic),
                "p_value": float(beta_corr.pvalue),
            },
        },
    }
    return subject_class_df, summary


def summarize_tau_window_analysis(
    module,
    tau_timecourse_df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    window_specs: List[Tuple[str, float, float]] = TAU_WINDOW_SPECS,
) -> Dict[str, object]:
    """按时间窗汇总 tau。

    对应补强实验中的“分时间段统计”：
    2.0--2.5 s、2.5--3.5 s、3.5--4.0 s 分别检验是否存在局部时间段可分性。
    """
    window_rows: List[Dict[str, object]] = []
    for window_name, start_time, end_time in window_specs:
        window_df = tau_timecourse_df.loc[
            (tau_timecourse_df["time_seconds"] >= start_time) & (tau_timecourse_df["time_seconds"] < end_time)
        ].copy()
        if window_df.empty:
            continue
        grouped = (
            window_df.groupby(["subject", "trial_index", "class_name"], as_index=False)["tau_mean"]
            .mean()
            .assign(window=window_name, window_start=start_time, window_end=end_time)
        )
        window_rows.extend(grouped.to_dict(orient="records"))

    if not window_rows:
        return {}

    window_trial_df = pd.DataFrame(window_rows).sort_values(["window", "subject", "class_name", "trial_index"]).reset_index(drop=True)
    window_trial_df.to_csv(output_dir / f"{prefix}_tau_window_trial_metrics.csv", index=False)

    subject_class_window_df = (
        window_trial_df.groupby(["subject", "class_name", "window"], as_index=False)["tau_mean"]
        .mean()
        .sort_values(["window", "subject", "class_name"])
        .reset_index(drop=True)
    )
    subject_class_window_df.to_csv(output_dir / f"{prefix}_tau_window_subject_class_summary.csv", index=False)

    summary: Dict[str, object] = {"windows": {}}
    class_order = module.LABEL_ORDER
    for window_name, start_time, end_time in window_specs:
        window_subject_df = subject_class_window_df.loc[subject_class_window_df["window"] == window_name]
        if window_subject_df.empty:
            continue
        tau_pivot = window_subject_df.pivot(index="subject", columns="class_name", values="tau_mean")
        friedman = friedmanchisquare(*[tau_pivot[label].to_numpy() for label in class_order])
        pairwise: Dict[str, Dict[str, float]] = {}
        for class_a, class_b in combinations(class_order, 2):
            a_values = tau_pivot[class_a].to_numpy()
            b_values = tau_pivot[class_b].to_numpy()
            test = ttest_rel(a_values, b_values)
            diff = a_values - b_values
            pairwise[f"{class_a}_vs_{class_b}"] = {
                "t_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "cohen_d": float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)),
                "mean_diff": float(np.mean(diff)),
            }
        class_summary: Dict[str, Dict[str, float]] = {}
        for label in class_order:
            values = window_trial_df.loc[
                (window_trial_df["window"] == window_name) & (window_trial_df["class_name"] == label),
                "tau_mean",
            ].to_numpy(dtype=float)
            class_summary[label] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)),
                "count": int(values.size),
            }
        summary["windows"][window_name] = {
            "time_range_seconds": [float(start_time), float(end_time)],
            "friedman": {
                "statistic": float(friedman.statistic),
                "p_value": float(friedman.pvalue),
            },
            "pairwise_subject_level": pairwise,
            "class_trial_summary": class_summary,
        }

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(window_specs), figsize=(12.0, 3.8), sharey=True)
    if len(window_specs) == 1:
        axes = [axes]
    for axis, (window_name, start_time, end_time) in zip(axes, window_specs):
        plot_df = window_trial_df.loc[window_trial_df["window"] == window_name].copy()
        if plot_df.empty:
            axis.set_visible(False)
            continue
        sns.boxplot(
            data=plot_df,
            x="class_name",
            y="tau_mean",
            order=class_order,
            ax=axis,
            color="#9ecae1",
            fliersize=1.5,
            linewidth=0.8,
        )
        axis.set_title(f"{start_time:.1f}-{end_time:.1f}s")
        axis.set_xlabel("")
        axis.set_ylabel("Window-averaged Tau")
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_tau_window_distributions.pdf", bbox_inches="tight")
    plt.close(fig)
    return summary


def summarize_tau_timecourse(
    timecourse_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """汇总 tau 的时间轨迹。

    对应论文补充材料中的 time-resolved tau 图，用来判断是否存在稳定时间分离。
    """
    subject_class_time = (
        timecourse_df.groupby(["subject", "class_name", "time_index", "time_seconds"], as_index=False)["tau_mean"]
        .mean()
        .sort_values(["subject", "class_name", "time_index"])
    )
    summary_df = (
        subject_class_time.groupby(["class_name", "time_index", "time_seconds"], as_index=False)
        .agg(mean_tau=("tau_mean", "mean"), std_tau=("tau_mean", "std"))
    )
    summary_df.to_csv(output_dir / "tau_timecourse_summary.csv", index=False)
    subject_class_time.to_csv(output_dir / "tau_timecourse_subject_level.csv", index=False)

    # 对应论文主文中“subject-level 峰值时间约 2.32--3.04 s、早期窗口均值约 1.49”的补充汇总。
    # 这里采用 subject-level 口径：先在每个 subject/class 内找峰值，再跨 subject 汇总，
    # 避免“先全体平均再取峰值”改变正文陈述的含义。
    coarse_windows = [
        ("early", 2.0, 2.5),
        ("mid", 2.5, 4.0),
        ("late", 4.0, 6.0),
    ]
    window_rows: List[Dict[str, float | str]] = []
    for class_name, class_group in subject_class_time.groupby("class_name"):
        peak_by_subject = (
            class_group.sort_values(["subject", "tau_mean"], ascending=[True, False])
            .groupby("subject", as_index=False)
            .first()[["subject", "tau_mean", "time_seconds"]]
        )
        window_rows.append(
            {
                "class_name": class_name,
                "window": "peak",
                "mean_tau": float(peak_by_subject["tau_mean"].mean()),
                "std_tau": float(peak_by_subject["tau_mean"].std(ddof=1)) if len(peak_by_subject) > 1 else 0.0,
                "mean_time_seconds": float(peak_by_subject["time_seconds"].mean()),
                "std_time_seconds": float(peak_by_subject["time_seconds"].std(ddof=1)) if len(peak_by_subject) > 1 else 0.0,
            }
        )
        for window_name, start_time, end_time in coarse_windows:
            window_group = class_group.loc[
                (class_group["time_seconds"] >= start_time) & (class_group["time_seconds"] < end_time)
            ]
            if window_group.empty:
                continue
            subject_window = window_group.groupby("subject", as_index=False)["tau_mean"].mean()
            window_rows.append(
                {
                    "class_name": class_name,
                    "window": window_name,
                    "mean_tau": float(subject_window["tau_mean"].mean()),
                    "std_tau": float(subject_window["tau_mean"].std(ddof=1)) if len(subject_window) > 1 else 0.0,
                    "mean_time_seconds": float((start_time + end_time) / 2.0),
                    "std_time_seconds": 0.0,
                }
            )
    pd.DataFrame(window_rows).to_csv(output_dir / "tau_time_window_summary.csv", index=False)

    sns.set_theme(style="whitegrid")
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    palette = {
        "left_hand": "#1f77b4",
        "right_hand": "#ff7f0e",
        "feet": "#2ca02c",
        "tongue": "#d62728",
    }
    for class_name, class_group in summary_df.groupby("class_name"):
        axis.plot(class_group["time_seconds"], class_group["mean_tau"], label=class_name.replace("_", " ").title(), color=palette[class_name], linewidth=2)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Mean Tau")
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "tau_timecourse_by_class.pdf", bbox_inches="tight")
    plt.close(fig)

    return {
        class_name: {
            "mean_tau": float(class_group["mean_tau"].mean()),
            "peak_tau": float(class_group["mean_tau"].max()),
            "peak_time_seconds": float(class_group.loc[class_group["mean_tau"].idxmax(), "time_seconds"]),
        }
        for class_name, class_group in summary_df.groupby("class_name")
    }


def save_tau_trial_histogram(
    module,
    tau_trial_df: pd.DataFrame,
    output_path: Path,
    figure_title: str = "Trial-averaged Tau by Class",
) -> Dict[str, Dict[str, float]]:
    """绘制 trial-averaged tau 分布图。"""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), sharex=False, sharey=False)
    summary: Dict[str, Dict[str, float]] = {}
    colors = {
        "left_hand": "#1f77b4",
        "right_hand": "#ff7f0e",
        "feet": "#2ca02c",
        "tongue": "#d62728",
    }
    for axis, class_name in zip(axes.flatten(), module.LABEL_ORDER):
        class_values = tau_trial_df.loc[tau_trial_df["class_name"] == class_name, "tau_mean"].to_numpy(dtype=float)
        if class_values.size == 0:
            axis.set_visible(False)
            continue
        sns.histplot(class_values, bins=35, stat="density", kde=True, ax=axis, color=colors[class_name], edgecolor=None)
        median = float(np.median(class_values))
        q1 = float(np.quantile(class_values, 0.25))
        q3 = float(np.quantile(class_values, 0.75))
        axis.axvline(median, color="black", linestyle="--", linewidth=1)
        axis.set_title(class_name.replace("_", " ").title())
        axis.set_xlabel("Trial-averaged Tau")
        axis.set_ylabel("Density")
        summary[class_name] = {
            "median": median,
            "q1": q1,
            "q3": q3,
            "count": int(class_values.size),
        }
    fig.suptitle(figure_title, y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return summary


def summarize_structured_metrics(
    module,
    structured_rows: List[Dict[str, object]],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Dict[str, float]]]]:
    """汇总结构化扰动的 seed-level 与 subject-level 指标。"""
    structured_seed_df = pd.DataFrame(structured_rows)
    subject_structured_df = (
        structured_seed_df.groupby(["subject", "model", "perturbation"], as_index=False)[["accuracy", "f1"]]
        .mean()
        .sort_values(["subject", "model", "perturbation"])
        .reset_index(drop=True)
    )
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_name, model_group in subject_structured_df.groupby("model"):
        summary[model_name] = {}
        for perturbation, perturb_group in model_group.groupby("perturbation"):
            values = perturb_group["accuracy"].to_numpy()
            ci_low, ci_high = module.confidence_interval_95(values)
            summary[model_name][perturbation] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
            }
    return structured_seed_df, subject_structured_df, summary


def paired_structured_tests(
    structured_df: pd.DataFrame,
    module,
    model_a: str,
    model_b: str,
) -> Dict[str, Dict[str, float]]:
    """对结构化扰动结果做配对统计检验。"""
    results: Dict[str, Dict[str, float]] = {}
    for perturbation, perturb_group in structured_df.groupby("perturbation"):
        if not {model_a, model_b}.issubset(set(perturb_group["model"].unique())):
            continue
        comparison_df = perturb_group[["subject", "model", "accuracy"]].copy()
        results[perturbation] = module.paired_test(comparison_df, model_a, model_b)
    return results


def run_sessionwise(config: SessionwiseConfig) -> Dict[str, object]:
    """运行严格的 session-wise 主实验。

    这部分直接支撑论文最关键的三类结论：
    1. 更严格协议下的模型排序；
    2. tau 的非判别性；
    3. 结构化扰动下的行为边界。
    """
    repo_root = Path(__file__).resolve().parents[1]
    module = load_core_module(repo_root)
    module.seed_everything(config.seed)
    device = module.get_device(config.device)

    data_dir = repo_root / config.data_dir
    output_dir = repo_root / config.output_dir
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = output_dir / "sessionwise_metrics.csv"
    prediction_path = output_dir / "predictions.csv"
    structured_path = output_dir / "structured_perturbation_metrics.csv"
    tau_trial_progress_path = output_dir / "_progress_tau_trial_metrics.csv"
    tau_timecourse_progress_path = output_dir / "_progress_tau_timecourse_metrics.csv"
    tau_motor_trial_progress_path = output_dir / "_progress_tau_motor_trial_metrics.csv"
    tau_motor_timecourse_progress_path = output_dir / "_progress_tau_motor_timecourse_metrics.csv"

    rows: List[Dict[str, object]] = load_records(fold_path)
    prediction_rows: List[Dict[str, object]] = load_records(prediction_path)
    parameter_counts: Dict[str, int] = {}
    tau_trial_tables: List[pd.DataFrame] = []
    tau_timecourse_tables: List[pd.DataFrame] = []
    tau_motor_trial_tables: List[pd.DataFrame] = []
    tau_motor_timecourse_tables: List[pd.DataFrame] = []
    structured_rows: List[Dict[str, object]] = load_records(structured_path)
    for path, target in [
        (tau_trial_progress_path, tau_trial_tables),
        (tau_timecourse_progress_path, tau_timecourse_tables),
        (tau_motor_trial_progress_path, tau_motor_trial_tables),
        (tau_motor_timecourse_progress_path, tau_motor_timecourse_tables),
    ]:
        existing_df = load_dataframe_if_exists(path)
        if existing_df is not None and not existing_df.empty:
            target.append(existing_df)
    completed = completed_model_subjects(rows)
    total_runs = len(config.subjects) * len(config.models)
    run_index = len(completed)
    if completed:
        print(f"resuming session-wise from {len(completed)} completed runs in {fold_path}", flush=True)

    # 外层循环对应论文中的 9 个 subject。
    for subject in config.subjects:
        X_raw, y, sessions = load_subject_session_data(module, subject, data_dir, cache_dir)
        X_raw = module.downsample_trials(X_raw, config.downsample_factor)
        train_mask = sessions == "0train"
        test_mask = sessions == "1test"
        X_train_full = X_raw[train_mask]
        y_train_full = y[train_mask]
        X_test = X_raw[test_mask]
        y_test = y[test_mask]

        # 训练 session 内部再切验证集，保持与 pooled 主实验相同的模型选择逻辑。
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=config.val_fraction,
            random_state=config.seed + subject,
        )
        train_idx, val_idx = next(splitter.split(np.zeros(len(X_train_full)), y_train_full))
        mean, std = module.compute_standardizer(X_train_full[train_idx])
        X_train = module.apply_standardizer(X_train_full[train_idx], mean, std)
        X_val = module.apply_standardizer(X_train_full[val_idx], mean, std)
        X_test_std = module.apply_standardizer(X_test, mean, std)

        n_channels = X_train.shape[1]
        n_samples = X_train.shape[2]
        if not parameter_counts:
            for model_name in config.models:
                parameter_counts[model_name] = module.get_parameter_count(
                    model_name=model_name,
                    n_channels=n_channels,
                    n_samples=n_samples,
                    n_classes=len(module.LABEL_ORDER),
                    cfc_hidden_size=config.cfc_hidden_size,
                    lstm_hidden_size=config.lstm_hidden_size,
                    cfc_dt=config.cfc_dt,
                    cfc_tau_init=config.cfc_tau_init,
                )

        train_loader = module.build_loader(X_train, y_train_full[train_idx], config.batch_size, True, device)
        val_loader = module.build_loader(X_val, y_train_full[val_idx], config.batch_size, False, device)
        test_loader = module.build_loader(X_test_std, y_test, config.batch_size, False, device)

        # 模型循环对应论文 session-wise 主表中的全部 baseline。
        for model_name in config.models:
            if (subject, model_name) in completed:
                continue
            run_index += 1
            model_seed = config.seed + subject * 100 + config.models.index(model_name)
            module.seed_everything(model_seed)
            print(f"[{run_index}/{total_runs}] subject={subject} model={model_name} device={device.type}", flush=True)
            if module.is_classical_model(model_name):
                fit_info = module.fit_riemann_tslr(
                    x_train=X_train,
                    y_train=y_train_full[train_idx],
                    x_val=X_val,
                    y_val=y_train_full[val_idx],
                )
                metrics = module.evaluate_classical_model(
                    model=fit_info["model"],
                    x=X_test_std,
                    y=y_test,
                    return_predictions=True,
                )
                runtime_model = fit_info["model"]
            else:
                runtime_model = module.build_model(
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
                metrics = module.evaluate_model(runtime_model, test_loader, device, return_predictions=True)
            rows.append(
                {
                    "subject": subject,
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
                protocol="sessionwise",
            )
            # tau 分析只对 CfC-style 进行，因为这是论文里唯一显式输出时间常数的模型。
            if model_name == "cfc":
                tau_trial_df, tau_timecourse_df = collect_cfc_trial_analysis(
                        module=module,
                        model=runtime_model,
                        loader=test_loader,
                        raw_trials=X_test,
                        subject=subject,
                        device=device,
                        sfreq=250.0 / config.downsample_factor,
                        analysis_scope="global",
                    )
                tau_trial_tables.append(tau_trial_df)
                tau_timecourse_tables.append(tau_timecourse_df)
                motor_indices = channel_indices(BCI_IV_2A_CHANNELS, MOTOR_CHANNEL_NAMES)
                X_test_motor = keep_only_channels(X_test, motor_indices)
                X_test_motor_std = keep_only_channels(X_test_std, motor_indices)
                motor_loader = module.build_loader(X_test_motor_std, y_test, config.batch_size, False, device)
                tau_motor_trial_df, tau_motor_timecourse_df = collect_cfc_trial_analysis(
                    module=module,
                    model=runtime_model,
                    loader=motor_loader,
                    raw_trials=X_test_motor,
                    subject=subject,
                    device=device,
                    sfreq=250.0 / config.downsample_factor,
                    analysis_scope="motor_channels",
                )
                tau_motor_trial_tables.append(tau_motor_trial_df)
                tau_motor_timecourse_tables.append(tau_motor_timecourse_df)
            sfreq = 250.0 / config.downsample_factor
            # 多随机种子扰动平均，对应论文 robustness 部分的 controlled perturbation study。
            for perturb_seed in range(config.structured_repeats):
                perturbation_specs = [
                    (
                        "band_limited_5db",
                        add_band_limited_noise(
                            X_test,
                            sfreq=sfreq,
                            snr_db=5.0,
                            rng=np.random.default_rng(
                                config.seed + subject * 1000 + config.models.index(model_name) * 100 + perturb_seed
                            ),
                        ),
                    ),
                    (
                        "channel_dropout_30pct",
                        apply_channel_dropout(
                            X_test,
                            drop_fraction=0.30,
                            rng=np.random.default_rng(
                                config.seed + subject * 2000 + config.models.index(model_name) * 100 + perturb_seed
                            ),
                        ),
                    ),
                ]
                for perturbation_name, perturbed_raw in perturbation_specs:
                    perturbed_std = module.apply_standardizer(perturbed_raw, mean, std)
                    if module.is_classical_model(model_name):
                        perturb_metrics = module.evaluate_classical_model(runtime_model, perturbed_std, y_test)
                    else:
                        perturbed_loader = module.build_loader(perturbed_std, y_test, config.batch_size, False, device)
                        perturb_metrics = module.evaluate_model(runtime_model, perturbed_loader, device)
                    structured_rows.append(
                        {
                            "subject": subject,
                            "model": model_name,
                            "perturbation": perturbation_name,
                            "seed": perturb_seed,
                            "accuracy": perturb_metrics["accuracy"],
                            "f1": perturb_metrics["f1"],
                    }
                )
            completed.add((subject, model_name))
            pd.DataFrame(rows).to_csv(fold_path, index=False)
            pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)
            if structured_rows:
                pd.DataFrame(structured_rows).to_csv(structured_path, index=False)
            if tau_trial_tables:
                pd.concat(tau_trial_tables, ignore_index=True).to_csv(tau_trial_progress_path, index=False)
            if tau_timecourse_tables:
                pd.concat(tau_timecourse_tables, ignore_index=True).to_csv(tau_timecourse_progress_path, index=False)
            if tau_motor_trial_tables:
                pd.concat(tau_motor_trial_tables, ignore_index=True).to_csv(tau_motor_trial_progress_path, index=False)
            if tau_motor_timecourse_tables:
                pd.concat(tau_motor_timecourse_tables, ignore_index=True).to_csv(tau_motor_timecourse_progress_path, index=False)

    # 下方开始把原始 session-wise 结果整理成正文表格、tau 统计和 robustness 附件。
    subject_df, summary = summarize_subject_metrics(rows)
    subject_df.to_csv(fold_path, index=False)
    prediction_df = pd.DataFrame(prediction_rows)
    module.save_prediction_artifacts(prediction_df, output_dir)
    stat_tests = {}
    model_pairs = list(combinations(config.models, 2))
    available_models = set(config.models)
    for model_a, model_b in model_pairs:
        if {model_a, model_b}.issubset(available_models):
            stat_tests[f"{model_a}_vs_{model_b}"] = module.paired_test(subject_df, model_a, model_b)
    module.apply_holm_correction(stat_tests, p_value_key="p_value", output_key="holm_p_value")
    module.apply_holm_correction(stat_tests, p_value_key="wilcoxon_p_value", output_key="wilcoxon_holm_p_value")
    pd.DataFrame([{"comparison": key, **value} for key, value in stat_tests.items()]).to_csv(
        output_dir / "stat_tests.csv",
        index=False,
    )

    subject_stability = {}
    if len(available_models) >= 2:
        subject_stability = module.save_subject_accuracy_artifacts(subject_df, output_dir)

    tau_analysis_summary: Dict[str, object] = {}
    if tau_trial_tables:
        tau_trial_df = pd.concat(tau_trial_tables, ignore_index=True)
        tau_trial_df.to_csv(output_dir / "tau_trial_metrics.csv", index=False)
        tau_subject_class_df, tau_analysis_summary = summarize_tau_trial_analysis(module, tau_trial_df)
        tau_subject_class_df.to_csv(output_dir / "tau_subject_class_summary.csv", index=False)
        tau_analysis_summary["trial_histogram"] = save_tau_trial_histogram(
            module,
            tau_trial_df,
            output_dir / "tau_dist_placeholder.pdf",
            figure_title="Global Trial-averaged Tau by Class",
        )
        tau_analysis_summary["windowed_analysis"] = summarize_tau_window_analysis(
            module,
            tau_timecourse_df=pd.concat(tau_timecourse_tables, ignore_index=True) if tau_timecourse_tables else pd.DataFrame(),
            output_dir=output_dir,
            prefix="global",
        )
        (output_dir / "tau_stats.json").write_text(json.dumps(tau_analysis_summary, indent=2), encoding="utf-8")
    if tau_motor_trial_tables:
        tau_motor_trial_df = pd.concat(tau_motor_trial_tables, ignore_index=True)
        tau_motor_trial_df.to_csv(output_dir / "tau_motor_trial_metrics.csv", index=False)
        tau_motor_subject_class_df, tau_motor_summary = summarize_tau_trial_analysis(module, tau_motor_trial_df)
        tau_motor_subject_class_df.to_csv(output_dir / "tau_motor_subject_class_summary.csv", index=False)
        tau_motor_summary["trial_histogram"] = save_tau_trial_histogram(
            module,
            tau_motor_trial_df,
            output_dir / "tau_motor_dist.pdf",
            figure_title="Motor-channel Trial-averaged Tau by Class",
        )
        tau_analysis_summary["motor_channel_analysis"] = {
            "channels": MOTOR_CHANNEL_NAMES,
            **tau_motor_summary,
        }
        if tau_motor_timecourse_tables:
            tau_analysis_summary["motor_channel_analysis"]["windowed_analysis"] = summarize_tau_window_analysis(
                module,
                tau_timecourse_df=pd.concat(tau_motor_timecourse_tables, ignore_index=True),
                output_dir=output_dir,
                prefix="motor",
            )
        (output_dir / "tau_stats.json").write_text(json.dumps(tau_analysis_summary, indent=2), encoding="utf-8")
    if tau_timecourse_tables:
        tau_timecourse_df = pd.concat(tau_timecourse_tables, ignore_index=True)
        tau_analysis_summary["timecourse_summary"] = summarize_tau_timecourse(tau_timecourse_df, output_dir)
        (output_dir / "tau_stats.json").write_text(json.dumps(tau_analysis_summary, indent=2), encoding="utf-8")

    structured_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    structured_tests: Dict[str, Dict[str, Dict[str, float]]] = {}
    if structured_rows:
        structured_seed_df, structured_subject_df, structured_summary = summarize_structured_metrics(module, structured_rows)
        structured_seed_df.to_csv(output_dir / "structured_perturbation_metrics.csv", index=False)
        structured_subject_df.to_csv(output_dir / "structured_perturbation_subject_summary.csv", index=False)
        pd.DataFrame(
            [
                {
                    "model": model_name,
                    "perturbation": perturbation,
                    "accuracy_mean": metrics["mean"],
                    "accuracy_std": metrics["std"],
                    "accuracy_ci95_low": metrics["ci95_low"],
                    "accuracy_ci95_high": metrics["ci95_high"],
                }
                for model_name, model_summary in structured_summary.items()
                for perturbation, metrics in model_summary.items()
            ]
        ).to_csv(output_dir / "structured_perturbation_summary.csv", index=False)
        for model_a, model_b in model_pairs:
            if {model_a, model_b}.issubset(available_models):
                structured_tests[f"{model_a}_vs_{model_b}"] = paired_structured_tests(
                    structured_subject_df,
                    module,
                    model_a,
                    model_b,
                )
        structured_entries = [
            (comparison_name, perturbation_name)
            for comparison_name, perturbation_map in structured_tests.items()
            for perturbation_name in perturbation_map.keys()
        ]
        adjusted_p = module.holm_adjust(
            [structured_tests[comparison_name][perturbation_name]["p_value"] for comparison_name, perturbation_name in structured_entries]
        )
        adjusted_w = module.holm_adjust(
            [
                structured_tests[comparison_name][perturbation_name]["wilcoxon_p_value"]
                for comparison_name, perturbation_name in structured_entries
            ]
        )
        for (comparison_name, perturbation_name), holm_p, holm_w in zip(structured_entries, adjusted_p, adjusted_w):
            structured_tests[comparison_name][perturbation_name]["holm_p_value"] = float(holm_p)
            structured_tests[comparison_name][perturbation_name]["wilcoxon_holm_p_value"] = float(holm_w)
        pd.DataFrame(
            [
                {
                    "comparison": comparison_name,
                    "perturbation": perturbation_name,
                    **metrics,
                }
                for comparison_name, perturbation_map in structured_tests.items()
                for perturbation_name, metrics in perturbation_map.items()
            ]
        ).to_csv(output_dir / "structured_perturbation_stats.csv", index=False)

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "config": asdict(config),
        "parameter_counts": parameter_counts,
        "summary": summary,
        "stat_tests": stat_tests,
        "subject_stability": subject_stability,
        "tau_analysis": tau_analysis_summary,
        "structured_perturbation_summary": structured_summary,
        "structured_perturbation_tests": structured_tests,
    }
    (output_dir / "sessionwise_results_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> SessionwiseConfig:
    """解析 session-wise 主实验命令行参数。"""
    parser = argparse.ArgumentParser(description="Run session-wise MI-EEG comparison (session 1 train, session 2 test).")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument(
        "--models",
        nargs="*",
        default=["shallow_convnet", "riemann_tslr", "eegnet", "tiny_transformer", "hybrid_cfc", "cfc", "lstm"],
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
    parser.add_argument("--cfc-dt", type=float, default=1.0)
    parser.add_argument("--cfc-tau-init", type=float, default=1.0)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--structured-repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/bspc_sessionwise")
    args = parser.parse_args()
    return SessionwiseConfig(
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
        cfc_dt=args.cfc_dt,
        cfc_tau_init=args.cfc_tau_init,
        downsample_factor=args.downsample_factor,
        structured_repeats=args.structured_repeats,
        device=args.device,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    run_sessionwise(parse_args())
