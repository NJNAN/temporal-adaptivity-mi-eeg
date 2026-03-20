from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from moabb.paradigms import MotorImagery
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

try:
    from moabb.datasets import BNCI2014_001 as BNCI2014Dataset
except ImportError:
    from moabb.datasets import BNCI2014001 as BNCI2014Dataset


LABEL_ORDER = ["left_hand", "right_hand", "feet", "tongue"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABEL_ORDER)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}
SNR_LEVELS = (10, 5, 0)


@dataclass
class ExperimentConfig:
    subjects: List[int]
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
    smoke_test: bool


class EEGNet(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int, dropout: float = 0.5) -> None:
        super().__init__()
        f1 = 8
        d = 2
        f2 = 16
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, 64), padding="same", bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, kernel_size=(n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(f1 * d, f1 * d, kernel_size=(1, 16), padding="same", groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            feature_dim = int(np.prod(self.features(dummy).shape[1:]))
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x = x.unsqueeze(1)
        features = self.features(x).flatten(1)
        logits = self.classifier(features)
        if return_aux:
            return logits, {}
        return logits


class LSTMClassifier(nn.Module):
    def __init__(self, n_channels: int, hidden_size: int, n_classes: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.4,
        )
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x = x.transpose(1, 2)
        outputs, _ = self.lstm(x)
        pooled = torch.cat([outputs.mean(dim=1), outputs.amax(dim=1)], dim=1)
        logits = self.classifier(self.dropout(pooled))
        if return_aux:
            return logits, {}
        return logits


class AdaptiveCfCCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_size + hidden_size)
        self.candidate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.tau_mlp = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.zeros_(self.tau_mlp[-1].weight)
        nn.init.constant_(self.tau_mlp[-1].bias, inverse_softplus(1.0))

    def forward(self, x_t: torch.Tensor, hidden: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        joint = self.norm(torch.cat([x_t, hidden], dim=1))
        tau = F.softplus(self.tau_mlp(joint)) + 1e-3
        decay = torch.exp(-dt / tau)
        candidate = self.candidate(joint)
        hidden = decay * hidden + (1.0 - decay) * candidate
        return hidden, tau


class CfCClassifier(nn.Module):
    def __init__(self, n_channels: int, hidden_size: int, n_classes: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_channels, hidden_size)
        self.cell = AdaptiveCfCCell(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_size * 2, n_classes)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x = x.transpose(1, 2)
        x = torch.tanh(self.input_proj(x))
        batch_size, seq_len, _ = x.shape
        hidden = x.new_zeros(batch_size, self.hidden_size)
        hidden_steps = []
        tau_steps = [] if return_aux else None
        for step in range(seq_len):
            hidden, tau = self.cell(x[:, step, :], hidden, dt=1.0)
            hidden_steps.append(hidden)
            if return_aux:
                tau_steps.append(tau)
        hidden_seq = torch.stack(hidden_steps, dim=1)
        pooled = torch.cat([hidden_seq.mean(dim=1), hidden_seq.amax(dim=1)], dim=1)
        logits = self.classifier(self.dropout(pooled))
        if return_aux:
            return logits, {"tau": torch.stack(tau_steps, dim=1)}
        return logits


class HybridCfCClassifier(nn.Module):
    def __init__(self, n_channels: int, hidden_size: int, n_classes: int) -> None:
        super().__init__()
        temporal_filters = 8
        spatial_filters = 16
        self.frontend = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, 64), padding="same", bias=False),
            nn.BatchNorm2d(temporal_filters),
            nn.Conv2d(
                temporal_filters,
                spatial_filters,
                kernel_size=(n_channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25),
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                kernel_size=(1, 16),
                padding="same",
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(spatial_filters, spatial_filters, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25),
        )
        self.bridge = nn.Linear(spatial_filters, hidden_size)
        self.cell = AdaptiveCfCCell(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_size, n_classes)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x = x.unsqueeze(1)
        x = self.frontend(x)
        x = x.squeeze(2).transpose(1, 2)
        x = torch.tanh(self.bridge(x))
        batch_size, seq_len, _ = x.shape
        hidden = x.new_zeros(batch_size, self.hidden_size)
        for step in range(seq_len):
            hidden, _ = self.cell(x[:, step, :], hidden, dt=1.0)
        logits = self.classifier(self.dropout(hidden))
        if return_aux:
            return logits, {}
        return logits


def inverse_softplus(value: float) -> float:
    return math.log(math.exp(value) - 1.0)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_mne_path(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(data_dir.resolve())


def clear_subject_downloads(subject: int, data_dir: Path) -> None:
    download_dir = data_dir / "MNE-bnci-data" / "database" / "data-sets" / "001-2014"
    if not download_dir.exists():
        return
    prefix = f"A{subject:02d}"
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


def load_subject_data(subject: int, data_dir: Path, cache_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"subject_{subject:02d}.npz"
    if cache_file.exists():
        cached = np.load(cache_file)
        return cached["X"].astype(np.float32), cached["y"].astype(np.int64)

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ensure_mne_path(data_dir)
    dataset = BNCI2014Dataset()
    paradigm = MotorImagery(n_classes=4, fmin=8, fmax=30, tmin=0.0, tmax=4.0)
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
            y_int = np.asarray([LABEL_TO_INDEX[str(label)] for label in y], dtype=np.int64)
            X = X.astype(np.float32)
            np.savez_compressed(cache_file, X=X, y=y_int)
            return X, y_int
        except Exception as exc:
            last_error = exc
            print(f"subject={subject} data preparation failed on attempt {attempt}/4: {exc}", flush=True)
            clear_subject_downloads(subject, data_dir)
            time.sleep(3 * attempt)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to prepare data for subject {subject}.")


def prepare_subject_cache(subjects: List[int], data_dir: Path, cache_dir: Path) -> None:
    for subject in subjects:
        cache_file = cache_dir / f"subject_{subject:02d}.npz"
        if cache_file.exists():
            print(f"subject={subject} cache already available", flush=True)
            continue
        print(f"preparing subject={subject} cache", flush=True)
        load_subject_data(subject, data_dir, cache_dir)


def compute_standardizer(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    standardized = (x - mean) / std
    return np.clip(standardized, -6.0, 6.0).astype(np.float32)


def downsample_trials(x: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return x
    return x[:, :, ::factor].copy()


def build_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def build_model(
    model_name: str,
    n_channels: int,
    n_samples: int,
    n_classes: int,
    cfc_hidden_size: int,
    lstm_hidden_size: int,
) -> nn.Module:
    if model_name == "eegnet":
        return EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes)
    if model_name == "lstm":
        return LSTMClassifier(n_channels=n_channels, hidden_size=lstm_hidden_size, n_classes=n_classes)
    if model_name == "cfc":
        return CfCClassifier(n_channels=n_channels, hidden_size=cfc_hidden_size, n_classes=n_classes)
    if model_name == "hybrid_cfc":
        return HybridCfCClassifier(n_channels=n_channels, hidden_size=cfc_hidden_size, n_classes=n_classes)
    raise ValueError(f"Unsupported model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def accuracy_and_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred) * 100.0),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def add_gaussian_noise(x: np.ndarray, snr_db: int, rng: np.random.Generator) -> np.ndarray:
    signal_power = np.mean(np.square(x), axis=(1, 2), keepdims=True)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, 1.0, size=x.shape).astype(np.float32) * np.sqrt(noise_power).astype(np.float32)
    return x + noise


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    min_epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> Dict[str, object]:
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_val_accuracy = float("-inf")
    best_epoch = 0
    epochs_since_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_examples = 0
        for features, targets in train_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_total += float(loss.item()) * targets.size(0)
            train_examples += targets.size(0)

        val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step()

        val_loss = val_metrics["loss"]
        val_accuracy = val_metrics["accuracy"]
        accuracy_improved = val_accuracy > best_val_accuracy + 1e-4
        tied_accuracy = abs(val_accuracy - best_val_accuracy) <= 1e-4
        loss_improved = val_loss < best_val_loss - 1e-4
        if accuracy_improved or (tied_accuracy and loss_improved):
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_since_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            if epoch >= min_epochs:
                epochs_since_improvement += 1

        if epoch >= min_epochs and epochs_since_improvement >= patience:
            break

    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    collect_tau: bool = False,
    tau_store: Dict[str, List[float]] | None = None,
    tau_rng: np.random.Generator | None = None,
    max_tau_per_class: int = 75000,
    sample_per_class_per_batch: int = 2500,
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    losses: List[float] = []
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if collect_tau:
                logits, aux = model(features, return_aux=True)
            else:
                logits = model(features)
                aux = {}

            loss = criterion(logits, targets)
            predictions = torch.argmax(logits, dim=1)
            losses.append(float(loss.item()) * targets.size(0))
            y_true.append(targets.cpu().numpy())
            y_pred.append(predictions.cpu().numpy())

            if collect_tau and tau_store is not None and tau_rng is not None:
                tau_tensor = aux["tau"].detach().cpu().numpy()
                labels = targets.cpu().numpy()
                for class_index in np.unique(labels):
                    class_name = INDEX_TO_LABEL[int(class_index)]
                    remaining = max_tau_per_class - len(tau_store[class_name])
                    if remaining <= 0:
                        continue
                    class_values = tau_tensor[labels == class_index].reshape(-1)
                    if class_values.size == 0:
                        continue
                    sample_size = min(sample_per_class_per_batch, class_values.size, remaining)
                    if class_values.size > sample_size:
                        sample_indices = tau_rng.choice(class_values.size, size=sample_size, replace=False)
                        class_values = class_values[sample_indices]
                    tau_store[class_name].extend(class_values.astype(np.float32).tolist())

    y_true_array = np.concatenate(y_true)
    y_pred_array = np.concatenate(y_pred)
    metrics = accuracy_and_f1(y_true_array, y_pred_array)
    metrics["loss"] = float(sum(losses) / len(y_true_array))
    return metrics


def format_metric(mean: float, std: float, digits: int = 1) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def summarize_subject_metrics(fold_rows: List[Dict[str, object]]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    fold_df = pd.DataFrame(fold_rows)
    subject_summary_df = (
        fold_df.groupby(["subject", "model"], as_index=False)[["accuracy", "f1"]]
        .mean()
        .sort_values(["subject", "model"])
        .reset_index(drop=True)
    )
    summary: Dict[str, Dict[str, float]] = {}
    for model_name, group in subject_summary_df.groupby("model"):
        accuracy_values = group["accuracy"].to_numpy()
        f1_values = group["f1"].to_numpy()
        accuracy_std = float(np.std(accuracy_values, ddof=1)) if len(accuracy_values) > 1 else 0.0
        f1_std = float(np.std(f1_values, ddof=1)) if len(f1_values) > 1 else 0.0
        summary[model_name] = {
            "accuracy_mean": float(np.mean(accuracy_values)),
            "accuracy_std": accuracy_std,
            "f1_mean": float(np.mean(f1_values)),
            "f1_std": f1_std,
        }
    return subject_summary_df, summary


def save_subject_accuracy_artifacts(
    subject_summary_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_wide = subject_summary_df.pivot(index="subject", columns="model", values="accuracy").reset_index()
    subject_wide.to_csv(output_dir / "subject_accuracy_table.csv", index=False)

    display_map = {
        "eegnet": "EEGNet",
        "hybrid_cfc": "Hybrid-CfC",
        "cfc": "CfC",
        "lstm": "LSTM",
    }
    order = ["EEGNet", "Hybrid-CfC", "CfC", "LSTM"]
    palette = {
        "EEGNet": "#1f77b4",
        "Hybrid-CfC": "#ff7f0e",
        "CfC": "#2ca02c",
        "LSTM": "#d62728",
    }
    plot_df = subject_summary_df.copy()
    plot_df["model_display"] = plot_df["model"].map(display_map)

    sns.set_theme(style="whitegrid")
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    sns.boxplot(
        data=plot_df,
        x="model_display",
        y="accuracy",
        order=order,
        hue="model_display",
        palette=palette,
        dodge=False,
        legend=False,
        width=0.55,
        fliersize=0,
        ax=axis,
    )
    sns.stripplot(
        data=plot_df,
        x="model_display",
        y="accuracy",
        order=order,
        color="black",
        size=4,
        alpha=0.7,
        ax=axis,
    )
    axis.set_xlabel("Model")
    axis.set_ylabel("Subject-wise Accuracy (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "subject_accuracy_boxplot.pdf", bbox_inches="tight")
    plt.close(fig)

    subject_pivot = subject_summary_df.pivot(index="subject", columns="model", values="accuracy")
    return {
        "cfc_gt_lstm_subjects": int((subject_pivot["cfc"] > subject_pivot["lstm"]).sum()),
        "hybrid_gt_cfc_subjects": int((subject_pivot["hybrid_cfc"] > subject_pivot["cfc"]).sum()),
        "hybrid_gt_eegnet_subjects": int((subject_pivot["hybrid_cfc"] > subject_pivot["eegnet"]).sum()),
        "cfc_gt_eegnet_subjects": int((subject_pivot["cfc"] > subject_pivot["eegnet"]).sum()),
    }


def summarize_noise_metrics(
    noise_rows: List[Dict[str, object]]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Dict[str, float]]]]:
    noise_df = pd.DataFrame(noise_rows)
    subject_noise_df = (
        noise_df.groupby(["subject", "model", "snr"], as_index=False)["accuracy"]
        .mean()
        .sort_values(["subject", "model", "snr"])
    )
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_name, model_group in subject_noise_df.groupby("model"):
        summary[model_name] = {}
        for snr_value, snr_group in model_group.groupby("snr"):
            values = snr_group["accuracy"].to_numpy()
            summary[model_name][str(int(snr_value))] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
    return subject_noise_df, summary


def paired_noise_tests(subject_noise_df: pd.DataFrame, model_a: str, model_b: str) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for snr_value, snr_group in subject_noise_df.groupby("snr"):
        pivot = snr_group.pivot(index="subject", columns="model", values="accuracy")
        a_values = pivot[model_a].to_numpy()
        b_values = pivot[model_b].to_numpy()
        if len(a_values) < 2:
            summary[str(int(snr_value))] = {
                "t_statistic": float("nan"),
                "p_value": float("nan"),
                "mean_diff": float("nan"),
            }
            continue
        test = ttest_rel(a_values, b_values)
        summary[str(int(snr_value))] = {
            "t_statistic": float(test.statistic),
            "p_value": float(test.pvalue),
            "mean_diff": float(np.mean(a_values - b_values)),
        }
    return summary


def paired_test(subject_summary_df: pd.DataFrame, model_a: str, model_b: str) -> Dict[str, float]:
    pivot = subject_summary_df.pivot(index="subject", columns="model", values="accuracy")
    a_values = pivot[model_a].to_numpy()
    b_values = pivot[model_b].to_numpy()
    if len(a_values) < 2:
        return {
            "t_statistic": float("nan"),
            "p_value": float("nan"),
            "cohen_d": float("nan"),
        }
    test = ttest_rel(a_values, b_values)
    diff = a_values - b_values
    cohen_d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-8))
    return {
        "t_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "cohen_d": cohen_d,
    }


def save_tau_figure(tau_store: Dict[str, List[float]], output_path: Path) -> Dict[str, Dict[str, float]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False, sharey=False)
    tau_summary: Dict[str, Dict[str, float]] = {}
    colors = {
        "left_hand": "#1f77b4",
        "right_hand": "#ff7f0e",
        "feet": "#2ca02c",
        "tongue": "#d62728",
    }

    for axis, label in zip(axes.flatten(), LABEL_ORDER):
        values = np.asarray(tau_store[label], dtype=np.float32)
        if values.size == 0:
            axis.set_visible(False)
            continue
        sns.histplot(values, bins=50, stat="density", kde=True, ax=axis, color=colors[label], edgecolor=None)
        median = float(np.median(values))
        q1 = float(np.quantile(values, 0.25))
        q3 = float(np.quantile(values, 0.75))
        tau_summary[label] = {
            "median": median,
            "q1": q1,
            "q3": q3,
            "count": int(values.size),
        }
        axis.axvline(median, color="black", linestyle="--", linewidth=1)
        axis.set_title(label.replace("_", " ").title())
        axis.set_xlabel("Learned Tau")
        axis.set_ylabel("Density")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return tau_summary


def load_progress_csv(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def load_tau_store(path: Path) -> Dict[str, List[float]]:
    if not path.exists():
        return {label: [] for label in LABEL_ORDER}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return {label: list(map(float, loaded.get(label, []))) for label in LABEL_ORDER}


def persist_progress(
    fold_rows: List[Dict[str, object]],
    noise_rows: List[Dict[str, object]],
    tau_store: Dict[str, List[float]],
    fold_path: Path,
    noise_path: Path,
    tau_path: Path,
) -> None:
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    pd.DataFrame(noise_rows).to_csv(noise_path, index=False)
    tau_path.write_text(json.dumps(tau_store), encoding="utf-8")


def run_experiment(config: ExperimentConfig) -> Dict[str, object]:
    seed_everything(config.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = get_device(config.device)
    root_dir = Path(__file__).resolve().parents[1]
    data_dir = root_dir / config.data_dir
    output_dir = root_dir / config.output_dir
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics_path = output_dir / "fold_metrics.csv"
    noise_metrics_path = output_dir / "noise_metrics.csv"
    tau_store_path = output_dir / "tau_samples.json"

    prepare_subject_cache(config.subjects, data_dir, cache_dir)

    fold_rows = load_progress_csv(fold_metrics_path)
    noise_rows = load_progress_csv(noise_metrics_path)
    tau_store = load_tau_store(tau_store_path)
    completed_runs = {
        (int(row["subject"]), int(row["fold"]), str(row["model"]))
        for row in fold_rows
    }
    tau_rng = np.random.default_rng(config.seed + 99)

    model_names = ["eegnet", "lstm", "cfc", "hybrid_cfc"]
    total_runs = len(config.subjects) * config.num_folds * len(model_names)
    run_index = len(completed_runs)
    parameter_counts: Dict[str, int] = {}

    for subject in config.subjects:
        X_raw, y = load_subject_data(subject=subject, data_dir=data_dir, cache_dir=cache_dir)
        X_raw = downsample_trials(X_raw, config.downsample_factor)
        n_trials, n_channels, n_samples = X_raw.shape
        if not parameter_counts:
            for model_name in model_names:
                probe_model = build_model(
                    model_name=model_name,
                    n_channels=n_channels,
                    n_samples=n_samples,
                    n_classes=len(LABEL_ORDER),
                    cfc_hidden_size=config.cfc_hidden_size,
                    lstm_hidden_size=config.lstm_hidden_size,
                )
                parameter_counts[model_name] = count_parameters(probe_model)
        splitter = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed + subject)

        for fold_idx, (train_val_idx, test_idx) in enumerate(splitter.split(np.zeros(n_trials), y), start=1):
            val_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=config.val_fraction,
                random_state=config.seed + subject * 10 + fold_idx,
            )
            inner_train_idx, val_idx = next(
                val_splitter.split(np.zeros(len(train_val_idx)), y[train_val_idx])
            )
            train_idx = train_val_idx[inner_train_idx]
            val_idx = train_val_idx[val_idx]

            mean, std = compute_standardizer(X_raw[train_idx])
            X_train = apply_standardizer(X_raw[train_idx], mean, std)
            X_val = apply_standardizer(X_raw[val_idx], mean, std)
            X_test = apply_standardizer(X_raw[test_idx], mean, std)

            train_loader = build_loader(X_train, y[train_idx], config.batch_size, True, device)
            val_loader = build_loader(X_val, y[val_idx], config.batch_size, False, device)
            test_loader = build_loader(X_test, y[test_idx], config.batch_size, False, device)

            for model_name in model_names:
                run_key = (subject, fold_idx, model_name)
                if run_key in completed_runs:
                    print(
                        f"[resume {run_index}/{total_runs}] subject={subject} fold={fold_idx}/{config.num_folds} "
                        f"model={model_name} already complete",
                        flush=True,
                    )
                    continue

                run_index += 1
                model_seed = config.seed + subject * 100 + fold_idx * 10 + model_names.index(model_name)
                seed_everything(model_seed)
                model = build_model(
                    model_name=model_name,
                    n_channels=n_channels,
                    n_samples=n_samples,
                    n_classes=len(LABEL_ORDER),
                    cfc_hidden_size=config.cfc_hidden_size,
                    lstm_hidden_size=config.lstm_hidden_size,
                )
                print(
                    f"[{run_index}/{total_runs}] subject={subject} fold={fold_idx}/{config.num_folds} "
                    f"model={model_name} device={device.type}",
                    flush=True,
                )
                fit_info = train_one_model(
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
                clean_metrics = evaluate_model(
                    model=model,
                    loader=test_loader,
                    device=device,
                    collect_tau=model_name == "cfc",
                    tau_store=tau_store if model_name == "cfc" else None,
                    tau_rng=tau_rng if model_name == "cfc" else None,
                )
                fold_rows.append(
                    {
                        "subject": subject,
                        "fold": fold_idx,
                        "model": model_name,
                        "accuracy": clean_metrics["accuracy"],
                        "f1": clean_metrics["f1"],
                        "best_epoch": fit_info["best_epoch"],
                        "best_val_loss": fit_info["best_val_loss"],
                        "best_val_accuracy": fit_info["best_val_accuracy"],
                    }
                )

                for snr_db in SNR_LEVELS:
                    noise_rng = np.random.default_rng(config.seed + subject * 1000 + fold_idx * 100 + snr_db)
                    X_noisy = add_gaussian_noise(X_raw[test_idx], snr_db=snr_db, rng=noise_rng)
                    X_noisy = apply_standardizer(X_noisy, mean, std)
                    noisy_loader = build_loader(X_noisy, y[test_idx], config.batch_size, False, device)
                    noisy_metrics = evaluate_model(model=model, loader=noisy_loader, device=device)
                    noise_rows.append(
                        {
                            "subject": subject,
                            "fold": fold_idx,
                            "model": model_name,
                            "snr": snr_db,
                            "accuracy": noisy_metrics["accuracy"],
                        }
                    )

                completed_runs.add(run_key)
                persist_progress(
                    fold_rows=fold_rows,
                    noise_rows=noise_rows,
                    tau_store=tau_store,
                    fold_path=fold_metrics_path,
                    noise_path=noise_metrics_path,
                    tau_path=tau_store_path,
                )

    subject_summary_df, metric_summary = summarize_subject_metrics(fold_rows)
    subject_stability = save_subject_accuracy_artifacts(subject_summary_df, output_dir)
    subject_noise_df, noise_summary = summarize_noise_metrics(noise_rows)
    stat_tests = {
        "cfc_vs_lstm": paired_test(subject_summary_df, "cfc", "lstm"),
        "cfc_vs_eegnet": paired_test(subject_summary_df, "cfc", "eegnet"),
        "hybrid_vs_cfc": paired_test(subject_summary_df, "hybrid_cfc", "cfc"),
        "hybrid_vs_eegnet": paired_test(subject_summary_df, "hybrid_cfc", "eegnet"),
    }
    noise_stat_tests = {
        "cfc_vs_eegnet": paired_noise_tests(subject_noise_df, "cfc", "eegnet"),
        "hybrid_vs_eegnet": paired_noise_tests(subject_noise_df, "hybrid_cfc", "eegnet"),
    }
    tau_summary = save_tau_figure(tau_store, output_dir / "tau_dist_placeholder.pdf")

    noise_summary_rows = [
        {
            "model": model_name,
            "snr": int(snr_key),
            "mean": snr_summary["mean"],
            "std": snr_summary["std"],
        }
        for model_name, model_summary in noise_summary.items()
        for snr_key, snr_summary in model_summary.items()
    ]

    pd.DataFrame(fold_rows).to_csv(fold_metrics_path, index=False)
    pd.DataFrame(noise_rows).to_csv(noise_metrics_path, index=False)
    subject_summary_df.to_csv(output_dir / "subject_summary.csv", index=False)
    subject_noise_df.to_csv(output_dir / "noise_subject_level.csv", index=False)
    pd.DataFrame(noise_summary_rows).to_csv(output_dir / "noise_subject_summary.csv", index=False)
    (output_dir / "stability_and_noise_stats.json").write_text(
        json.dumps(
            {
                "subject_stability": subject_stability,
                "noise_stat_tests": noise_stat_tests,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "config": asdict(config),
        "parameter_counts": parameter_counts,
        "summary": metric_summary,
        "subject_stability": subject_stability,
        "noise_summary": noise_summary,
        "noise_stat_tests": noise_stat_tests,
        "stat_tests": stat_tests,
        "tau_summary": tau_summary,
        "label_order": LABEL_ORDER,
    }
    (output_dir / "results_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run MI-EEG experiments for EEGNet, LSTM, and CfC.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a short validation experiment.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--subjects", type=int, nargs="*", default=None)
    parser.add_argument("--num-folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260318)
    parser.add_argument("--cfc-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/full")
    args = parser.parse_args()

    if args.smoke_test:
        subjects = args.subjects or [1]
        return ExperimentConfig(
            subjects=subjects,
            num_folds=args.num_folds or 2,
            epochs=args.epochs or 3,
            patience=args.patience or 2,
            min_epochs=args.min_epochs or 3,
            batch_size=args.batch_size or 64,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            val_fraction=args.val_fraction,
            seed=args.seed,
            cfc_hidden_size=args.cfc_hidden_size,
            lstm_hidden_size=args.lstm_hidden_size,
            downsample_factor=args.downsample_factor,
            device=args.device,
            data_dir=args.data_dir,
            output_dir="outputs/smoke",
            smoke_test=True,
        )

    return ExperimentConfig(
        subjects=args.subjects or list(range(1, 10)),
        num_folds=args.num_folds or 5,
        epochs=args.epochs or 80,
        patience=args.patience or 20,
        min_epochs=args.min_epochs or 25,
        batch_size=args.batch_size or 64,
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
        smoke_test=False,
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    config = parse_args()
    run_experiment(config)


if __name__ == "__main__":
    main()
