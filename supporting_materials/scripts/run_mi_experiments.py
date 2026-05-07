"""论文主实验脚本。

对应论文：
1. pooled subject-wise 5-fold 主实验。
2. 表格中的模型参数量、主结果、统计检验和噪声鲁棒性。
3. 基础 tau 直方图、逐被试箱线图、混淆矩阵等补充图表。

这份脚本是整篇论文最核心的实现入口。更细的章节映射见
`CODE_TO_PAPER_MAPPING.md`，但这里的注释会直接指出每段代码服务于论文哪一部分。
"""

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
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from moabb.paradigms import MotorImagery
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.stats import t as student_t
from scipy.stats import ttest_rel, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

try:
    from moabb.datasets import BNCI2014_001 as BNCI2014Dataset
except ImportError:
    from moabb.datasets import BNCI2014001 as BNCI2014Dataset


# 论文实验中的标签定义、模型显示名、表格顺序和绘图配色。
LABEL_ORDER = ["left_hand", "right_hand", "feet", "tongue"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABEL_ORDER)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}
SNR_LEVELS = (10, 5, 0)
RIEMANN_C_VALUES = (0.1, 1.0, 10.0)
CLASSICAL_MODELS = {"riemann_tslr"}
MODEL_DISPLAY_NAMES = {
    "eegnet": "EEGNet",
    "shallow_convnet": "Shallow ConvNet",
    "tiny_transformer": "Tiny-Transformer",
    "hybrid_cfc": "Hybrid-CfC-style",
    "cfc": "CfC-style",
    "gru": "GRU",
    "lstm": "LSTM",
    "riemann_tslr": "Riemann-TSLR",
}
MODEL_ORDER = [
    "Shallow ConvNet",
    "Riemann-TSLR",
    "EEGNet",
    "Tiny-Transformer",
    "Hybrid-CfC-style",
    "CfC-style",
    "GRU",
    "LSTM",
]
MODEL_PALETTE = {
    "Shallow ConvNet": "#6b5b95",
    "Riemann-TSLR": "#8c564b",
    "EEGNet": "#1f77b4",
    "Tiny-Transformer": "#17becf",
    "Hybrid-CfC-style": "#ff7f0e",
    "CfC-style": "#2ca02c",
    "GRU": "#9467bd",
    "LSTM": "#d62728",
}


@dataclass
class ExperimentConfig:
    """主实验配置。

    对应论文 Dataset/Setup 部分中的统一训练协议。
    """
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
    smoke_test: bool


class EEGNet(nn.Module):
    """EEGNet 基线。

    对应论文主表中的紧凑 CNN baseline，用来代表强空间-频谱归纳偏置。
    """
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


class ShallowConvNet(nn.Module):
    """Shallow ConvNet 基线。

    对应论文中表现最强的浅层 band-power CNN，对 boundary claim 很关键。
    """
    def __init__(self, n_channels: int, n_samples: int, n_classes: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, 25), bias=False),
            nn.Conv2d(40, 40, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(40),
            nn.Identity(),
        )
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            feature_map = self._forward_features(dummy)
            feature_dim = int(np.prod(feature_map.shape[1:]))
        self.classifier = nn.Linear(feature_dim, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x = x.unsqueeze(1)
        features = self._forward_features(x).flatten(1)
        logits = self.classifier(features)
        if return_aux:
            return logits, {}
        return logits


class LSTMClassifier(nn.Module):
    """LSTM 基线。

    对应论文里“CfC-style 稳定优于标准 recurrent baseline”的参照对象。
    """
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


class GRUClassifier(nn.Module):
    """GRU 补充对照。

    对应修稿阶段对“CfC-style 是否只是比 LSTM 门控更强”的额外检验。
    """
    def __init__(self, n_channels: int, hidden_size: int, n_classes: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
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
        outputs, _ = self.gru(x)
        pooled = torch.cat([outputs.mean(dim=1), outputs.amax(dim=1)], dim=1)
        logits = self.classifier(self.dropout(pooled))
        if return_aux:
            return logits, {}
        return logits


class TinyTransformerClassifier(nn.Module):
    """轻量 attention baseline。

    对应修稿阶段新增的“更强序列建模能力是否会改变结论”对照。
    该模型保持紧凑，只用于提供一个最小 transformer/attention 参照。
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int, d_model: int = 64) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_samples, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(d_model * 2, n_classes)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x = x.transpose(1, 2)
        x = self.input_proj(x) + self.pos_embed[:, : x.size(1), :]
        x = self.encoder(x)
        x = self.norm(x)
        pooled = torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=1)
        logits = self.classifier(self.dropout(pooled))
        if return_aux:
            return logits, {}
        return logits


class AdaptiveCfCCell(nn.Module):
    """CfC-style 单元。

    对应论文方法里的 `CfC-style exponential-decay update`：
    使用 input-dependent、per-unit 的 tau 来实现离散采样下的指数混合。
    """
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
        """执行单时间步更新。

        这一步直接对应论文公式中的 `exp(-Δt / τ)` 更新规则。
        """
        joint = self.norm(torch.cat([x_t, hidden], dim=1))
        tau = F.softplus(self.tau_mlp(joint)) + 1e-3
        decay = torch.exp(-dt / tau)
        candidate = self.candidate(joint)
        hidden = decay * hidden + (1.0 - decay) * candidate
        return hidden, tau


class CfCClassifier(nn.Module):
    """纯时间建模的 CfC-style 分类器。

    对应论文中的核心连续时间模型，用来检验“temporal adaptivity 是否足以支撑判别”。
    """
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
    """诊断性 Hybrid-CfC-style 模型。

    对应论文中用于隔离“轻量空间前端 + 连续时间单元”作用的 diagnostic design，
    不是为了追求最优 hybrid 性能。
    """
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
    """把目标初值映射到 softplus 之前的偏置空间。

    用于把 tau 初始化到论文希望的正值范围附近。
    """
    return math.log(math.exp(value) - 1.0)


def seed_everything(seed: int) -> None:
    """统一设置随机种子。

    对应论文复现性要求中的随机控制入口。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    """选择运行设备，对应论文中的 CPU/CUDA 运行口径。"""
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_mne_path(data_dir: Path) -> None:
    """配置 MNE/MOABB 数据缓存目录。"""
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(data_dir.resolve())


def clear_subject_downloads(subject: int, data_dir: Path) -> None:
    """清理损坏的被试下载文件，避免 MOABB 数据准备失败。"""
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
    """载入单被试 IV-2a 数据。

    对应论文的主数据集准备：4 类 MI、8-30 Hz、标准 trial 切片。
    """
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
    """预先准备全部被试缓存，降低整轮实验的 IO 和下载不稳定性。"""
    for subject in subjects:
        cache_file = cache_dir / f"subject_{subject:02d}.npz"
        if cache_file.exists():
            print(f"subject={subject} cache already available", flush=True)
            continue
        print(f"preparing subject={subject} cache", flush=True)
        load_subject_data(subject, data_dir, cache_dir)


def compute_standardizer(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算训练集 per-channel 标准化统计量。

    对应论文中强调的 training-only standardization。
    """
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """应用训练集标准化，并裁剪到固定范围。

    对应论文方法里 `per-channel normalization + clipping to [-6, 6]`。
    """
    standardized = (x - mean) / std
    return np.clip(standardized, -6.0, 6.0).astype(np.float32)


def downsample_trials(x: np.ndarray, factor: int) -> np.ndarray:
    """对 trial 做时间下采样。

    对应论文里把 250 Hz 数据降到 125 Hz 的实现。
    """
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
    """把标准化后的 trial 打包成 DataLoader。

    对应论文统一训练协议中的输入管线。
    """
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
    """按论文模型名构建对应网络。"""
    if model_name == "eegnet":
        return EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes)
    if model_name == "shallow_convnet":
        return ShallowConvNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes)
    if model_name == "tiny_transformer":
        return TinyTransformerClassifier(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes)
    if model_name == "lstm":
        return LSTMClassifier(n_channels=n_channels, hidden_size=lstm_hidden_size, n_classes=n_classes)
    if model_name == "gru":
        return GRUClassifier(n_channels=n_channels, hidden_size=lstm_hidden_size, n_classes=n_classes)
    if model_name == "cfc":
        return CfCClassifier(n_channels=n_channels, hidden_size=cfc_hidden_size, n_classes=n_classes)
    if model_name == "hybrid_cfc":
        return HybridCfCClassifier(n_channels=n_channels, hidden_size=cfc_hidden_size, n_classes=n_classes)
    raise ValueError(f"Unsupported model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量，对应论文表格中的 params 列。"""
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def is_classical_model(model_name: str) -> bool:
    """区分神经网络模型与经典几何基线。"""
    return model_name in CLASSICAL_MODELS


def get_model_display_name(model_name: str) -> str:
    """返回论文中使用的模型显示名。"""
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def riemann_parameter_count(n_channels: int, n_classes: int) -> int:
    """按显式线性分类头估算 Riemann-TSLR 参数量。"""
    tangent_dim = n_channels * (n_channels + 1) // 2
    return int(n_classes * tangent_dim + n_classes)


def get_parameter_count(
    model_name: str,
    n_channels: int,
    n_samples: int,
    n_classes: int,
    cfc_hidden_size: int,
    lstm_hidden_size: int,
) -> int:
    """统一获取论文所有模型的参数量。"""
    if model_name == "riemann_tslr":
        return riemann_parameter_count(n_channels=n_channels, n_classes=n_classes)
    probe_model = build_model(
        model_name=model_name,
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        cfc_hidden_size=cfc_hidden_size,
        lstm_hidden_size=lstm_hidden_size,
    )
    return count_parameters(probe_model)


def build_riemann_tslr_pipeline(c_value: float) -> Pipeline:
    """构建 Riemann-TSLR 基线。

    对应论文中唯一新增的 classical / geometric baseline。
    """
    classifier = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
    )
    return Pipeline(
        steps=[
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ]
    )


def fit_riemann_tslr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, object]:
    """在给定 train/val 划分上选择最优 Riemann-TSLR 正则强度。"""
    best_model: Pipeline | None = None
    best_c = float("nan")
    best_val_accuracy = float("-inf")
    best_val_f1 = float("-inf")

    for c_value in RIEMANN_C_VALUES:
        model = build_riemann_tslr_pipeline(c_value)
        model.fit(x_train, y_train)
        y_val_pred = model.predict(x_val)
        val_metrics = accuracy_and_f1(y_val, y_val_pred)
        accuracy_improved = val_metrics["accuracy"] > best_val_accuracy + 1e-4
        tied_accuracy = abs(val_metrics["accuracy"] - best_val_accuracy) <= 1e-4
        f1_improved = val_metrics["f1"] > best_val_f1 + 1e-6
        if accuracy_improved or (tied_accuracy and f1_improved):
            best_model = model
            best_c = float(c_value)
            best_val_accuracy = val_metrics["accuracy"]
            best_val_f1 = val_metrics["f1"]

    if best_model is None:
        raise RuntimeError("Riemann-TSLR selection failed to produce a fitted model.")
    return {
        "model": best_model,
        "best_c": best_c,
        "best_val_accuracy": best_val_accuracy,
        "best_val_f1": best_val_f1,
        "best_epoch": float("nan"),
        "best_val_loss": float("nan"),
    }


def accuracy_and_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算论文主文统一使用的 Accuracy 与 macro-F1。"""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred) * 100.0),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def compute_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算 per-class F1，对应补充材料中的类别级指标。"""
    _, _, f1_values, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(LABEL_ORDER))),
        zero_division=0,
    )
    return {INDEX_TO_LABEL[index]: float(score) for index, score in enumerate(f1_values)}


def confidence_interval_95(values: Sequence[float]) -> Tuple[float, float]:
    """计算 95% 置信区间，对应补充图表中的误差带。"""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        value = float(array[0])
        return value, value
    sem = float(np.std(array, ddof=1) / math.sqrt(array.size))
    margin = float(student_t.ppf(0.975, df=array.size - 1) * sem)
    mean = float(np.mean(array))
    return mean - margin, mean + margin


def add_prediction_rows(
    rows: List[Dict[str, object]],
    *,
    subject: int,
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold: int | None = None,
    protocol: str,
) -> None:
    """保存 trial-level 预测结果。

    对应论文补充材料中的混淆矩阵和 per-class F1 计算底表。
    """
    for index, (true_label, pred_label) in enumerate(zip(y_true.tolist(), y_pred.tolist())):
        row: Dict[str, object] = {
            "subject": subject,
            "model": model,
            "protocol": protocol,
            "trial_position": index,
            "true_label": int(true_label),
            "true_label_name": INDEX_TO_LABEL[int(true_label)],
            "pred_label": int(pred_label),
            "pred_label_name": INDEX_TO_LABEL[int(pred_label)],
        }
        if fold is not None:
            row["fold"] = fold
        rows.append(row)


def save_prediction_artifacts(predictions_df: pd.DataFrame, output_dir: Path) -> None:
    """导出 per-class F1 与混淆矩阵。

    对应论文补充材料中对类别错误模式的说明。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    if predictions_df.empty:
        return

    per_class_rows: List[Dict[str, object]] = []
    for (subject, model_name), group in predictions_df.groupby(["subject", "model"]):
        per_class_scores = compute_per_class_f1(
            group["true_label"].to_numpy(dtype=np.int64),
            group["pred_label"].to_numpy(dtype=np.int64),
        )
        for class_name, class_f1 in per_class_scores.items():
            per_class_rows.append(
                {
                    "subject": int(subject),
                    "model": model_name,
                    "model_display": get_model_display_name(model_name),
                    "class_name": class_name,
                    "f1": class_f1,
                }
            )

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(output_dir / "per_class_f1_subject.csv", index=False)
    per_class_summary_df = (
        per_class_df.groupby(["model", "model_display", "class_name"], as_index=False)["f1"]
        .agg(
            f1_mean="mean",
            f1_std=lambda series: float(np.std(series.to_numpy(dtype=float), ddof=1)) if len(series) > 1 else 0.0,
        )
        .sort_values(["model_display", "class_name"])
    )
    per_class_summary_df.to_csv(output_dir / "per_class_f1_summary.csv", index=False)

    confusion_rows: List[Dict[str, object]] = []
    available_models = list(predictions_df["model"].drop_duplicates())
    for model_name in available_models:
        group = predictions_df.loc[predictions_df["model"] == model_name]
        matrix = confusion_matrix(
            group["true_label"].to_numpy(dtype=np.int64),
            group["pred_label"].to_numpy(dtype=np.int64),
            labels=list(range(len(LABEL_ORDER))),
        )
        row_totals = np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
        normalized = matrix / row_totals
        for true_index, true_label in enumerate(LABEL_ORDER):
            for pred_index, pred_label in enumerate(LABEL_ORDER):
                confusion_rows.append(
                    {
                        "model": model_name,
                        "model_display": get_model_display_name(model_name),
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": int(matrix[true_index, pred_index]),
                        "row_normalized": float(normalized[true_index, pred_index]),
                    }
                )
    confusion_df = pd.DataFrame(confusion_rows)
    confusion_df.to_csv(output_dir / "confusion_matrices.csv", index=False)

    display_order = [name for name in MODEL_ORDER if name in confusion_df["model_display"].unique()]
    n_models = len(display_order)
    if n_models == 0:
        return
    ncols = 2
    nrows = math.ceil(n_models / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.6 * ncols / 2, 3.6 * nrows))
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)
    for axis in axes_array.flatten():
        axis.set_visible(False)
    for axis, display_name in zip(axes_array.flatten(), display_order):
        axis.set_visible(True)
        model_group = confusion_df.loc[confusion_df["model_display"] == display_name]
        matrix = (
            model_group.pivot(index="true_label", columns="pred_label", values="row_normalized")
            .reindex(index=LABEL_ORDER, columns=LABEL_ORDER)
            .to_numpy(dtype=float)
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=False,
            xticklabels=[label.replace("_", " ").title() for label in LABEL_ORDER],
            yticklabels=[label.replace("_", " ").title() for label in LABEL_ORDER],
            ax=axis,
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_title(display_name)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.pdf", bbox_inches="tight")
    plt.close(fig)


def add_gaussian_noise(x: np.ndarray, snr_db: int, rng: np.random.Generator) -> np.ndarray:
    """加入高斯噪声。

    对应早期 pooled 噪声鲁棒性分析；后续更完整的结构化扰动在其他脚本里实现。
    """
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
    """训练单个神经网络模型。

    对应论文统一训练协议：Adam、cosine annealing、early stopping、gradient clipping。
    """
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
        # 训练和验证严格遵守论文里的 train/val 划分，不向测试集泄漏统计量。
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
    return_predictions: bool = False,
    tau_store: Dict[str, List[float]] | None = None,
    tau_rng: np.random.Generator | None = None,
    max_tau_per_class: int = 75000,
    sample_per_class_per_batch: int = 2500,
) -> Dict[str, float]:
    """评估神经网络模型。

    这里既服务于主精度结果，也服务于 CfC-style 的 tau 采样与补充分析。
    """
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

            # tau 只对 CfC-style 收集，并限制每类样本量，避免补充图过大。
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
    if return_predictions:
        metrics["y_true"] = y_true_array
        metrics["y_pred"] = y_pred_array
    return metrics


def evaluate_classical_model(
    model: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
    return_predictions: bool = False,
) -> Dict[str, object]:
    """评估经典几何基线。"""
    predictions = model.predict(x).astype(np.int64)
    metrics = accuracy_and_f1(y, predictions)
    metrics["loss"] = float("nan")
    if return_predictions:
        metrics["y_true"] = y.astype(np.int64)
        metrics["y_pred"] = predictions
    return metrics


def format_metric(mean: float, std: float, digits: int = 1) -> str:
    """把均值和标准差格式化成论文表格字符串。"""
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def summarize_subject_metrics(fold_rows: List[Dict[str, object]]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """把 fold 级结果汇总到被试级。

    对应论文主表中的 `mean ± std across subjects`。
    """
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
    """导出逐被试结果表和箱线图。

    对应论文补充材料中回答“是否由少数 subject 驱动”的证据。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_wide = subject_summary_df.pivot(index="subject", columns="model", values="accuracy").reset_index()
    subject_wide.to_csv(output_dir / "subject_accuracy_table.csv", index=False)
    plot_df = subject_summary_df.copy()
    plot_df["model_display"] = plot_df["model"].map(get_model_display_name)
    available_order = [label for label in MODEL_ORDER if label in set(plot_df["model_display"])]

    sns.set_theme(style="whitegrid")
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    sns.boxplot(
        data=plot_df,
        x="model_display",
        y="accuracy",
        order=available_order,
        hue="model_display",
        palette=MODEL_PALETTE,
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
        order=available_order,
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
    stability: Dict[str, int] = {}
    if {"cfc", "lstm"}.issubset(subject_pivot.columns):
        stability["cfc_gt_lstm_subjects"] = int((subject_pivot["cfc"] > subject_pivot["lstm"]).sum())
    if {"hybrid_cfc", "cfc"}.issubset(subject_pivot.columns):
        stability["hybrid_gt_cfc_subjects"] = int((subject_pivot["hybrid_cfc"] > subject_pivot["cfc"]).sum())
    if {"hybrid_cfc", "eegnet"}.issubset(subject_pivot.columns):
        stability["hybrid_gt_eegnet_subjects"] = int((subject_pivot["hybrid_cfc"] > subject_pivot["eegnet"]).sum())
    if {"cfc", "eegnet"}.issubset(subject_pivot.columns):
        stability["cfc_gt_eegnet_subjects"] = int((subject_pivot["cfc"] > subject_pivot["eegnet"]).sum())
    if {"shallow_convnet", "eegnet"}.issubset(subject_pivot.columns):
        stability["shallow_gt_eegnet_subjects"] = int((subject_pivot["shallow_convnet"] > subject_pivot["eegnet"]).sum())
    if {"riemann_tslr", "cfc"}.issubset(subject_pivot.columns):
        stability["riemann_gt_cfc_subjects"] = int((subject_pivot["riemann_tslr"] > subject_pivot["cfc"]).sum())
    if {"riemann_tslr", "eegnet"}.issubset(subject_pivot.columns):
        stability["riemann_gt_eegnet_subjects"] = int((subject_pivot["riemann_tslr"] > subject_pivot["eegnet"]).sum())
    return stability


def summarize_noise_metrics(
    noise_rows: List[Dict[str, object]]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Dict[str, float]]]]:
    """汇总 pooled 高斯噪声实验的被试级结果。"""
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
    """对各个 SNR 条件下的噪声结果做成对比较。"""
    summary: Dict[str, Dict[str, float]] = {}
    for snr_value, snr_group in subject_noise_df.groupby("snr"):
        pivot = snr_group.pivot(index="subject", columns="model", values="accuracy")
        if model_a not in pivot.columns or model_b not in pivot.columns:
            summary[str(int(snr_value))] = {
                "t_statistic": float("nan"),
                "p_value": float("nan"),
                "mean_diff": float("nan"),
            }
            continue
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


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """实现 Holm 多重比较校正。"""
    array = np.asarray(p_values, dtype=float)
    adjusted = np.full(array.shape, np.nan, dtype=float)
    valid_mask = np.isfinite(array)
    if not np.any(valid_mask):
        return adjusted.tolist()

    valid_values = array[valid_mask]
    order = np.argsort(valid_values)
    sorted_values = valid_values[order]
    m = len(sorted_values)
    sorted_adjusted = np.empty(m, dtype=float)
    for index, p_value in enumerate(sorted_values):
        sorted_adjusted[index] = min((m - index) * p_value, 1.0)
    sorted_adjusted = np.maximum.accumulate(sorted_adjusted)
    reverse = np.empty(m, dtype=int)
    reverse[order] = np.arange(m)
    adjusted_values = sorted_adjusted[reverse]
    adjusted[valid_mask] = adjusted_values
    return adjusted.tolist()


def apply_holm_correction(
    test_dict: Dict[str, Dict[str, float]],
    p_value_key: str = "p_value",
    output_key: str = "holm_p_value",
) -> Dict[str, Dict[str, float]]:
    """把 Holm 校正写回统计结果字典。"""
    keys = list(test_dict.keys())
    adjusted = holm_adjust([test_dict[key].get(p_value_key, float("nan")) for key in keys])
    for key, adjusted_value in zip(keys, adjusted):
        test_dict[key][output_key] = float(adjusted_value)
    return test_dict


def paired_test(subject_summary_df: pd.DataFrame, model_a: str, model_b: str) -> Dict[str, float]:
    """执行论文里的 paired t-test、Wilcoxon 和 Cohen's d。"""
    pivot = subject_summary_df.pivot(index="subject", columns="model", values="accuracy")
    a_values = pivot[model_a].to_numpy()
    b_values = pivot[model_b].to_numpy()
    if len(a_values) < 2:
        return {
            "t_statistic": float("nan"),
            "p_value": float("nan"),
            "cohen_d": float("nan"),
            "wilcoxon_statistic": float("nan"),
            "wilcoxon_p_value": float("nan"),
            "mean_diff": float("nan"),
            "n_subjects": int(len(a_values)),
        }
    test = ttest_rel(a_values, b_values)
    diff = a_values - b_values
    cohen_d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-8))
    if np.allclose(diff, 0.0):
        wilcoxon_statistic = 0.0
        wilcoxon_p_value = 1.0
    else:
        wilcoxon_result = wilcoxon(a_values, b_values, zero_method="wilcox", alternative="two-sided", mode="auto")
        wilcoxon_statistic = float(wilcoxon_result.statistic)
        wilcoxon_p_value = float(wilcoxon_result.pvalue)
    return {
        "t_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "cohen_d": cohen_d,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p_value": wilcoxon_p_value,
        "mean_diff": float(np.mean(diff)),
        "n_subjects": int(len(a_values)),
    }


def save_tau_figure(tau_store: Dict[str, List[float]], output_path: Path) -> Dict[str, Dict[str, float]]:
    """生成 tau 直方图。

    对应论文早期 pooled 分析中的 tau 分布图和分位数统计。
    """
    if sum(len(values) for values in tau_store.values()) == 0:
        return {}
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
    """读取中间进度文件，用于断点续跑整轮实验。"""
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def load_tau_store(path: Path) -> Dict[str, List[float]]:
    """读取 CfC-style 的 tau 采样缓存。"""
    if not path.exists():
        return {label: [] for label in LABEL_ORDER}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return {label: list(map(float, loaded.get(label, []))) for label in LABEL_ORDER}


def persist_progress(
    fold_rows: List[Dict[str, object]],
    prediction_rows: List[Dict[str, object]],
    noise_rows: List[Dict[str, object]],
    tau_store: Dict[str, List[float]],
    fold_path: Path,
    prediction_path: Path,
    noise_path: Path,
    tau_path: Path,
) -> None:
    """把 pooled 主实验中间结果持久化，支持长时间运行后的恢复。"""
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)
    pd.DataFrame(noise_rows).to_csv(noise_path, index=False)
    tau_path.write_text(json.dumps(tau_store), encoding="utf-8")


def run_experiment(config: ExperimentConfig) -> Dict[str, object]:
    """运行 pooled 主实验。

    对应论文中：
    1. pooled subject-wise 5-fold 主结果；
    2. 配套的统计检验；
    3. 噪声分析、tau 直方图和补充图表导出。
    """
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
    prediction_path = output_dir / "predictions.csv"
    noise_metrics_path = output_dir / "noise_metrics.csv"
    tau_store_path = output_dir / "tau_samples.json"

    prepare_subject_cache(config.subjects, data_dir, cache_dir)

    fold_rows = load_progress_csv(fold_metrics_path)
    prediction_rows = load_progress_csv(prediction_path)
    noise_rows = load_progress_csv(noise_metrics_path)
    tau_store = load_tau_store(tau_store_path)
    completed_runs = {
        (int(row["subject"]), int(row["fold"]), str(row["model"]))
        for row in fold_rows
    }
    tau_rng = np.random.default_rng(config.seed + 99)

    model_names = list(config.models)
    total_runs = len(config.subjects) * config.num_folds * len(model_names)
    run_index = len(completed_runs)
    parameter_counts: Dict[str, int] = {}

    # 外层循环对应论文里的 subject-dependent 评估。
    for subject in config.subjects:
        X_raw, y = load_subject_data(subject=subject, data_dir=data_dir, cache_dir=cache_dir)
        X_raw = downsample_trials(X_raw, config.downsample_factor)
        n_trials, n_channels, n_samples = X_raw.shape
        if not parameter_counts:
            for model_name in model_names:
                parameter_counts[model_name] = get_parameter_count(
                    model_name=model_name,
                    n_channels=n_channels,
                    n_samples=n_samples,
                    n_classes=len(LABEL_ORDER),
                    cfc_hidden_size=config.cfc_hidden_size,
                    lstm_hidden_size=config.lstm_hidden_size,
                )
        # pooled 协议：先合并两次 session，再做被试内 stratified K-fold。
        splitter = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed + subject)

        for fold_idx, (train_val_idx, test_idx) in enumerate(splitter.split(np.zeros(n_trials), y), start=1):
            # 每个 outer fold 内再切一份验证集，用于统一早停与 classical baseline 选择。
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

            # 内层模型循环直接对应论文主表中的各个 baseline。
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
                print(
                    f"[{run_index}/{total_runs}] subject={subject} fold={fold_idx}/{config.num_folds} "
                    f"model={model_name} device={device.type}",
                    flush=True,
                )
                if is_classical_model(model_name):
                    fit_info = fit_riemann_tslr(
                        x_train=X_train,
                        y_train=y[train_idx],
                        x_val=X_val,
                        y_val=y[val_idx],
                    )
                    clean_metrics = evaluate_classical_model(
                        model=fit_info["model"],
                        x=X_test,
                        y=y[test_idx],
                        return_predictions=True,
                    )
                    runtime_model = fit_info["model"]
                else:
                    runtime_model = build_model(
                        model_name=model_name,
                        n_channels=n_channels,
                        n_samples=n_samples,
                        n_classes=len(LABEL_ORDER),
                        cfc_hidden_size=config.cfc_hidden_size,
                        lstm_hidden_size=config.lstm_hidden_size,
                    )
                    fit_info = train_one_model(
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
                    clean_metrics = evaluate_model(
                        model=runtime_model,
                        loader=test_loader,
                        device=device,
                        collect_tau=model_name == "cfc",
                        return_predictions=True,
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
                        "best_c": float(fit_info.get("best_c", float("nan"))),
                    }
                )
                add_prediction_rows(
                    prediction_rows,
                    subject=subject,
                    model=model_name,
                    y_true=clean_metrics["y_true"],
                    y_pred=clean_metrics["y_pred"],
                    fold=fold_idx,
                    protocol="pooled",
                )

                # 这里是 pooled 高斯噪声实验；更严格的结构化扰动在 session-wise 脚本里实现。
                for snr_db in SNR_LEVELS:
                    noise_rng = np.random.default_rng(config.seed + subject * 1000 + fold_idx * 100 + snr_db)
                    X_noisy = add_gaussian_noise(X_raw[test_idx], snr_db=snr_db, rng=noise_rng)
                    X_noisy = apply_standardizer(X_noisy, mean, std)
                    if is_classical_model(model_name):
                        noisy_metrics = evaluate_classical_model(runtime_model, X_noisy, y[test_idx])
                    else:
                        noisy_loader = build_loader(X_noisy, y[test_idx], config.batch_size, False, device)
                        noisy_metrics = evaluate_model(model=runtime_model, loader=noisy_loader, device=device)
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
                    prediction_rows=prediction_rows,
                    noise_rows=noise_rows,
                    tau_store=tau_store,
                    fold_path=fold_metrics_path,
                    prediction_path=prediction_path,
                    noise_path=noise_metrics_path,
                    tau_path=tau_store_path,
                )

    # 下方开始把原始 fold 结果整理成论文正文和补充材料中的表格/统计量。
    subject_summary_df, metric_summary = summarize_subject_metrics(fold_rows)
    subject_stability = save_subject_accuracy_artifacts(subject_summary_df, output_dir)
    prediction_df = pd.DataFrame(prediction_rows)
    save_prediction_artifacts(prediction_df, output_dir)
    subject_noise_df, noise_summary = summarize_noise_metrics(noise_rows)
    stat_tests = {
        f"{model_a}_vs_{model_b}": paired_test(subject_summary_df, model_a, model_b)
        for model_a, model_b in combinations(model_names, 2)
    }
    apply_holm_correction(stat_tests, p_value_key="p_value", output_key="holm_p_value")
    apply_holm_correction(stat_tests, p_value_key="wilcoxon_p_value", output_key="wilcoxon_holm_p_value")
    noise_stat_tests: Dict[str, Dict[str, Dict[str, float]]] = {}
    if {"cfc", "eegnet"}.issubset(set(model_names)):
        noise_stat_tests["cfc_vs_eegnet"] = paired_noise_tests(subject_noise_df, "cfc", "eegnet")
    if {"hybrid_cfc", "eegnet"}.issubset(set(model_names)):
        noise_stat_tests["hybrid_cfc_vs_eegnet"] = paired_noise_tests(subject_noise_df, "hybrid_cfc", "eegnet")
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
    prediction_df.to_csv(prediction_path, index=False)
    pd.DataFrame(noise_rows).to_csv(noise_metrics_path, index=False)
    subject_summary_df.to_csv(output_dir / "subject_summary.csv", index=False)
    subject_noise_df.to_csv(output_dir / "noise_subject_level.csv", index=False)
    pd.DataFrame(noise_summary_rows).to_csv(output_dir / "noise_subject_summary.csv", index=False)
    pd.DataFrame(
        [
            {"comparison": name, **metrics}
            for name, metrics in stat_tests.items()
        ]
    ).to_csv(output_dir / "stat_tests.csv", index=False)
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
    """解析主实验命令行参数。"""
    parser = argparse.ArgumentParser(description="Run pooled MI-EEG experiments under a unified protocol.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a short validation experiment.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--subjects", type=int, nargs="*", default=None)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model list to evaluate. Defaults to Shallow ConvNet, Riemann-TSLR, EEGNet, Hybrid-CfC-style, CfC-style, and LSTM.",
    )
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
    parser.add_argument("--output-dir", default="outputs/bspc_pooled")
    args = parser.parse_args()

    if args.smoke_test:
        subjects = args.subjects or [1]
        models = args.models or ["shallow_convnet", "riemann_tslr", "eegnet", "tiny_transformer", "hybrid_cfc", "cfc", "lstm"]
        return ExperimentConfig(
            subjects=subjects,
            models=models,
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
            output_dir="outputs/bspc_smoke",
            smoke_test=True,
        )

    models = args.models or ["shallow_convnet", "riemann_tslr", "eegnet", "tiny_transformer", "hybrid_cfc", "cfc", "lstm"]
    return ExperimentConfig(
        subjects=args.subjects or list(range(1, 10)),
        models=models,
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
    """命令行入口，对应论文 pooled 主实验的可复现启动点。"""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    config = parse_args()
    run_experiment(config)


if __name__ == "__main__":
    main()
