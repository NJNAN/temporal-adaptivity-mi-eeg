"""Session-wise MI-Mamba/readout sensitivity controls."""

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
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn


@dataclass
class Config:
    subjects: List[int]
    variants: List[str]
    epochs: int
    patience: int
    min_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    val_fraction: float
    seed: int
    cfc_hidden_size: int
    downsample_factor: int
    device: str
    data_dir: str
    output_dir: str


def load_module(repo_root: Path, filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, repo_root / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def seed_cache(repo_root: Path, target_cache: Path) -> None:
    source_cache = repo_root / "outputs" / "bspc_sessionwise_full_rerun" / "cache"
    if not source_cache.exists() or source_cache.resolve() == target_cache.resolve():
        return
    target_cache.mkdir(parents=True, exist_ok=True)
    for source_file in source_cache.glob("subject_*.npz"):
        target_file = target_cache / source_file.name
        if not target_file.exists():
            shutil.copy2(source_file, target_file)


class AttentionPool(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class MIMambaVariant(nn.Module):
    def __init__(self, core, n_channels: int, n_classes: int, d_model: int = 64, d_state: int = 16, pooling: str = "meanmax") -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 31), padding="same", bias=False),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=(n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25),
            nn.Conv2d(32, 32, kernel_size=(1, 15), padding="same", groups=32, bias=False),
            nn.Conv2d(32, d_model, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.25),
        )
        self.ssm = nn.Sequential(
            core.SelectiveSSMBlock(d_model=d_model, d_state=d_state, dropout=0.2),
            core.SelectiveSSMBlock(d_model=d_model, d_state=d_state, dropout=0.2),
        )
        self.norm = nn.LayerNorm(d_model)
        self.pooling = pooling
        if pooling == "attention":
            self.pool = AttentionPool(d_model)
            classifier_dim = d_model
        elif pooling == "last":
            classifier_dim = d_model
        else:
            classifier_dim = d_model * 2
        self.classifier = nn.Linear(classifier_dim, n_classes)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x = self.frontend(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
        x = self.norm(self.ssm(x))
        if self.pooling == "attention":
            pooled = self.pool(x)
        elif self.pooling == "last":
            pooled = x[:, -1, :]
        else:
            pooled = torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=1)
        logits = self.classifier(pooled)
        if return_aux:
            return logits, {}
        return logits


class TinyTransformerAttention(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int, d_model: int = 64) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_samples, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
        self.pool = AttentionPool(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x = x.transpose(1, 2)
        x = self.input_proj(x) + self.pos_embed[:, : x.size(1), :]
        x = self.norm(self.encoder(x))
        logits = self.classifier(self.dropout(self.pool(x)))
        if return_aux:
            return logits, {}
        return logits


class CfCFinalState(nn.Module):
    def __init__(self, core, n_channels: int, hidden_size: int, n_classes: int, dt: float = 1.0, tau_init: float = 1.0) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_channels, hidden_size)
        self.cell = core.AdaptiveCfCCell(hidden_size, hidden_size, tau_init=tau_init)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_size, n_classes)
        self.hidden_size = hidden_size
        self.dt = dt

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x = torch.tanh(self.input_proj(x.transpose(1, 2)))
        hidden = x.new_zeros(x.size(0), self.hidden_size)
        for step in range(x.size(1)):
            hidden, _ = self.cell(x[:, step, :], hidden, dt=self.dt)
        logits = self.classifier(self.dropout(hidden))
        if return_aux:
            return logits, {}
        return logits


def build_variant(core, variant: str, n_channels: int, n_samples: int, n_classes: int, hidden_size: int) -> nn.Module:
    if variant == "mi_mamba_d8_meanmax":
        return MIMambaVariant(core, n_channels, n_classes, d_state=8, pooling="meanmax")
    if variant == "mi_mamba_d16_meanmax":
        return MIMambaVariant(core, n_channels, n_classes, d_state=16, pooling="meanmax")
    if variant == "mi_mamba_d32_meanmax":
        return MIMambaVariant(core, n_channels, n_classes, d_state=32, pooling="meanmax")
    if variant == "mi_mamba_d16_attention":
        return MIMambaVariant(core, n_channels, n_classes, d_state=16, pooling="attention")
    if variant == "mi_mamba_d16_last":
        return MIMambaVariant(core, n_channels, n_classes, d_state=16, pooling="last")
    if variant == "tiny_transformer_attention":
        return TinyTransformerAttention(n_channels, n_samples, n_classes)
    if variant == "cfc_final":
        return CfCFinalState(core, n_channels, hidden_size, n_classes)
    raise ValueError(f"Unknown variant: {variant}")


DISPLAY = {
    "mi_mamba_d8_meanmax": "MI-Mamba-style d_state=8 mean-max",
    "mi_mamba_d16_meanmax": "MI-Mamba-style d_state=16 mean-max",
    "mi_mamba_d32_meanmax": "MI-Mamba-style d_state=32 mean-max",
    "mi_mamba_d16_attention": "MI-Mamba-style d_state=16 attention",
    "mi_mamba_d16_last": "MI-Mamba-style d_state=16 last",
    "tiny_transformer_attention": "Tiny-Transformer attention",
    "cfc_final": "CfC-style final-state",
}


def summarize(rows: List[Dict[str, object]], core, output_dir: Path) -> tuple[dict, dict]:
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "readout_mamba_sensitivity_metrics.csv", index=False)
    summary = {}
    summary_rows = []
    for model, group in df.groupby("model"):
        acc = group["accuracy"].to_numpy(dtype=float)
        f1 = group["f1"].to_numpy(dtype=float)
        item = {
            "model": model,
            "model_display": DISPLAY[model],
            "accuracy_mean": float(acc.mean()),
            "accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
            "f1_mean": float(f1.mean()),
            "f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
        }
        summary[model] = item
        summary_rows.append(item)
    pd.DataFrame(summary_rows).sort_values("accuracy_mean", ascending=False).to_csv(output_dir / "readout_mamba_sensitivity_summary.csv", index=False)
    stats = {}
    for a, b in combinations(sorted(df["model"].unique()), 2):
        stats[f"{a}_vs_{b}"] = core.paired_test(df[["subject", "model", "accuracy"]], a, b)
    core.apply_holm_correction(stats, p_value_key="p_value", output_key="holm_p_value")
    pd.DataFrame([{"comparison": key, **value} for key, value in stats.items()]).to_csv(output_dir / "readout_mamba_sensitivity_stats.csv", index=False)
    return summary, stats


def run(config: Config) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    core = load_module(repo_root, "run_mi_experiments.py", "mi_core_readout_sensitivity")
    sessionwise = load_module(repo_root, "run_sessionwise_mi_comparison.py", "mi_session_readout_sensitivity")
    core.seed_everything(config.seed)
    device = core.get_device(config.device)
    output_dir = repo_root / config.output_dir
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_cache(repo_root, cache_dir)
    metrics_path = output_dir / "readout_mamba_sensitivity_metrics.csv"
    rows = pd.read_csv(metrics_path).to_dict(orient="records") if metrics_path.exists() else []
    completed = {(int(row["subject"]), str(row["model"])) for row in rows if not pd.isna(row.get("accuracy", np.nan))}
    total_runs = len(config.subjects) * len(config.variants)
    run_index = len(completed)
    parameter_counts: Dict[str, int] = {}

    for subject in config.subjects:
        x_raw, y, sessions = sessionwise.load_subject_session_data(core, subject, repo_root / config.data_dir, cache_dir)
        x_raw = core.downsample_trials(x_raw, config.downsample_factor)
        train_mask = sessions == "0train"
        test_mask = sessions == "1test"
        x_train_full = x_raw[train_mask]
        y_train_full = y[train_mask]
        x_test = x_raw[test_mask]
        y_test = y[test_mask]
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=config.val_fraction, random_state=config.seed + subject)
        train_idx, val_idx = next(splitter.split(np.zeros(len(x_train_full)), y_train_full))
        mean, std = core.compute_standardizer(x_train_full[train_idx])
        x_train = core.apply_standardizer(x_train_full[train_idx], mean, std)
        x_val = core.apply_standardizer(x_train_full[val_idx], mean, std)
        x_test_std = core.apply_standardizer(x_test, mean, std)
        train_loader = core.build_loader(x_train, y_train_full[train_idx], config.batch_size, True, device)
        val_loader = core.build_loader(x_val, y_train_full[val_idx], config.batch_size, False, device)
        test_loader = core.build_loader(x_test_std, y_test, config.batch_size, False, device)
        for variant in config.variants:
            if (subject, variant) in completed:
                continue
            run_index += 1
            core.seed_everything(config.seed + subject * 100 + config.variants.index(variant))
            print(f"[{run_index}/{total_runs}] subject={subject} variant={variant}", flush=True)
            model = build_variant(core, variant, x_train.shape[1], x_train.shape[2], len(core.LABEL_ORDER), config.cfc_hidden_size)
            parameter_counts.setdefault(variant, core.count_parameters(model))
            fit_info = core.train_one_model(model, train_loader, val_loader, device, config.epochs, config.patience, config.min_epochs, config.learning_rate, config.weight_decay)
            metrics = core.evaluate_model(model, test_loader, device)
            rows.append(
                {
                    "subject": subject,
                    "model": variant,
                    "model_display": DISPLAY[variant],
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                    "best_epoch": fit_info["best_epoch"],
                    "best_val_accuracy": fit_info["best_val_accuracy"],
                    "best_val_loss": fit_info["best_val_loss"],
                }
            )
            completed.add((subject, variant))
            pd.DataFrame(rows).to_csv(metrics_path, index=False)

    summary, stats = summarize(rows, core, output_dir)
    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "session-wise readout and MI-Mamba-style sensitivity",
        "config": asdict(config),
        "parameter_counts": parameter_counts,
        "summary": summary,
        "stat_tests": stats,
    }
    (output_dir / "readout_mamba_sensitivity_results_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Run MI-Mamba/readout session-wise sensitivity controls.")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--variants", nargs="*", default=[
        "mi_mamba_d8_meanmax",
        "mi_mamba_d16_meanmax",
        "mi_mamba_d32_meanmax",
        "mi_mamba_d16_attention",
        "mi_mamba_d16_last",
        "tiny_transformer_attention",
        "cfc_final",
    ])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--cfc-hidden-size", type=int, default=128)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/revision_readout_mamba_sensitivity")
    return Config(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
