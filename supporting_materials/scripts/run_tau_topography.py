"""Channel-wise tau sensitivity analysis for the CfC-style model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit


BCI_IV_2A_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]


@dataclass
class TauTopographyConfig:
    subjects: List[int]
    epochs: int
    patience: int
    min_epochs: int
    batch_size: int
    seed: int
    cfc_hidden_size: int
    cfc_dt: float
    cfc_tau_init: float
    downsample_factor: int
    device: str
    data_dir: str
    output_dir: str


def load_module(repo_root: Path, filename: str, module_name: str):
    script_path = repo_root / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def seed_cache(repo_root: Path, target_cache: Path) -> None:
    source_cache = repo_root / "outputs" / "bspc_sessionwise" / "cache"
    if not source_cache.exists() or source_cache.resolve() == target_cache.resolve():
        return
    target_cache.mkdir(parents=True, exist_ok=True)
    for source_file in source_cache.glob("subject_*.npz"):
        target_file = target_cache / source_file.name
        if not target_file.exists():
            shutil.copy2(source_file, target_file)


def mean_tau(model, loader, device) -> np.ndarray:
    values = []
    model.eval()
    with torch.no_grad():
        for features, _ in loader:
            _, aux = model(features.to(device), return_aux=True)
            values.append(aux["tau"].detach().cpu().numpy().mean(axis=(1, 2)))
    return np.concatenate(values)


def save_topomap(summary_df: pd.DataFrame, output_dir: Path) -> str:
    values = summary_df.set_index("channel").loc[BCI_IV_2A_CHANNELS, "tau_sensitivity_mean"].to_numpy()
    try:
        import mne

        info = mne.create_info(BCI_IV_2A_CHANNELS, sfreq=125.0, ch_types="eeg")
        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, on_missing="ignore")
        fig, axis = plt.subplots(figsize=(6, 5))
        mne.viz.plot_topomap(values, info, axes=axis, show=False, names=BCI_IV_2A_CHANNELS)
        axis.set_title("Channel-wise Tau Sensitivity")
    except Exception:
        fig, axis = plt.subplots(figsize=(9, 4))
        axis.bar(BCI_IV_2A_CHANNELS, values)
        axis.set_ylabel("Mean |Delta tau|")
        axis.set_title("Channel-wise Tau Sensitivity")
        axis.tick_params(axis="x", rotation=45)
    path = output_dir / "tau_occlusion_topomap_global.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run_tau_topography(config: TauTopographyConfig):
    repo_root = Path(__file__).resolve().parents[1]
    core = load_module(repo_root, "run_mi_experiments.py", "mi_core_tau_topography")
    sessionwise = load_module(repo_root, "run_sessionwise_mi_comparison.py", "mi_session_tau_topography")
    device = core.get_device(config.device)
    data_dir = repo_root / config.data_dir
    output_dir = repo_root / config.output_dir
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_cache(repo_root, cache_dir)
    rows = []
    for subject in config.subjects:
        core.seed_everything(config.seed + subject)
        x_raw, y, sessions = sessionwise.load_subject_session_data(core, subject, data_dir, cache_dir)
        x_raw = core.downsample_trials(x_raw, config.downsample_factor)
        train_mask = sessions == "0train"
        test_mask = sessions == "1test"
        x_train_full = x_raw[train_mask]
        y_train_full = y[train_mask]
        x_test = x_raw[test_mask]
        y_test = y[test_mask]
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=config.seed + subject)
        train_idx, val_idx = next(splitter.split(np.zeros(len(x_train_full)), y_train_full))
        mean, std = core.compute_standardizer(x_train_full[train_idx])
        x_train = core.apply_standardizer(x_train_full[train_idx], mean, std)
        x_val = core.apply_standardizer(x_train_full[val_idx], mean, std)
        x_test_std = core.apply_standardizer(x_test, mean, std)
        train_loader = core.build_loader(x_train, y_train_full[train_idx], config.batch_size, True, device)
        val_loader = core.build_loader(x_val, y_train_full[val_idx], config.batch_size, False, device)
        test_loader = core.build_loader(x_test_std, y_test, config.batch_size, False, device)
        model = core.build_model(
            "cfc",
            n_channels=x_test_std.shape[1],
            n_samples=x_test_std.shape[2],
            n_classes=len(core.LABEL_ORDER),
            cfc_hidden_size=config.cfc_hidden_size,
            lstm_hidden_size=128,
            cfc_dt=config.cfc_dt,
            cfc_tau_init=config.cfc_tau_init,
        )
        core.train_one_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=config.epochs,
            patience=config.patience,
            min_epochs=config.min_epochs,
            learning_rate=1e-3,
            weight_decay=1e-4,
        )
        full_tau = mean_tau(model, test_loader, device)
        for channel_index, channel_name in enumerate(BCI_IV_2A_CHANNELS):
            occluded = x_test_std.copy()
            occluded[:, channel_index, :] = 0.0
            occluded_loader = core.build_loader(occluded, y_test, config.batch_size, False, device)
            occluded_tau = mean_tau(model, occluded_loader, device)
            sensitivity = np.abs(full_tau - occluded_tau)
            rows.append(
                {
                    "subject": subject,
                    "channel": channel_name,
                    "channel_index": channel_index,
                    "tau_sensitivity_mean": float(sensitivity.mean()),
                    "tau_sensitivity_std": float(sensitivity.std(ddof=1)) if len(sensitivity) > 1 else 0.0,
                }
            )
        pd.DataFrame(rows).to_csv(output_dir / "tau_occlusion_channel_subject.csv", index=False)
    subject_df = pd.DataFrame(rows)
    summary_df = (
        subject_df.groupby(["channel", "channel_index"], as_index=False)["tau_sensitivity_mean"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "tau_sensitivity_mean", "std": "tau_sensitivity_std"})
    )
    summary_df.to_csv(output_dir / "tau_occlusion_channel_summary.csv", index=False)
    figure_path = save_topomap(summary_df, output_dir)
    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": asdict(config),
        "figure": figure_path,
        "note": "Values are channel-wise sensitivity of hidden-state tau, not electrode-specific learned tau parameters.",
    }
    (output_dir / "tau_topography_stats.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> TauTopographyConfig:
    parser = argparse.ArgumentParser(description="Run channel-wise tau occlusion/sensitivity analysis.")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--cfc-hidden-size", type=int, default=128)
    parser.add_argument("--cfc-dt", type=float, default=1.0)
    parser.add_argument("--cfc-tau-init", type=float, default=1.0)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/revision_tau_topography")
    args = parser.parse_args()
    return TauTopographyConfig(**vars(args))


if __name__ == "__main__":
    run_tau_topography(parse_args())
