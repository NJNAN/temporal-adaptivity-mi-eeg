"""Session-wise preprocessing band sensitivity controls."""

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

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass
class Config:
    subjects: List[int]
    models: List[str]
    bands: List[str]
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
    device: str
    data_dir: str
    output_dir: str


BAND_MAP: Dict[str, Tuple[float, float]] = {
    "mu_beta_8_30": (8.0, 30.0),
    "broad_4_40": (4.0, 40.0),
    "broad_1_45": (1.0, 45.0),
}


def load_core(repo_root: Path):
    spec = importlib.util.spec_from_file_location("mi_core_band_sensitivity", repo_root / "scripts" / "run_mi_experiments.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_subject_session_data(core, subject: int, data_dir: Path, cache_dir: Path, band_name: str):
    fmin, fmax = BAND_MAP[band_name]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"subject_{subject:02d}_{band_name}.npz"
    if cache_file.exists():
        cached = np.load(cache_file, allow_pickle=True)
        return cached["X"].astype(np.float32), cached["y"].astype(np.int64), cached["session"].astype(str)
    core.ensure_mne_path(data_dir)
    dataset = core.BNCI2014Dataset()
    paradigm = core.MotorImagery(n_classes=4, fmin=fmin, fmax=fmax, tmin=0.0, tmax=4.0)
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
    y_int = np.asarray([core.LABEL_TO_INDEX[str(label)] for label in y], dtype=np.int64)
    sessions = metadata["session"].astype(str).to_numpy()
    X = X.astype(np.float32)
    np.savez_compressed(cache_file, X=X, y=y_int, session=sessions)
    return X, y_int, sessions


def completed_keys(rows: List[Dict[str, object]]) -> set[tuple[str, int, str]]:
    return {
        (str(row["band"]), int(row["subject"]), str(row["model"]))
        for row in rows
        if not pd.isna(row.get("accuracy", np.nan))
    }


def summarize(rows: List[Dict[str, object]], core, output_dir: Path) -> tuple[dict, dict]:
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "band_sensitivity_metrics.csv", index=False)
    summary = {}
    summary_rows = []
    for (band, model), group in df.groupby(["band", "model"]):
        acc = group["accuracy"].to_numpy(dtype=float)
        f1 = group["f1"].to_numpy(dtype=float)
        item = {
            "band": band,
            "model": model,
            "model_display": core.get_model_display_name(model),
            "accuracy_mean": float(acc.mean()),
            "accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
            "f1_mean": float(f1.mean()),
            "f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
        }
        summary[f"{band}:{model}"] = item
        summary_rows.append(item)
    pd.DataFrame(summary_rows).sort_values(["band", "accuracy_mean"], ascending=[True, False]).to_csv(output_dir / "band_sensitivity_summary.csv", index=False)
    stats = {}
    for model, group in df.groupby("model"):
        for band_a, band_b in combinations(sorted(group["band"].unique()), 2):
            subset = group[["subject", "band", "accuracy"]].rename(columns={"band": "model"})
            stats[f"{model}:{band_a}_vs_{band_b}"] = core.paired_test(subset, band_a, band_b)
    core.apply_holm_correction(stats, p_value_key="p_value", output_key="holm_p_value")
    pd.DataFrame([{"comparison": key, **value} for key, value in stats.items()]).to_csv(output_dir / "band_sensitivity_stats.csv", index=False)
    return summary, stats


def run(config: Config) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    core = load_core(repo_root)
    core.seed_everything(config.seed)
    device = core.get_device(config.device)
    output_dir = repo_root / config.output_dir
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "band_sensitivity_metrics.csv"
    rows = pd.read_csv(metrics_path).to_dict(orient="records") if metrics_path.exists() else []
    completed = completed_keys(rows)
    total_runs = len(config.bands) * len(config.subjects) * len(config.models)
    run_index = len(completed)
    parameter_counts: Dict[str, int] = {}

    for band in config.bands:
        for subject in config.subjects:
            x_raw, y, sessions = load_subject_session_data(core, subject, repo_root / config.data_dir, cache_dir, band)
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
            for model_name in config.models:
                if (band, subject, model_name) in completed:
                    continue
                run_index += 1
                core.seed_everything(config.seed + subject * 100 + config.models.index(model_name))
                print(f"[{run_index}/{total_runs}] band={band} subject={subject} model={model_name}", flush=True)
                if core.is_classical_model(model_name):
                    fit_info = core.fit_riemann_tslr(x_train, y_train_full[train_idx], x_val, y_train_full[val_idx])
                    metrics = core.evaluate_classical_model(fit_info["model"], x_test_std, y_test)
                else:
                    model = core.build_model(model_name, x_train.shape[1], x_train.shape[2], len(core.LABEL_ORDER), config.cfc_hidden_size, config.lstm_hidden_size, config.cfc_dt, config.cfc_tau_init)
                    parameter_counts.setdefault(model_name, core.count_parameters(model))
                    fit_info = core.train_one_model(model, train_loader, val_loader, device, config.epochs, config.patience, config.min_epochs, config.learning_rate, config.weight_decay)
                    metrics = core.evaluate_model(model, test_loader, device)
                rows.append(
                    {
                        "band": band,
                        "fmin": BAND_MAP[band][0],
                        "fmax": BAND_MAP[band][1],
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
                completed.add((band, subject, model_name))
                pd.DataFrame(rows).to_csv(metrics_path, index=False)

    summary, stats = summarize(rows, core, output_dir)
    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "session-wise preprocessing band sensitivity",
        "config": asdict(config),
        "parameter_counts": parameter_counts,
        "summary": summary,
        "stat_tests": stats,
    }
    (output_dir / "band_sensitivity_results_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Run session-wise band sensitivity controls.")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--models", nargs="*", default=["shallow_convnet", "riemann_tslr", "eegnet", "mi_mamba", "tiny_transformer", "cfc", "lstm"])
    parser.add_argument("--bands", nargs="*", default=["mu_beta_8_30", "broad_4_40", "broad_1_45"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--cfc-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--cfc-dt", type=float, default=1.0)
    parser.add_argument("--cfc-tau-init", type=float, default=1.0)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/revision_band_sensitivity")
    return Config(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
