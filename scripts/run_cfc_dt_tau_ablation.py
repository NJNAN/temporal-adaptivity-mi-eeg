"""CfC Delta-t and tau-initialization ablation wrapper."""

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

import pandas as pd


@dataclass
class AblationConfig:
    models: List[str]
    subjects: List[int]
    dt_values: List[float]
    tau_init_values: List[float]
    epochs: int
    patience: int
    min_epochs: int
    batch_size: int
    seed: int
    device: str
    data_dir: str
    output_dir: str


def load_sessionwise(repo_root: Path):
    script_path = repo_root / "scripts" / "run_sessionwise_mi_comparison.py"
    spec = importlib.util.spec_from_file_location("mi_sessionwise_ablation", script_path)
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


def run_ablation(config: AblationConfig):
    repo_root = Path(__file__).resolve().parents[1]
    sessionwise = load_sessionwise(repo_root)
    output_dir = repo_root / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "ablation_summary.csv"
    rows = pd.read_csv(summary_path).to_dict(orient="records") if summary_path.exists() else []
    completed = {
        (float(row["dt"]), float(row["tau_init"]), str(row["model"]))
        for row in rows
        if {"dt", "tau_init", "model"}.issubset(row)
    }
    for dt_value in config.dt_values:
        for tau_init in config.tau_init_values:
            combo_done = all((float(dt_value), float(tau_init), model_name) in completed for model_name in config.models)
            combo_name = f"dt_{dt_value:g}_tau_{tau_init:g}".replace(".", "p")
            combo_dir = output_dir / combo_name
            if combo_done:
                print(f"skipping completed ablation combo {combo_name}", flush=True)
                continue
            seed_cache(repo_root, combo_dir / "cache")
            print(f"running ablation combo {combo_name}", flush=True)
            session_config = sessionwise.SessionwiseConfig(
                subjects=config.subjects,
                models=config.models,
                epochs=config.epochs,
                patience=config.patience,
                min_epochs=config.min_epochs,
                batch_size=config.batch_size,
                learning_rate=1e-3,
                weight_decay=1e-4,
                val_fraction=0.15,
                seed=config.seed,
                cfc_hidden_size=128,
                lstm_hidden_size=128,
                cfc_dt=dt_value,
                cfc_tau_init=tau_init,
                downsample_factor=2,
                structured_repeats=1,
                device=config.device,
                data_dir=config.data_dir,
                output_dir=str(combo_dir.relative_to(repo_root)),
            )
            result = sessionwise.run_sessionwise(session_config)
            for model_name, metrics in result["summary"].items():
                rows.append(
                    {
                        "dt": dt_value,
                        "tau_init": tau_init,
                        "model": model_name,
                        **metrics,
                    }
                )
            pd.DataFrame(rows).to_csv(output_dir / "ablation_summary.csv", index=False)
    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": asdict(config),
        "rows": rows,
    }
    (output_dir / "ablation_stats.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> AblationConfig:
    parser = argparse.ArgumentParser(description="Run CfC dt/tau-init session-wise ablation.")
    parser.add_argument("--models", nargs="*", default=["cfc", "hybrid_cfc", "ss_cfc"])
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--dt-values", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    parser.add_argument("--tau-init-values", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/revision_cfc_dt_tau_ablation")
    args = parser.parse_args()
    return AblationConfig(**vars(args))


if __name__ == "__main__":
    run_ablation(parse_args())
