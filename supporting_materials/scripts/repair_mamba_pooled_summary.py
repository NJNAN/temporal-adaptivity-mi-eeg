"""Repair a pooled revision summary after appending models in the same output dir.

The pooled runner supports resume, so it is easy to run a subset later and leave
`results_summary.json` with the latest subset config while `fold_metrics.csv`
contains all completed models. This script treats `fold_metrics.csv` as the
source of truth and rebuilds the JSON summary, statistics, and parameter counts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import pandas as pd


def load_core(repo_root: Path):
    script_path = repo_root / "scripts" / "run_mi_experiments.py"
    spec = importlib.util.spec_from_file_location("mi_exp_core_repair", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair revision_mamba_pooled results_summary.json.")
    parser.add_argument("--output-dir", default="outputs/revision_mamba_pooled")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--cfc-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--cfc-dt", type=float, default=1.0)
    parser.add_argument("--cfc-tau-init", type=float, default=1.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    core = load_core(repo_root)
    output_dir = repo_root / args.output_dir
    fold_path = output_dir / "fold_metrics.csv"
    if not fold_path.exists():
        raise FileNotFoundError(fold_path)

    fold_rows = pd.read_csv(fold_path).to_dict(orient="records")
    fold_df = pd.DataFrame(fold_rows)
    models = args.models or list(fold_df["model"].drop_duplicates())
    subject_summary_df, summary = core.summarize_subject_metrics(fold_rows)
    subject_summary_df.to_csv(output_dir / "subject_summary.csv", index=False)

    cache_dir = output_dir / "cache"
    x_first, _ = core.load_subject_data(args.subjects[0], repo_root / args.data_dir, cache_dir)
    x_first = core.downsample_trials(x_first, args.downsample_factor)
    n_channels = x_first.shape[1]
    n_samples = x_first.shape[2]
    parameter_counts = {
        model_name: core.get_parameter_count(
            model_name=model_name,
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=len(core.LABEL_ORDER),
            cfc_hidden_size=args.cfc_hidden_size,
            lstm_hidden_size=args.lstm_hidden_size,
            cfc_dt=args.cfc_dt,
            cfc_tau_init=args.cfc_tau_init,
        )
        for model_name in models
    }

    stat_tests = {}
    for model_a, model_b in combinations(models, 2):
        if {model_a, model_b}.issubset(set(subject_summary_df["model"])):
            stat_tests[f"{model_a}_vs_{model_b}"] = core.paired_test(subject_summary_df, model_a, model_b)
    core.apply_holm_correction(stat_tests, p_value_key="p_value", output_key="holm_p_value")
    core.apply_holm_correction(stat_tests, p_value_key="wilcoxon_p_value", output_key="wilcoxon_holm_p_value")
    pd.DataFrame([{"comparison": key, **value} for key, value in stat_tests.items()]).to_csv(
        output_dir / "stat_tests.csv",
        index=False,
    )

    repaired = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repair_note": "Rebuilt from fold_metrics.csv because the output directory was populated by multiple resumed/subset runs.",
        "device": "cuda",
        "config": {
            "subjects": args.subjects,
            "models": models,
            "num_folds": int(fold_df["fold"].max()) if "fold" in fold_df else 5,
            "epochs": 80,
            "patience": 20,
            "min_epochs": 25,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "val_fraction": 0.15,
            "seed": 20260318,
            "cfc_hidden_size": args.cfc_hidden_size,
            "lstm_hidden_size": args.lstm_hidden_size,
            "cfc_dt": args.cfc_dt,
            "cfc_tau_init": args.cfc_tau_init,
            "downsample_factor": args.downsample_factor,
            "device": "cuda",
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "smoke_test": False,
        },
        "parameter_counts": parameter_counts,
        "summary": {model_name: summary[model_name] for model_name in models if model_name in summary},
        "stat_tests": stat_tests,
    }
    (output_dir / "results_summary.json").write_text(json.dumps(repaired, indent=2), encoding="utf-8")
    print(json.dumps({"models": models, "parameter_counts": parameter_counts}, indent=2), flush=True)


if __name__ == "__main__":
    main()
