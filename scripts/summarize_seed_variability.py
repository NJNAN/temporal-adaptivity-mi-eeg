"""汇总多次 session-wise clean repeat 的 seed variability。

对应论文：
1. 关于“主排序不是由单个幸运 seed 造成”的补充说明。
2. supporting materials 中的 seed-level summary / ranking / variability JSON。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_seed_run(directory: Path, metrics_name: str) -> pd.DataFrame:
    """读取单个 repeat 目录，并补齐 seed 字段。"""
    metrics_path = directory / metrics_name
    summary_path = directory / "results_summary.json"
    df = pd.read_csv(metrics_path)
    if "seed" not in df.columns:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        df["seed"] = int(summary["config"]["seed"])
    return df


def rank_string(group: pd.DataFrame) -> str:
    """把单个 seed 下的模型排序压成可直接写入论文的一行文本。"""
    ordered = group.sort_values("accuracy_mean", ascending=False)["model_display"].tolist()
    return " > ".join(ordered)


def main() -> None:
    """汇总多个 repeat seed 的稳定性结果。"""
    parser = argparse.ArgumentParser(description="Summarize session-wise seed variability across multiple run directories.")
    parser.add_argument("--run-dir", nargs="+", required=True, help="Directories containing metrics and results_summary.json.")
    parser.add_argument("--metrics-name", type=str, default="sessionwise_clean_metrics.csv")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables: List[pd.DataFrame] = []
    for run_dir_str in args.run_dir:
        run_dir = Path(run_dir_str)
        metrics_name = args.metrics_name
        if not (run_dir / metrics_name).exists() and (run_dir / "sessionwise_metrics.csv").exists():
            metrics_name = "sessionwise_metrics.csv"
        df = load_seed_run(run_dir, metrics_name)
        tables.append(df)

    combined = pd.concat(tables, ignore_index=True)
    if args.models:
        combined = combined.loc[combined["model"].isin(args.models)].copy()
    if "model_display" not in combined.columns:
        combined["model_display"] = combined["model"]
    combined = combined.sort_values(["seed", "subject", "model"]).reset_index(drop=True)
    combined.to_csv(output_dir / "seed_subject_metrics.csv", index=False)

    by_seed = (
        combined.groupby(["seed", "model", "model_display"], as_index=False)[["accuracy", "f1"]]
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=1)) if len(s) > 1 else 0.0),
            f1_mean=("f1", "mean"),
            f1_std=("f1", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=1)) if len(s) > 1 else 0.0),
        )
        .sort_values(["seed", "accuracy_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    by_seed.to_csv(output_dir / "seed_model_summary.csv", index=False)

    across_seed = (
        by_seed.groupby(["model", "model_display"], as_index=False)[["accuracy_mean", "f1_mean"]]
        .agg(
            accuracy_mean=("accuracy_mean", "mean"),
            accuracy_seed_std=("accuracy_mean", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=1)) if len(s) > 1 else 0.0),
            accuracy_min=("accuracy_mean", "min"),
            accuracy_max=("accuracy_mean", "max"),
            f1_mean=("f1_mean", "mean"),
            f1_seed_std=("f1_mean", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=1)) if len(s) > 1 else 0.0),
            f1_min=("f1_mean", "min"),
            f1_max=("f1_mean", "max"),
        )
        .sort_values("accuracy_mean", ascending=False)
        .reset_index(drop=True)
    )
    across_seed.to_csv(output_dir / "seed_variability_summary.csv", index=False)

    rank_rows = []
    for seed_value, group in by_seed.groupby("seed"):
        rank_rows.append({"seed": int(seed_value), "ranking": rank_string(group)})
    rank_df = pd.DataFrame(rank_rows).sort_values("seed").reset_index(drop=True)
    rank_df.to_csv(output_dir / "seed_rankings.csv", index=False)

    pairwise_relations: Dict[str, Dict[str, int]] = {}
    pivot = by_seed.pivot(index="seed", columns="model", values="accuracy_mean")
    for model_a in pivot.columns:
        for model_b in pivot.columns:
            if model_a >= model_b:
                continue
            relation_key = f"{model_a}_gt_{model_b}"
            pairwise_relations[relation_key] = {
                "count": int((pivot[model_a] > pivot[model_b]).sum()),
                "num_seeds": int(len(pivot)),
            }

    summary = {
        "num_seeds": int(combined["seed"].nunique()),
        "seeds": [int(seed) for seed in sorted(combined["seed"].unique().tolist())],
        "models": combined["model"].drop_duplicates().tolist(),
        "rankings": rank_df.to_dict(orient="records"),
        "pairwise_relations": pairwise_relations,
    }
    (output_dir / "seed_variability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
