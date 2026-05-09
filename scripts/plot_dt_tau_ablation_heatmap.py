"""Plot heatmaps for the completed CfC dt/tau-initialization ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODEL_ORDER = ["cfc", "hybrid_cfc", "ss_cfc"]
MODEL_TITLES = {
    "cfc": "CfC-style",
    "hybrid_cfc": "Hybrid-CfC-style",
    "ss_cfc": "SpatialSpectral-CfC",
}


def plot_metric(summary: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Render one row of heatmaps, one panel per model."""
    sns.set_theme(style="white", font_scale=0.9)
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(10.5, 3.2), constrained_layout=True)

    for axis, model_name in zip(axes, MODEL_ORDER):
        model_df = summary.loc[summary["model"] == model_name]
        heatmap_df = model_df.pivot(index="tau_init", columns="dt", values=metric).sort_index(ascending=False)
        sns.heatmap(
            heatmap_df,
            ax=axis,
            annot=True,
            fmt=".1f" if metric.startswith("accuracy") else ".3f",
            cmap="viridis",
            cbar=model_name == MODEL_ORDER[-1],
            linewidths=0.5,
            linecolor="white",
        )
        axis.set_title(MODEL_TITLES.get(model_name, model_name))
        axis.set_xlabel(r"$\Delta t$")
        axis.set_ylabel(r"$\tau_{\mathrm{init}}$" if model_name == MODEL_ORDER[0] else "")

    fig.suptitle("CfC dt/tau-initialization ablation", y=1.05)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dt/tau ablation heatmaps.")
    parser.add_argument(
        "--summary",
        default="outputs/revision_cfc_dt_tau_ablation/ablation_summary.csv",
        help="Path to ablation_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/revision_cfc_dt_tau_ablation",
        help="Directory for generated heatmap PDFs.",
    )
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    output_dir = Path(args.output_dir)
    plot_metric(summary, "accuracy_mean", output_dir / "dt_tau_accuracy_heatmap.pdf")
    plot_metric(summary, "f1_mean", output_dir / "dt_tau_f1_heatmap.pdf")


if __name__ == "__main__":
    main()
