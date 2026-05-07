"""导出论文佐证材料与复现包。

对应论文：
1. Supporting materials / reproducibility package 的全部目录结构。
2. 主文中关于“提供脚本、fold 划分、seed 配置与补充结果”的描述。

本脚本不产生新实验结果，而是把已经完成的各类实验产物重新整理为
投稿可附带的 `paper_ready` 与 `supporting_materials` 两套包。
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def load_module(path: Path, module_name: str):
    """动态加载脚本，避免复制主实验中的显示名和统计工具。"""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_summary_table(summary: dict, parameter_counts: dict, display_name_fn, model_order: list[str]) -> pd.DataFrame:
    """把论文 JSON 汇总结果整理成正文表格可直接引用的 CSV。"""
    rows = []
    for model_name, metrics in summary.items():
        rows.append(
            {
                "model": model_name,
                "model_display": display_name_fn(model_name),
                "params": int(parameter_counts[model_name]),
                "accuracy_mean": metrics["accuracy_mean"],
                "accuracy_std": metrics["accuracy_std"],
                "f1_mean": metrics["f1_mean"],
                "f1_std": metrics["f1_std"],
            }
        )
    df = pd.DataFrame(rows)
    order = {name: index for index, name in enumerate(model_order)}
    df["sort_key"] = df["model_display"].map(order)
    df = df.sort_values(["sort_key", "model_display"]).drop(columns=["sort_key"])
    return df.reset_index(drop=True)


def copy_if_exists(source: Path, destination: Path) -> None:
    """仅在源文件存在时复制，避免补充材料导出因可选实验缺失而中断。"""
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree_files(source_root: Path, target_root: Path, relative_paths: list[str]) -> None:
    """按相对路径列表复制文件树，用于同步 supporting materials。"""
    for relative_path in relative_paths:
        copy_if_exists(source_root / relative_path, target_root / relative_path)


def build_tau_window_summary(path: Path) -> dict:
    """把论文正文使用的 subject-level tau 窗口汇总整理成可嵌入 JSON 的结构。"""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for class_name, class_df in df.groupby("class_name"):
        summary[class_name] = {}
        for _, row in class_df.iterrows():
            summary[class_name][str(row["window"])] = {
                "mean_tau": float(row["mean_tau"]),
                "std_tau": float(row["std_tau"]),
                "mean_time_seconds": None if pd.isna(row["mean_time_seconds"]) else float(row["mean_time_seconds"]),
                "std_time_seconds": None if pd.isna(row["std_time_seconds"]) else float(row["std_time_seconds"]),
            }
    return summary


def git_commit(repo_root: Path) -> str:
    """记录当前代码版本，对应论文复现说明中的 commit hash。"""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    """汇总论文正文、补充材料和复现包所需的全部产物。"""
    repo_root = Path(__file__).resolve().parents[1]
    core = load_module(repo_root / "scripts" / "run_mi_experiments.py", "mi_exp_core_export")
    sessionwise = load_module(repo_root / "scripts" / "run_sessionwise_mi_comparison.py", "mi_exp_session_export")

    pooled_dir = repo_root / "outputs" / "bspc_pooled"
    session_dir = repo_root / "outputs" / "bspc_sessionwise"
    grouped_dir = repo_root / "outputs" / "bspc_grouped_cv"
    gru_pooled_dir = repo_root / "outputs" / "bspc_gru_pooled"
    gru_session_dir = repo_root / "outputs" / "bspc_gru_sessionwise"
    sweep_dir = repo_root / "outputs" / "bspc_perturbation_sweep"
    temporal_shuffle_dir = repo_root / "outputs" / "bspc_temporal_shuffle"
    efficiency_dir = repo_root / "outputs" / "bspc_efficiency"
    bnci_aux_dir = repo_root / "outputs" / "bnci2014_004_aux"
    tau_control_dir = repo_root / "outputs" / "bspc_tau_controls"
    paper_dir = repo_root / "outputs" / "paper_ready"
    support_dir = repo_root / "supporting_materials"
    paper_dir.mkdir(parents=True, exist_ok=True)
    support_dir.mkdir(parents=True, exist_ok=True)

    pooled_summary = json.loads((pooled_dir / "results_summary.json").read_text(encoding="utf-8"))
    session_summary = json.loads((session_dir / "sessionwise_results_summary.json").read_text(encoding="utf-8"))
    grouped_summary = json.loads((grouped_dir / "results_summary.json").read_text(encoding="utf-8"))
    gru_pooled_summary = json.loads((gru_pooled_dir / "results_summary.json").read_text(encoding="utf-8"))
    gru_session_summary = json.loads((gru_session_dir / "sessionwise_results_summary.json").read_text(encoding="utf-8"))

    pooled_table_order = [
        "Shallow ConvNet",
        "Riemann-TSLR",
        "EEGNet",
        "Hybrid-CfC-style",
        "Tiny-Transformer",
        "CfC-style",
        "LSTM",
    ]
    session_table_order = [
        "Riemann-TSLR",
        "Shallow ConvNet",
        "Hybrid-CfC-style",
        "Tiny-Transformer",
        "EEGNet",
        "CfC-style",
        "LSTM",
    ]
    grouped_table_order = [
        "Riemann-TSLR",
        "Shallow ConvNet",
        "EEGNet",
        "Hybrid-CfC-style",
        "Tiny-Transformer",
        "CfC-style",
        "LSTM",
    ]

    main_table = build_summary_table(
        pooled_summary["summary"],
        pooled_summary["parameter_counts"],
        core.get_model_display_name,
        pooled_table_order,
    )
    main_table.to_csv(paper_dir / "main_table.csv", index=False)
    session_table = build_summary_table(
        session_summary["summary"],
        session_summary["parameter_counts"],
        core.get_model_display_name,
        session_table_order,
    )
    session_table.to_csv(paper_dir / "sessionwise_table.csv", index=False)
    grouped_table = build_summary_table(
        grouped_summary["summary"],
        grouped_summary["parameter_counts"],
        core.get_model_display_name,
        grouped_table_order,
    )
    grouped_table.to_csv(paper_dir / "grouped_cv_table.csv", index=False)
    recurrent_control_table = pd.DataFrame(
        [
            {
                "protocol": "pooled",
                "model": model_name,
                "model_display": core.get_model_display_name(model_name),
                "params": int(
                    pooled_summary["parameter_counts"].get(model_name, gru_pooled_summary["parameter_counts"].get(model_name))
                ),
                "accuracy_mean": (
                    pooled_summary["summary"].get(model_name, gru_pooled_summary["summary"].get(model_name))["accuracy_mean"]
                ),
                "accuracy_std": (
                    pooled_summary["summary"].get(model_name, gru_pooled_summary["summary"].get(model_name))["accuracy_std"]
                ),
                "f1_mean": pooled_summary["summary"].get(model_name, gru_pooled_summary["summary"].get(model_name))["f1_mean"],
                "f1_std": pooled_summary["summary"].get(model_name, gru_pooled_summary["summary"].get(model_name))["f1_std"],
            }
            for model_name in ["cfc", "gru", "lstm"]
        ]
        + [
            {
                "protocol": "sessionwise",
                "model": model_name,
                "model_display": core.get_model_display_name(model_name),
                "params": int(
                    session_summary["parameter_counts"].get(model_name, gru_session_summary["parameter_counts"].get(model_name))
                ),
                "accuracy_mean": (
                    session_summary["summary"].get(model_name, gru_session_summary["summary"].get(model_name))["accuracy_mean"]
                ),
                "accuracy_std": (
                    session_summary["summary"].get(model_name, gru_session_summary["summary"].get(model_name))["accuracy_std"]
                ),
                "f1_mean": session_summary["summary"].get(model_name, gru_session_summary["summary"].get(model_name))["f1_mean"],
                "f1_std": session_summary["summary"].get(model_name, gru_session_summary["summary"].get(model_name))["f1_std"],
            }
            for model_name in ["cfc", "gru", "lstm"]
        ]
    )
    recurrent_control_table.to_csv(paper_dir / "recurrent_control_table.csv", index=False)
    pd.read_csv(sweep_dir / "sweep_summary.csv").to_csv(paper_dir / "perturbation_sweep_summary.csv", index=False)
    pd.read_csv(sweep_dir / "sweep_stats.csv").to_csv(paper_dir / "perturbation_sweep_stats.csv", index=False)
    structured_table = pd.read_csv(session_dir / "structured_perturbation_summary.csv")
    structured_order = {
        "shallow_convnet": 0,
        "riemann_tslr": 1,
        "eegnet": 2,
        "tiny_transformer": 3,
        "cfc": 4,
        "hybrid_cfc": 5,
        "lstm": 6,
    }
    structured_table["sort_key"] = structured_table["model"].map(structured_order)
    structured_table = structured_table.sort_values(["sort_key", "perturbation"]).drop(columns=["sort_key"])
    structured_table.to_csv(paper_dir / "structured_perturbation_table.csv", index=False)
    if (temporal_shuffle_dir / "temporal_shuffle_summary.csv").exists():
        pd.read_csv(temporal_shuffle_dir / "temporal_shuffle_summary.csv").to_csv(
            paper_dir / "temporal_shuffle_summary.csv",
            index=False,
        )
        pd.read_csv(temporal_shuffle_dir / "temporal_shuffle_stats.csv").to_csv(
            paper_dir / "temporal_shuffle_stats.csv",
            index=False,
        )
        pd.read_csv(temporal_shuffle_dir / "temporal_shuffle_subject_summary.csv").to_csv(
            paper_dir / "temporal_shuffle_subject_summary.csv",
            index=False,
        )
        (paper_dir / "temporal_shuffle_results_summary.json").write_text(
            (temporal_shuffle_dir / "temporal_shuffle_results_summary.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    if (tau_control_dir / "tau_stats.json").exists():
        copy_if_exists(tau_control_dir / "tau_stats.json", paper_dir / "tau_local_window_stats.json")
        copy_if_exists(
            tau_control_dir / "tau_motor_subject_class_summary.csv",
            paper_dir / "tau_motor_subject_class_summary.csv",
        )
        copy_if_exists(
            tau_control_dir / "global_tau_window_subject_class_summary.csv",
            paper_dir / "global_tau_window_subject_class_summary.csv",
        )
        copy_if_exists(
            tau_control_dir / "motor_tau_window_subject_class_summary.csv",
            paper_dir / "motor_tau_window_subject_class_summary.csv",
        )

    pd.read_csv(pooled_dir / "stat_tests.csv").to_csv(paper_dir / "pooled_stats.csv", index=False)
    pd.read_csv(session_dir / "stat_tests.csv").to_csv(paper_dir / "sessionwise_stats.csv", index=False)
    pd.read_csv(grouped_dir / "stat_tests.csv").to_csv(paper_dir / "grouped_cv_stats.csv", index=False)
    pd.read_csv(session_dir / "structured_perturbation_stats.csv").to_csv(
        paper_dir / "structured_perturbation_stats.csv",
        index=False,
    )
    pooled_recurrent_df = pd.concat(
        [
            pd.read_csv(pooled_dir / "subject_summary.csv").query("model in ['cfc', 'lstm']"),
            pd.read_csv(gru_pooled_dir / "subject_summary.csv"),
        ],
        ignore_index=True,
    )
    session_recurrent_df = pd.concat(
        [
            pd.read_csv(session_dir / "sessionwise_metrics.csv").query("model in ['cfc', 'lstm']"),
            pd.read_csv(gru_session_dir / "sessionwise_metrics.csv"),
        ],
        ignore_index=True,
    )
    recurrent_control_stats = {
        "pooled": {
            "cfc_vs_gru": core.paired_test(pooled_recurrent_df[["subject", "model", "accuracy"]], "cfc", "gru"),
            "gru_vs_lstm": core.paired_test(pooled_recurrent_df[["subject", "model", "accuracy"]], "gru", "lstm"),
            "cfc_vs_lstm": core.paired_test(pooled_recurrent_df[["subject", "model", "accuracy"]], "cfc", "lstm"),
        },
        "sessionwise": {
            "cfc_vs_gru": core.paired_test(session_recurrent_df[["subject", "model", "accuracy"]], "cfc", "gru"),
            "gru_vs_lstm": core.paired_test(session_recurrent_df[["subject", "model", "accuracy"]], "gru", "lstm"),
            "cfc_vs_lstm": core.paired_test(session_recurrent_df[["subject", "model", "accuracy"]], "cfc", "lstm"),
        },
    }
    for test_family in recurrent_control_stats.values():
        core.apply_holm_correction(test_family, p_value_key="p_value", output_key="holm_p_value")
        core.apply_holm_correction(test_family, p_value_key="wilcoxon_p_value", output_key="wilcoxon_holm_p_value")
    recurrent_control_export = {
        "metadata": {
            "note": (
                "Holm correction in this file is applied within the recurrent-only control family "
                "{cfc_vs_gru, gru_vs_lstm, cfc_vs_lstm}. This differs from pooled_stats.csv and "
                "sessionwise_stats.csv, where Holm correction is applied within the full benchmark family."
            ),
            "pooled_family": ["cfc_vs_gru", "gru_vs_lstm", "cfc_vs_lstm"],
            "sessionwise_family": ["cfc_vs_gru", "gru_vs_lstm", "cfc_vs_lstm"],
        },
        **recurrent_control_stats,
    }
    (paper_dir / "recurrent_control_stats.json").write_text(
        json.dumps(recurrent_control_export, indent=2),
        encoding="utf-8",
    )

    bnci_aux_results = None
    if (bnci_aux_dir / "results_summary.json").exists():
        bnci_aux_results = json.loads((bnci_aux_dir / "results_summary.json").read_text(encoding="utf-8"))
        bnci_aux_table = build_summary_table(
            bnci_aux_results["summary"],
            bnci_aux_results["parameter_counts"],
            core.get_model_display_name,
            ["Shallow ConvNet", "EEGNet", "Riemann-TSLR", "LSTM", "Tiny-Transformer", "CfC-style"],
        )
        bnci_aux_table.to_csv(paper_dir / "bnci2014_004_aux_summary.csv", index=False)
        pd.read_csv(bnci_aux_dir / "stat_tests.csv").to_csv(paper_dir / "bnci2014_004_aux_stats.csv", index=False)
        pd.read_csv(bnci_aux_dir / "aux_metrics.csv").to_csv(paper_dir / "bnci2014_004_aux_metrics.csv", index=False)
        (paper_dir / "bnci2014_004_results_summary.json").write_text(
            json.dumps(bnci_aux_results, indent=2),
            encoding="utf-8",
        )

    tau_analysis_export = dict(session_summary.get("tau_analysis", {}))
    if "timecourse_summary" in tau_analysis_export:
        tau_analysis_export["global_mean_timecourse_summary"] = tau_analysis_export.pop("timecourse_summary")
    tau_window_summary = build_tau_window_summary(session_dir / "tau_time_window_summary.csv")
    if tau_window_summary:
        tau_analysis_export["subject_level_window_summary"] = tau_window_summary
        tau_analysis_export["time_summary_note"] = (
            "The manuscript timing sentence uses subject-level peak/window summaries from "
            "tau_time_window_summary.csv. The global_mean_timecourse_summary, when present, is based "
            "on class-averaged timecourses and should not be compared directly to the manuscript values."
        )

    key_stats = {
        "pooled": {
            "summary": pooled_summary["summary"],
            "stat_tests": pooled_summary["stat_tests"],
            "stat_test_family": "full seven-model pooled benchmark family",
        },
        "sessionwise": {
            "summary": session_summary["summary"],
            "stat_tests": session_summary["stat_tests"],
            "stat_test_family": "full seven-model session-wise benchmark family",
        },
        "grouped_cv": {
            "summary": grouped_summary["summary"],
            "stat_tests": grouped_summary["stat_tests"],
            "group_definition": grouped_summary.get("group_definition", ""),
            "stat_test_family": "full seven-model grouped pooled benchmark family",
        },
        "recurrent_controls": {
            "pooled": gru_pooled_summary["summary"],
            "sessionwise": gru_session_summary["summary"],
            "stats": recurrent_control_export,
        },
        "tau_analysis": tau_analysis_export,
        "structured_perturbation_summary": session_summary.get("structured_perturbation_summary", {}),
        "structured_perturbation_tests": session_summary.get("structured_perturbation_tests", {}),
        "perturbation_sweep": json.loads((sweep_dir / "results_summary.json").read_text(encoding="utf-8")),
    }
    if (tau_control_dir / "tau_stats.json").exists():
        key_stats["tau_locality_controls"] = json.loads(
            (tau_control_dir / "tau_stats.json").read_text(encoding="utf-8")
        )
    if (temporal_shuffle_dir / "temporal_shuffle_results_summary.json").exists():
        key_stats["temporal_shuffle_control"] = json.loads(
            (temporal_shuffle_dir / "temporal_shuffle_results_summary.json").read_text(encoding="utf-8")
        )
    if bnci_aux_results is not None:
        key_stats["auxiliary_bnci2014_004"] = {
            "summary": bnci_aux_results["summary"],
            "stat_tests": bnci_aux_results["stat_tests"],
            "protocol": bnci_aux_results["protocol"],
        }
    (paper_dir / "key_stats.json").write_text(json.dumps(key_stats, indent=2), encoding="utf-8")

    data_dir = repo_root / "data"
    cache_dir = session_dir / "cache"
    pooled_rows = []
    session_rows = []
    pooled_seed = int(pooled_summary["config"]["seed"])
    session_seed = int(session_summary["config"]["seed"])
    downsample_factor = int(session_summary["config"]["downsample_factor"])
    val_fraction = float(session_summary["config"]["val_fraction"])
    subjects = list(range(1, 10))

    for subject in subjects:
        X, y, sessions = sessionwise.load_subject_session_data(core, subject, data_dir, cache_dir)
        X = core.downsample_trials(X, downsample_factor)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=pooled_seed + subject)
        for fold_idx, (train_val_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
            inner = StratifiedShuffleSplit(
                n_splits=1,
                test_size=val_fraction,
                random_state=pooled_seed + subject * 10 + fold_idx,
            )
            train_inner_idx, val_inner_idx = next(inner.split(X[train_val_idx], y[train_val_idx]))
            assignment = {}
            for idx in train_val_idx[train_inner_idx]:
                assignment[int(idx)] = "train"
            for idx in train_val_idx[val_inner_idx]:
                assignment[int(idx)] = "val"
            for idx in test_idx:
                assignment[int(idx)] = "test"
            for trial_index in range(len(y)):
                pooled_rows.append(
                    {
                        "subject": subject,
                        "fold": fold_idx,
                        "trial_index": trial_index,
                        "session": sessions[trial_index],
                        "label_index": int(y[trial_index]),
                        "label_name": core.INDEX_TO_LABEL[int(y[trial_index])],
                        "split": assignment[trial_index],
                    }
                )

        train_mask = sessions == "0train"
        test_mask = sessions == "1test"
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_fraction,
            random_state=session_seed + subject,
        )
        train_inner_idx, val_inner_idx = next(inner.split(X[train_mask], y[train_mask]))
        session_assignment = {}
        train_pool_indices = [idx for idx, is_train in enumerate(train_mask) if is_train]
        for local_idx in train_inner_idx:
            session_assignment[train_pool_indices[int(local_idx)]] = "train"
        for local_idx in val_inner_idx:
            session_assignment[train_pool_indices[int(local_idx)]] = "val"
        for idx, is_test in enumerate(test_mask):
            if is_test:
                session_assignment[idx] = "test"
        for trial_index in range(len(y)):
            session_rows.append(
                {
                    "subject": subject,
                    "trial_index": trial_index,
                    "session": sessions[trial_index],
                    "label_index": int(y[trial_index]),
                    "label_name": core.INDEX_TO_LABEL[int(y[trial_index])],
                    "split": session_assignment[trial_index],
                }
            )

    pd.DataFrame(pooled_rows).to_csv(paper_dir / "pooled_fold_assignments.csv", index=False)
    pd.DataFrame(session_rows).to_csv(paper_dir / "sessionwise_assignments.csv", index=False)
    pd.read_csv(grouped_dir / "grouped_fold_assignments.csv").to_csv(
        paper_dir / "grouped_fold_assignments.csv",
        index=False,
    )

    pd.read_csv(pooled_dir / "subject_summary.csv").to_csv(paper_dir / "pooled_subject_scores.csv", index=False)
    pd.read_csv(session_dir / "sessionwise_metrics.csv").to_csv(paper_dir / "sessionwise_subject_scores.csv", index=False)
    pd.read_csv(grouped_dir / "subject_summary.csv").to_csv(paper_dir / "grouped_subject_scores.csv", index=False)

    seed_info = {
        "git_commit": git_commit(repo_root),
        "pooled_cv_seed": pooled_seed,
        "sessionwise_seed": session_seed,
        "grouped_cv_seed": int(grouped_summary["config"]["seed"]),
        "downsample_factor": downsample_factor,
        "val_fraction": val_fraction,
        "structured_repeats": int(session_summary["config"]["structured_repeats"]),
        "subjects": subjects,
        "scripts": {
            "pooled": "scripts/run_mi_experiments.py",
            "sessionwise": "scripts/run_sessionwise_mi_comparison.py",
            "grouped_cv": "scripts/run_grouped_pooled_control.py",
            "perturbation_sweep": "scripts/run_structured_perturbation_sweep.py",
            "temporal_shuffle": "scripts/run_temporal_shuffle_control.py",
            "bnci_aux": "scripts/run_bnci2014_004_aux.py",
            "efficiency": "scripts/benchmark_model_efficiency.py",
            "export": "scripts/export_reproducibility_artifacts.py",
        },
    }
    (paper_dir / "seed_config.json").write_text(json.dumps(seed_info, indent=2), encoding="utf-8")

    # 下方目录结构直接对应论文提交时的 supporting materials 分栏。
    sections = {
        "paper_tables": [
            (paper_dir / "main_table.csv", "main_table.csv"),
            (paper_dir / "sessionwise_table.csv", "sessionwise_table.csv"),
            (paper_dir / "grouped_cv_table.csv", "grouped_cv_table.csv"),
            (paper_dir / "recurrent_control_table.csv", "recurrent_control_table.csv"),
            (paper_dir / "perturbation_sweep_summary.csv", "perturbation_sweep_summary.csv"),
            (paper_dir / "temporal_shuffle_summary.csv", "temporal_shuffle_summary.csv"),
            (paper_dir / "bnci2014_004_aux_summary.csv", "bnci2014_004_aux_summary.csv"),
            (paper_dir / "bnci2014_004_aux_stats.csv", "bnci2014_004_aux_stats.csv"),
            (paper_dir / "structured_perturbation_table.csv", "structured_perturbation_table.csv"),
            (paper_dir / "pooled_stats.csv", "pooled_stats.csv"),
            (paper_dir / "sessionwise_stats.csv", "sessionwise_stats.csv"),
            (paper_dir / "grouped_cv_stats.csv", "grouped_cv_stats.csv"),
            (paper_dir / "perturbation_sweep_stats.csv", "perturbation_sweep_stats.csv"),
            (paper_dir / "temporal_shuffle_stats.csv", "temporal_shuffle_stats.csv"),
            (paper_dir / "structured_perturbation_stats.csv", "structured_perturbation_stats.csv"),
            (paper_dir / "recurrent_control_stats.json", "recurrent_control_stats.json"),
            (paper_dir / "tau_local_window_stats.json", "tau_local_window_stats.json"),
            (paper_dir / "key_stats.json", "key_stats.json"),
        ],
        "subject_results": [
            (paper_dir / "pooled_subject_scores.csv", "pooled_subject_scores.csv"),
            (paper_dir / "sessionwise_subject_scores.csv", "sessionwise_subject_scores.csv"),
            (paper_dir / "grouped_subject_scores.csv", "grouped_subject_scores.csv"),
            (gru_pooled_dir / "subject_summary.csv", "gru_pooled_subject_scores.csv"),
            (gru_session_dir / "sessionwise_metrics.csv", "gru_sessionwise_subject_scores.csv"),
            (paper_dir / "pooled_fold_assignments.csv", "pooled_fold_assignments.csv"),
            (paper_dir / "sessionwise_assignments.csv", "sessionwise_assignments.csv"),
            (paper_dir / "grouped_fold_assignments.csv", "grouped_fold_assignments.csv"),
            (pooled_dir / "subject_accuracy_boxplot.pdf", "pooled_subject_accuracy_boxplot.pdf"),
            (pooled_dir / "subject_accuracy_table.csv", "pooled_subject_accuracy_table.csv"),
            (session_dir / "subject_accuracy_boxplot.pdf", "sessionwise_subject_accuracy_boxplot.pdf"),
            (session_dir / "subject_accuracy_table.csv", "sessionwise_subject_accuracy_table.csv"),
            (grouped_dir / "subject_accuracy_boxplot.pdf", "grouped_subject_accuracy_boxplot.pdf"),
            (grouped_dir / "subject_accuracy_table.csv", "grouped_subject_accuracy_table.csv"),
            (pooled_dir / "per_class_f1_subject.csv", "pooled_per_class_f1_subject.csv"),
            (pooled_dir / "per_class_f1_summary.csv", "pooled_per_class_f1_summary.csv"),
            (paper_dir / "bnci2014_004_aux_metrics.csv", "bnci2014_004_aux_metrics.csv"),
            (paper_dir / "temporal_shuffle_subject_summary.csv", "temporal_shuffle_subject_summary.csv"),
            (pooled_dir / "confusion_matrices.csv", "pooled_confusion_matrices.csv"),
            (pooled_dir / "confusion_matrices.pdf", "pooled_confusion_matrices.pdf"),
            (session_dir / "per_class_f1_subject.csv", "sessionwise_per_class_f1_subject.csv"),
            (session_dir / "per_class_f1_summary.csv", "sessionwise_per_class_f1_summary.csv"),
            (session_dir / "confusion_matrices.csv", "sessionwise_confusion_matrices.csv"),
            (session_dir / "confusion_matrices.pdf", "sessionwise_confusion_matrices.pdf"),
            (grouped_dir / "per_class_f1_subject.csv", "grouped_per_class_f1_subject.csv"),
            (grouped_dir / "per_class_f1_summary.csv", "grouped_per_class_f1_summary.csv"),
            (grouped_dir / "confusion_matrices.csv", "grouped_confusion_matrices.csv"),
            (grouped_dir / "confusion_matrices.pdf", "grouped_confusion_matrices.pdf"),
        ],
        "tau_analysis": [
            (session_dir / "tau_stats.json", "tau_stats.json"),
            (session_dir / "tau_trial_metrics.csv", "tau_trial_metrics.csv"),
            (session_dir / "tau_subject_class_summary.csv", "tau_subject_class_summary.csv"),
            (session_dir / "tau_timecourse_summary.csv", "tau_timecourse_summary.csv"),
            (session_dir / "tau_timecourse_subject_level.csv", "tau_timecourse_subject_level.csv"),
            (session_dir / "tau_time_window_summary.csv", "tau_time_window_summary.csv"),
            (session_dir / "tau_timecourse_by_class.pdf", "tau_timecourse_by_class.pdf"),
            (session_dir / "tau_dist_placeholder.pdf", "tau_dist_placeholder.pdf"),
            (tau_control_dir / "tau_stats.json", "tau_local_window_stats.json"),
            (tau_control_dir / "tau_motor_subject_class_summary.csv", "tau_motor_subject_class_summary.csv"),
            (tau_control_dir / "tau_motor_dist.pdf", "tau_motor_dist.pdf"),
            (
                tau_control_dir / "global_tau_window_subject_class_summary.csv",
                "global_tau_window_subject_class_summary.csv",
            ),
            (
                tau_control_dir / "motor_tau_window_subject_class_summary.csv",
                "motor_tau_window_subject_class_summary.csv",
            ),
            (tau_control_dir / "global_tau_window_distributions.pdf", "global_tau_window_distributions.pdf"),
            (tau_control_dir / "motor_tau_window_distributions.pdf", "motor_tau_window_distributions.pdf"),
        ],
        "robustness": [
            (session_dir / "structured_perturbation_metrics.csv", "structured_perturbation_metrics.csv"),
            (session_dir / "structured_perturbation_subject_summary.csv", "structured_perturbation_subject_summary.csv"),
            (session_dir / "structured_perturbation_summary.csv", "structured_perturbation_summary.csv"),
            (session_dir / "structured_perturbation_stats.csv", "structured_perturbation_stats.csv"),
            (sweep_dir / "clean_subject_metrics.csv", "perturbation_sweep_clean_subject_metrics.csv"),
            (sweep_dir / "sweep_metrics.csv", "perturbation_sweep_metrics.csv"),
            (sweep_dir / "sweep_subject_summary.csv", "perturbation_sweep_subject_summary.csv"),
            (sweep_dir / "sweep_summary.csv", "perturbation_sweep_summary.csv"),
            (sweep_dir / "sweep_stats.csv", "perturbation_sweep_stats.csv"),
            (sweep_dir / "band_noise_accuracy_sweep.pdf", "band_noise_accuracy_sweep.pdf"),
            (sweep_dir / "channel_dropout_accuracy_sweep.pdf", "channel_dropout_accuracy_sweep.pdf"),
            (temporal_shuffle_dir / "temporal_shuffle_metrics.csv", "temporal_shuffle_metrics.csv"),
            (temporal_shuffle_dir / "temporal_shuffle_drop.pdf", "temporal_shuffle_drop.pdf"),
            (temporal_shuffle_dir / "temporal_shuffle_results_summary.json", "temporal_shuffle_results_summary.json"),
        ],
        "efficiency": [
            (efficiency_dir / "benchmark.csv", "benchmark.csv"),
            (efficiency_dir / "benchmark.json", "benchmark.json"),
        ],
        "reproducibility": [
            (paper_dir / "seed_config.json", "seed_config.json"),
            (paper_dir / "bnci2014_004_results_summary.json", "bnci2014_004_results_summary.json"),
            (repo_root / "REPRODUCIBILITY.md", "REPRODUCIBILITY.md"),
        ],
        "scripts": [
            (repo_root / "scripts" / "run_mi_experiments.py", "run_mi_experiments.py"),
            (repo_root / "scripts" / "run_sessionwise_mi_comparison.py", "run_sessionwise_mi_comparison.py"),
            (repo_root / "scripts" / "run_grouped_pooled_control.py", "run_grouped_pooled_control.py"),
            (repo_root / "scripts" / "run_structured_perturbation_sweep.py", "run_structured_perturbation_sweep.py"),
            (repo_root / "scripts" / "run_bnci2014_004_aux.py", "run_bnci2014_004_aux.py"),
            (repo_root / "scripts" / "run_temporal_shuffle_control.py", "run_temporal_shuffle_control.py"),
            (repo_root / "scripts" / "benchmark_model_efficiency.py", "benchmark_model_efficiency.py"),
            (repo_root / "scripts" / "export_reproducibility_artifacts.py", "export_reproducibility_artifacts.py"),
        ],
        "manuscript": [
            (repo_root / "lnn_mi_eeg_paper (2).tex", "lnn_mi_eeg_paper.tex"),
            (repo_root / "references.bib", "references.bib"),
        ],
    }

    for section, files in sections.items():
        target_dir = support_dir / section
        target_dir.mkdir(parents=True, exist_ok=True)
        for source, target_name in files:
            if source.exists():
                copy_if_exists(source, target_dir / target_name)

    print(paper_dir)
    print(support_dir)


if __name__ == "__main__":
    main()
