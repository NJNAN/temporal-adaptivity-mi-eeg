# Reproducibility Notes

This repository includes the scripts, seeds, and exported score tables used for the current MI-EEG letter draft.

## Main scripts

- `scripts/run_mi_experiments.py`
  - Main pooled 5-fold experiment for `Shallow ConvNet`, `Riemann-TSLR`, `EEGNet`, `Hybrid-CfC-style`, `Tiny-Transformer`, `CfC-style`, and `LSTM`
- `scripts/run_sessionwise_mi_comparison.py`
  - Session-wise experiment (`session 1 -> train/val`, `session 2 -> test`)
  - Also exports `tau` statistics, time-resolved `tau`, coarse `tau` window summaries, per-class F1, confusion matrices, and structured perturbation results
- `scripts/run_grouped_pooled_control.py`
  - Grouped pooled control using `session+run` identifiers in the outer split
  - Evaluates the seven main benchmark models: `Shallow ConvNet`, `Riemann-TSLR`, `EEGNet`, `Hybrid-CfC-style`, `Tiny-Transformer`, `CfC-style`, and `LSTM`
- `scripts/run_structured_perturbation_sweep.py`
  - Session-wise perturbation sweep over multiple SNR levels and channel-dropout ratios
  - Evaluates `Shallow ConvNet`, `Riemann-TSLR`, `EEGNet`, `CfC-style`, `GRU`, and `LSTM`
- `scripts/run_temporal_shuffle_control.py`
  - Supplementary session-wise temporal-shuffle control on representative models
  - Evaluates `Shallow ConvNet`, `Riemann-TSLR`, `EEGNet`, `CfC-style`, and `LSTM`
- `scripts/run_sessionwise_clean_subset.py`
  - Clean session-wise rerun used for repeat-seed variability checks
  - Keeps split and training seeds identical across models within each subject for a given repeat seed
- `scripts/summarize_seed_variability.py`
  - Aggregates repeat-seed session-wise reruns into ranking and variability summaries
- `scripts/run_bnci2014_004_aux.py`
  - Supporting-only binary-MI sanity check on BNCI2014-004 using the same preprocessing and training conventions where possible
- `scripts/benchmark_model_efficiency.py`
  - Practical GPU throughput / memory snapshot plus approximate per-trial latency and CPU `Riemann-TSLR` latency
- `scripts/export_reproducibility_artifacts.py`
  - Exports paper-ready tables, split assignments, and the supporting-materials package
- `scripts/check_environment.py`
  - Records Python, package, PyTorch/CUDA, GPU, Git commit, and expected-output availability for a reproducibility sanity check
- `scripts/run_loso_cross_subject.py`
  - Revision experiment for leave-one-subject-out cross-subject evaluation
- `scripts/run_cfc_dt_tau_ablation.py`
  - Revision wrapper for `Delta t` and `tau` initialization ablations
- `scripts/run_tau_topography.py`
  - Revision analysis for channel-wise sensitivity of hidden-state `tau`

## Seeds and split configuration

- Pooled subject-wise CV seed: `20260318`
- Session-wise seed: `20260320`
- Supplementary repeat seeds for clean session-wise variability: `42`, `43`
- Downsampling: `250 Hz -> 125 Hz`
- Structured perturbation repeats: `5`
- Validation fraction:
  - Pooled inner split: `0.15`
  - Session-wise train-session split: `0.15`

Full exported configuration is in:

- `outputs/paper_ready/seed_config.json`

## Exported reproducibility artifacts

- `outputs/paper_ready/pooled_fold_assignments.csv`
- `outputs/paper_ready/sessionwise_assignments.csv`
- `outputs/paper_ready/grouped_fold_assignments.csv`
- `outputs/paper_ready/pooled_subject_scores.csv`
- `outputs/paper_ready/sessionwise_subject_scores.csv`
- `outputs/paper_ready/grouped_subject_scores.csv`
- `outputs/paper_ready/recurrent_control_table.csv`
- `outputs/paper_ready/recurrent_control_stats.json`
- `outputs/paper_ready/tau_local_window_stats.json`
- `outputs/paper_ready/sessionwise_seed_model_summary.csv`
- `outputs/paper_ready/sessionwise_seed_variability_summary.csv`
- `outputs/paper_ready/sessionwise_seed_rankings.csv`
- `outputs/paper_ready/sessionwise_seed_variability_summary.json`
- `outputs/paper_ready/bnci2014_004_aux_summary.csv`
- `outputs/paper_ready/bnci2014_004_aux_stats.csv`
- `outputs/paper_ready/bnci2014_004_results_summary.json`
- `outputs/paper_ready/perturbation_sweep_summary.csv`
- `outputs/paper_ready/perturbation_sweep_stats.csv`
- `outputs/paper_ready/temporal_shuffle_summary.csv`
- `outputs/paper_ready/temporal_shuffle_stats.csv`
- `outputs/paper_ready/temporal_shuffle_subject_summary.csv`
- `outputs/paper_ready/temporal_shuffle_results_summary.json`
- `outputs/paper_ready/main_table.csv`
- `outputs/paper_ready/sessionwise_table.csv`
- `outputs/paper_ready/sessionwise_extended_revision_table.csv`
- `outputs/paper_ready/revision_mamba_hybrid_sessionwise_table.csv`
- `outputs/paper_ready/grouped_cv_table.csv`
- `outputs/paper_ready/structured_perturbation_table.csv`
- `outputs/paper_ready/pooled_stats.csv`
- `outputs/paper_ready/sessionwise_stats.csv`
- `outputs/paper_ready/grouped_cv_stats.csv`
- `outputs/paper_ready/structured_perturbation_stats.csv`
- `outputs/paper_ready/revision_mamba_pooled_table.csv`
- `outputs/paper_ready/revision_mamba_pooled_subject_scores.csv`
- `outputs/paper_ready/revision_mamba_hybrid_sessionwise_metrics.csv`
- `outputs/paper_ready/sessionwise_extended_revision_metrics.csv`
- `outputs/paper_ready/revision_loso_table.csv`
- `outputs/paper_ready/revision_loso_metrics.csv`
- `outputs/paper_ready/revision_loso_assignments.csv`
- `outputs/paper_ready/revision_cfc_dt_tau_ablation_summary.csv`
- `outputs/paper_ready/revision_tau_occlusion_channel_summary.csv`
- `outputs/paper_ready/revision_tau_occlusion_channel_subject.csv`
- `outputs/paper_ready/revision_tau_occlusion_topomap_global.pdf`
- `outputs/paper_ready/key_stats.json`
- `outputs/paper_ready/environment_check.json`
- `outputs/paper_ready/artifact_manifest.csv`

The same curated revision artifacts are copied into `supporting_materials/` by:

```powershell
python scripts/export_reproducibility_artifacts.py
```

Key submission-facing copies are:

- `supporting_materials/paper_tables/revision_mamba_pooled_table.csv`
- `supporting_materials/paper_tables/revision_loso_table.csv`
- `supporting_materials/paper_tables/revision_cfc_dt_tau_ablation_summary.csv`
- `supporting_materials/paper_tables/revision_tau_occlusion_channel_summary.csv`
- `supporting_materials/subject_results/revision_loso_metrics.csv`
- `supporting_materials/tau_analysis/revision_tau_occlusion_topomap_global.pdf`
- `supporting_materials/reproducibility/artifact_manifest.csv`

## Representative commands

```powershell
python scripts/run_mi_experiments.py --device cuda --output-dir outputs/bspc_pooled
python scripts/run_sessionwise_mi_comparison.py --device cuda --output-dir outputs/bspc_sessionwise_full_rerun
python scripts/run_grouped_pooled_control.py --models shallow_convnet riemann_tslr eegnet hybrid_cfc tiny_transformer cfc lstm --device cuda --output-dir outputs/bspc_grouped_cv
python scripts/run_mi_experiments.py --models gru --device cuda --output-dir outputs/bspc_gru_pooled
python scripts/run_sessionwise_mi_comparison.py --models gru --device cuda --output-dir outputs/bspc_gru_sessionwise
python scripts/run_structured_perturbation_sweep.py --device cuda --output-dir outputs/bspc_perturbation_sweep
python scripts/run_temporal_shuffle_control.py --device cuda --output-dir outputs/bspc_temporal_shuffle
python scripts/run_sessionwise_clean_subset.py --models shallow_convnet riemann_tslr eegnet cfc lstm --seed 42 --output-dir outputs/seed_variability/sessionwise_seed_42
python scripts/run_sessionwise_clean_subset.py --models shallow_convnet riemann_tslr eegnet cfc lstm --seed 43 --output-dir outputs/seed_variability/sessionwise_seed_43
python scripts/summarize_seed_variability.py --run-dir outputs/seed_variability/sessionwise_seed_42 outputs/seed_variability/sessionwise_seed_43 --output-dir outputs/seed_variability/summary_42_43
python scripts/run_bnci2014_004_aux.py --models shallow_convnet riemann_tslr eegnet tiny_transformer cfc lstm --output-dir outputs/bnci2014_004_aux --seed 42
python scripts/benchmark_model_efficiency.py
python scripts/export_reproducibility_artifacts.py
python scripts/check_environment.py --output outputs/paper_ready/environment_check.json
python scripts/run_mi_experiments.py --models shallow_convnet riemann_tslr eegnet mi_mamba tiny_transformer cfc lstm --device cuda --output-dir outputs/revision_mamba_pooled
python scripts/run_sessionwise_mi_comparison.py --models mi_mamba ss_cfc ss_head --device cuda --output-dir outputs/revision_mamba_hybrid_sessionwise
python scripts/run_loso_cross_subject.py --models shallow_convnet riemann_tslr eegnet mi_mamba tiny_transformer cfc lstm --device cuda --output-dir outputs/revision_loso
python scripts/run_cfc_dt_tau_ablation.py --models cfc hybrid_cfc ss_cfc --dt-values 0.5 1.0 2.0 --tau-init-values 0.5 1.0 2.0 --device cuda --output-dir outputs/revision_cfc_dt_tau_ablation
python scripts/run_tau_topography.py --device cuda --output-dir outputs/revision_tau_topography
```

## Local CUDA environment used for the revision run

The current Windows workstation run used:

```powershell
cd "D:\作业\lnn论文1"
$env:PYTHONNOUSERSITE = "1"
$py = "D:\conda\envs\lnn-mi-eeg\python.exe"
& $py scripts/check_environment.py --output outputs/paper_ready/environment_check.json
```

The environment was verified with PyTorch `2.11.0+cu128` and an NVIDIA GeForce RTX 4060 Laptop GPU. `PYTHONNOUSERSITE=1` is recommended on this machine to prevent user-site packages from shadowing the conda environment.

## Notes

- The pooled subject-wise CV is trial-level stratified within each subject after pooling both sessions. It is trial-disjoint and uses training-only normalization, but it does not enforce run-level grouping.
- The grouped pooled control keeps the same preprocessing and model-selection rules, but uses `session+run` groups in the outer split to quantify sensitivity to trial shuffling across runs.
- The session-wise protocol trains on session 1, uses a validation split drawn from session 1 only, and tests on session 2.
- The supplementary repeat-seed session-wise check uses the same outer protocol as the main session-wise benchmark, but reruns the clean evaluation with repeat seeds `42` and `43`; for a given subject and repeat seed, all models share the same split seed and training seed. This is intended as a limited repeatability sanity check, not a full multi-seed study.
- The auxiliary BNCI2014-004 run is intentionally kept outside the main tables because it is a binary-MI sanity check with a different channel set and protocol scale; it is included only to test whether the main boundary claim qualitatively survives a second dataset. In that auxiliary run, Shallow ConvNet remains first, while CfC-style and LSTM become nearly tied.
- The revision code now includes `MI-Mamba-style`, `SpatialSpectral-Head`, and `SpatialSpectral-CfC` baselines. `MI-Mamba-style` is a shared-protocol selective-SSM surrogate implemented in PyTorch for this repository; it should not be described as a byte-for-byte reproduction of the original MI-Mamba implementation.
- The LOSO script holds out one subject for testing and uses a different training subject for validation, so no held-out-subject trials are used for normalization, validation, or early stopping.
- The `tau` topography script reports channel-wise sensitivity of hidden-state time constants by occluding standardized input channels. These values are not electrode-specific learned `tau` parameters.
- The `--cfc-dt` and `--cfc-tau-init` options are available in the pooled and session-wise runners and are swept by `run_cfc_dt_tau_ablation.py`.
- The supplementary `GRU` control uses the same hidden size, recurrent depth, dropout, and mean-max pooling readout as the LSTM baseline.
- `outputs/paper_ready/sessionwise_stats.csv` and `outputs/paper_ready/pooled_stats.csv` apply Holm correction within the full benchmark comparison family for that protocol.
- `outputs/paper_ready/recurrent_control_stats.json` applies Holm correction only within the recurrent-only control family `{cfc_vs_gru, gru_vs_lstm, cfc_vs_lstm}`, so shared comparisons such as `cfc_vs_lstm` can differ from the full-benchmark CSVs.
- The perturbation sweep evaluates `SNR = 20, 10, 5, 0 dB` and channel dropout fractions `0.1, 0.3, 0.5`, with `5` random seeds per condition.
- The temporal-shuffle control keeps training unchanged and randomly permutes the within-trial time index at test time, using the same permutation across all channels within a trial. It is included as a supplementary diagnostic rather than a new main protocol.
- `Riemann-TSLR` uses `Covariances(estimator="oas") -> TangentSpace(metric="riemann") -> StandardScaler(train-only) -> LogisticRegression`, with `C \in {0.1, 1, 10}` selected on the same validation split used by the neural models.
- `tau` correlations in the paper are based on subject-class summaries (`9 subjects x 4 classes = 36` points), not per-time-step pooled samples.
- `outputs/bspc_sessionwise/tau_time_window_summary.csv` is generated from the session-wise `tau` timecourse and summarizes subject-level coarse early/mid/late/peak windows used for the manuscript's descriptive timing sentence.
- In `outputs/paper_ready/key_stats.json`, the manuscript-facing timing summary is stored under `tau_analysis.subject_level_window_summary`; any `global_mean_timecourse_summary` entry is a separate class-averaged timecourse view and should not be compared directly to the manuscript's subject-level timing sentence.
