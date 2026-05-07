# Supporting Materials

This folder collects the evidence package for the current MI-EEG letter draft.

## Layout

- `paper_tables/`
  - Final paper-ready tables and key statistics, including the grouped pooled control, recurrent GRU control, tau locality/window controls, perturbation sweep, temporal-shuffle control, session-wise seed-variability summary, and the BNCI2014-004 auxiliary sanity check
- `subject_results/`
  - Per-subject scores, pooled/session-wise/grouped split assignments, per-class F1 summaries, confusion matrices, and boxplots
- `tau_analysis/`
  - `tau` histograms, subject-class summaries, motor-channel locality controls, time-resolved trajectories, and windowed temporal summaries
- `robustness/`
  - Structured perturbation results (band-limited noise, channel dropout, and temporal-shuffle control) with multi-seed averaging, corrected statistics, and sweep curves
- `efficiency/`
  - Practical GPU timing, approximate per-trial latency, and CPU `Riemann-TSLR` latency
- `reproducibility/`
  - Seeds, repeat-seed summaries, and reproducibility notes
- `scripts/`
  - Core scripts used to generate the results in this package, including the BNCI2014-004 auxiliary runner
- `manuscript/`
  - Current LaTeX draft and bibliography

## Most important files

- `paper_tables/main_table.csv`
- `paper_tables/sessionwise_table.csv`
- `paper_tables/grouped_cv_table.csv`
- `paper_tables/recurrent_control_table.csv`
- `paper_tables/sessionwise_seed_model_summary.csv`
- `paper_tables/sessionwise_seed_variability_summary.csv`
- `paper_tables/perturbation_sweep_summary.csv`
- `paper_tables/temporal_shuffle_summary.csv`
- `paper_tables/tau_local_window_stats.json`
- `paper_tables/bnci2014_004_aux_summary.csv`
- `paper_tables/bnci2014_004_aux_stats.csv`
- `paper_tables/structured_perturbation_table.csv`
- `paper_tables/pooled_stats.csv`
- `paper_tables/sessionwise_stats.csv`
- `paper_tables/grouped_cv_stats.csv`
- `paper_tables/perturbation_sweep_stats.csv`
- `paper_tables/temporal_shuffle_stats.csv`
- `paper_tables/key_stats.json`
- `subject_results/pooled_subject_scores.csv`
- `subject_results/sessionwise_subject_scores.csv`
- `subject_results/grouped_subject_scores.csv`
- `subject_results/gru_pooled_subject_scores.csv`
- `subject_results/gru_sessionwise_subject_scores.csv`
- `subject_results/bnci2014_004_aux_metrics.csv`
- `subject_results/temporal_shuffle_subject_summary.csv`
- `subject_results/pooled_per_class_f1_summary.csv`
- `subject_results/sessionwise_per_class_f1_summary.csv`
- `subject_results/grouped_per_class_f1_summary.csv`
- `subject_results/pooled_confusion_matrices.pdf`
- `subject_results/sessionwise_confusion_matrices.pdf`
- `subject_results/grouped_confusion_matrices.pdf`
- `tau_analysis/tau_stats.json`
- `tau_analysis/tau_local_window_stats.json`
- `tau_analysis/tau_motor_subject_class_summary.csv`
- `tau_analysis/global_tau_window_subject_class_summary.csv`
- `tau_analysis/motor_tau_window_subject_class_summary.csv`
- `tau_analysis/tau_timecourse_by_class.pdf`
- `tau_analysis/tau_time_window_summary.csv`
- `robustness/band_noise_accuracy_sweep.pdf`
- `robustness/channel_dropout_accuracy_sweep.pdf`
- `robustness/temporal_shuffle_drop.pdf`
- `efficiency/benchmark.csv`
- `reproducibility/seed_config.json`
- `reproducibility/seed_variability_summary.json`
- `reproducibility/seed_rankings.csv`
- `reproducibility/bnci2014_004_results_summary.json`
- `manuscript/lnn_mi_eeg_paper.tex`

## Notes

- `manuscript/lnn_mi_eeg_paper.tex` is copied from the current working draft `lnn_mi_eeg_paper (2).tex`.
- The pooled CV uses trial-level stratified folds after pooling both sessions.
- The grouped pooled control uses `session+run` identifiers in the outer split and now covers all seven main benchmark models to quantify how much the pooled ranking depends on trial shuffling across runs.
- The session-wise protocol trains on session 1, validates on a split from session 1, and tests on session 2.
- The supplementary seed-variability check reruns the clean session-wise benchmark with repeat seeds `42` and `43`, while keeping split and training seeds identical across models within each subject.
- The supplementary GRU control uses the same hidden size, recurrent depth, dropout, and mean-max pooling readout as the LSTM baseline.
- The supplementary tau-locality control reruns the session-wise CfC-style analysis with motor-related channels only (`C3`, `C4`, `CP3`, `CP4`) and with explicit early/mid/late temporal windows to test whether the negative tau result is an artifact of global averaging.
- The auxiliary BNCI2014-004 sanity check is a supporting-only binary-MI result and is not used as a third main benchmark table because its label space and channel count differ from BCI IV-2a; in that auxiliary run, Shallow ConvNet remains first while the CfC-style vs. LSTM gap largely disappears.
- The perturbation sweep evaluates `SNR = 20, 10, 5, 0 dB` and channel-dropout fractions `0.1, 0.3, 0.5` with five random seeds per subject-model condition.
- The temporal-shuffle control keeps training fixed and randomizes within-trial time order only at test time, using one permutation per trial shared across all channels. It is included as a supplementary diagnostic to test whether preserving temporal order creates any distinct CfC-style advantage.
- `CfC-style` denotes the implemented exponential-decay variant studied in this repository; it is intentionally distinguished from every canonical smoothed-gate CfC instantiation.
