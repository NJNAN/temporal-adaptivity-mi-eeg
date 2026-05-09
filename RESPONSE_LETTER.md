# Response to Reviewers

We thank the reviewers for the constructive comments. We revised both the experimental package and the manuscript to narrow the claim, add missing controls, and strengthen reproducibility. Below we summarize the response structure and the evidence now available in the repository.

## Comment 1: Scope beyond standard four-class IV-2a

**Reviewer comment.** The study is confined to a standard four-class MI paradigm, so the limits of temporal adaptivity may not hold in more complex, multi-class, or asynchronous MI.

**Response.** We agree and have narrowed the manuscript claim. The revised abstract, introduction, and conclusion now state that the conclusion applies to standard cue-locked four-class MI decoding under the tested preprocessing and validation protocols. We explicitly do not claim that temporal adaptivity is unimportant for asynchronous MI, continuous control, event-onset detection, or paradigms where timing is part of the target.

**Manuscript changes.** The limitation paragraph now discusses asynchronous MI, continuous control, onset timing, and state transitions as settings where temporal adaptivity may be more central.

## Comment 2: Missing Mamba / ODE / CDE comparisons

**Reviewer comment.** The manuscript cites Mamba-style models but did not compare against them.

**Response.** We added a shared-protocol `MI-Mamba-style` selective-SSM baseline. It is implemented as a PyTorch surrogate under exactly the same preprocessing, training-only normalization, split logic, early stopping, and statistics as the other models; it is not described as a byte-for-byte reproduction of the original MI-Mamba code.

**Results.**

- Pooled: MI-Mamba-style reaches 59.0% accuracy and 0.575 macro F1. It improves the temporal-model tier but remains below Shallow ConvNet (68.8%) and Riemann-TSLR (68.3%).
- Session-wise: MI-Mamba-style reaches 48.4% accuracy and 0.447 macro F1, again below Riemann-TSLR (62.8%) and Shallow ConvNet (59.5%).
- Grouped-pooled: MI-Mamba-style reaches 57.0% accuracy and 0.554 macro F1. This leads the revision-only grouped extension but remains below the original grouped leaders, Riemann-TSLR (67.8%) and Shallow ConvNet (66.2%).

**Manuscript changes.** The methods and results now include MI-Mamba-style, and the conclusion was refined to say that a modern SSM improves the temporal tier but does not overturn the spatial/geometric top tier in this setting.

## Comment 3: Cross-subject / LOSO experiments

**Reviewer comment.** All results were within-subject; a cross-subject or leave-one-subject-out setting is needed.

**Response.** We added a leave-one-subject-out (LOSO) experiment. In each fold, one subject is held out for testing, normalization is estimated only from training subjects, and validation is selected from the training subjects.

**Results.** In unaligned LOSO, EEGNet leads at 45.1%, followed by Shallow ConvNet at 41.4%, CfC-style at 39.8%, LSTM at 39.2%, MI-Mamba-style at 38.6%, Tiny-Transformer at 38.0%, and Riemann-TSLR at 34.2%. Because Riemann-TSLR was unexpectedly low, we added an alignment diagnostic: unsupervised Euclidean alignment raises Riemann-TSLR from 34.2% to 47.9% (paired p = 0.001). We therefore interpret LOSO as a subject-shift and alignment stress test rather than evidence for a temporal-model advantage.

## Comment 4: Tau topography

**Reviewer comment.** The manuscript should visualize whether learned tau adapts differently over motor cortices versus peripheral channels.

**Response.** We added channel-wise tau sensitivity analysis. Because the CfC-style time constants are hidden-state variables rather than electrode-specific learned parameters, the topography is framed as channel-wise sensitivity of hidden-state tau, not as learned electrode tau.

**Results.** The exported topography artifacts are generated under `outputs/paper_ready/` and mirrored in the tracked supporting package:

- `supporting_materials/paper_tables/revision_tau_occlusion_channel_summary.csv`
- `supporting_materials/tau_analysis/revision_tau_occlusion_channel_subject.csv`
- `supporting_materials/tau_analysis/revision_tau_occlusion_topomap_global.pdf`

The manuscript now states that motor-adjacent channels can influence tau, but this does not become stable class discrimination.

## Comment 5: Hybrid-CfC diagnostic strength

**Reviewer comment.** The minimal Hybrid-CfC performs poorly; either justify it as diagnostic or implement a stronger hybrid.

**Response.** We did both. We clarified that Hybrid-CfC is diagnostic only and should not be read as evidence against all spatial-continuous hybrids. We also added two stronger controls:

- `SpatialSpectral-Head`, a spatial-spectral frontend without CfC recurrence.
- `SpatialSpectral-CfC`, the same frontend with a CfC backend.

**Results.**

- Session-wise: SpatialSpectral-Head reaches 44.6%; SpatialSpectral-CfC reaches 38.3%.
- Pooled: SpatialSpectral-Head reaches 47.5%; SpatialSpectral-CfC reaches 42.1%.
- Grouped-pooled: SpatialSpectral-Head reaches 47.1%; SpatialSpectral-CfC reaches 40.4%. SpatialSpectral-Head exceeds SpatialSpectral-CfC after Holm correction (p = 0.025), again showing no synergy from the current CfC backend.

These controls test whether adding the CfC backend to a stronger spatial-spectral frontend provides consistent synergy.

## Comment 6: Delta t and tau initialization

**Reviewer comment.** The default Delta t = 1.0 and tau initialization may be arbitrary.

**Response.** We completed a 3 x 3 x 3 ablation over `Delta t in {0.5, 1.0, 2.0}`, `tau_init in {0.5, 1.0, 2.0}`, and models `{CfC-style, Hybrid-CfC-style, SpatialSpectral-CfC}`.

**Results.**

- Best pure CfC-style: 45.6% at `Delta t = 2.0`, `tau_init = 0.5`.
- Best Hybrid-CfC-style: 52.2% at `Delta t = 0.5`, `tau_init = 2.0`.
- Best SpatialSpectral-CfC: 39.1% at `Delta t = 1.0`, `tau_init = 1.0`.

The ablation does not rescue pure CfC-style performance. We also added heatmap figures, generated under `outputs/revision_cfc_dt_tau_ablation/` and mirrored in the tracked supporting package:

- `supporting_materials/tau_analysis/revision_cfc_dt_tau_accuracy_heatmap.pdf`
- `supporting_materials/tau_analysis/revision_cfc_dt_tau_f1_heatmap.pdf`

## Comment 7: Reproducibility

**Reviewer comment.** Reproducibility is a major concern for a methods-focused letter.

**Response.** We added and expanded:

- `scripts/check_environment.py`
- `scripts/run_all_revision_experiments.ps1`
- `outputs/paper_ready/artifact_manifest.csv`, mirrored as `supporting_materials/reproducibility/artifact_manifest.csv`
- `REPRODUCIBILITY.md`
- `supporting_materials/reproducibility/REPRODUCIBILITY.md`
- submission-facing copies under `supporting_materials/`

The manuscript file was also renamed to the canonical `lnn_mi_eeg_paper.tex`.

## Comment 8: arXiv preprint publication status

**Reviewer comment.** Some arXiv preprints should be checked for final publication status.

**Response.** We checked the main cited preprints where publication status affected the manuscript framing. MI-Mamba is treated as a published article. Gui et al.'s `EEGMamba: Bidirectional State Space Model with Mixture of Experts for EEG Multi-task Classification` remains an arXiv/OpenReview submission, so the manuscript now cites the formally published Neural Networks 2025 EEGMamba foundation-model article where an EEGMamba reference is needed. The MOABB benchmark citation was also corrected to Chevallier et al. and retained as the official HAL/arXiv working-paper citation used by the MOABB documentation.

## Current Revision Checklist

- [x] MI-Mamba pooled experiment
- [x] MI-Mamba grouped-pooled experiment
- [x] MI-Mamba session-wise experiment
- [x] LOSO experiment
- [x] Riemann LOSO alignment diagnostic
- [x] Tau channel-wise sensitivity/topography
- [x] Completed Delta t / tau initialization ablation
- [x] Delta t / tau heatmap figures
- [x] SpatialSpectral-CfC and SpatialSpectral-Head pooled experiment
- [x] SpatialSpectral-CfC and SpatialSpectral-Head grouped-pooled experiment
- [x] MI-Mamba efficiency benchmark
- [x] One-command revision runner
- [x] Canonical manuscript filename
