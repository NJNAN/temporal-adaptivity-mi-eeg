# On the Limits of Temporal Adaptivity in MI-EEG Decoding

This repository contains the manuscript code and supporting evidence for a motor-imagery EEG decoding study. The project tests whether stronger temporal modeling, especially a CfC-style continuous-time recurrent unit, is enough to improve standard cue-locked MI-EEG classification.

The current evidence compares CfC-style recurrence with LSTM, GRU controls, Tiny-Transformer, EEGNet, Shallow ConvNet, and a compact Riemannian tangent-space logistic-regression baseline under shared preprocessing and controlled evaluation protocols.

## Main Claim

In the current BCI Competition IV-2a setting, temporal adaptivity alone does not appear to set the decoding performance ceiling. The strongest results are driven more consistently by spatial-spectral and geometric inductive biases, especially Shallow ConvNet and Riemann-TSLR.

This is a scoped claim for standard cue-locked four-class MI decoding. It is not a universal claim about asynchronous MI, continuous control, or every possible temporal/SSM/hybrid architecture.

## Repository Layout

```text
.
├── scripts/                         # Experiment, analysis, export, and plotting scripts
├── supporting_materials/            # Curated small evidence package for the paper
│   ├── manuscript/                  # LaTeX draft and bibliography copy
│   ├── paper_tables/                # Paper-ready CSV/JSON tables
│   ├── subject_results/             # Subject-level score tables and confusion matrices
│   ├── tau_analysis/                # Tau statistics and figures
│   ├── robustness/                  # Perturbation and temporal-shuffle results
│   ├── efficiency/                  # Runtime benchmark snapshot
│   └── reproducibility/             # Seeds, split notes, and exported configs
├── lnn_mi_eeg_paper (2).tex         # Current manuscript draft
├── references.bib                   # Bibliography
├── REPRODUCIBILITY.md               # Detailed reproducibility notes
├── CODE_TO_PAPER_MAPPING.md         # Mapping from claims/tables to scripts and outputs
└── REVIEWER_RESPONSE_PLAN.md        # Planned response to reviewer requests
```

Large raw datasets and full intermediate experiment outputs are intentionally not tracked. They are regenerated or downloaded locally.

## Data

The experiments use public MOABB-accessible datasets:

- BCI Competition IV Dataset 2a / BNCI2014-001
- BNCI2014-004 as a supporting binary-MI sanity check

The local MOABB cache is ignored by Git through `.gitignore`. On first run, MOABB may download the data into `data/MNE-bnci-data/...` depending on your local configuration.

## Environment

Recommended setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn torch moabb pyriemann reportlab
```

CUDA is optional but strongly recommended for the neural-model experiments. The scripts accept `--device cpu`, but full runs will be slow on CPU.

## Quick Smoke Test

Run a short single-subject validation pass:

```powershell
python scripts/run_mi_experiments.py --smoke-test --device cuda
```

If CUDA is unavailable:

```powershell
python scripts/run_mi_experiments.py --smoke-test --device cpu
```

## Main Reproduction Commands

Pooled within-subject 5-fold benchmark:

```powershell
python scripts/run_mi_experiments.py --device cuda --output-dir outputs/bspc_pooled
```

Session-wise benchmark:

```powershell
python scripts/run_sessionwise_mi_comparison.py --device cuda --output-dir outputs/bspc_sessionwise
```

Grouped pooled control using `session+run` groups:

```powershell
python scripts/run_grouped_pooled_control.py --models shallow_convnet riemann_tslr eegnet hybrid_cfc tiny_transformer cfc lstm --device cuda --output-dir outputs/bspc_grouped_cv
```

Structured perturbation sweep:

```powershell
python scripts/run_structured_perturbation_sweep.py --device cuda --output-dir outputs/bspc_perturbation_sweep
```

Temporal-shuffle control:

```powershell
python scripts/run_temporal_shuffle_control.py --device cuda --output-dir outputs/bspc_temporal_shuffle
```

BNCI2014-004 auxiliary experiment:

```powershell
python scripts/run_bnci2014_004_aux.py --models shallow_convnet riemann_tslr eegnet tiny_transformer cfc lstm --output-dir outputs/bnci2014_004_aux --seed 42
```

Export paper-ready tables and supporting artifacts:

```powershell
python scripts/export_reproducibility_artifacts.py
```

More detail is available in `REPRODUCIBILITY.md`.

## Key Existing Results

The curated result tables are in `supporting_materials/paper_tables/`.

Representative current tables:

- `main_table.csv`: pooled 5-fold results
- `grouped_cv_table.csv`: grouped pooled control
- `sessionwise_table.csv`: train-session to test-session results
- `structured_perturbation_table.csv`: band-limited noise and channel dropout
- `temporal_shuffle_summary.csv`: within-trial temporal-order control
- `bnci2014_004_aux_summary.csv`: auxiliary binary-MI dataset

The manuscript currently reports that Shallow ConvNet and Riemann-TSLR form the strongest tier across the main IV-2a protocols, while CfC-style recurrence improves over LSTM in some settings but does not close the gap to spatial/geometric baselines.

## Reviewer-Driven Revision Plan

`REVIEWER_RESPONSE_PLAN.md` lists the planned additions for a stronger revision:

- Mamba / MI-Mamba-style head-to-head comparison
- leave-one-subject-out or cross-subject evaluation
- tau topography or channel-wise tau attribution
- `Delta t` and `tau` initialization ablation
- stronger spatial-spectral + CfC hybrid control
- stronger open-source reproducibility packaging
- final-publication-status checks for arXiv references

## GitHub Upload Notes

Before pushing, check what will be tracked:

```powershell
git status --short
git check-ignore -v data outputs tmp
```

Do not commit:

- `data/`
- full `outputs/`
- `*.npz`, `*.npy`, model checkpoints
- temporary LaTeX/build/cache files

The intended public repository should contain the scripts, manuscript sources, curated small supporting materials, and documentation.

## Citation

This work is currently a manuscript project. A formal citation entry can be added after the paper has a stable preprint or publication record.
