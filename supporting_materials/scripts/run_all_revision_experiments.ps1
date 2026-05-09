param(
    [string]$Python = "D:\conda\envs\lnn-mi-eeg\python.exe",
    [string]$Device = "cuda",
    [switch]$SkipLongRuns
)

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"

Set-Location (Resolve-Path "$PSScriptRoot\..")

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Args
    )
    Write-Host ""
    Write-Host "==> $Name"
    & $Python @Args
}

Invoke-Step "Environment check" @("scripts/check_environment.py")

if (-not $SkipLongRuns) {
    Invoke-Step "MI-Mamba pooled benchmark" @(
        "scripts/run_mi_experiments.py",
        "--models", "shallow_convnet", "riemann_tslr", "eegnet", "mi_mamba", "hybrid_cfc", "tiny_transformer", "cfc", "lstm",
        "--device", $Device,
        "--output-dir", "outputs/revision_mamba_pooled"
    )

    Invoke-Step "SpatialSpectral pooled controls" @(
        "scripts/run_mi_experiments.py",
        "--models", "ss_head", "ss_cfc",
        "--device", $Device,
        "--output-dir", "outputs/revision_spatialspectral_pooled"
    )

    Invoke-Step "MI-Mamba and SpatialSpectral grouped controls" @(
        "scripts/run_grouped_pooled_control.py",
        "--models", "mi_mamba", "ss_head", "ss_cfc",
        "--device", $Device,
        "--output-dir", "outputs/revision_mamba_hybrid_grouped"
    )

    Invoke-Step "Full session-wise MI-Mamba and SpatialSpectral controls" @(
        "scripts/run_sessionwise_mi_comparison.py",
        "--models", "mi_mamba", "ss_cfc", "ss_head",
        "--device", $Device,
        "--output-dir", "outputs/revision_mamba_hybrid_sessionwise"
    )

    Invoke-Step "LOSO cross-subject benchmark" @(
        "scripts/run_loso_cross_subject.py",
        "--models", "shallow_convnet", "riemann_tslr", "eegnet", "mi_mamba", "tiny_transformer", "cfc", "lstm",
        "--device", $Device,
        "--output-dir", "outputs/revision_loso"
    )

    Invoke-Step "Riemann LOSO alignment diagnostic" @(
        "scripts/run_loso_riemann_alignment_check.py",
        "--output-dir", "outputs/revision_loso_riemann_alignment"
    )

    Invoke-Step "CfC dt/tau ablation" @(
        "scripts/run_cfc_dt_tau_ablation.py",
        "--models", "cfc", "hybrid_cfc", "ss_cfc",
        "--dt-values", "0.5", "1.0", "2.0",
        "--tau-init-values", "0.5", "1.0", "2.0",
        "--device", $Device,
        "--output-dir", "outputs/revision_cfc_dt_tau_ablation"
    )

    Invoke-Step "Tau topography" @(
        "scripts/run_tau_topography.py",
        "--device", $Device,
        "--output-dir", "outputs/revision_tau_topography"
    )

    Invoke-Step "Efficiency benchmark" @("scripts/benchmark_model_efficiency.py")
}

Invoke-Step "dt/tau heatmaps" @(
    "scripts/plot_dt_tau_ablation_heatmap.py",
    "--summary", "outputs/revision_cfc_dt_tau_ablation/ablation_summary.csv",
    "--output-dir", "outputs/revision_cfc_dt_tau_ablation"
)

Invoke-Step "Export reproducibility artifacts" @("scripts/export_reproducibility_artifacts.py")
