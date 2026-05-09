param(
    [string]$Python = "D:\conda\envs\lnn-mi-eeg\python.exe",
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Args
    )
    Write-Host "===== $Name ====="
    & $Python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Invoke-Step "1 full-model aligned LOSO" @(
    "scripts/run_loso_alignment_all_models.py",
    "--models", "shallow_convnet", "riemann_tslr", "eegnet", "mi_mamba", "tiny_transformer", "cfc", "lstm",
    "--variants", "standard", "euclidean_alignment",
    "--device", $Device,
    "--output-dir", "outputs/revision_loso_alignment_all"
)

Invoke-Step "2 tuned Shallow/EEGNet sanity check" @(
    "scripts/run_sessionwise_mi_comparison.py",
    "--models", "shallow_convnet", "eegnet",
    "--epochs", "160",
    "--patience", "40",
    "--min-epochs", "40",
    "--device", $Device,
    "--output-dir", "outputs/revision_tuned_shallow_eegnet"
)

Invoke-Step "3-4 MI-Mamba/readout sensitivity" @(
    "scripts/run_readout_mamba_sensitivity.py",
    "--variants", "mi_mamba_d8_meanmax", "mi_mamba_d16_meanmax", "mi_mamba_d32_meanmax", "mi_mamba_d16_attention", "mi_mamba_d16_last", "tiny_transformer_attention", "cfc_final",
    "--device", $Device,
    "--output-dir", "outputs/revision_readout_mamba_sensitivity"
)

Invoke-Step "5 band sensitivity" @(
    "scripts/run_band_sensitivity_sessionwise.py",
    "--models", "shallow_convnet", "riemann_tslr", "eegnet", "mi_mamba", "tiny_transformer", "cfc", "lstm",
    "--bands", "mu_beta_8_30", "broad_4_40", "broad_1_45",
    "--device", $Device,
    "--output-dir", "outputs/revision_band_sensitivity"
)

Invoke-Step "6 full model seed 42" @(
    "scripts/run_sessionwise_mi_comparison.py",
    "--models", "shallow_convnet", "riemann_tslr", "eegnet", "mi_mamba", "tiny_transformer", "cfc", "lstm", "ss_head", "ss_cfc",
    "--seed", "42",
    "--device", $Device,
    "--output-dir", "outputs/revision_sessionwise_seed42_full"
)

Invoke-Step "6 full model seed 43" @(
    "scripts/run_sessionwise_mi_comparison.py",
    "--models", "shallow_convnet", "riemann_tslr", "eegnet", "mi_mamba", "tiny_transformer", "cfc", "lstm", "ss_head", "ss_cfc",
    "--seed", "43",
    "--device", $Device,
    "--output-dir", "outputs/revision_sessionwise_seed43_full"
)

Invoke-Step "export artifacts" @(
    "scripts/export_reproducibility_artifacts.py"
)
