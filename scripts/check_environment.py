"""Check the local runtime used to reproduce the MI-EEG experiments.

The script is intentionally lightweight: it does not download data or run a
training job. It records Python/package versions, CUDA visibility, GPU details,
and the presence of expected result directories.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "torch",
    "mne",
    "moabb",
    "pyriemann",
    "matplotlib",
    "seaborn",
]

EXPECTED_OUTPUTS = [
    "outputs/bspc_pooled",
    "outputs/bspc_sessionwise",
    "outputs/bspc_grouped_cv",
    "outputs/revision_mamba_pooled",
    "outputs/revision_loso",
    "outputs/revision_tau_topography",
    "outputs/revision_cfc_dt_tau_ablation",
    "outputs/paper_ready",
    "supporting_materials",
]


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return f"UNAVAILABLE: {exc}"


def torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {"installed": False}
    try:
        import torch
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        info["import_error"] = str(exc)
        return info

    info.update(
        {
            "installed": True,
            "version": torch.__version__,
            "cuda_compiled": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "cudnn_version": torch.backends.cudnn.version(),
        }
    )
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_mib": int(props.total_memory // (1024 * 1024)),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    info["devices"] = devices
    return info


def build_report(repo_root: Path) -> dict[str, Any]:
    return {
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
            "platform": platform.platform(),
        },
        "git_commit": git_commit(repo_root),
        "packages": {name: package_version(name) for name in REQUIRED_PACKAGES},
        "torch": torch_info(),
        "paths": {
            relative_path: (repo_root / relative_path).exists()
            for relative_path in EXPECTED_OUTPUTS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MI-EEG experiment environment.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    report = build_report(repo_root)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        output_path = repo_root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
