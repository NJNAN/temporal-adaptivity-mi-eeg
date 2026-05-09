"""效率快照脚本。

对应论文：
1. Results/Discussion 中关于不同模型训练与推理代价的定量说明。
2. Supporting materials 中的 `benchmark.csv` 与 `benchmark.json`。

本脚本不参与主精度结论，而是给出当前 PyTorch eager-mode 实现下的
训练步耗时、单次前向耗时和显存峰值，用来支撑论文里对
“CfC-style 当前实现存在明显运行开销”的讨论。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch


def load_core_module(repo_root: Path):
    """加载主实验脚本，复用论文正文所有模型定义与工具函数。"""
    script_path = repo_root / "scripts" / "run_mi_experiments.py"
    spec = importlib.util.spec_from_file_location("mi_exp_core", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sync_if_needed(device: torch.device) -> None:
    """在 CUDA 上显式同步，保证计时对应论文中的真实 wall-clock 开销。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_torch_model(
    module,
    model_name: str,
    device: torch.device,
    batch_size: int,
    n_channels: int,
    n_samples: int,
) -> dict:
    """测量神经网络模型的前向与训练步耗时。

    对应论文的 efficiency snapshot：
    - `batch_size=64` 近似吞吐场景。
    - `batch_size=1` 近似单 trial 延迟。
    """
    module.seed_everything(20260321 + abs(hash((model_name, batch_size))) % 1000)
    model = module.build_model(
        model_name=model_name,
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=len(module.LABEL_ORDER),
        cfc_hidden_size=128,
        lstm_hidden_size=128,
    ).to(device)
    model.train()

    x = torch.randn(batch_size, n_channels, n_samples, device=device)
    y = torch.randint(0, len(module.LABEL_ORDER), (batch_size,), device=device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    warmup_steps = 10
    eval_steps = 60 if batch_size == 1 else 40

    for _ in range(warmup_steps):
        # 预热阶段不写入结果，只用于稳定 cuDNN 与缓存行为。
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    sync_if_needed(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(eval_steps):
            _ = model(x)
    sync_if_needed(device)
    forward_ms = (time.perf_counter() - start) * 1000.0 / eval_steps

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sync_if_needed(device)
    start = time.perf_counter()
    for _ in range(eval_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    sync_if_needed(device)
    train_step_ms = (time.perf_counter() - start) * 1000.0 / eval_steps

    peak_memory_mb = float("nan")
    if device.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    return {
        "model": model_name,
        "model_display": module.get_model_display_name(model_name),
        "device": str(device),
        "batch_size": batch_size,
        "params": module.count_parameters(model),
        "forward_ms": forward_ms,
        "train_step_ms": train_step_ms,
        "peak_memory_mb": peak_memory_mb,
    }


def benchmark_riemann_cpu(module, x_train, y_train, x_val, y_val, x_test) -> dict:
    """测量 Riemann-TSLR 的 CPU 推理耗时。

    对应论文中“经典几何基线只单列 CPU latency，不与 GPU 显存直接硬比较”的说明。
    """
    fit_info = module.fit_riemann_tslr(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    model = fit_info["model"]
    repeats = 200
    start = time.perf_counter()
    for _ in range(repeats):
        _ = model.predict(x_test[:1])
    forward_ms = (time.perf_counter() - start) * 1000.0 / repeats
    return {
        "model": "riemann_tslr",
        "model_display": module.get_model_display_name("riemann_tslr"),
        "device": "cpu",
        "batch_size": 1,
        "params": module.riemann_parameter_count(x_train.shape[1], len(module.LABEL_ORDER)),
        "forward_ms": forward_ms,
        "train_step_ms": float("nan"),
        "peak_memory_mb": float("nan"),
        "best_c": float(fit_info["best_c"]),
    }


def main() -> None:
    """生成论文效率表与补充材料所需的原始 benchmark 文件。"""
    repo_root = Path(__file__).resolve().parents[1]
    module = load_core_module(repo_root)
    device = module.get_device("cuda")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the neural-model benchmark.")

    cache_dir = repo_root / "outputs" / "bspc_sessionwise" / "cache"
    X_raw, y, sessions = load_session_subject_data(repo_root, module, cache_dir)
    X_raw = module.downsample_trials(X_raw, 2)
    train_mask = sessions == "0train"
    x_train_full = X_raw[train_mask]
    y_train_full = y[train_mask]
    x_test = X_raw[~train_mask]
    mean, std = module.compute_standardizer(x_train_full[: int(0.85 * len(x_train_full))])
    x_train = module.apply_standardizer(x_train_full[: int(0.85 * len(x_train_full))], mean, std)
    y_train = y_train_full[: int(0.85 * len(y_train_full))]
    x_val = module.apply_standardizer(x_train_full[int(0.85 * len(x_train_full)) :], mean, std)
    y_val = y_train_full[int(0.85 * len(y_train_full)) :]
    x_test_std = module.apply_standardizer(x_test, mean, std)

    n_channels = x_train.shape[1]
    n_samples = x_train.shape[2]

    models = ["shallow_convnet", "eegnet", "mi_mamba", "tiny_transformer", "hybrid_cfc", "cfc", "lstm"]
    rows = []
    for batch_size in (64, 1):
        for model_name in models:
            rows.append(
                benchmark_torch_model(
                    module,
                    model_name,
                    device=device,
                    batch_size=batch_size,
                    n_channels=n_channels,
                    n_samples=n_samples,
                )
            )
    rows.append(benchmark_riemann_cpu(module, x_train, y_train, x_val, y_val, x_test_std))

    output_dir = repo_root / "outputs" / "bspc_efficiency"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "benchmark.csv", index=False)
    (output_dir / "benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


def load_session_subject_data(repo_root: Path, module, cache_dir: Path):
    """复用 session-wise 脚本的数据载入逻辑，保证效率测试与论文主协议一致。"""
    sessionwise_path = repo_root / "scripts" / "run_sessionwise_mi_comparison.py"
    spec = importlib.util.spec_from_file_location("mi_exp_sessionwise_benchmark", sessionwise_path)
    sessionwise = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sessionwise
    spec.loader.exec_module(sessionwise)
    return sessionwise.load_subject_session_data(module, 1, repo_root / "data", cache_dir)


if __name__ == "__main__":
    main()
