# 2026-05-08 Revision Experiment Log

本文档记录 2026-05-08 针对审稿意见新增和补跑的实验、当前结果、论文修改含义与后续任务。当前所有结果均来自本机 `D:\作业\lnn论文1` 工作区。

## 1. 实验环境

- Conda 环境：`D:\conda\envs\lnn-mi-eeg`
- Python：3.11
- PyTorch：`2.11.0+cu128`
- CUDA：可用，`torch.cuda.is_available() = True`
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- 关键依赖：`moabb`, `mne`, `pyriemann`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`
- 运行隔离：实验命令使用 `PYTHONNOUSERSITE=1`，避免用户级 Python 包污染 conda 环境

常用环境命令：

```powershell
cd "D:\作业\lnn论文1"
$env:PYTHONNOUSERSITE='1'
$py = "D:\conda\envs\lnn-mi-eeg\python.exe"
```

## 2. 已完成实验

### 2.1 MI-Mamba-style pooled head-to-head

目的：回应审稿意见 2，补充 Mamba / MI-Mamba 类模型对比。

输出目录：

```text
outputs/revision_mamba_pooled/
```

命令：

```powershell
& $py scripts/run_mi_experiments.py --models shallow_convnet riemann_tslr eegnet mi_mamba tiny_transformer cfc lstm --device cuda --output-dir outputs/revision_mamba_pooled
& $py scripts/run_mi_experiments.py --models hybrid_cfc --device cuda --output-dir outputs/revision_mamba_pooled
```

最终 pooled 5-fold 结果：

| Model | Accuracy mean | Accuracy std | Macro-F1 mean | Macro-F1 std |
|---|---:|---:|---:|---:|
| Shallow ConvNet | 68.75 | 16.90 | 0.686 | 0.170 |
| Riemann-TSLR | 68.27 | 14.79 | 0.681 | 0.149 |
| MI-Mamba-style | 59.04 | 18.99 | 0.575 | 0.195 |
| EEGNet | 55.82 | 19.93 | 0.539 | 0.205 |
| Hybrid-CfC-style | 52.93 | 18.54 | 0.518 | 0.193 |
| Tiny-Transformer | 51.89 | 15.11 | 0.504 | 0.160 |
| CfC-style | 49.40 | 16.66 | 0.474 | 0.178 |
| LSTM | 42.40 | 13.42 | 0.396 | 0.151 |

关键解读：

- `MI-Mamba-style` 明显强于 `CfC-style`, `LSTM`, `Tiny-Transformer`，说明审稿人指出的 selective SSM / Mamba 类模型确实值得补充。
- 但 `MI-Mamba-style` 仍低于 `Shallow ConvNet` 和 `Riemann-TSLR` 约 9 个百分点。
- `Hybrid-CfC-style` 比纯 `CfC-style` 高约 3.5 个百分点，说明轻量空间前端有帮助，但仍不能接近空间-谱/几何 top tier。

利好程度：高。

论文修改含义：

- 主表应加入 `MI-Mamba-style`。
- Methods 中需要说明 `MI-Mamba-style` 是 shared-protocol surrogate，不是原论文官方完整复现。
- 结论应从“temporal models cannot compete”改为“temporal adaptivity alone does not set the performance ceiling; modern SSM hybrids improve but do not surpass compact spatial/geometric baselines under this protocol”。

### 2.2 LOSO cross-subject evaluation

目的：回应审稿意见 3，补充 leave-one-subject-out 跨被试评估。

输出目录：

```text
outputs/revision_loso/
```

命令：

```powershell
& $py scripts/run_loso_cross_subject.py --models shallow_convnet riemann_tslr eegnet mi_mamba tiny_transformer cfc lstm --device cuda --output-dir outputs/revision_loso
```

完整性：

- `loso_metrics.csv`：63 行
- 9 个 held-out/test subject，每个 subject 7 个模型
- `loso_results_summary.json` 已生成
- `loso_subject_summary.csv` 已生成
- `loso_stats.csv` 已生成

最终 LOSO 结果：

| Model | Accuracy mean | Accuracy std | Macro-F1 mean | Macro-F1 std |
|---|---:|---:|---:|---:|
| EEGNet | 45.08 | 15.66 | 0.413 | 0.186 |
| Shallow ConvNet | 41.38 | 13.82 | 0.381 | 0.168 |
| CfC-style | 39.80 | 14.81 | 0.332 | 0.180 |
| LSTM | 39.16 | 12.34 | 0.335 | 0.152 |
| MI-Mamba-style | 38.62 | 10.93 | 0.348 | 0.119 |
| Tiny-Transformer | 38.02 | 8.99 | 0.338 | 0.110 |
| Riemann-TSLR | 34.16 | 6.36 | 0.278 | 0.075 |

关键解读：

- LOSO 明显比 within-subject pooled 更难，所有模型都下降。
- 模型差异在 cross-subject 下被压缩，Holm correction 后 pairwise 差异不稳定。
- temporal models 没有形成稳定优势；`CfC-style`, `LSTM`, `MI-Mamba-style`, `Tiny-Transformer` 都集中在 38-40% 左右。
- `Riemann-TSLR` 在 LOSO 中明显下降，说明 covariance geometry 在跨被试未对齐条件下不一定稳健。

利好程度：中高。

论文修改含义：

- 不能写“Shallow/Riemann 在所有协议下都第一”。
- 应写：within-subject 中 spatial-spectral/geometric inductive bias 明显主导；LOSO 中 subject shift 成为额外瓶颈，模型差异压缩，temporal adaptivity 仍没有稳定独立优势。
- 建议新增 `Cross-subject Evaluation` 小节或 supplementary table。

### 2.3 Tau topography / channel-wise tau sensitivity

目的：回应审稿意见 4，可视化 CfC learned tau 是否受 motor cortex 附近通道影响。

输出目录：

```text
outputs/revision_tau_topography/
```

命令：

```powershell
& $py scripts/run_tau_topography.py --device cuda --output-dir outputs/revision_tau_topography
```

生成文件：

```text
tau_occlusion_topomap_global.pdf
tau_occlusion_channel_summary.csv
tau_occlusion_channel_subject.csv
tau_topography_stats.json
```

完整性：

- `tau_occlusion_channel_subject.csv`：198 行
- 9 subjects x 22 channels
- `tau_occlusion_topomap_global.pdf` 已生成

Top channels by tau sensitivity：

| Rank | Channel | Mean sensitivity |
|---:|---|---:|
| 1 | C3 | 0.0289 |
| 2 | CP3 | 0.0260 |
| 3 | CP4 | 0.0255 |
| 4 | FC1 | 0.0213 |
| 5 | POz | 0.0200 |
| 6 | CP1 | 0.0193 |
| 7 | Cz | 0.0187 |
| 8 | CPz | 0.0185 |
| 9 | CP2 | 0.0180 |
| 10 | FC3 | 0.0167 |

关键解读：

- C3, CP3, CP4, Cz, CPz 等 sensorimotor 附近通道排位较高。
- 这说明 hidden-state tau 对运动相关通道有空间敏感性。
- 但这不是“每个电极学习了一个 tau”，而是“遮挡某通道后 hidden-state tau 的变化幅度”。
- 该结果不能推翻前文 tau 不稳定区分类别的分析，只能说明 tau 有合理的 sensorimotor spatial influence。

利好程度：高，但必须谨慎表述。

论文修改含义：

- 增加一张 tau topography 图，优先放 supplement，若篇幅允许可放正文。
- 图注必须写：`channel-wise sensitivity of hidden-state tau, not electrode-specific learned tau parameters`。
- 推荐写法：`Tau is spatially influenced by sensorimotor channels, but this sensitivity does not establish class-discriminative tau structure.`

## 3. 进行中实验

### 3.1 Delta t / tau initialization ablation

目的：回应审稿意见 6，排除 `Delta t=1.0` 和 `tau_init=1.0` 任意设置导致 CfC 表现受限的可能。

输出目录：

```text
outputs/revision_cfc_dt_tau_ablation/
```

命令：

```powershell
& $py scripts/run_cfc_dt_tau_ablation.py --models cfc hybrid_cfc ss_cfc --dt-values 0.5 1.0 2.0 --tau-init-values 0.5 1.0 2.0 --device cuda --output-dir outputs/revision_cfc_dt_tau_ablation
```

当前状态：

- 已完成全部 27 行。
- 覆盖 `dt = 0.5, 1.0, 2.0`、`tau_init = 0.5, 1.0, 2.0`、`model = cfc, hybrid_cfc, ss_cfc`。
- 已同步到 `outputs/paper_ready/revision_cfc_dt_tau_ablation_summary.csv` 和 `supporting_materials/paper_tables/revision_cfc_dt_tau_ablation_summary.csv`。

最佳结果：

| dt | tau_init | Model | Accuracy mean | Accuracy std | Macro-F1 mean | Macro-F1 std |
|---:|---:|---|---:|---:|---:|---:|
| 0.5 | 2.0 | Hybrid-CfC-style | 52.24 | 15.93 | 0.510 | 0.162 |
| 2.0 | 0.5 | CfC-style | 45.60 | 13.11 | 0.419 | 0.156 |
| 1.0 | 1.0 | SpatialSpectral-CfC | 39.12 | 12.54 | 0.360 | 0.142 |

当前解读：

- CfC-style 没有被 `dt/tau_init` 调参救回；最佳 45.60% 仍低于 spatial/geometric baselines。
- Hybrid-CfC-style 在较大 `tau_init` 下有一定提升，说明 memory scale tuning 会影响 hybrid 表现。
- SpatialSpectral-CfC 当前实现没有稳定协同收益。
- 可用于回应：`Delta t=1.0` 或 `tau_init=1.0` 不是 CfC-style 性能受限的唯一解释，调参会改变数值但不改变主要排序。

利好程度：目前中等偏利好，最终结论待完整 sweep。

## 4. 论文应如何修改

### 4.1 摘要和结论需要收缩主张

不建议继续写成：

```text
Temporal models cannot compete with spatial models.
```

建议改成：

```text
In standard cue-locked four-class MI decoding, temporal adaptivity alone does not set the performance ceiling. Spatial-spectral and geometric inductive biases dominate within-subject decoding, while cross-subject evaluation shows that subject shift compresses model differences and no temporal model, including CfC-style recurrence or an MI-Mamba-style surrogate, yields a reliable standalone advantage.
```

### 4.2 Results 需要新增或更新的表

建议至少更新四个结果位置：

1. Pooled main table
   - 加入 `MI-Mamba-style`
   - 更新 `Hybrid-CfC-style`
   - 使用 `outputs/revision_mamba_pooled/results_summary.json`

2. Cross-subject / LOSO table
   - 使用 `outputs/revision_loso/loso_subject_summary.csv`
   - 强调 subject shift 和 model ranking compression

3. Tau topography figure
   - 使用 `outputs/revision_tau_topography/tau_occlusion_topomap_global.pdf`
   - 图注必须说明是 channel-wise sensitivity

4. Delta t / tau initialization ablation
   - 已完成 27 行 sweep
   - 论文中可写成：memory-scale tuning modulates performance but does not overturn the main ranking

### 4.3 Methods 需要补充的模型说明

需要新增或扩写：

- `MI-Mamba-style selective SSM surrogate`
- `LOSO cross-subject protocol`
- `Channel-wise tau occlusion sensitivity`
- `Delta t / tau initialization ablation`
- `SpatialSpectral-CfC` 和 `SpatialSpectral-Head` 的定位

### 4.4 对 Hybrid-CfC 的表述

不要把当前 Hybrid-CfC 写成“证明 hybrid 不行”。

建议写成：

```text
The minimal Hybrid-CfC diagnostic improves over pure CfC but remains below the strongest spatial-spectral/geometric baselines, suggesting that a lightweight spatial frontend helps but does not by itself close the inductive-bias gap.
```

## 5. 对审稿意见的回应映射

| 审稿意见 | 当前状态 | 结果倾向 |
|---|---|---|
| 1. IV-2a 外推边界 | 主要靠改稿收缩主张 | 利好，结论更稳 |
| 2. Mamba / MI-Mamba 缺失 | pooled 和 sessionwise 已完成，grouped 未完成 | 高利好 |
| 3. Cross-subject / LOSO | 已完成 | 中高利好 |
| 4. Tau topography | 已完成 | 高利好，但需谨慎解释 |
| 5. Hybrid-CfC 太弱 | Hybrid-CfC、SS-Head、SS-CfC sessionwise 已补 | 中等利好 |
| 6. Delta t / tau init 任意 | 已完成 27 行 sweep | 中高利好 |
| 7. Reproducibility | README/requirements 已有，仍需 environment check 和 manifest | 需继续补 |
| 8. arXiv 状态 | Mamba 已初步更新，仍需最终核查 | 需继续补 |

## 6. 当前总判断

总体利好程度：8/10。

新增结果没有推翻论文核心结论，反而让结论更细、更能抗审稿：

- `MI-Mamba-style` 提升了 temporal/SSM baseline，但在 pooled 和 session-wise 中仍不超过 Shallow/Riemann。
- LOSO 显示 cross-subject 主要问题是 subject shift，temporal models 没有稳定优势。
- Tau topography 显示 tau 受 sensorimotor channels 影响，但不等于 class-discriminative biomarker。
- Hybrid-CfC 小幅优于 pure CfC，说明空间前端有帮助，但不足以追上强 spatial/geometric priors。

推荐最终主线：

```text
Temporal adaptivity is useful but not sufficient. In standard cue-locked four-class MI decoding, the strongest within-subject gains align with spatial-spectral and geometric inductive biases; cross-subject evaluation further shows that subject shift compresses model differences and no temporal model achieves a reliable standalone advantage.
```

## 7. 下一步任务

1. full session-wise 主实验已重跑完成，当前有效来源为 `outputs/bspc_sessionwise_full_rerun`。
2. 修 `run_loso_cross_subject.py`：
   - 增加 `heldout_subject` 别名列。
   - 增加真正的断点续跑读取逻辑。
3. 更新 `lnn_mi_eeg_paper (2).tex`：
   - pooled 表。
   - LOSO 表。
   - tau topography 图。
   - limitations 和 conclusion。
4. 同步 `supporting_materials/manuscript/lnn_mi_eeg_paper.tex`。
5. 更新 `README.md` 和 `REPRODUCIBILITY.md`。
6. 增加 `scripts/check_environment.py` 和 artifact manifest。
7. 复核 BibTeX 中 arXiv preprint 的最终发表状态。
8. commit 并 push 到 GitHub。

## 8. 2026-05-09 补救后新增 session-wise 结果

有效主结果目录：

```text
outputs/bspc_sessionwise_full_rerun/
outputs/revision_mamba_hybrid_sessionwise/
```

合并 10 模型 session-wise 排序：

| Model | Accuracy mean | Accuracy std | Macro-F1 mean |
|---|---:|---:|---:|
| Riemann-TSLR | 62.81 | 12.73 | 0.617 |
| Shallow ConvNet | 59.45 | 15.43 | 0.585 |
| MI-Mamba-style | 48.38 | 17.96 | 0.447 |
| Tiny-Transformer | 46.84 | 15.85 | 0.428 |
| EEGNet | 45.52 | 15.54 | 0.426 |
| CfC-style | 45.10 | 12.90 | 0.430 |
| Hybrid-CfC-style | 45.02 | 12.28 | 0.418 |
| SpatialSpectral-Head | 44.56 | 14.82 | 0.409 |
| LSTM | 38.81 | 10.36 | 0.359 |
| SpatialSpectral-CfC | 38.31 | 12.13 | 0.348 |

新 tau 结果不再是初始化噪声：

- Friedman subject-class test: `p = 0.706`
- tau vs mu power: `r = 0.792`, `p = 8.95e-09`
- tau vs beta power: `r = 0.108`, `p = 0.529`

论文结论需要从“tau 与 mu/beta correlations vanish”改成更精确的表述：tau 与 mu-band 状态有关，但没有形成稳定 class biomarker。
