# 2026-05-09 Experiment Integrity Remediation Plan

本文档记录 2026-05-09 对当前实验输出完整性审查后发现的问题、影响范围和补救方案。核心结论是：项目代码和数据仍可恢复，但当前 `outputs/` 中部分关键产物已经被 smoke test 或续跑合并覆盖，不能直接作为论文证据使用。

## 1. 总体判断

当前最严重的问题不是模型结论本身已经被证明错误，而是若干本地输出目录的 provenance 已经混乱：

- `outputs/bspc_sessionwise/` 被一次 1-subject、1-epoch、CPU smoke test 覆盖。
- `outputs/paper_ready/sessionwise_table.csv` 从损坏的 session-wise JSON 重新导出，已经无效。
- `outputs/bspc_sessionwise/tau_*` 基于未充分训练的模型，不能支撑 tau 机制结论。
- `outputs/revision_mamba_pooled/results_summary.json` 的 summary 与 config/parameter_counts 不一致。
- `outputs/revision_cfc_dt_tau_ablation/` 仍在运行，当前结果只是中间快照。

因此，论文中凡是引用 `outputs/bspc_sessionwise/` 或当前 `outputs/paper_ready/sessionwise_*` 的结果，都必须暂停使用，直到完成重跑和重新导出。

## 2. 问题清单与严重程度

### 2.1 Session-wise 主实验被覆盖

**发现**

`outputs/bspc_sessionwise/sessionwise_results_summary.json` 当前配置为：

- subjects: `[1]`
- models: `["mi_mamba", "ss_cfc", "cfc"]`
- epochs: `1`
- patience: `1`
- device: `cpu`

这说明该目录被 smoke test 或小规模检查覆盖，不再是论文所需的完整 session-wise 主实验。

**影响**

直接失效：

- `outputs/bspc_sessionwise/sessionwise_results_summary.json`
- `outputs/bspc_sessionwise/sessionwise_metrics.csv`
- `outputs/bspc_sessionwise/stat_tests.csv`
- `outputs/bspc_sessionwise/tau_stats.json`
- `outputs/bspc_sessionwise/tau_timecourse_*`
- `outputs/paper_ready/sessionwise_table.csv`
- `outputs/paper_ready/sessionwise_stats.csv`

间接影响：

- session-wise 主表
- tau 机制分析
- tau 与 mu/beta 功率相关分析
- session-wise confusion matrix / per-class F1
- 从坏目录重新导出的 paper-ready 包

**严重程度**

一级严重。该问题会直接导致论文中最严格的 within-subject session-wise 协议证据不成立。

**补救方案**

不要在原目录继续试错。先重跑到新目录：

```powershell
cd "D:\作业\lnn论文1"
$env:PYTHONNOUSERSITE = "1"
$py = "D:\conda\envs\lnn-mi-eeg\python.exe"

& $py scripts/run_sessionwise_mi_comparison.py `
  --models shallow_convnet riemann_tslr eegnet hybrid_cfc tiny_transformer cfc lstm `
  --device cuda `
  --output-dir outputs/bspc_sessionwise_full_rerun
```

重跑完成并验证后，再决定是否复制/替换为正式目录 `outputs/bspc_sessionwise/`。

**验收标准**

`sessionwise_results_summary.json` 必须满足：

- subjects 为 1-9 共 9 个被试。
- models 包含 7 个主模型。
- epochs 为 80，patience 为 20，min_epochs 为 25。
- device 为 cuda。
- accuracy 不应全部接近 25% 随机水平。

建议检查命令：

```powershell
& $py -c "import json, pathlib; p=pathlib.Path('outputs/bspc_sessionwise_full_rerun/sessionwise_results_summary.json'); d=json.loads(p.read_text()); print(d['config']); print(d['summary'].keys())"
```

### 2.2 Tau 机制分析来自未训练模型

**发现**

当前 `outputs/bspc_sessionwise/tau_stats.json` 中 tau 均值约为 `1.001`，标准差约为 `1e-5`。这接近 `tau_init=1.0` 的初始化状态，说明该 tau 分析不代表训练后模型行为。

**影响**

以下结论不能使用：

- tau 不形成稳定 class-discriminative structure。
- tau 与 mu/beta power 不相关。
- tau timecourse / early-mid-late window 描述。
- 由坏 session-wise 目录导出的 tau supporting figures/tables。

**严重程度**

一级严重。tau 是论文机制解释的重要论据，必须重跑。

**补救方案**

完整重跑 session-wise 后，使用新目录重新生成 tau 统计、timecourse、window summary 和 correlation。若脚本默认生成 tau 文件，则随 session-wise 主实验一起恢复；否则补单独 tau 分析脚本。

同时保留当前 `outputs/revision_tau_topography/` 作为候选补充结果，但论文中必须写清：

- topography 是 channel-wise sensitivity of hidden-state tau。
- 不是 electrode-specific learned tau。
- 最终仍要和正常训练后的 session-wise tau 结果相互校验。

**验收标准**

- tau summary 不应全部停留在初始化值附近。
- 每个 class 至少有 9 subject x 4 class 的 subject-class 汇总。
- pairwise/tau-power 统计不能因为只有一个 subject 而出现全 NaN。

### 2.3 Paper-ready session-wise 表已损坏

**发现**

`outputs/paper_ready/sessionwise_table.csv` 当前只含 3 个模型，且准确率约 25%。这是从坏掉的 `outputs/bspc_sessionwise` 导出的。

**影响**

当前 `outputs/paper_ready/` 不能作为最终投稿包直接使用。

**严重程度**

一级严重，但属于导出层问题。只要上游 session-wise 重跑成功，可以重新导出修复。

**补救方案**

在完整 session-wise 重跑前，不再运行全量 `scripts/export_reproducibility_artifacts.py` 覆盖 supporting materials。重跑完成后：

```powershell
& $py scripts/export_reproducibility_artifacts.py
```

如果仍需保留旧有效 supporting 表，导出前先备份：

```powershell
Copy-Item supporting_materials supporting_materials_backup_before_export -Recurse
```

**验收标准**

- `outputs/paper_ready/sessionwise_table.csv` 含 7 个主模型。
- `supporting_materials/paper_tables/sessionwise_table.csv` 与新 paper-ready 表一致。
- `artifact_manifest.csv` 包含新的 session-wise 文件 hash。

### 2.4 `revision_mamba_pooled` 元数据不一致

**发现**

`outputs/revision_mamba_pooled/results_summary.json` 的 `summary` 包含 8 个模型，但 `config.models` 只显示 `["hybrid_cfc"]`，`parameter_counts` 也只有 `hybrid_cfc`。

原因很可能是：先跑了一批模型，之后在同一 output-dir 单独补跑 `hybrid_cfc`，结果 summary 读取了已有 fold rows，但 config/parameter_counts 被最后一次运行覆盖。

**影响**

- accuracy/f1 可能来自真实 fold metrics，可作为临时参考。
- 但 `results_summary.json` 的 config、parameter_counts 和 provenance 不可信。
- `revision_mamba_pooled_table.csv` 中 params 大量为空。

**严重程度**

二级严重。它削弱 MI-Mamba pooled 对比的可复现性，但不一定说明数值全部错误。

**补救方案 A：最稳**

重新全量跑到新目录：

```powershell
& $py scripts/run_mi_experiments.py `
  --models shallow_convnet riemann_tslr eegnet mi_mamba hybrid_cfc tiny_transformer cfc lstm `
  --device cuda `
  --output-dir outputs/revision_mamba_pooled_full_rerun
```

**补救方案 B：省时间**

写 repair 脚本，从 `fold_metrics.csv` 重建：

- summary
- stat_tests
- full config
- complete parameter_counts

但论文最终投稿更推荐方案 A。

**验收标准**

- `config.models` 与 summary 模型一致。
- `parameter_counts` 覆盖所有模型。
- table 中 params 不为空。

### 2.5 `Delta t / tau_init` ablation 已完成

**发现**

`outputs/revision_cfc_dt_tau_ablation/ablation_summary.csv` 已补齐 27 行，覆盖：

- `dt = 0.5, 1.0, 2.0`
- `tau_init = 0.5, 1.0, 2.0`
- `model = cfc, hybrid_cfc, ss_cfc`

**影响**

现在可以用该表回应审稿人关于 `Delta t=1.0` 任意性的质疑。

**严重程度**

已解除。该项从“未完成风险”变成“补充实验可用”。

**结果摘要**

最佳设置：

- `cfc`: `dt=2.0, tau_init=0.5`, accuracy `45.60%`
- `hybrid_cfc`: `dt=0.5, tau_init=2.0`, accuracy `52.24%`
- `ss_cfc`: `dt=1.0, tau_init=1.0`, accuracy `39.12%`

解释：

- CfC 没有被 `dt/tau_init` 调参救回；最好结果仍低于 spatial/geometric baselines。
- Hybrid-CfC 对较大的 `tau_init` 有一定收益，说明 temporal scale tuning 会影响 hybrid 表现。
- SS-CfC 当前实现没有表现出稳定协同收益。

**已同步文件**

- `outputs/paper_ready/revision_cfc_dt_tau_ablation_summary.csv`
- `outputs/paper_ready/revision_cfc_dt_tau_ablation_stats.json`
- `supporting_materials/paper_tables/revision_cfc_dt_tau_ablation_summary.csv`
- `supporting_materials/paper_tables/revision_cfc_dt_tau_ablation_stats.json`

检查命令：

```powershell
Import-Csv outputs/revision_cfc_dt_tau_ablation/ablation_summary.csv | Measure-Object
Import-Csv outputs/revision_cfc_dt_tau_ablation/ablation_summary.csv | Format-Table -AutoSize
```

**验收标准**

已满足：`ablation_summary.csv` 为 27 行，并覆盖所有 dt/tau_init 组合。

### 2.6 LOSO 中 Riemann-TSLR 排名异常

**发现**

LOSO 结果中 Riemann-TSLR 排名最后：

- EEGNet: 45.08%
- Shallow ConvNet: 41.38%
- CfC-style: 39.80%
- LSTM: 39.16%
- MI-Mamba-style: 38.62%
- Tiny-Transformer: 38.02%
- Riemann-TSLR: 34.16%

这与 pooled/grouped 中 Riemann-TSLR 稳居前二的结果不一致。

**影响**

不能再写成“Riemann-TSLR across all protocols is top-tier”。更准确的说法应是：

- within-subject pooled/grouped 中 Riemann-TSLR 很强。
- LOSO cross-subject 中，未做跨被试 alignment 的 Riemann-TSLR 受 subject shift 影响明显。

**严重程度**

二级风险。它可能是合理现象，也可能暗示 LOSO 几何 pipeline 需要 alignment 控制。

**补救方案**

代码层面先复查：

- LOSO normalization 是否只用训练 subject。
- Riemann pipeline 的 `StandardScaler` 是否只 fit 在训练 subject tangent features 上。
- held-out subject 是否没有进入 validation 或 hyperparameter selection。

推荐补一个 aligned Riemann control：

- Euclidean Alignment 或 Riemannian mean alignment。
- 输出 `outputs/revision_loso_aligned_riemann/`。
- 只需跑 Riemann-TSLR 和 aligned Riemann-TSLR，也可加 Shallow/EEGNet 做参考。

**验收标准**

- 若 alignment 后 Riemann 上升，论文中解释原始 LOSO 的 subject-shift 问题。
- 若 alignment 后仍低，说明 IV-2a LOSO 下 Riemann tangent-space baseline 确实不稳。

## 3. 当前仍可保留的有效工作

以下工作不应因为上述问题全部丢弃：

- LOSO 七模型结果：可暂用，但 Riemann 需谨慎解释。
- `run_loso_cross_subject.py` 的 resume/heldout_subject 修复：有效。
- `scripts/check_environment.py`：有效。
- README、REPRODUCIBILITY、artifact manifest 结构：有效。
- `supporting_materials/` 的新增 revision 文件结构：有效，但需要在最终导出后刷新。
- `outputs/revision_tau_topography/`：可作为候选补充结果，但需与正常 session-wise tau 重跑后互相校验。
- `dt/tau` 当前后台任务：继续等待，不要中途把中间表写成最终结论。

## 4. 建议补救执行顺序

### 第一步：等待当前 dt/tau sweep 完成

不要同时开新的 GPU 训练任务，避免显存、温度和速度问题。

实时查看：

```powershell
Get-Content outputs/revision_job_logs/dt_tau.out.log -Tail 30 -Wait
```

### 第二步：完整重跑 session-wise

```powershell
& $py scripts/run_sessionwise_mi_comparison.py `
  --models shallow_convnet riemann_tslr eegnet hybrid_cfc tiny_transformer cfc lstm `
  --device cuda `
  --output-dir outputs/bspc_sessionwise_full_rerun
```

通过验收后，再替换正式目录或修改 export 脚本指向新目录。

### 第三步：重建 tau 机制结果

确认新 session-wise 输出包含：

- `tau_stats.json`
- `tau_trial_metrics.csv`
- `tau_subject_class_summary.csv`
- `tau_timecourse_summary.csv`
- `tau_timecourse_subject_level.csv`
- `tau_time_window_summary.csv`

### 第四步：修复 MI-Mamba pooled metadata

优先新目录全量重跑：

```powershell
& $py scripts/run_mi_experiments.py `
  --models shallow_convnet riemann_tslr eegnet mi_mamba hybrid_cfc tiny_transformer cfc lstm `
  --device cuda `
  --output-dir outputs/revision_mamba_pooled_full_rerun
```

### 第五步：补 aligned Riemann LOSO

这是补救 LOSO 异常解释的最好实验。可以只跑几何 baseline 的 aligned/un-aligned 对照。

### 第六步：重新导出 paper_ready 和 supporting_materials

在所有关键结果通过验收后再执行：

```powershell
& $py scripts/check_environment.py --output outputs/paper_ready/environment_check.json
& $py scripts/export_reproducibility_artifacts.py
```

导出后检查：

```powershell
Get-Content outputs/paper_ready/sessionwise_table.csv
Get-Content supporting_materials/paper_tables/sessionwise_table.csv
Import-Csv supporting_materials/reproducibility/artifact_manifest.csv | Measure-Object
```

## 5. 论文写作层面的立即约束

在补救完成前，论文中不要写：

- “session-wise 结果显示……”
- “tau 与 class/mu/beta 的关系显示……”
- “所有协议中 Riemann-TSLR 均强于 temporal models”
- “dt/tau ablation 已证明……”
- “MI-Mamba pooled 表已经完全可复现”

可以暂时写：

- “LOSO preliminary results suggest that cross-subject shift compresses all model differences.”
- “The current topography analysis is treated as a channel-wise sensitivity diagnostic and will be cross-checked after the full session-wise rerun.”
- “The dt/tau initialization sweep is in progress and will be reported only after all planned combinations finish.”

## 6. 最终验收清单

- [x] `outputs/bspc_sessionwise_full_rerun/sessionwise_results_summary.json` 为 9 subjects、7 models、CUDA、80 epochs。
- [x] 新 session-wise 表不是随机水平，且含 7 个模型。
- [x] 新 tau 统计不是初始化值，且统计检验不再因 1 subject 出现全 NaN。
- [x] `dt/tau` ablation summary 为 27 行。
- [x] `revision_mamba_pooled` 的 config、summary、parameter_counts 一致。
- [x] LOSO Riemann 异常有代码复查记录，并已有 aligned Riemann control。
- [x] `outputs/paper_ready/` 从通过验收的结果重新生成。
- [x] `supporting_materials/` 与 paper-ready 结果同步。
- [x] `artifact_manifest.csv` 更新。
- [x] 论文正文和 response letter 只引用通过验收的表和图。

## 8. 2026-05-09 full session-wise 补救结果

### 8.1 原 7 模型 full session-wise 已重跑完成

输出目录：

```text
outputs/bspc_sessionwise_full_rerun/
```

验收结果：

- `63/63` runs 完成。
- 9 subjects。
- 7 models。
- `epochs=80`, `patience=20`, `min_epochs=25`。
- `device=cuda`。
- `tau` 不再停留在初始化值附近。

主结果：

| Model | Accuracy mean | Accuracy std | F1 mean |
|---|---:|---:|---:|
| Riemann-TSLR | 62.81 | 12.73 | 0.617 |
| Shallow ConvNet | 59.45 | 15.43 | 0.585 |
| Tiny-Transformer | 46.84 | 15.85 | 0.428 |
| EEGNet | 45.52 | 15.54 | 0.426 |
| CfC-style | 45.10 | 12.90 | 0.430 |
| Hybrid-CfC-style | 45.02 | 12.28 | 0.418 |
| LSTM | 38.81 | 10.36 | 0.359 |

### 8.2 MI-Mamba / stronger hybrid session-wise 已补跑完成

输出目录：

```text
outputs/revision_mamba_hybrid_sessionwise/
```

结果：

| Model | Accuracy mean | Accuracy std | F1 mean |
|---|---:|---:|---:|
| MI-Mamba-style | 48.38 | 17.96 | 0.447 |
| SpatialSpectral-Head | 44.56 | 14.82 | 0.409 |
| SpatialSpectral-CfC | 38.31 | 12.13 | 0.348 |

合并 10 模型 session-wise 排序：

| Model | Accuracy mean |
|---|---:|
| Riemann-TSLR | 62.81 |
| Shallow ConvNet | 59.45 |
| MI-Mamba-style | 48.38 |
| Tiny-Transformer | 46.84 |
| EEGNet | 45.52 |
| CfC-style | 45.10 |
| Hybrid-CfC-style | 45.02 |
| SpatialSpectral-Head | 44.56 |
| LSTM | 38.81 |
| SpatialSpectral-CfC | 38.31 |

解释：

- MI-Mamba-style 是新增 sequence/SSM 类模型中最强的，但仍低于 Riemann-TSLR 和 Shallow ConvNet。
- SpatialSpectral-Head 接近 CfC/EEGNet 中间层。
- SpatialSpectral-CfC 低于 frontend-only head，说明当前 CfC 后端没有给这个 frontend 带来稳定协同收益。

### 8.3 Tau 机制结论更新

新 `tau_stats.json` 来自完整 session-wise 重跑，不再是初始化噪声。

关键结果：

- Friedman subject-class test: `p = 0.706`，仍不支持稳定类别分离。
- class mean tau:
  - left hand: `1.565`
  - right hand: `1.574`
  - feet: `1.631`
  - tongue: `1.616`
- pairwise raw p-values all `>= 0.152`。
- tau vs mu power: `r = 0.792`, `p = 8.95e-09`。
- tau vs beta power: `r = 0.108`, `p = 0.529`。

论文写法需要从“tau 与 mu/beta correlations vanish”改为：

> tau tracks part of the mu-band sensorimotor state but does not form a stable class biomarker.

### 8.4 已同步文件

- `supporting_materials/paper_tables/sessionwise_table.csv`
- `supporting_materials/paper_tables/sessionwise_extended_revision_table.csv`
- `supporting_materials/paper_tables/revision_mamba_hybrid_sessionwise_table.csv`
- `supporting_materials/subject_results/sessionwise_subject_scores.csv`
- `supporting_materials/subject_results/sessionwise_extended_revision_metrics.csv`
- `supporting_materials/subject_results/revision_mamba_hybrid_sessionwise_metrics.csv`
- `supporting_materials/tau_analysis/tau_stats.json`
- `supporting_materials/tau_analysis/tau_timecourse_by_class.pdf`
- `supporting_materials/reproducibility/sessionwise_results_summary.json`
- `supporting_materials/reproducibility/revision_mamba_hybrid_sessionwise_results_summary.json`

## 7. 2026-05-09 已执行的补救动作

- 新增 `scripts/repair_mamba_pooled_summary.py`：从 `fold_metrics.csv` 重建 `revision_mamba_pooled/results_summary.json`、`stat_tests.csv`、`subject_summary.csv` 和完整 `parameter_counts`，用于修复“config 只显示 hybrid_cfc 但 summary 有 8 个模型”的 provenance 问题。
- 新增 `scripts/run_loso_riemann_alignment_check.py`：对 LOSO Riemann-TSLR 做 standard vs unsupervised Euclidean Alignment 诊断，用来判断 34% 是否主要来自跨被试 tangent-space/协方差对齐问题。
- 更新 `scripts/run_cfc_dt_tau_ablation.py`：增加组合级断点跳过逻辑。以后如果长 sweep 中断，重启时会跳过 `ablation_summary.csv` 中已经完整出现的 dt/tau/model 组合。
- 更新 `lnn_mi_eeg_paper.tex` 和 `supporting_materials/manuscript/lnn_mi_eeg_paper.tex`：删除 tau 分析段落中的旧硬编码统计值，改为明确说明这些数值必须等完整 session-wise 重跑后刷新。
- 在 CfC 方法描述中补充默认 `dt=1.0, tau_init=1.0` 的初始 retention 解释：初始每步保留因子约为 `exp(-1)=0.37`，若 tau 保持在初始化附近，约 5 个 recurrent updates 后旧状态贡献低于 1%。这将作为 ablation 解释的模型设计边界。

### 7.1 LOSO Riemann alignment 诊断结果

已运行：

```powershell
& $py scripts/run_loso_riemann_alignment_check.py --output-dir outputs/revision_loso_riemann_alignment
```

输出：

- `outputs/revision_loso_riemann_alignment/riemann_alignment_loso_metrics.csv`
- `outputs/revision_loso_riemann_alignment/riemann_alignment_loso_summary.csv`
- `outputs/revision_loso_riemann_alignment/riemann_alignment_loso_stats.csv`
- `outputs/revision_loso_riemann_alignment/riemann_alignment_loso_results_summary.json`

结果：

| Variant | Accuracy mean | Accuracy std | F1 mean | F1 std |
|---|---:|---:|---:|---:|
| standard | 34.16 | 6.36 | 0.278 | 0.075 |
| euclidean_alignment | 47.90 | 13.14 | 0.473 | 0.136 |

Paired standard vs Euclidean Alignment:

- mean difference: `-13.73` percentage points for standard minus aligned
- paired t-test: `p = 0.00113`
- Wilcoxon: `p = 0.00391`

解释：

LOSO 中 Riemann-TSLR 跌到 34% 不是标准化泄漏导致的“虚高/虚低”证据，而更像跨被试 covariance/tangent-space alignment 问题。无监督 Euclidean Alignment 使用每个 subject 的未标注 trials 做 test-time alignment，因此不能和完全无适配 LOSO 主表直接混为同一协议；但它很好地解释了为什么 within-subject 中强的 Riemann-TSLR 在原始 LOSO 中掉到最后。论文应写成：Riemann-TSLR is strong within subject, but raw LOSO without subject alignment is unfavorable to tangent-space geometry; an unsupervised EA diagnostic substantially improves it.

### 7.2 MI-Mamba pooled metadata 修复结果

已运行：

```powershell
& $py scripts/repair_mamba_pooled_summary.py `
  --output-dir outputs/revision_mamba_pooled `
  --models shallow_convnet riemann_tslr eegnet mi_mamba hybrid_cfc tiny_transformer cfc lstm
```

修复后：

- `results_summary.json` 的 `config.models` 覆盖 8 个模型。
- `parameter_counts` 覆盖 8 个模型。
- JSON 中新增 `repair_note`，说明该 summary 是从 `fold_metrics.csv` 重建的。
- 该结果不再应被解释为“混入了 bspc_pooled”；更准确地说，它是同一 revision output-dir 内多次 subset/resume 运行后，从完整 `fold_metrics.csv` 重新汇总得到的结果。
