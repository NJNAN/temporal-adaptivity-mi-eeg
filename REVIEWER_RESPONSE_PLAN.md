# 审稿意见解决计划

本文档根据当前项目代码、`lnn_mi_eeg_paper.tex`、`references.bib`、`REPRODUCIBILITY.md` 和 `supporting_materials/` 梳理。目标不是立刻改论文，而是把 8 条审稿意见拆成可执行的补实验、改稿和复现任务。

## 0. 当前稿件状态判断

当前论文已经具备的证据：

- 主数据集：BCI Competition IV-2a / BNCI2014-001，9 被试，4 类 MI，22 通道。
- 辅助数据集：BNCI2014-004，二分类 MI sanity check。
- 已有协议：pooled 5-fold、session+run grouped-pooled、session-wise train session 1 -> test session 2。
- 已有模型：CfC-style、Hybrid-CfC-style、LSTM、GRU 补充控制、Tiny-Transformer、EEGNet、Shallow ConvNet、Riemann-TSLR。
- 已有机制分析：trial-level `tau`、time-resolved `tau`、`tau` 与 mu/beta 功率相关、motor-channel restriction、temporal shuffle、structured perturbation、seed variability、efficiency snapshot。
- 已有复现材料：`REPRODUCIBILITY.md`、`outputs/paper_ready/`、`supporting_materials/`、split assignment CSV、seed config、paper tables。

当前最薄弱的点：

- 没有 Mamba / MI-Mamba / EEGMamba 的直接实验。
- 没有 cross-subject / leave-one-subject-out。
- `tau` 不是按电极直接学习的，目前只有全局和 motor-channel restriction，没有 scalp topography。
- `Delta t=1.0` 和 `tau` 初始化硬编码在 `AdaptiveCfCCell` 中，没有扫参。
- Hybrid-CfC 目前只是轻量诊断模型，不能代表强空间-时间混合架构。
- 复现材料已经有雏形，但 methods letter 会要求更像“可一键复现包”。
- 文稿结论需要进一步收缩，避免被理解为“所有 MI / 所有 temporal model 都没用”。

优先级建议：

1. 必做：Mamba head-to-head、LOSO/cross-subject、Delta t/tau init ablation、tau topography、复现包强化、引用状态核查。
2. 强烈建议：实现一个更强的 spatial-spectral + CfC hybrid，或把当前 Hybrid-CfC 明确降级为 diagnostic-only 且不作为负面证据。
3. 主要靠改稿解决：复杂、多类、异步 MI 的外推边界。

## 1. 审稿意见 1：标准四分类 IV-2a 的外推边界

审稿意见要点：

> 当前只在标准四类 MI 范式上测试，不能说明更复杂、多类或异步 MI 中 temporal adaptivity 也受限。

当前项目状态：

- 论文已经在 Conclusion 限制段中写到“不声称 temporal modeling 在所有 EEG 中都不重要，只针对标准四分类 MI”。
- 但摘要和引言里的语气仍偏强，容易被读成对 MI-EEG temporal modeling 的普遍否定。
- BNCI2014-004 只是二分类辅助检查，不能解决“更复杂 / 异步 MI”质疑。

解决方案：

- 修改摘要、引言和结论，把主张限定为：
  - standard cue-based four-class MI。
  - 当前 preprocessing / session-wise / compact baseline 条件。
  - temporal adaptivity alone is not a primary source in this setting。
- 在 Limitations 中新增一段专门回应：
  - 异步 MI 需要同时解决 event detection、idle state rejection、onset timing 和 non-cue-aligned decoding。
  - 多动作或连续控制范式可能更依赖 onset latency、duration、state transition。
  - 本文不能排除 temporal adaptivity 在这些场景中更重要。
- 不建议在本轮强行加入异步 MI 数据集，除非期刊明确要求。原因是异步协议会改变任务定义，从 trial classification 变成 detection / segmentation，工作量和论文主题都会变大。

需要改的文件：

- `lnn_mi_eeg_paper.tex`
- `supporting_materials/manuscript/lnn_mi_eeg_paper.tex`

建议增加的句子方向：

> These results should be read as a boundary claim for cue-locked, standard four-class MI decoding, not as a universal claim about asynchronous MI, continuous control, or paradigms in which onset timing and state transitions are part of the target variable.

## 2. 审稿意见 2：缺少 Mamba / MI-Mamba / EEGMamba 对比

审稿意见要点：

> 引言和 limitations 提到 Mamba、ODE/CDE，但没有纳入对比。MI-Mamba 是当前混合模型，缺席会削弱“时间模型不如空间模型”的结论。

当前项目状态：

- `references.bib` 已经引用 `guo2025mimamba`，且该条是正式期刊论文：Ann. N. Y. Acad. Sci. 1544(1):242-253, DOI `10.1111/nyas.15288`。
- 当前代码中没有任何 Mamba 模型。
- 论文 limitations 承认未包含 Mamba / MI-Mamba / EEGMamba。
- 这个问题不能只靠 limitations 解决，最好补实验。

推荐补实验设计：

- 新增一个现代 selective-SSM 对照：
  - 首选：`MI-Mamba-style`，因为它直接针对 MI-EEG，并且是 CNN + Mamba 的 hybrid。
  - 备选：`EEGMamba-style`，但要注意当前引用的 Gui et al. EEGMamba 是多任务 EEG 模型，不是专门的 IV-2a MI baseline；实现成本更高，且任务设定更不一样。
- 为避免不公平比较，不直接拿 MI-Mamba 论文里的 80.59% 与本文数值比；必须在本项目同一 preprocessing、split、normalization、seed、early stopping、统计检验下重跑。
- 结果解释要谨慎：
  - 如果 MI-Mamba-style 接近或超过 Shallow/Riemann：结论应改为“纯 temporal adaptivity 不够，现代 hybrid temporal-spatial model 可以受益于空间/频谱前端”。
  - 如果 MI-Mamba-style 仍未超过 Shallow/Riemann：结论会更强，但仍要说明它是 one implementation, not all SSMs。

代码任务：

- 在 `scripts/run_mi_experiments.py` 中新增模型名：
  - `mi_mamba` 或 `mamba_mi`
  - display name 建议：`MI-Mamba-style`
- 更新 `MODEL_ORDER`、`MODEL_DISPLAY_NAMES`、`build_model()`、parameter counting。
- 抽象一个 Mamba dependency 检查：
  - 首选安装 `mamba-ssm`。
  - 如果 Windows/CUDA 安装失败，采用纯 PyTorch 的 simplified selective SSM block，但论文必须标注为 `MI-Mamba-style surrogate`。
- 将新模型接入：
  - pooled: `scripts/run_mi_experiments.py`
  - grouped-pooled: `scripts/run_grouped_pooled_control.py`
  - session-wise: `scripts/run_sessionwise_mi_comparison.py`
  - efficiency: `scripts/benchmark_model_efficiency.py`
  - export: `scripts/export_reproducibility_artifacts.py`

建议命令：

```powershell
python scripts/run_mi_experiments.py --models mi_mamba shallow_convnet riemann_tslr eegnet tiny_transformer cfc lstm --device cuda --output-dir outputs/revision_mamba_pooled
python scripts/run_grouped_pooled_control.py --models mi_mamba shallow_convnet riemann_tslr eegnet tiny_transformer cfc lstm --device cuda --output-dir outputs/revision_mamba_grouped
python scripts/run_sessionwise_mi_comparison.py --models mi_mamba shallow_convnet riemann_tslr eegnet tiny_transformer cfc lstm --device cuda --output-dir outputs/revision_mamba_sessionwise
```

论文改动：

- 表 1 / 表 2 增加 `MI-Mamba-style`。
- Methods 新增 `Selective SSM / MI-Mamba-style baseline` 小节。
- Introduction 保留 MI-Mamba 引用，但不再把它只放在 limitations。
- Conclusion 如果结果允许，改成：
  - “temporal adaptivity alone” 不足。
  - “hybrid spatial-spectral/SSM designs” 是未来方向。

引用状态初查：

- Mamba 原始论文现在已有 COLM 2024 OpenReview 页面，应从纯 arXiv 条目更新为会议论文或至少加注 COLM 2024：https://openreview.net/forum?id=tEYskw1VY2
- MI-Mamba 已是正式期刊论文，当前 Bib 基本可保留。
- 当前引用的 Gui et al. `EEGMamba: Bidirectional State Space Model with Mixture of Experts...` 在 arXiv 页面仍显示为 arXiv v2，没有发现同名正式出版信息：https://arxiv.org/abs/2407.20254

## 3. 审稿意见 3：缺少 cross-subject / LOSO

审稿意见要点：

> 所有结果都是 within-subject，需要 cross-subject 或 leave-one-subject-out。

当前项目状态：

- 当前有 pooled、grouped pooled、session-wise，但都在每个 subject 内训练/测试。
- EEGNet 原论文和很多 MI 论文会报告 cross-subject，因此审稿人这个要求合理。

推荐补实验设计：

- 新增 LOSO：每次留出 1 个 subject 测试，其余 8 个 subject 训练。
- 数据处理原则：
  - 所有 subject 使用相同 22 通道和同一 2-6 s post-cue window。
  - train-only normalization：均值方差只用 8 个训练 subject 估计。
  - validation 从训练 subject 中按 subject-group 切出，不能从 held-out subject 取。
  - 可先用 session 1+2 全部训练 subject 数据训练，然后 held-out subject 两个 session 测试；也可以做 stricter variant：训练 subject session 1/2 -> held-out session 2，具体要在文中写清。
- 首轮推荐模型：
  - Shallow ConvNet
  - Riemann-TSLR
  - EEGNet
  - CfC-style
  - LSTM
  - Tiny-Transformer
  - MI-Mamba-style
- Hybrid-CfC 可作为补充，不一定进入主表。

代码任务：

- 新增 `scripts/run_loso_cross_subject.py`。
- 尽量复用 `run_mi_experiments.py` 中的模型、标准化、训练、统计函数。
- 输出：
  - `outputs/revision_loso/loso_metrics.csv`
  - `outputs/revision_loso/loso_subject_summary.csv`
  - `outputs/revision_loso/loso_results_summary.json`
  - `outputs/revision_loso/loso_stats.csv`
  - `outputs/revision_loso/loso_assignments.csv`
- 更新 `export_reproducibility_artifacts.py`，把 LOSO 表复制到 `outputs/paper_ready/` 和 `supporting_materials/`。

建议命令：

```powershell
python scripts/run_loso_cross_subject.py --models shallow_convnet riemann_tslr eegnet mi_mamba tiny_transformer cfc lstm --device cuda --output-dir outputs/revision_loso
```

论文改动：

- Methods 新增 `Cross-subject evaluation`。
- Results 新增一个小表或 supplement table。
- 主结论要允许 LOSO 可能更低：
  - 如果所有深度模型都崩得更厉害，而 Riemann 仍强：强化几何 inductive bias。
  - 如果 MI-Mamba/EEGNet 在 LOSO 明显更强：结论转为“temporal-only 不够，跨被试需要更强空间归一化/适配”。

## 4. 审稿意见 4：需要 `tau` topography

审稿意见要点：

> 已经证明 `tau` 与 mu/beta power 不相关，但应可视化 learned `tau` topography，看 motor cortices 和 peripheral channels 是否不同。

当前项目状态：

- `run_sessionwise_mi_comparison.py` 已有：
  - `BCI_IV_2A_CHANNELS`
  - `MOTOR_CHANNEL_NAMES = ["C3", "C4", "CP3", "CP4"]`
  - global tau
  - motor-channel restriction
  - windowed tau
- 但当前 `tau` 是 hidden-unit 级别，不是 electrode-indexed。因此不能直接把 hidden `tau` 当成电极 `tau` 画 scalp map。

推荐解决方案：

- 明确在论文中说明：
  - CfC-style 的 `tau` 是 hidden-state time constant。
  - 头皮图展示的是“通道对 `tau` 的影响 / attribution”，不是每个电极直接学习了一个 `tau`。
- 生成两类 topography：
  1. Occlusion-based tau sensitivity：
     - 对每个 test trial 和每个 channel，把该 channel 置零或替换为 train mean。
     - 重新前向，计算 `abs(tau_full - tau_occluded_channel)`。
     - 对 time、hidden units、trials、subjects 汇总为每个 channel 一个值。
  2. Gradient-based tau attribution：
     - 计算 `d mean_tau / d input` 的绝对值。
     - 对 time 和 trial 汇总到 channel。
- 优先把 occlusion map 放主文或补充材料，因为它更容易解释。

代码任务：

- 新增 `scripts/run_tau_topography.py`。
- 使用 MNE 的 montage 画 BCI IV-2a 22 通道 topomap。
- 输出：
  - `outputs/revision_tau_topography/tau_occlusion_topomap_global.pdf`
  - `outputs/revision_tau_topography/tau_occlusion_topomap_by_class.pdf`
  - `outputs/revision_tau_topography/tau_occlusion_channel_summary.csv`
  - `outputs/revision_tau_topography/tau_gradient_channel_summary.csv`
  - `outputs/revision_tau_topography/tau_topography_stats.json`

建议命令：

```powershell
python scripts/run_tau_topography.py --checkpoint-source outputs/bspc_sessionwise --output-dir outputs/revision_tau_topography
```

注意事项：

- 如果当前训练脚本没有保存 model checkpoints，需要先补 checkpoint 保存，或在 topography 脚本中重新训练 CfC-style session-wise 后立即分析。
- 图注必须避免写成“learned tau per electrode”；应写成“channel-wise sensitivity of the learned hidden-state time constants”。

论文改动：

- `Temporal Adaptivity versus Class Discriminability` 小节增加 topography 图或 supplementary figure。
- 如果 C3/C4 附近没有明显增强：支持当前结论。
- 如果 motor cortex 附近增强但不分 class：改成“tau is spatially influenced by sensorimotor channels but not class-discriminative”。

## 5. 审稿意见 5：Hybrid-CfC 太弱，需要证明 diagnostic 充分或实现更强 hybrid

审稿意见要点：

> Hybrid-CfC 表现差，若称为 diagnostic，需要解释为何这个最小 hybrid 足够；否则应该实现更强 hybrid 来测试空间和时间先验是否协同。

当前项目状态：

- 当前 Hybrid-CfC 已经有 EEGNet-style frontend，但论文也承认它是 minimal diagnostic。
- 它不能代表所有 spatial-continuous hybrids。
- 如果继续把它当作“hybrid 不行”的证据，会被审稿人抓住。

推荐解决方案：

- 不把当前 Hybrid-CfC 当作否定 hybrid 的证据。
- 新增一个更强但预注册、范围有限的 hybrid：
  - 名称建议：`SpatialSpectral-CfC` 或 `SS-CfC`。
  - 前端：Shallow ConvNet / filter-bank inspired temporal convolution + depthwise spatial convolution。
  - 中间：把 pooled temporal feature map 切成短 token sequence。
  - 后端：CfC-style recurrent block。
  - 对照：同一前端 + mean-max/linear head，不接 CfC。
- 关键问题不是“能不能刷榜”，而是：
  - spatial-spectral frontend alone 已经能解释多少性能？
  - 加 CfC 是否带来稳定增益？

代码任务：

- 在 `run_mi_experiments.py` 中新增：
  - `SpatialSpectralFrontend`
  - `SpatialSpectralCfCClassifier`
  - `SpatialSpectralHeadOnlyClassifier`
- 加入 `build_model()`：
  - `ss_cfc`
  - `ss_head`
- 接入 pooled、session-wise、grouped、efficiency。

建议命令：

```powershell
python scripts/run_sessionwise_mi_comparison.py --models ss_head ss_cfc shallow_convnet riemann_tslr cfc lstm --device cuda --output-dir outputs/revision_strong_hybrid_sessionwise
python scripts/run_grouped_pooled_control.py --models ss_head ss_cfc shallow_convnet riemann_tslr cfc lstm --device cuda --output-dir outputs/revision_strong_hybrid_grouped
```

论文改动：

- Methods 中把现有 `Hybrid-CfC-style` 改为 `Minimal Hybrid-CfC-style diagnostic`。
- 新增 `Spatial-spectral CfC hybrid`。
- Results 中增加 hybrid ablation 表：
  - `frontend-only`
  - `frontend+CfC`
  - `pure CfC`
  - `Shallow`
  - `Riemann`
- 结论根据结果调整：
  - 若 `ss_cfc > ss_head`：承认 temporal adaptivity can be synergistic after spatial-spectral encoding。
  - 若二者接近：说明主要收益来自 frontend，不是 CfC temporal adaptivity。

## 6. 审稿意见 6：`Delta t=1.0` 和 `tau` 初始化任意

审稿意见要点：

> `Delta t=1.0` 任意，应做 `Delta t` 和 `tau` 初始化 ablation，排除调参不足。

当前项目状态：

- `AdaptiveCfCCell` 中 `tau_mlp[-1].bias` 初始化为 `inverse_softplus(1.0)`。
- `CfCClassifier.forward()` 中每步传入的 `dt` 固定为 1.0。
- 命令行没有 `--cfc-dt`、`--cfc-tau-init`。

代码任务：

- 修改 `AdaptiveCfCCell.__init__()`：
  - 增加 `tau_init: float = 1.0`。
  - `inverse_softplus(tau_init)` 初始化 bias。
- 修改 `CfCClassifier` 和 `HybridCfCClassifier`：
  - 增加 `dt: float = 1.0`。
  - forward 中使用 `self.dt`。
- 修改 `build_model()`：
  - 增加 `cfc_dt`、`cfc_tau_init` 参数。
- 修改所有 runner 的 config / CLI：
  - `--cfc-dt`
  - `--cfc-tau-init`
- 新增 sweep 脚本 `scripts/run_cfc_dt_tau_ablation.py`。

推荐 sweep：

- `Delta t`: 0.5, 1.0, 2.0
- `tau_init`: 0.5, 1.0, 2.0
- 模型：
  - `cfc`
  - `hybrid_cfc`
  - 如果新增 `ss_cfc`，也纳入补充表。
- 协议：
  - 主：session-wise。
  - 辅：pooled 或 grouped 可只跑 `cfc`，避免计算量过大。

建议命令：

```powershell
python scripts/run_cfc_dt_tau_ablation.py --models cfc hybrid_cfc --dt-values 0.5 1.0 2.0 --tau-init-values 0.5 1.0 2.0 --device cuda --output-dir outputs/revision_cfc_dt_tau_ablation
```

输出：

- `outputs/revision_cfc_dt_tau_ablation/ablation_metrics.csv`
- `outputs/revision_cfc_dt_tau_ablation/ablation_summary.csv`
- `outputs/revision_cfc_dt_tau_ablation/ablation_stats.json`
- `outputs/revision_cfc_dt_tau_ablation/dt_tau_heatmap.pdf`

论文改动：

- Methods 增加 ablation setup。
- Results 或 supplement 增加 heatmap。
- 如果某一组明显提升，需要更新主表或说明主表使用 best validation-selected `dt/tau_init`。
- 如果没有显著提升，可以更有力回应“不是因为 `Delta t=1.0` 调参差”。

## 7. 审稿意见 7：methods letter 的 reproducibility 不足

审稿意见要点：

> 复现性是主要问题。

当前项目状态：

- 已有 `REPRODUCIBILITY.md` 和 `supporting_materials/reproducibility/`。
- 已有 seeds、split assignments、paper-ready tables。
- 但还缺少严格 methods/release 级别的环境锁定和一键执行入口。
- 当前稿件已统一为根目录 `lnn_mi_eeg_paper.tex`，supporting copy 位于 `supporting_materials/manuscript/lnn_mi_eeg_paper.tex`。

必须补强：

- 新增根目录 `README.md`：
  - 项目目的。
  - 数据下载/缓存说明。
  - 环境安装。
  - 快速 smoke test。
  - 完整复现实验命令。
  - 输出目录说明。
- 新增环境文件：
  - `requirements.txt`
  - 如有 CUDA/Mamba 依赖，再加 `environment.yml` 或 `requirements-mamba.txt`。
- 新增 `scripts/check_environment.py`：
  - 打印 Python、PyTorch、CUDA、MOABB、MNE、sklearn、pyRiemann、mamba-ssm 版本。
- 新增 `scripts/run_all_revision_experiments.ps1`：
  - 不一定默认跑全部耗时实验，但列出可复制命令。
  - 支持 `-Smoke` 模式。
- 新增 artifact manifest：
  - `outputs/paper_ready/artifact_manifest.csv`
  - 包含文件路径、生成脚本、生成时间、hash、是否正文使用。
- 更新 `.gitignore`：
  - 忽略 `__pycache__/`、`.pyc`、大数据缓存、临时 PDF/PNG。
  - 保留 `supporting_materials/` 中投稿需要的小表和图。
- 统一论文文件名：
  - 已把旧草稿文件名统一为 `lnn_mi_eeg_paper.tex`。
  - supporting copy 同步。

论文改动：

- Methods 末尾增加 `Reproducibility and code availability` 小节。
- 写清：
  - exact seeds。
  - train-only standardization。
  - validation selection。
  - split assignment files。
  - statistical correction family。
  - hardware for efficiency benchmark。

## 8. 审稿意见 8：检查 arXiv preprint 是否已有正式发表

审稿意见要点：

> 一些 arXiv preprints 应检查是否已有最终出版状态。

当前项目状态：

- `references.bib` 中仍有若干 `@misc` 或 `note={also available as arXiv...}`。
- 部分已经是正式出版但还保留 arXiv note，这没问题；但纯 arXiv 条目需要核查。

初步核查结果：

| Bib key | 当前状态 | 计划 |
|---|---|---|
| `gu2023mamba` | OpenReview 显示 COLM 2024 已发表 | 更新为 COLM 2024 conference entry，保留 arXiv 号可选 |
| `gui2024eegmamba` | arXiv v2 / OpenReview under review，未发现同题正式出版；另有不同题目的 `EEGMamba: An EEG foundation model with Mamba` 已发表于 Neural Networks 2025 | 正文改引正式发表的 `wang2025eegmamba`；Gui et al. 不再作为正文参考文献 |
| `chevallier2024moabb` | 官方 MOABB 文档与 HAL 页面仍给出 HAL/arXiv 工作论文；未找到可靠期刊 DOI | 修正作者为 Chevallier et al.，保留 HAL/arXiv 条目 |
| `ingolfsson2020eegtcnet` | 有 IEEE SMC 2020 conference record | 更新为 conference paper，不只写 arXiv note |
| `ju2022graphcspnet` | 已有 IEEE TNNLS DOI，当前条目基本正确 | 保留 article，arXiv note 可留 |
| `chen2025nsa` | 当前条目写 IJCAI 2025 + DOI | 检查 DOI/页码是否最终稳定 |

需要改的文件：

- `references.bib`
- `lnn_mi_eeg_paper.tex` 中相关引用上下文。

建议做法：

- 不要为了显得更新而引用二手页面。
- 优先顺序：publisher page / DOI / OpenReview / arXiv / official project docs。
- 修改后重新编译，检查 BibTeX 不报错。

## 9. 建议执行顺序

第一阶段：低风险改稿和复现整理

- 收缩摘要、引言、结论中的泛化表述。
- 新增 `Reproducibility and code availability` 小节。
- 统一 TeX 文件名。
- 补 `README.md`、`requirements.txt`、环境检查脚本。
- 更新明显已有正式出版状态的 Bib 条目。

第二阶段：最高优先级补实验

- 实现并跑 `MI-Mamba-style`。
- 实现并跑 LOSO cross-subject。
- 实现 `Delta t/tau_init` ablation。
- 实现 `tau` topography。

第三阶段：hybrid 争议补强

- 实现 `SpatialSpectral-CfC` 和 `frontend-only` 对照。
- 把当前 `Hybrid-CfC-style` 改写为 minimal diagnostic，不再承担强结论。

第四阶段：整合论文和回复信

- 更新主表、补充表和图。
- 根据新结果重写核心 claim。
- 准备逐条 response letter：
  - 每条都写“we agree / we have revised / we added experiment / result shows ...”。
  - 对不能补的异步 MI，明确解释 scope boundary。

## 10. 最终论文结论的推荐边界

不建议继续写成：

> temporal models cannot compete with spatial models.

建议改成：

> In cue-locked standard MI decoding, temporal adaptivity by itself is not sufficient to set the performance ceiling. The strongest gains arise when the model encodes spatial-spectral or geometric structure; modern temporal modules such as selective SSMs should therefore be evaluated as part of hybrid inductive-bias designs rather than as isolated temporal-capacity replacements.

这句话更能承受 Mamba、hybrid 和 cross-subject 实验结果的变化。

## 11. 验收清单

- [ ] `MI-Mamba-style` 在 pooled、grouped、session-wise 至少三种协议下完成。
- [ ] LOSO/cross-subject 完成，输出 subject-level 表和统计检验。
- [ ] `tau` topography 完成，并在图注中说明是 channel-wise tau attribution/sensitivity。
- [ ] `Delta t/tau_init` sweep 完成，能回答是否因超参导致 CfC 失败。
- [ ] 当前 Hybrid-CfC 的定位被修正，或新增 strong hybrid 对照。
- [ ] README、requirements、environment check、run commands、artifact manifest 完成。
- [x] `references.bib` 中 Mamba、EEG-TCNet、MOABB/EEGMamba 等状态复核完成。
- [ ] 论文结论收缩到 cue-locked IV-2a / current protocol 范围。
- [ ] supporting materials 同步更新。
- [ ] response letter 能逐条引用新增表格、图和文件。
