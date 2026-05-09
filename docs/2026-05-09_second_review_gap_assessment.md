# 2026-05-09 第二轮技术审查意见补救评估

本文档针对新一轮技术意见做可执行分解。核心判断是：这些意见不会直接推翻当前主结论，但会限制结论强度。最需要补的是全模型 cross-subject alignment 公平性、MI-Mamba-style 细节、以及“低于近期 SOTA”的解释。

## 总体结论

当前论文已经适合作为一个受控 negative/boundary study，但不能写成“temporal models cannot compete”或“spatial/geometric priors set the universal ceiling”。更稳的表述是：

> 在当前 cue-locked IV-2a、8--30 Hz、统一训练协议和统一 readout 下，temporal adaptivity alone 没有解释主要性能差异；结果更符合 protocol-dependent spatial-spectral/geometric inductive bias 的解释。

这轮意见中，最硬的实验缺口是 LOSO fairness。当前只给 Riemann-TSLR 做了 Euclidean Alignment diagnostic。由于已有工作显示 EA 对 deep EEG decoders 也有效，若继续投稿 revision，建议新增 aligned LOSO for all model families，至少覆盖 EEGNet、Shallow ConvNet、Tiny-Transformer、MI-Mamba-style、CfC-style、LSTM。

## 1. CfC-style 和 MI-Mamba-style 细节不足

### 问题严重性

中等偏严重。CfC-style 已经说明是 variant，但 MI-Mamba-style 的 state size、A/B/C 参数化、离散化/扫描形式写得太短。审稿人会问：负结果是否来自一个过弱 surrogate，而不是 Mamba-like model 本身。

### 当前代码事实

`scripts/run_mi_experiments.py` 中 `SelectiveSSMBlock` 当前配置：

- `d_model = 64`
- `d_state = 16`
- block 数量：2
- 输入投影：`Linear(d_model, 2*d_model)`，拆成 content 和 gate
- 局部卷积：depthwise `Conv1d`, kernel size 5, padding 2
- A 参数：`a = -softplus(a_log)`，`a_log` shape 为 `(d_model, d_state)`，初始 0
- 离散更新：固定单位步长 `state = exp(a) * state + B(x_t) * x_t`
- B/C 参数：input-dependent linear projections，shape 变为 `(batch, time, d_model, d_state)`
- 输出：`sum(C_t * state_t)` 后乘 sigmoid gate，再 `out_proj`
- dropout：0.2
- readout：temporal mean + max pooling

### 补救动作

- 已建议正文统一写 `MI-Mamba-style selective-SSM surrogate`，不能写成 exact MI-Mamba。
- 应在 `REPRODUCIBILITY.md` 或 supporting materials 中加入上面的完整 SSM 配置。
- 如页数允许，正文 Methods 加一句：state size 16, two blocks, input-dependent B/C, negative softplus A, fixed-step exponential scan。
- Response letter 中也要承认它是 shared-protocol surrogate，不是 original CUDA Mamba/MI-Mamba reimplementation。

## 2. 8--30 Hz narrowband preprocessing 可能偏向 band-power/covariance

### 问题严重性

中等。这个意见合理，因为 8--30 Hz 本来就强化 mu/beta ERD/ERS，也可能削弱 broadband temporal signatures。它不会推翻“在标准 MI band 下 temporal adaptivity alone 不足”的结论，但会限制泛化范围。

### 补救动作

优先级 A：正文 limitation 明确：

- 结果只适用于 standard mu/beta band-pass MI decoding。
- raw/broadband input could reveal temporal signatures not visible here。
- 因此不能声称 temporal adaptivity 对所有 EEG setting 不重要。

优先级 B：如果有时间，跑一个小型 broadband sensitivity：

```powershell
python scripts/run_mi_experiments.py --models shallow_convnet eegnet mi_mamba cfc lstm --device cuda --output-dir outputs/revision_broadband_pooled
```

但当前代码的数据准备默认 8--30 Hz，真正做 broadband 需要给 preprocessing 增加 band 参数，不能只改命令。

## 3. 统一 mean-max pooling 可能不适合 SSM/Transformer

### 问题严重性

中等。统一 readout 保证控制变量，但确实可能不是每类模型最优。

### 补救动作

- 正文写成 controlled readout，不是 architecture-optimized readout。
- 可补一个轻量 ablation：Tiny-Transformer attention pooling、MI-Mamba last-token/attention pooling、CfC final-state pooling。
- 如果没有时间跑，必须在 limitation 中承认 alternative heads could change absolute performance and rank within temporal models。

## 4. Session-wise accuracy 低于近期 SOTA

### 问题严重性

严重但可解释。Shallow ConvNet 59.5% 和 EEGNet 45.5% 低于许多 SOTA 报告。这个点如果不解释，会让审稿人怀疑训练不足，并进一步怀疑 temporal models 也被低估。

### 可能原因

- 当前是严格 train session 1 -> test session 2，不做 cropped training。
- 不做 subject-specific heavy tuning。
- 不做 strong augmentation、mixup、pretraining、多分支滤波或 architecture-specific tricks。
- 所有模型共享 80 epoch、early stopping、train-only normalization 的统一协议。
- 目标是 controlled comparison，不是刷新 IV-2a SOTA。

### 补救动作

正文需要非常明确：

- 本文不是 SOTA benchmark。
- 当前 compact baselines intentionally avoid cropped training and architecture-specific tuning。
- SOTA-like models such as EEGEncoder、DBConformer、DFBRTS、Graph-CSPNet 等支持“strong spatial-spectral/geometric/hybrid engineering matters”，但不能直接和本统一协议数值横比。

如果要实验证明不欠优化，建议：

1. 跑 Shallow ConvNet cropped/augmentation enhanced session-wise。
2. 跑 EEGNet tuned learning rate/dropout/cropping。
3. 至少给出一个 “stronger Shallow/EEGNet sanity check”。

## 5. LOSO fairness：只给 Riemann 做 EA 不够

### 问题严重性

最高。当前 LOSO 解释是 diagnostic，但如果正文呈现 ranking，审稿人会要求所有模型都在 aligned 和 unaligned 下公平比较。

### 文献依据

- Junqueira et al. `arXiv:2401.10746` 系统评估 EA + deep learning，报告 target-subject decoding 提升 4.33%，并显著减少收敛时间。
- Wimpff et al. `arXiv:2311.18520` 探讨 online test-time adaptation for EEG MI，包含 alignment、adaptive batch normalization、entropy minimization 等 unsupervised adaptation。

### 补救动作

新增脚本建议：

```text
scripts/run_loso_alignment_all_models.py
```

建议最小模型集：

- `eegnet`
- `shallow_convnet`
- `tiny_transformer`
- `mi_mamba`
- `cfc`
- `lstm`
- `riemann_tslr`

建议 variant：

- `standard`
- `euclidean_alignment`

若时间允许，再加入：

- `riemann_alignment` 或 source+target RA
- adaptive BN / entropy-min OTTA 作为独立后续

关键注意：

- EA 使用 held-out subject 的 unlabeled test trials，必须明确是 test-time unsupervised alignment/adaptation setting。
- aligned and unaligned rankings should be reported separately。
- 不应把 aligned result 和原 standard LOSO 混成一个主排名。

## 6. Full model set seed 数不足

### 问题严重性

中等。当前 repeat seed 只覆盖 subset，不能声称全模型 rank stability。

### 补救动作

优先补文：

- repeat-seed check is limited and subset-based。
- rank stability for all revision-only models remains residual uncertainty。

若补实验：

```powershell
python scripts/run_sessionwise_mi_comparison.py --models shallow_convnet riemann_tslr eegnet mi_mamba tiny_transformer cfc lstm ss_head ss_cfc --seed 42 --device cuda --output-dir outputs/revision_sessionwise_seed42_full
python scripts/run_sessionwise_mi_comparison.py --models shallow_convnet riemann_tslr eegnet mi_mamba tiny_transformer cfc lstm ss_head ss_cfc --seed 43 --device cuda --output-dir outputs/revision_sessionwise_seed43_full
```

这个成本很高，建议只在 journal 要求时跑。

## 7. “revision/regenerated/artifact-like”措辞

### 问题严重性

轻到中等。审稿版可以说 regenerated/revision，但投稿正文最好像一篇独立 letter。

### 补救动作

正文中替换：

- `revision-inclusive rerun` -> `MI-Mamba-inclusive comparison` 或 `extended comparison`
- `regenerated full session-wise run` -> `session-wise evaluation`
- `supporting materials as revision_*` -> `supporting materials`

路径名可以留在 Reproducibility/README，不要塞进正文主叙述。

## 8. Tau topography 只有描述图，缺量化摘要

### 问题严重性

中等。图已经有 colorbar，但正文没有给 channel-level 数值摘要。

### 当前可补信息

`supporting_materials/paper_tables/revision_tau_occlusion_channel_summary.csv` 包含 channel-wise sensitivity mean/std。

建议正文加一句：

> The strongest channel-wise sensitivities are summarized in the supporting table, with the top channels concentrated around motor-adjacent central/parietal electrodes rather than forming a class-specific map.

更好：直接列 top-3 channel 和 mean 值，但要先从 CSV 读出。

## 9. 新增相关工作

### 已核查信息

- Euclidean Alignment for DL EEG decoding: `arXiv:2401.10746`, title `A Systematic Evaluation of Euclidean Alignment with Deep Learning for EEG Decoding`，arXiv v4 2024-05-23，DataCite DOI `10.48550/arXiv.2401.10746`。
- OTTA: `arXiv:2311.18520`, title `Calibration-free online test-time adaptation for electroencephalography motor imagery decoding`，6-page BCI 2024 conference version，DataCite DOI `10.48550/arXiv.2311.18520`。
- DFBRTS: `arXiv:2310.19198`, title `Enhancing Motor Imagery Decoding in Brain Computer Interfaces using Riemann Tangent Space Mapping and Cross Frequency Coupling`。
- EEGEncoder: `arXiv:2404.14869`, title `EEGEncoder: Advancing BCI with Transformer-Based Motor Imagery Classification`。
- Mamba: `arXiv:2312.00752`, already in manuscript/reference list as the base selective-SSM paper.

### 补救动作

正文 4 页限制很紧，不建议把所有新工作塞进主文。可在 response letter 和 supporting related-work note 中写：

- These works are not omitted as irrelevant; they represent architecture-optimized or adaptation-enhanced systems outside the controlled comparison.
- Their high reported accuracies reinforce the conclusion that task-tailored spatial-spectral/geometric/hybrid engineering matters.

## 建议执行顺序

1. 立即修文：去掉正文中 `revision/regenerated` 式措辞；补 MI-Mamba-style SSM 配置；补 narrowband/readout/SOTA/LOSO alignment limitation。
2. 立即补材料：`REPRODUCIBILITY.md` 增加 MI-Mamba-style implementation card 和 alignment fairness note。
3. 中等成本实验：写并跑 `run_loso_alignment_all_models.py`。
4. 高成本实验：full model set repeat seeds 或 stronger Shallow/EEGNet tuning。
5. 可选实验：broadband/raw input sensitivity、alternative readout head ablation。

## 当前可接受的 response stance

可以在回复信中承认：

- The original LOSO result is diagnostic, not a final aligned cross-subject benchmark.
- MI-Mamba-style is a surrogate; its negative result should not be generalized to all selective SSM designs.
- 8--30 Hz preprocessing and unified mean-max pooling are controlled-design choices that may limit temporal models.
- Recent high-performing architectures are not contradicted by this study; they often add stronger spatial-spectral/geometric/hybrid priors or adaptation machinery, which is consistent with the manuscript's refined thesis.

