# 论文与代码实现对应说明

本文档详细说明当前论文 [D:/作业/lnn论文1/lnn_mi_eeg_paper.tex](D:/作业/lnn论文1/lnn_mi_eeg_paper.tex) 中每个主要结论、实验部分、图表和补充证据，分别由哪些代码实现、输出到哪些文件、以及这些代码在整个实验流水线里承担什么作用。

文档目标不是简单列目录，而是回答下面几个问题：

1. 论文每一节到底对应哪些脚本和函数。
2. 每段方法文字在代码里真正做了什么。
3. 每个表、图、统计量来自哪里。
4. 哪些脚本是主结果，哪些只是补充控制实验。
5. 当前论文没有做的内容，在代码里也没有做什么。

---

## 1. 项目总体结构

当前论文的证据链由 4 层组成：

1. 主实验层  
   对应 pooled、grouped-pooled、session-wise 三个主协议。

2. 机制分析层  
   对应 `tau` 分布、`tau` 与频带功率相关、time-resolved `tau`。

3. 补充控制层  
   对应 GRU control、seed variability、temporal shuffle、BNCI2014-004 辅助 sanity check。

4. 整理导出层  
   对应 `paper_ready` 和 `supporting_materials`，用于把原始产物整理成投稿可引用的表格、图和复现包。

核心脚本分工如下：

- [D:/作业/lnn论文1/scripts/run_mi_experiments.py](D:/作业/lnn论文1/scripts/run_mi_experiments.py)  
  主 pooled 5-fold 实验，兼作整个项目的“核心库”。绝大多数模型定义、训练函数、评估函数、统计函数都在这里。

- [D:/作业/lnn论文1/scripts/run_sessionwise_mi_comparison.py](D:/作业/lnn论文1/scripts/run_sessionwise_mi_comparison.py)  
  主 session-wise 实验，同时负责 `tau` 分析、structured perturbation、per-class F1、confusion matrix。

- [D:/作业/lnn论文1/scripts/run_grouped_pooled_control.py](D:/作业/lnn论文1/scripts/run_grouped_pooled_control.py)  
  grouped pooled control，用来检查 pooled trial-level CV 是否因 run/session 混洗而偏乐观。

- [D:/作业/lnn论文1/scripts/run_structured_perturbation_sweep.py](D:/作业/lnn论文1/scripts/run_structured_perturbation_sweep.py)  
  多强度扰动 sweep。

- [D:/作业/lnn论文1/scripts/run_temporal_shuffle_control.py](D:/作业/lnn论文1/scripts/run_temporal_shuffle_control.py)  
  补充 temporal shuffle 控制，用来测试“打乱试次内时间顺序后，模型是否还保留优势”。

- [D:/作业/lnn论文1/scripts/run_sessionwise_clean_subset.py](D:/作业/lnn论文1/scripts/run_sessionwise_clean_subset.py)  
  repeat-seed session-wise clean rerun，用来做 seed variability sanity check。

- [D:/作业/lnn论文1/scripts/summarize_seed_variability.py](D:/作业/lnn论文1/scripts/summarize_seed_variability.py)  
  汇总 repeat seeds 的稳定性结果。

- [D:/作业/lnn论文1/scripts/run_bnci2014_004_aux.py](D:/作业/lnn论文1/scripts/run_bnci2014_004_aux.py)  
  第二个小数据集的辅助 sanity check，不进入主结论，只作为 supporting-only evidence。

- [D:/作业/lnn论文1/scripts/benchmark_model_efficiency.py](D:/作业/lnn论文1/scripts/benchmark_model_efficiency.py)  
  效率快照：forward、train step、batch size 1 latency、CPU 上的 `Riemann-TSLR` 延迟。

- [D:/作业/lnn论文1/scripts/export_reproducibility_artifacts.py](D:/作业/lnn论文1/scripts/export_reproducibility_artifacts.py)  
  把所有散落在 `outputs/` 的结果整理成 `outputs/paper_ready/` 和 `supporting_materials/`。

---

## 2. 论文核心论点，对应哪些代码

论文现在的核心论点不是“哪个模型赢”，而是：

> 在标准四分类 MI-EEG 上，continuous-time temporal adaptivity 不是主要的判别信息来源。  
> 它能改善 recurrent baseline，但不能替代 spatial / spectral / geometric structure。

这个论点在代码上不是由单一脚本证明的，而是由以下几组证据共同支撑：

### 2.1 `CfC-style > LSTM`

由以下脚本提供：

- [D:/作业/lnn论文1/scripts/run_mi_experiments.py](D:/作业/lnn论文1/scripts/run_mi_experiments.py)
- [D:/作业/lnn论文1/scripts/run_sessionwise_mi_comparison.py](D:/作业/lnn论文1/scripts/run_sessionwise_mi_comparison.py)
- [D:/作业/lnn论文1/scripts/run_grouped_pooled_control.py](D:/作业/lnn论文1/scripts/run_grouped_pooled_control.py)

也就是：

- pooled 5-fold 下比一次
- grouped pooled 下再比一次
- session-wise 下再比一次

这样不是单协议结论，而是跨协议结论。

### 2.2 `CfC-style` 仍然不如 `Shallow ConvNet / Riemann-TSLR`

由以下脚本提供：

- pooled 主结果：  
  [D:/作业/lnn论文1/scripts/run_mi_experiments.py](D:/作业/lnn论文1/scripts/run_mi_experiments.py)

- grouped control：  
  [D:/作业/lnn论文1/scripts/run_grouped_pooled_control.py](D:/作业/lnn论文1/scripts/run_grouped_pooled_control.py)

- session-wise 主泛化结果：  
  [D:/作业/lnn论文1/scripts/run_sessionwise_mi_comparison.py](D:/作业/lnn论文1/scripts/run_sessionwise_mi_comparison.py)

这里的重要点不是“某一次比输”，而是：

- pooled：输
- grouped：输
- session-wise：还输

所以代码层面支撑的是“结构性边界”，不是偶然排序。

### 2.3 `tau` 没有形成稳定类判别结构

由以下代码负责：

- trial-level `tau` 提取：  
  [D:/作业/lnn论文1/scripts/run_sessionwise_mi_comparison.py](D:/作业/lnn论文1/scripts/run_sessionwise_mi_comparison.py) 中 `collect_cfc_trial_analysis`

- class-level / subject-level 聚合与 Friedman / pairwise 统计：  
  同文件中的 `summarize_tau_trial_analysis`

- time-resolved `tau` 曲线：  
  同文件中的 `summarize_tau_timecourse`

这部分不是“看图说话”，而是三段证据链：

1. 分布不分开
2. 统计不显著
3. 与 `mu/beta` 关联也不稳

所以论文才可以写成：

> temporal adaptivity does not constitute a reliable source of class-discriminative information

### 2.4 `temporal shuffle` 没有给 `CfC-style` 带来独特优势

由以下脚本提供：

- [D:/作业/lnn论文1/scripts/run_temporal_shuffle_control.py](D:/作业/lnn论文1/scripts/run_temporal_shuffle_control.py)

这不是主协议，而是补充机制控制。它回答的不是“谁最高”，而是：

> 如果把试次内时间顺序打乱，是否说明 `CfC-style` 真正在利用某种独特的 temporal order 信息？

代码给出的答案是：

- `Riemann-TSLR` 不变
- `Shallow / EEGNet / CfC / LSTM` 都掉
- 但 `CfC-style` 没有因此出现独特优势

所以这部分不是在说“时间没用”，而是在说：

> preserving temporal order does not create a distinct CfC-style advantage

---

## 3. 论文章节与代码逐段对应

下面按论文的章节顺序说明。

---

## 4. `Introduction` 对应的代码证据

引言本身当然不是代码实现，但引言里的每个论断都必须在代码里有对应证据。

### 4.1 “continuous-time models expose adaptive time scales”

代码对应：

- [D:/作业/lnn论文1/scripts/run_mi_experiments.py](D:/作业/lnn论文1/scripts/run_mi_experiments.py)
  - `AdaptiveCfCCell`
  - `CfCClassifier`
  - `HybridCfCClassifier`

这里真正暴露 `tau` 的地方在 `AdaptiveCfCCell.forward`：

- 输入是当前 step 的输入和前一时刻 hidden state
- 输出是新的 hidden 和当前 step 的 `tau`
- `tau` 通过 `softplus` 保证正值

也就是说，引言里说 “CfC-style exposes adaptive time scales” 并不是抽象口号，而是这几个类真正返回了 `tau`。

### 4.2 “the study tests whether temporal adaptivity itself is a major source of discriminative power”

这句在代码上对应三类实验：

1. 主分类性能比较  
   看 `CfC-style` 是否真的比传统 recurrent 好、是否能逼近 top tier。

2. `tau` 可解释性分析  
   看 `tau` 是否和类别相关。

3. temporal shuffle control  
   看时间顺序被破坏后，`CfC-style` 是否有独特优势。

对应脚本：

- [D:/作业/lnn论文1/scripts/run_mi_experiments.py](D:/作业/lnn论文1/scripts/run_mi_experiments.py)
- [D:/作业/lnn论文1/scripts/run_sessionwise_mi_comparison.py](D:/作业/lnn论文1/scripts/run_sessionwise_mi_comparison.py)
- [D:/作业/lnn论文1/scripts/run_temporal_shuffle_control.py](D:/作业/lnn论文1/scripts/run_temporal_shuffle_control.py)

---

## 5. `Models Under Test` 对应实现

这一节是最直接的“方法 -> 代码”映射。

### 5.1 `CfC-style Unit`

代码入口：

- [D:/作业/lnn论文1/scripts/run_mi_experiments.py](D:/作业/lnn论文1/scripts/run_mi_experiments.py)

具体实现：

- `AdaptiveCfCCell`
- `CfCClassifier`

#### 5.1.1 `AdaptiveCfCCell` 做了什么

它就是论文公式的直接实现：

1. 把当前输入和前一隐藏状态拼接起来
2. 做 `LayerNorm`
3. 走 candidate head，得到候选隐藏状态
4. 走 tau MLP，得到当前 step 的 `tau`
5. 用 `exp(-dt / tau)` 做指数衰减混合

对应论文中的变量关系：

- candidate head 对应 $\tilde{h}_t$
- `tau_mlp` 对应 $\tau_t$
- `decay = exp(-dt / tau)` 对应公式里的连续时间衰减项

#### 5.1.2 `CfCClassifier` 做了什么

它把 cell 变成一个完整分类器：

1. 输入形状从 `(batch, channels, time)` 转为 `(batch, time, channels)`
2. 先做一层 `input_proj`
3. 逐时间步循环调用 `AdaptiveCfCCell`
4. 把所有 hidden state 存下来
5. 用 mean-max pooling 读出
6. 接线性分类头

这正对应论文里写的：

- `CfC-style uses hidden size 128`
- `temporal mean-max pooling`
- `Delta t = 1.0`

### 5.2 `Hybrid-CfC-style`

代码入口：

- `HybridCfCClassifier`

这个类对应论文里“diagnostic hybrid”那段。

它不是完整搜索后的 hybrid，而是有意识地保持小：

1. 很浅的 EEGNet-style 前端
2. depthwise spatial convolution
3. compact separable temporal block
4. average pooling
5. 再把前端特征送入 CfC-style cell

论文里强调它是 diagnostic design，不是失败的新模型。代码上也确实如此：

- temporal filters 固定为 `8`
- spatial filters 固定为 `16`
- dropout 固定为 `0.25`
- 后端 CfC hidden 仍然是 `128`

也就是说，这个 hybrid 的目的不是“追最佳性能”，而是测试：

> 加一点轻量 spatial prior 后，纯 temporal recurrence 的结论会不会变

### 5.3 `LSTM` / `GRU` / `EEGNet` / `Shallow ConvNet`

代码入口：

- `LSTMClassifier`
- `GRUClassifier`
- `EEGNet`
- `ShallowConvNet`

它们对应论文中的四类 reference：

- 离散 recurrent
- compact CNN
- band-power CNN
- recurrent control

其中：

- `GRU` 不在主表里长期并列，而是补充 control
- `ShallowConvNet` 是论文里的 top-tier CNN
- `EEGNet` 是 compact CNN baseline

### 5.4 `Riemann-TSLR`

代码入口：

- `build_riemann_tslr_pipeline`
- `fit_riemann_tslr`
- `riemann_parameter_count`

这部分实现了论文中的 compact geometric baseline：

1. `Covariances(estimator="oas")`
2. `TangentSpace(metric="riemann")`
3. `StandardScaler()`
4. `LogisticRegression`

注意这里的 `StandardScaler` 是在 pipeline 里 fit 在训练数据上，因此论文里 “train-only feature standardization” 的说法是有代码支撑的。

---

## 6. `Dataset and Setup` 对应实现

### 6.1 数据集读取

主数据集：

- [D:/作业/lnn论文1/scripts/run_mi_experiments.py](D:/作业/lnn论文1/scripts/run_mi_experiments.py)
  - `load_subject_data`

session-wise 数据加载：

- [D:/作业/lnn论文1/scripts/run_sessionwise_mi_comparison.py](D:/作业/lnn论文1/scripts/run_sessionwise_mi_comparison.py)
  - `load_subject_session_data`

区别：

- `load_subject_data` 用于 pooled 结果，不显式保留 session label
- `load_subject_session_data` 会把 `session` 一起缓存下来，用于 `session 1 -> train / session 2 -> test`

### 6.2 MOABB 窗口与滤波

实现位置：

- `MotorImagery(n_classes=4, fmin=8, fmax=30, tmin=0.0, tmax=4.0)`

论文里写“2–6 s post-cue window, implemented as 0–4 s relative to MI onset in MOABB”，代码上就是这里：

- `tmin=0.0, tmax=4.0`
- `fmin=8, fmax=30`

也就是说：

- 论文的 2–6 s 是相对于 cue 的表述
- 代码的 0–4 s 是相对于 MI onset 的表述

二者是一致的。

### 6.3 下采样

实现位置：

- `downsample_trials`

论文里所有 `125 Hz` 结果都依赖这里：

- 原数据 250 Hz
- `factor = 2`
- 输出 125 Hz

### 6.4 训练集统计量标准化

实现位置：

- `compute_standardizer`
- `apply_standardizer`

逻辑：

1. 只用训练数据计算均值和方差
2. 统计维度是 `trial × time`
3. 保留 channel 维度，也就是 per-channel 标准化
4. 标准化后裁剪到 `[-6, 6]`

这正对应论文里：

- training-only standardization
- per-channel mean/std
- clipping to `[-6,6]`

### 6.5 数据划分

#### pooled 5-fold

实现位置：

- `run_experiment`
- `StratifiedKFold`
- 内层 `StratifiedShuffleSplit`

#### session-wise

实现位置：

- `run_sessionwise`

逻辑：

- session `0train` 作为训练池
- session `1test` 作为测试集
- 再从训练池里切 validation

#### grouped-pooled

实现位置：

- [D:/作业/lnn论文1/scripts/run_grouped_pooled_control.py](D:/作业/lnn论文1/scripts/run_grouped_pooled_control.py)

这个脚本不是重新定义模型，而是重新定义 outer split：

- group = `session + run`
- 每个被试 `2 sessions × 6 runs = 12 groups`

这正对应论文中“each subject contributes 12 groups”。

---

## 7. `Classification Results` 对应实现

### 7.1 pooled 主结果

主脚本：

- [D:/作业/lnn论文1/scripts/run_mi_experiments.py](D:/作业/lnn论文1/scripts/run_mi_experiments.py)

主输出目录：

- [D:/作业/lnn论文1/outputs/bspc_pooled](D:/作业/lnn论文1/outputs/bspc_pooled)

关键输出：

- `results_summary.json`
- `fold_metrics.csv`
- `subject_summary.csv`
- `stat_tests.csv`
- `predictions.csv`

论文里的 pooled 主表并不是直接从 `fold_metrics.csv` 手写的，而是经由：

- `subject_summary.csv` 先对每个 subject 聚合
- `results_summary.json` 保留均值/std
- [D:/作业/lnn论文1/scripts/export_reproducibility_artifacts.py](D:/作业/lnn论文1/scripts/export_reproducibility_artifacts.py)
  再导出成 [D:/作业/lnn论文1/outputs/paper_ready/main_table.csv](D:/作业/lnn论文1/outputs/paper_ready/main_table.csv)

### 7.2 session-wise 主结果

主脚本：

- [D:/作业/lnn论文1/scripts/run_sessionwise_mi_comparison.py](D:/作业/lnn论文1/scripts/run_sessionwise_mi_comparison.py)

主输出目录：

- [D:/作业/lnn论文1/outputs/bspc_sessionwise](D:/作业/lnn论文1/outputs/bspc_sessionwise)

关键输出：

- `sessionwise_results_summary.json`
- `sessionwise_metrics.csv`
- `stat_tests.csv`
- `predictions.csv`
- `subject_accuracy_boxplot.pdf`

导出后的 paper-ready 表：

- [D:/作业/lnn论文1/outputs/paper_ready/sessionwise_table.csv](D:/作业/lnn论文1/outputs/paper_ready/sessionwise_table.csv)

### 7.3 grouped-pooled control

主脚本：

- [D:/作业/lnn论文1/scripts/run_grouped_pooled_control.py](D:/作业/lnn论文1/scripts/run_grouped_pooled_control.py)

主输出目录：

- [D:/作业/lnn论文1/outputs/bspc_grouped_cv](D:/作业/lnn论文1/outputs/bspc_grouped_cv)

关键输出：

- `results_summary.json`
- `subject_summary.csv`
- `stat_tests.csv`
- `grouped_fold_assignments.csv`

对应论文里的作用：

- 不是新主表
- 是用来缓解 reviewer 对 pooled trial-level CV 的质疑

### 7.4 GRU control

不是单独大脚本，而是复用主脚本：

- pooled：  
  `python scripts/run_mi_experiments.py --models gru`

- session-wise：  
  `python scripts/run_sessionwise_mi_comparison.py --models gru`

最后通过：

- [D:/作业/lnn论文1/scripts/export_reproducibility_artifacts.py](D:/作业/lnn论文1/scripts/export_reproducibility_artifacts.py)

整理成：

- [D:/作业/lnn论文1/outputs/paper_ready/recurrent_control_table.csv](D:/作业/lnn论文1/outputs/paper_ready/recurrent_control_table.csv)
- [D:/作业/lnn论文1/outputs/paper_ready/recurrent_control_stats.json](D:/作业/lnn论文1/outputs/paper_ready/recurrent_control_stats.json)

---

## 8. 统计检验对应实现

论文里所有 paired comparison、Holm correction、Wilcoxon 都来自主核心脚本：

- `paired_test`
- `holm_adjust`
- `apply_holm_correction`

位置：

- [D:/作业/lnn论文1/scripts/run_mi_experiments.py](D:/作业/lnn论文1/scripts/run_mi_experiments.py)

要注意：Holm 校正不是“全项目唯一一套”。不同导出文件按不同比较家族分别校正：

- `pooled_stats.csv` / `sessionwise_stats.csv` / `grouped_cv_stats.csv`
  - 对应完整 benchmark family
- `recurrent_control_stats.json`
  - 只对应 recurrent-only family：`{cfc_vs_gru, gru_vs_lstm, cfc_vs_lstm}`

所以同一个比较（例如 `cfc_vs_lstm`）在不同文件里的 Holm 值可能不同，这不是结果冲突，而是多重比较家族不同。

### 8.1 `paired_test`

做的事：

1. 按 `subject` 对齐两个模型
2. 做 paired `t-test`
3. 计算 `cohen_d`
4. 做 `Wilcoxon signed-rank`

所以论文里一切“paired t-tests, Wilcoxon, Cohen’s d”都来自这里。

### 8.2 `holm_adjust` 和 `apply_holm_correction`

做的事：

- 把一组原始 `p` 值做 Holm 调整
- 写回到结果字典里

论文里所有：

- `holm_p_value`
- `wilcoxon_holm_p_value`

都来自这里。

---

## 9. `tau` 分析对应实现

这部分是论文最重要的机制证据之一。

### 9.1 trial-level `tau`

实现位置：

- `collect_cfc_trial_analysis`

做的事：

1. 在 `CfC-style` session-wise 测试集上前向传播
2. 从 `aux["tau"]` 取出每个时间步、每个 hidden unit 的 `tau`
3. 计算：
   - `tau_mean`
   - `mu_power`
   - `beta_power`
   - `correct`

输出：

- `tau_trial_metrics.csv`

### 9.2 class-level / subject-level 聚合

实现位置：

- `summarize_tau_trial_analysis`

做的事：

1. 先按 `subject × class` 聚合
2. 做 Friedman test
3. 做 class-pair raw paired t-tests
4. 计算 `tau` 的 trial-level summary
5. 计算 `tau` 与 `mu/beta` power 的 Pearson 相关

输出：

- `tau_subject_class_summary.csv`
- `tau_stats.json`

### 9.3 time-resolved `tau`

实现位置：

- `summarize_tau_timecourse`

做的事：

1. 把 `tau` 按时间索引展开
2. 按 `subject × class × time` 聚合
3. 生成总体均值时间曲线
4. 另外再生成 subject-level 的 coarse window / peak 汇总，供正文时间句使用

输出：

- `tau_timecourse_summary.csv`
- `tau_timecourse_subject_level.csv`
- `tau_timecourse_by_class.pdf`
- `tau_time_window_summary.csv`

这里要区分两种口径：

- `tau_timecourse_summary.csv`
  - 是 `class × time` 的总体均值时间曲线
  - 适合画 time-resolved 图
- `tau_time_window_summary.csv`
  - 是先在 `subject × class` 内求峰值或窗口均值，再跨 subject 汇总
  - 这是论文正文里 `2.32--3.04 s` 和 `约 1.49` 那句使用的口径

### 9.4 `tau` 分布图

实现位置：

- `save_tau_trial_histogram`

输出：

- `tau_dist_placeholder.pdf`

论文里 Fig. 2 就来自这里。

---

## 10. `Structured Perturbations` 对应实现

### 10.1 band-limited noise

实现位置：

- `add_band_limited_noise`

做的事：

1. 生成白噪声
2. 对噪声做 `8–30 Hz` band-pass
3. 按每个 trial 的 signal power 缩放到指定 `SNR`

这正对应论文里：

> noise power = signal power / 10^(SNR/10)

### 10.2 channel dropout

实现位置：

- `apply_channel_dropout`

做的事：

- 对每个 trial 随机选若干通道置零

### 10.3 session-wise structured perturbation 主表

实现位置：

- `run_sessionwise`

做的事：

1. 先训练 clean model
2. 对每个 perturbation seed 生成 perturbed test set
3. 重复评估
4. 汇总成 subject-level summary
5. 做 pairwise structured tests

输出：

- `structured_perturbation_metrics.csv`
- `structured_perturbation_subject_summary.csv`
- `structured_perturbation_summary.csv`
- `structured_perturbation_stats.csv`

论文里的 Table `Structured Perturbations` 就来自这部分。

---

## 11. `Perturbation Sweep` 对应实现

主脚本：

- [D:/作业/lnn论文1/scripts/run_structured_perturbation_sweep.py](D:/作业/lnn论文1/scripts/run_structured_perturbation_sweep.py)

它不是重复主表，而是做更广的 sweep：

- `SNR = 20, 10, 5, 0 dB`
- channel dropout = `10%, 30%, 50%`

输出：

- `sweep_metrics.csv`
- `sweep_subject_summary.csv`
- `sweep_summary.csv`
- `sweep_stats.csv`
- `band_noise_accuracy_sweep.pdf`
- `channel_dropout_accuracy_sweep.pdf`

论文中关于：

- “ranking compresses at 5 dB”
- “no single robustness winner”

这些不是来自主结构化扰动表，而是来自 sweep。

---

## 12. `Temporal Shuffle` 对应实现

主脚本：

- [D:/作业/lnn论文1/scripts/run_temporal_shuffle_control.py](D:/作业/lnn论文1/scripts/run_temporal_shuffle_control.py)

这部分是后来为增强机制论证新增的补充控制。

### 12.1 它在逻辑上回答什么

它回答的问题是：

> 如果把试次内时间顺序随机打乱，模型的性能变化能否说明 `CfC-style` 真正在利用一种独特的 temporal order 信息？

### 12.2 `apply_temporal_shuffle`

做的事：

- 对每个 trial 生成一个时间维 permutation
- 同一个 permutation 同时作用于该 trial 的所有通道

为什么这样设计：

- 这样测试的是 temporal order 本身
- 而不是把 channel 之间的结构也完全打烂

### 12.3 主实验流程

`run_temporal_shuffle_control`：

1. 沿用主 session-wise 训练流程
2. 训练模型保持 clean
3. 只在 test 端做 temporal shuffle
4. 用多个 shuffle seed 评估
5. 计算 clean vs shuffle 的 accuracy drop

### 12.4 输出

- [D:/作业/lnn论文1/outputs/bspc_temporal_shuffle/temporal_shuffle_metrics.csv](D:/作业/lnn论文1/outputs/bspc_temporal_shuffle/temporal_shuffle_metrics.csv)
- [D:/作业/lnn论文1/outputs/bspc_temporal_shuffle/temporal_shuffle_subject_summary.csv](D:/作业/lnn论文1/outputs/bspc_temporal_shuffle/temporal_shuffle_subject_summary.csv)
- [D:/作业/lnn论文1/outputs/bspc_temporal_shuffle/temporal_shuffle_summary.csv](D:/作业/lnn论文1/outputs/bspc_temporal_shuffle/temporal_shuffle_summary.csv)
- [D:/作业/lnn论文1/outputs/bspc_temporal_shuffle/temporal_shuffle_stats.csv](D:/作业/lnn论文1/outputs/bspc_temporal_shuffle/temporal_shuffle_stats.csv)
- [D:/作业/lnn论文1/outputs/bspc_temporal_shuffle/temporal_shuffle_drop.pdf](D:/作业/lnn论文1/outputs/bspc_temporal_shuffle/temporal_shuffle_drop.pdf)

论文里关于：

> randomizing within-trial order ... does not create any distinct CfC-style advantage

就来自这部分。

---

## 13. `Seed Variability` 对应实现

### 13.1 clean repeat runner

脚本：

- [D:/作业/lnn论文1/scripts/run_sessionwise_clean_subset.py](D:/作业/lnn论文1/scripts/run_sessionwise_clean_subset.py)

它做的不是主实验，而是：

- 在相同 session-wise 协议下
- 用 repeat seeds 重跑一个代表性模型子集

关键设计点：

- 同一个 repeat seed 下
- 同一个 subject 下
- 所有模型共享相同 split/train seed

这样 reviewer 才不会说“模型比较混入了 seed 偏差”。

### 13.2 汇总脚本

脚本：

- [D:/作业/lnn论文1/scripts/summarize_seed_variability.py](D:/作业/lnn论文1/scripts/summarize_seed_variability.py)

输出：

- `sessionwise_seed_model_summary.csv`
- `sessionwise_seed_variability_summary.csv`
- `sessionwise_seed_rankings.csv`
- `sessionwise_seed_variability_summary.json`

论文里关于：

- repeat seeds `42/43`
- 排序保持一致
- seed-level std 很小

都来自这里。

---

## 14. `BNCI2014-004` 辅助 sanity check 对应实现

主脚本：

- [D:/作业/lnn论文1/scripts/run_bnci2014_004_aux.py](D:/作业/lnn论文1/scripts/run_bnci2014_004_aux.py)

这部分的目的不是扩展主结论，而是：

> 看主边界结论在另一个更小的 binary MI 数据集上是否出现明显反例

输出：

- `aux_metrics.csv`
- `stat_tests.csv`
- `results_summary.json`

最后导出成：

- `bnci2014_004_aux_summary.csv`
- `bnci2014_004_aux_stats.csv`
- `bnci2014_004_results_summary.json`

论文里对它的定位非常保守：

- supporting-only
- sanity check
- 不改变 main boundary claim

代码和论文口径是一致的。

---

## 15. `Efficiency` 对应实现

主脚本：

- [D:/作业/lnn论文1/scripts/benchmark_model_efficiency.py](D:/作业/lnn论文1/scripts/benchmark_model_efficiency.py)

它做三类事情：

1. GPU batch size 64 forward / train step
2. GPU batch size 1 近似 per-trial latency
3. CPU 上 `Riemann-TSLR` 推理延迟

输出：

- `benchmark.csv`
- `benchmark.json`

论文里关于：

- `eager-mode FP32`
- `batch size 64`
- `batch size 1`
- `CfC` 的 per-step Python loop 开销

都是围绕这份结果写的。

---

## 16. `paper_ready` 和 `supporting_materials` 是怎么来的

主脚本：

- [D:/作业/lnn论文1/scripts/export_reproducibility_artifacts.py](D:/作业/lnn论文1/scripts/export_reproducibility_artifacts.py)

这个脚本相当于投稿整理器，不做新实验，只做：

1. 读 `outputs/bspc_*`
2. 生成 paper-ready 表格
3. 复制图、CSV、JSON、脚本、稿件
4. 形成 supporting package

### 16.1 `paper_ready`

目录：

- [D:/作业/lnn论文1/outputs/paper_ready](D:/作业/lnn论文1/outputs/paper_ready)

这里是论文最直接引用的中间产物：

- `main_table.csv`
- `sessionwise_table.csv`
- `grouped_cv_table.csv`
- `structured_perturbation_table.csv`
- `temporal_shuffle_summary.csv`
- `pooled_stats.csv`
- `sessionwise_stats.csv`
- `grouped_cv_stats.csv`
- `key_stats.json`

### 16.2 `supporting_materials`

目录：

- [D:/作业/lnn论文1/supporting_materials](D:/作业/lnn论文1/supporting_materials)

这里是投稿时可作为 supplemental package 的整理版本，分成：

- `paper_tables/`
- `subject_results/`
- `tau_analysis/`
- `robustness/`
- `efficiency/`
- `reproducibility/`
- `scripts/`
- `manuscript/`

也就是说，论文正文里用到的结论并不是散落在几十个目录，而是最终都被汇入了这两个统一出口。

---

## 17. 每个论文表/图，具体来自哪里

### 17.1 pooled 主表

论文位置：

- `Classification Results` 中的 pooled table

直接来源：

- [D:/作业/lnn论文1/outputs/paper_ready/main_table.csv](D:/作业/lnn论文1/outputs/paper_ready/main_table.csv)

上游来源：

- `outputs/bspc_pooled/results_summary.json`

### 17.2 session-wise 主表

直接来源：

- [D:/作业/lnn论文1/outputs/paper_ready/sessionwise_table.csv](D:/作业/lnn论文1/outputs/paper_ready/sessionwise_table.csv)

### 17.3 grouped pooled control

直接来源：

- [D:/作业/lnn论文1/outputs/paper_ready/grouped_cv_table.csv](D:/作业/lnn论文1/outputs/paper_ready/grouped_cv_table.csv)

### 17.4 structured perturbation table

直接来源：

- [D:/作业/lnn论文1/outputs/paper_ready/structured_perturbation_table.csv](D:/作业/lnn论文1/outputs/paper_ready/structured_perturbation_table.csv)

### 17.5 `tau` 图

直接来源：

- [D:/作业/lnn论文1/outputs/bspc_sessionwise/tau_dist_placeholder.pdf](D:/作业/lnn论文1/outputs/bspc_sessionwise/tau_dist_placeholder.pdf)

### 17.6 temporal shuffle 支撑句

直接来源：

- [D:/作业/lnn论文1/outputs/paper_ready/temporal_shuffle_summary.csv](D:/作业/lnn论文1/outputs/paper_ready/temporal_shuffle_summary.csv)
- [D:/作业/lnn论文1/outputs/paper_ready/temporal_shuffle_stats.csv](D:/作业/lnn论文1/outputs/paper_ready/temporal_shuffle_stats.csv)

### 17.7 efficiency 段

直接来源：

- [D:/作业/lnn论文1/outputs/bspc_efficiency/benchmark.csv](D:/作业/lnn论文1/outputs/bspc_efficiency/benchmark.csv)

注意：

- 当前论文对的是 `outputs/bspc_efficiency/benchmark.csv`
- 不是旧的 `outputs/efficiency/benchmark.csv`

---

## 18. 论文没有做、代码里也没有做什么

为了避免误读，这里明确列出当前论文没有实现的内容：

- 没有做 `LOSO / cross-subject`
- 没有做 canonical CfC 全变体对照
- 没有做 `ODE/CDE/SSM` 实验基线
- 没有做 cropped training
- 没有做强 augmentation
- 没有做 `torch.compile / vectorized scan / JIT` 的效率重写
- 没有把 `tau` 做成 electrode-specific topography
- 没有把 `tau` 解释成真实秒级 physiological time constant

因此论文里的相关说法都应该理解为：

- 当前实现下
- 当前协议下
- 当前数据集边界内

---

## 19. 如果你要重跑整篇论文，推荐顺序

### 19.1 主结果

1. pooled  
   [D:/作业/lnn论文1/scripts/run_mi_experiments.py](D:/作业/lnn论文1/scripts/run_mi_experiments.py)

2. session-wise  
   [D:/作业/lnn论文1/scripts/run_sessionwise_mi_comparison.py](D:/作业/lnn论文1/scripts/run_sessionwise_mi_comparison.py)

3. grouped pooled  
   [D:/作业/lnn论文1/scripts/run_grouped_pooled_control.py](D:/作业/lnn论文1/scripts/run_grouped_pooled_control.py)

### 19.2 补充控制

4. GRU control  
   复用 pooled / session-wise 脚本，`--models gru`

5. perturbation sweep  
   [D:/作业/lnn论文1/scripts/run_structured_perturbation_sweep.py](D:/作业/lnn论文1/scripts/run_structured_perturbation_sweep.py)

6. temporal shuffle  
   [D:/作业/lnn论文1/scripts/run_temporal_shuffle_control.py](D:/作业/lnn论文1/scripts/run_temporal_shuffle_control.py)

7. seed repeat  
   [D:/作业/lnn论文1/scripts/run_sessionwise_clean_subset.py](D:/作业/lnn论文1/scripts/run_sessionwise_clean_subset.py)
   + [D:/作业/lnn论文1/scripts/summarize_seed_variability.py](D:/作业/lnn论文1/scripts/summarize_seed_variability.py)

8. auxiliary BNCI2014-004  
   [D:/作业/lnn论文1/scripts/run_bnci2014_004_aux.py](D:/作业/lnn论文1/scripts/run_bnci2014_004_aux.py)

9. efficiency  
   [D:/作业/lnn论文1/scripts/benchmark_model_efficiency.py](D:/作业/lnn论文1/scripts/benchmark_model_efficiency.py)

10. 导出  
   [D:/作业/lnn论文1/scripts/export_reproducibility_artifacts.py](D:/作业/lnn论文1/scripts/export_reproducibility_artifacts.py)

---

## 20. 一句话总括：代码如何支撑论文

如果只用一句话概括整套实现：

> `run_mi_experiments.py` 提供模型、训练和统计核心；`run_sessionwise_mi_comparison.py` 提供最重要的严格协议与 `tau` 机制证据；其他脚本负责把“这个结论是不是稳定、是不是偶然、是不是被某个协议或seed撑出来的”逐层排除掉，最后由 `export_reproducibility_artifacts.py` 把这些证据整理成论文和补充材料可直接引用的形式。

---

## 21. 你接下来最可能会用到这个文档的场景

### 场景 A：改论文

你可以直接用本文档查：

- 某个结论来自哪个 CSV / JSON
- 某张图是谁画的
- 某个 reviewer 问题要去改哪个脚本

### 场景 B：答辩 / 回复审稿人

你可以直接按本文档说：

- 主结果来自哪个协议
- 统计怎么做的
- `tau` 怎么聚合的
- temporal shuffle 的设计为什么合理

### 场景 C：后续扩展

如果以后你要加：

- LOSO
- 新 SSM baseline
- 更强 hybrid
- electrode-specific `tau`

这份文档可以帮你定位最合适的入口脚本。
