# AdaDec3D Paper Narrative and Candidate Venues

> 本文档只讨论论文叙事与候选会议，不包含实验时间线。  
> 当前重点是 **Paper A：Observation Study**；AdaDec3D 方法作为后续 Paper B。

---

## 1. 两篇论文的总体关系

### Paper A：Observation Paper

**研究问题**

> Where does additional decoder computation actually improve 3D medical image segmentation?

Paper A 不提出 AdaDec3D，也不声称已经实现了动态计算节省。它研究的是一个更基础的问题：

> 在高效 3D 分割模型中，更强 decoder 带来的额外收益是否在空间上高度集中，以及这种收益能否被低成本、测试时可用的信号预测？

Paper A 的结果应当独立成立。即使后续 AdaDec3D 没有达到预期，这篇分析论文仍应提供：

- decoder 计算收益的空间分布；
- 边界、小器官和困难区域对额外 decoder capacity 的敏感性；
- 不确定性等低成本信号对“额外计算是否有用”的预测能力；
- 在不同数据集、backbone 和训练阶段上的稳定性；
- 面向 adaptive decoder 研究的评估协议和设计启示。

### Paper B：AdaDec3D Method Paper

Paper B 承接 Paper A 的结论：

> Since additional decoder computation is useful only in a predictable subset of regions, AdaDec3D conditionally allocates decoder capacity to those regions.

Paper B 必须进一步证明两件 Paper A 没有证明的事：

1. **Realizability**：机会分析中观察到的收益能够由可执行的条件 decoder 实现；
2. **Efficiency**：在包含 routing、ROI halo 和重复 tile 计算后，真实执行计算量、显存与延迟确实下降。

因此两篇论文的边界是：

| 问题 | Paper A | Paper B |
|---|---:|---:|
| decoder 收益是否空间异质 | ✓ | 作为前提引用 |
| 收益是否可预测 | ✓ | 作为 router 依据 |
| oracle selective allocation 是否存在 | ✓ | — |
| 可训练的 adaptive decoder | — | ✓ |
| 实际 executed MACs / latency 节省 | — | ✓ |
| accuracy–efficiency Pareto improvement | — | ✓ |

---

## 2. Paper A 的推荐标题

### 首选

**Where Does Decoder Computation Matter? A Marginal-Utility Study of Efficient 3D Medical Image Segmentation**

这个标题明确包含：

- 研究对象：decoder computation；
- 主要问题：where it matters；
- 方法性质：marginal-utility study；
- 应用范围：efficient 3D medical image segmentation。

### 备选

1. **Rethinking Decoder Redundancy in Efficient 3D Medical Image Segmentation**
2. **The Spatially Concentrated Utility of Decoder Computation in 3D Medical Image Segmentation**
3. **Understanding Where Additional Decoder Capacity Helps in Efficient 3D Medical Image Segmentation**
4. **A Small Fraction of Voxels Accounts for Most Decoder Gains in 3D Medical Image Segmentation**

第 4 个标题只有在 O9 得到足够强、跨 seed 稳定的 Pareto 结果后才能使用，不能预先假设具体比例。

---

## 3. 核心概念：Decoder Marginal Utility

论文不能只讨论“哪里容易错”或“哪里 entropy 高”。这些问题已有大量研究，单独不足以形成强贡献。

核心分析对象应定义为额外 decoder capacity 的逐 voxel 边际效用：

$$
\Delta(v) =
\ell_{\mathrm{Effi}}(v) -
\ell_{\mathrm{Full}}(v),
$$

其中：

- $\ell_{\mathrm{Effi}}(v)$ 是轻量 EffiDec3D 在 voxel $v$ 的损失；
- $\ell_{\mathrm{Full}}(v)$ 是 matched full decoder 在同一 voxel 的损失；
- $\Delta(v)>0$ 表示额外 decoder capacity 有益；
- $\Delta(v)<0$ 表示 full decoder 反而使该 voxel 变差；
- $\Delta(v)\approx0$ 表示额外计算没有实质收益。

同时报告离散标签转移：

- **Positive gain**：EffiDec3D 错、Full decoder 对；
- **Negative gain**：EffiDec3D 对、Full decoder 错；
- **Net gain**：positive gain − negative gain。

这样可以避免把任何 prediction disagreement 都错误解释成“full decoder 带来改善”。

### 关键区分

论文需要依次证明四层不同的 claim：

1. **Heterogeneity**：decoder gain 在 subject、organ 或 region 上并不均匀；
2. **Concentration**：少量区域贡献了大部分正向或净收益；
3. **Predictability**：测试时可用信号可以识别这些区域；
4. **Generalizability**：该现象不是单一数据集、backbone 或 checkpoint 的偶然结果。

Paper A 不声称第五层 **realizable efficiency**；这一层属于 AdaDec3D Paper。

---

## 4. 一句话 Thesis 与贡献

### Central thesis

> The marginal utility of decoder capacity is spatially concentrated and predictable: most additional gains from a full decoder arise from a small subset of anatomically difficult regions, while the remaining voxels receive little or no benefit.

如果实验只能证明 gain 集中，却不能证明部署时信号可预测它，应将 thesis 缩减为：

> The marginal utility of decoder capacity is strongly heterogeneous across spatial regions and anatomical structures.

### 推荐贡献写法

1. **A marginal-utility formulation of decoder computation**  
   不再仅比较整幅图像的 Dice/FLOPs，而是测量额外 decoder capacity 对 subject、organ、boundary 和 voxel region 的正向、负向及净收益。

2. **A systematic characterization of where decoder computation helps**  
   分析收益集中度、边界距离、器官大小、训练演化，并明确区分 segmentation difficulty 与 decoder-specific utility。

3. **A selective-allocation opportunity analysis**  
   绘制“被分配额外计算的 voxel/region 比例”与“恢复的 decoder gain 比例”之间的 Pareto 曲线，并与 random、boundary、confidence 等 controls 比较。

4. **Evidence for consistency and predictability**  
   使用 subject-level confidence interval、多个 seed、跨数据集和跨 backbone 实验，检验该现象是否稳定以及低成本信号能否预测 held-out marginal utility。

不要把“首次”“90% FLOPs”“只需 5% voxels”等措辞写进贡献，除非实验和文献检索都能支持。

---

## 5. Narrative 主线

### 5.1 Introduction 的五段结构

#### Paragraph 1 — 背景

3D medical segmentation 的 decoder 在多尺度特征融合和空间细节恢复中非常重要，但高分辨率 3D decoding 计算和显存成本高。

#### Paragraph 2 — 已有方法及其隐含假设

EffiDec3D 等方法通过统一通道压缩和分辨率限制显著降低 decoder 开销。这类 static efficient decoder 对所有 subject、organ 和 spatial location 分配相同计算预算，隐含假设是 decoder capacity 的效用在空间上大致均匀。

#### Paragraph 3 — 研究缺口

现有工作通常只报告整例 Dice、参数量和 FLOPs，因此无法回答：

> 哪些区域真正从额外 decoder 计算中获益？这些收益是否足够集中，并能否在不知道 ground truth 的测试阶段被预测？

“哪里容易错”不等于“哪里值得增加 decoder 计算”。困难 voxel 可能无法被更强 decoder 修复，而低 entropy voxel 也可能是 confidently wrong。

#### Paragraph 4 — 本文做什么

构造 matched lightweight/full decoder comparison，在相同数据和尽可能受控的 encoder 条件下定义 voxel-level marginal decoder utility。围绕空间集中度、解剖分布、训练演化和可预测性进行系统分析，并用 subject-level bootstrap 和跨域实验检验稳健性。

#### Paragraph 5 — 主要发现

最终按真实结果填写：

- `[X]%` 的候选区域解释了 `[Y]%` 的 positive/net decoder gain；
- gain 在 boundary 和 small/difficult organs 上更集中；
- entropy 或其他 deployable signal 在 held-out subjects 上优于 matched random/boundary controls；
- 这一趋势在 `[datasets]` 和 `[backbones]` 上保持一致/部分一致。

此处只报告事实，不引入 AdaDec3D 结构。最后一句可写：

> These findings motivate region-adaptive decoder allocation as a promising direction, while the design of such a model is left to future work.

### 5.2 Related Work 的组织

只保留三个小节：

1. **Efficient decoders for 3D medical segmentation**
2. **Uncertainty and difficulty estimation in medical segmentation**
3. **Conditional computation and adaptive inference**

Related Work 的落点应是：

- efficient decoder 研究“怎样整体缩小 decoder”；
- uncertainty 研究“哪里可能预测错误”；
- conditional computation 研究“怎样动态执行”；
- 本文研究的是两者之间尚未被充分回答的问题：**额外 decoder capacity 在哪里具有正边际效用**。

### 5.3 Study Design 的组织

不要按 O1–O11 写成十一组互相独立的小实验。正文合并成三个 scientific findings：

#### Finding I — Decoder utility is spatially concentrated

对应 O1、O2、O5、O9：

- error 与 entropy 的空间稀疏性；
- positive/negative/net decoder gain；
- gain concentration curve；
- oracle、deployable signal、random control 的 selective-allocation curve。

#### Finding II — Decoder utility has anatomical structure

对应 O3、O4、O10：

- uncertainty 与 error 的关系；
- boundary distance 分析；
- organ-wise gain；
- organ volume 与 difficulty/gain 的相关或偏相关分析。

重点不是简单得出“小器官更难”，而是回答：

> 在控制器官大小或 boundary proportion 后，routing signal 是否仍能解释 decoder gain？

#### Finding III — Concentration is stable and predictable

对应 O6、O7、O8、O11：

- 随训练 checkpoint 的变化；
- CT → MRI 跨数据集；
- UX-Net → SwinUNETR/MedNeXt 跨 backbone；
- entropy、margin、MC-dropout、boundary proxy 等 signal comparison。

O11 如果涉及训练 router 或方法组件，应移到 Paper B；Paper A 只比较无需 AdaDec3D 训练即可计算的 signals。

---

## 6. 必须出现的主图与主表

### Figure 1 — Motivation and definition

同一病例展示：

1. CT/MRI image；
2. ground truth；
3. EffiDec3D prediction；
4. full decoder prediction；
5. positive/negative decoder gain map；
6. entropy 或最佳 deployable signal。

图注必须强调 error map、uncertainty map 和 decoder gain map 是三个不同概念。

### Figure 2 — 全文核心 Pareto 图

横轴：

> Fraction of regions receiving additional decoder computation

纵轴：

> Fraction of total positive/net decoder gain recovered

曲线至少包括：

- oracle gain ranking；
- entropy 或最佳 deployable signal；
- confidence/margin；
- boundary proxy；
- random allocation（均值和 95% CI）。

这是最能支撑标题和 central thesis 的图。

### Figure 3 — Anatomical analysis

- per-organ decoder gain；
- boundary vs interior；
- organ size vs difficulty；
- organ size vs decoder gain；
- 必要时报告 partial correlation。

### Figure 4 — Generalization

使用统一坐标展示：

- BTCV / FeTA；
- UX-Net / SwinUNETR 或 MedNeXt；
- 不同训练 checkpoint；
- 多个随机 seed。

### Table 1 — Baselines

报告：

- Dice、HD95；
- parameters、static MACs/GFLOPs；
- latency、peak memory；
- seed-level mean ± std；
- full 与 efficient decoder 的训练协议是否完全匹配。

### Table 2 — Predictability

报告每个 deployable signal 对 held-out marginal utility 的：

- Pearson/Spearman correlation；
- AUROC/AUPRC（若将 positive gain 定义为二分类）；
- top-k gain recovery；
- 相对 matched random control 的提升；
- subject-level 95% confidence interval。

---

## 7. 实验结果如何决定文章强度

### Strong narrative

如果满足：

- gain concentration 明显；
- deployable signal 显著优于 controls；
- 跨 seed、backbone 或 dataset 稳定；

则可以主张：

> decoder capacity is over-provisioned for most regions, and its useful computation can be localized before executing the stronger decoder.

### Medium narrative

如果 gain 集中，但 entropy 等信号预测一般：

> decoder computation has strongly heterogeneous utility, but identifying beneficial regions remains an open problem.

这仍是有效 observation paper，但不能把它包装成 AdaDec3D 的直接可行性证明。

### Negative but publishable narrative

如果 entropy 只预测 error、不预测 decoder gain：

> segmentation uncertainty is not a reliable proxy for the marginal utility of decoder computation.

这会推翻当前 AdaDec3D 的 router 假设，但若 controls 和跨 backbone 实验完整，仍可作为有价值的 negative result / evaluation paper，尤其适合 WACV E&D Track。

### No-paper condition

若 Full 与 EffiDec3D 的差异：

- 不稳定且 seed variance 大于 decoder gain；
- 无法在 matched setting 中复现；
- 只在单一模型、单一器官或极少病例出现；

则不应强行声称普遍 decoder redundancy，应先修改实验控制或缩小 claim。

---

## 8. 常见叙事错误

1. **把 error 当作 decoder gain**  
   高误差区域不一定能被更强 decoder 修复。

2. **只报告 positive transitions**  
   必须同时报告 negative transitions 和 net gain。

3. **把 oracle hybrid prediction 当作真实加速**  
   Paper A 的 selective allocation 是 opportunity analysis，不是 executed FLOPs reduction。

4. **比较两个完全独立训练模型后声称差异来自 decoder**  
   主实验应尽量共享 encoder、初始化或训练协议；独立 end-to-end 模型仅作为生态有效性验证。

5. **将全部 voxel 当作独立样本计算显著性**  
   confidence interval 和 hypothesis test 必须以 subject 为重采样单位。

6. **在同一 validation set 上选择阈值并报告最终性能**  
   signal/预算选择和 confirmatory evaluation 需要 subject-level split、nested CV 或独立数据集。

7. **用未来式的预期数值写结论**  
   比例、相关系数与 Pareto 优势都应在实验完成后填入。

8. **在 Paper A 详细介绍 AdaDec3D**  
   最多在 Discussion 中说明研究启示，避免让 reviewer 认为论文缺少一个尚未实现的方法。

---

## 9. 候选会议

以下信息按 **2026-07-25** 可见的官方页面整理。CCF 等级应以学校采用的最新版目录和“regular/full paper 是否计入”规则再次确认；EI/Scopus 收录也应在录用和出版前复核，不能只依赖往届记录。

| 优先级 | 会议 | 投稿截止 | 举办地点 | 属性 | Narrative 匹配度 | 主要注意点 |
|---:|---|---|---|---|---|---|
| 1 | **WACV 2027 Round 2, E&D Track** | enrollment 2026-08-21；paper 2026-08-28 | Orlando, USA | IEEE/CVF；CCF 属性须单独核实 | 很高 | E&D Track 明确接受 rigorous reproduction、failure analysis、critical analysis 和 evaluation methodology；竞争强 |
| 2 | **IEEE ICASSP 2027** | 2026-09-16 | Toronto, Canada | 通常按 CCF-B；IEEE Xplore | 高 | Biomedical Imaging、Computational Imaging 在 scope 内；篇幅紧，需要保留最强一条结论 |
| 3 | **ICBBE 2026** | 2026-08-10 | Hokkaido, Japan | 非 CCF；官网称 ACM DL，并计划 EI Compendex/Scopus | 高（主题） | biomedical 主题直接；会议层级和认可度低于前两项，索引表述来自主办方，应复核最终 proceedings |
| 4 | **MMM 2027** | 2026-08-16 | Siem Reap, Cambodia | 通常按 CCF-C；Springer/LNCS 系列惯例需以本届 CFP 为准 | 中低 | healthcare in scope，但核心社区偏 multimedia modelling，3D medical decoder observation 不是最自然的受众 |

### 9.1 WACV 2027 Evaluation & Datasets Track

**推荐定位**

> A rigorous evaluation of where efficient 3D segmentation decoders fail and where additional decoder capacity provides measurable utility.

WACV 2027 的 E&D Track 明确包括：

- existing evaluation 的 limitation/failure-mode analysis；
- rigorous reproduction、auditing 和 stress testing；
- negative results 和 critical analyses；
- evaluation protocols 和 systematic analyses。

这与 observation paper 的体裁高度一致。Round 2 的 enrollment 为 **2026-08-21**，paper deadline 为 **2026-08-28 AoE**：

- [WACV 2027 Call for Papers](https://wacv.thecvf.com/Conferences/2027/CallForPapers)
- [WACV 2027 Dates](https://wacv.thecvf.com/Conferences/2027/Dates)

**注意**：应投 E&D Track，而不是强行包装成 Algorithms Track。CCF 收录情况存在不同版本和口径差异，应由学校或项目认定部门确认。

### 9.2 ICASSP 2027

**推荐定位**

> Predicting the marginal utility of decoder computation in volumetric biomedical image segmentation.

官方 CFP 包含 Biomedical Imaging & Signal Processing、Computational Imaging、Computer Vision 和 Image/Video/Multidimensional Signal Processing，full paper deadline 为 **2026-09-16**，会议于 **2027-05-16 至 05-21** 在 Toronto 举行：

- [ICASSP 2027 Call for Papers](https://2027.ieeeicassp.org/call-for-papers/)

**综合判断**：学术认可度高于 ICBBE/MMM，但短篇格式要求叙事非常聚焦。建议只保留：

- marginal utility 定义；
- gain concentration/Pareto 主图；
- deployable signal vs controls；
- 一项跨 backbone 或跨 dataset 验证。

### 9.3 ICBBE 2026

**推荐定位**

> Understanding the spatial utility of decoder computation for efficient 3D biomedical image segmentation.

ICBBE 官网列出：

- submission deadline：**2026-08-10**；
- conference：**2026-12-25 至 12-28**；
- location：Hokkaido, Japan；
- 2026 accepted papers 计划由 ACM conference proceedings 出版、归档 ACM DL，并送 EI Compendex 和 Scopus；
- 官网称 2016–2025 往届论文已采用 ACM proceedings 并被 EI Compendex/Scopus 收录。

来源：

- [ICBBE 2026 Official Website](https://www.icbbe.com/)
- [ICBBE 2026 Important Dates](https://www.icbbe.com/date.html)

**综合判断**：主题匹配、亚洲举行、适合作为 EI-oriented biomedical engineering 投稿，但它不是 CCF 会议。文档中应写“计划送检/官网声明”，不要承诺最终一定被 EI/Scopus 收录。

### 9.4 MMM 2027

**推荐定位**

如果投稿 MMM，需要把工作放在：

> efficient modelling and resource-aware analysis of volumetric medical visual content

而不是只强调医学器官分割。其主题匹配弱于 WACV/ICASSP/ICBBE，因此只建议作为后备。

本届 CFP 信息需在投稿前从 [MMM 2027 official site](https://mmm2027.net/) 再次核实。

---

## 10. 投稿选择建议

### 若优先考虑 CCF 与亚洲地点

在已确认 2026 年截止的会议中，**MMM 2027** 满足亚洲地点和通常采用的 CCF-C
口径，但主题匹配较弱。

### 若优先考虑 paper narrative 的天然匹配

**WACV 2027 E&D Track** 最合适，因为它明确接受 evaluation、reproduction、failure analysis 和 negative results；但应先确认本单位对其 CCF/成果类别的认定。

### 若优先考虑学术认可度

**ICASSP 2027** 值得优先于 ICBBE，但需要将完整 observation study 压缩成一条非常集中的 signal-processing 结论。

### 若优先考虑医学主题与 EI-oriented 发表

**ICBBE 2026** 可以作为候选，但应把它视作主题匹配的工程/生物医学会议，而不是 CCF 替代品。

### 推荐总体排序

综合 narrative fit、学术认可度和用户提出的地区/检索要求：

1. **WACV 2027 E&D Track**
2. **ICASSP 2027**
3. **ICBBE 2026**
4. **MMM 2027**

同一篇 archival paper 不得同时投稿多个会议。应选择一个主投 venue，再根据评审结果重写和转投，而不是并行提交。

---

## 11. 2026 年 10–12 月投稿窗口

### 11.1 结论

截至 **2026-07-25**，没有发现一个同时满足以下全部条件的已确认会议：

- 2026 年 10–12 月提交 regular/full paper；
- CCF-C 或以上；
- 在亚洲或北美举行；
- 与 3D medical image segmentation observation study 高度匹配。

不应为了满足月份而投向主题明显不符的 data mining、multimedia、industrial
electronics 或通用 AI 会议。对 observation paper 来说，reviewer community
是否理解 medical imaging evaluation，比单纯“在 CCF 目录中”更重要。

### 11.2 首选：IEEE ISBI 2027

**IEEE International Symposium on Biomedical Imaging (ISBI 2027)**

| 项目 | 信息 |
|---|---|
| Four-page paper deadline | **2026-10-26** |
| Conference location | Lausanne, Switzerland |
| Research fit | **很高** |
| Publication | IEEE conference proceedings；最终 IEEE Xplore/EI 状态应按本届出版信息复核 |
| CCF | 不作为 CCF-C 会议处理 |

ISBI 专注 biological and medical imaging 的数学、算法和计算问题，是当前
10–12 月窗口中与 Paper A 最匹配且来源可靠的会议：

- [ISBI 2027 Official Website and Important Dates](https://biomedicalimaging.org/2027/)

**适合的 narrative**

> Where does additional decoder computation improve volumetric biomedical image segmentation?

由于主论文只有四页，正文建议只保留：

1. matched Full/EffiDec comparison；
2. marginal decoder utility 定义；
3. positive、negative、net gain；
4. selective-allocation Pareto curve；
5. entropy/boundary/random controls；
6. 一项跨 backbone 或跨 dataset 验证。

O1–O11 的完整结果放入 supplementary material。ISBI 的缺点是举办地不在亚洲或
北美，但从主题匹配看，它明显优于为了 CCF 标签而选择的通用会议。

### 11.3 可观察但暂不作为确定目标：MIDL 2027

MIDL 与 observation-driven medical imaging work 高度匹配，往届 full-paper
deadline 通常在 12 月附近；但截至本文核查日期，MIDL 2027 官方主页只确认
**2027-07-14 至 07-16、Porto**，尚未发布可采信的 full-paper deadline：

- [MIDL 2027 Official Website](https://2027.midl.io/)

此外，MIDL 不应被写成 CCF-C 或 EI/SCI 会议。它的领域认可度很高，但不满足
用户当前的形式化检索条件，因此只能作为学术匹配型候选，等待正式 CFP。

### 11.4 不建议作为主投的 10–12 月会议

以下类型不纳入主清单：

- 只写 “submitted to EI/Scopus” 且没有稳定往届出版记录的新会议；
- 官网没有明确 proceedings publisher 的 Intelligent Medical Imaging 类会议；
- 虽为 IEEE/EI，但主题属于 industrial electronics、instrumentation、
  consumer electronics，和 decoder observation study 关系牵强的会议；
- AISTATS、PAKDD 等通用统计/数据挖掘会议：即使 CCF 等级满足，当前论文缺少
  通用学习理论或数据挖掘方法贡献，命中率会明显低于 ISBI；
- 仅收 abstract、poster 或 workshop paper 的渠道，因为这类成果未必计入
  CCF full/regular paper，也未必形成稳定 EI proceedings。

### 11.5 如果必须在 10–12 月提交并要求 SCI

SCI/SCIE 通常指期刊，不是会议。常规期刊全年滚动投稿，因此可以在 10–12 月
完成后直接提交，不需要等待 CFP。

| 期刊 | 匹配度 | 更适合的文章版本 | 注意点 |
|---|---:|---|---|
| **Computerized Medical Imaging and Graphics (CMIG)** | 高 | 深入的跨数据集/跨 backbone observation study | 需要突出 medical-imaging insight 和 evaluation methodology，而不只是 EffiDec3D 个案 |
| **Computer Methods and Programs in Biomedicine (CMPB)** | 高 | 完整实验协议、统计检验和可复现实验工具 | 强调可复用的方法学与软件价值 |
| **Biomedical Signal Processing and Control (BSPC)** | 中高 | 以 uncertainty、image analysis 和 efficiency 为中心 | 偏 application-led；需要明确实际 biomedical image analysis 意义 |
| **Computers in Biology and Medicine (CBM)** | 中高 | Paper A + 更完整的验证，或 Paper B 方法 | 官方 scope 明确拒绝轻微架构修改及数据划分不严谨的 segmentation paper；必须依靠强 observation insight、严格 split 和充分 SOTA |

官方 scope：

- [CMIG](https://www.sciencedirect.com/journal/computerized-medical-imaging-and-graphics)
- [CMPB](https://www.sciencedirect.com/journal/computer-methods-and-programs-in-biomedicine)
- [BSPC](https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control)
- [CBM](https://www.sciencedirect.com/journal/computers-in-biology-and-medicine)

**建议**

- 若希望在 2026 年 10 月形成短而强的会议论文：优先 **ISBI 2027**；
- 若实验能够覆盖 O1–O11、多个 seed、跨数据集和跨 backbone：不必强行压缩，
  可在 10–12 月直接投稿 **CMIG 或 CMPB**；
- 若学校硬性要求 CCF-C full paper：当前 10–12 月没有足够匹配且已确认的目标，
  应等待后续 CFP，而不是投主题不符的会议。

---

## 12. Paper A 摘要骨架

> Efficient decoders substantially reduce the cost of volumetric medical image segmentation, but existing evaluations report only case-level accuracy and aggregate computation. It therefore remains unclear where additional decoder capacity is useful and whether its benefit can be predicted before the computation is executed. We formulate the voxel-wise marginal utility of decoder capacity by comparing matched lightweight and full decoders, explicitly separating positive, negative, and net prediction transitions. Across [datasets], [backbones], and [seeds], we analyze the concentration, anatomical distribution, training evolution, and predictability of decoder gains. We find that [result 1], with [X]% of candidate regions accounting for [Y]% of net gain. These gains are concentrated in [result 2] and can be identified by [signal] significantly better than random, boundary, and confidence controls on held-out subjects [result 3]. Our findings show that decoder computation has spatially heterogeneous utility and establish an evaluation framework for future region-adaptive decoders.

在实验完成前保留方括号，不提前填写预期数值。

---

## 13. Paper B 的一句话衔接

Paper B 的开场不需要重复 Paper A 的全部观察，只需：

> Our previous analysis showed that the marginal utility of 3D decoder computation is spatially concentrated and predictable. Based on this finding, we introduce AdaDec3D, a region-adaptive decoder that routes additional capacity only to regions with predicted positive utility.

Paper B 的标题可以保留：

**AdaDec3D: Difficulty-Aware Regional Decoder Allocation for Efficient 3D Medical Image Segmentation**
