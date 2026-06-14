# AdaDec3D: Uncertainty-Guided Adaptive Decoder for Efficient 3D Medical Image Segmentation

**全称**：AdaDec3D: Uncertainty-Guided Adaptive Decoder Routing and ROI Refinement for Efficient 3D Medical Image Segmentation

**类型**：Research Proposal

**基于**：EffiDec3D（CVPR 2025）/ 3D UX-Net / SwinUNETR / SwinUNETRv2

**目标期刊/会议**：MICCAI / IEEE TMI / JBHI

**优先级**：⭐⭐⭐⭐ 高

---

## 1. 动机与问题定义

### 1.1 EffiDec3D 的现有设计

EffiDec3D（CVPR 2025）通过两个**静态**优化显著降低了 3D 分割解码器的计算开销：

| 优化策略 | 实现方式 | 效果 |
|---|---|---|
| **Channel Reduction** | 将解码器各阶段通道数统一压缩到 C_reduced（encoder 最小通道数） | −96.4% Params |
| **Resolution Restriction** | 解码最高只到 D/2×H/2×W/2，最后用双线性上采样恢复 | −93.0% FLOPs |

但这两个优化均为**训练时固定、推理时不变**的静态配置。

### 1.2 核心问题

EffiDec3D 隐含了一个根本性假设：

> **所有体素、所有器官、所有患者需要等量的解码器计算。**

这一假设在三个维度上均不成立：

**问题 1：解剖结构难度差异巨大**

EffiDec3D 消融实验（Table 2 & Table 4）已直接揭示：
- **大器官**（肝脏、脾脏、肾脏）：即便在 D/4 分辨率下，DICE 仍维持在 80% 以上
- **小结构**（食管、胰腺、血管、肾上腺）：从 D/2 降至 D/4 分辨率，DICE 下降 2–4%

二者不应占用相同的解码器资源。

**问题 2：分割不确定性高度非均匀**

绝大多数体素属于背景或简单区域，仅边界、小病灶、模糊结构区域存在真正的分割歧义。对全体素等量分配计算是显著的资源浪费。

**问题 3：分辨率决策过于粗粒度**

EffiDec3D 的 `resolution_factor` 是全局开关，无法对同一图像中的不同区域差异化处理——ROI 区域需要高分辨率，背景区域可接受低分辨率。

### 1.3 AdaDec3D 的目标

在 EffiDec3D 的架构基础上，引入**不确定性引导的自适应解码**：

```
EffiDec3D 静态设计
    ↓ 升级
AdaDec3D 动态设计

静态通道数   →  Uncertainty-conditioned MoE Decoder（多专家动态选择）
静态分辨率   →  ROI-aware Progressive Refinement（不确定区域选择性精化）
全局统一     →  Uncertainty-Guided Routing（体素级计算分配）
```

---

## 2. AdaDec3D 框架

### 2.1 整体架构

```
Input Volume
↓
Encoder (frozen from EffiDec3D baseline)
↓
EffiDec3D Coarse Decoder
↓
Coarse Prediction + Uncertainty Head
↓
Uncertainty Map (voxel-wise)
↓
Adaptive Router
├── Expert Decoder S (32ch, D/2 res)   ← 低不确定性区域
├── Expert Decoder M (64ch, D/2 res)   ← 中不确定性区域
└── Expert Decoder L (96ch, D res)     ← 高不确定性区域（边界/小结构）
↓
ROI-aware Refinement（对高不确定区域施加全分辨率精化）
↓
Final Prediction
```

三个核心创新模块：
1. **Uncertainty Estimation** — 轻量不确定性头，估计体素级分割难度
2. **Uncertainty-Guided MoE Decoder** — 多专家解码器，按不确定性路由计算资源
3. **ROI-aware Progressive Refinement** — 粗→精两阶段，仅对不确定区域施加高分辨率精化

---

## 3. 模块 1：Uncertainty Estimation

### 设计

在粗解码器输出（C 通道 softmax 概率图）上，计算体素级预测熵：

```python
def estimate_uncertainty(prob_map):
    # prob_map: [B, C, D/2, H/2, W/2]
    entropy = -torch.sum(
        prob_map * torch.log(prob_map + 1e-8), dim=1
    )  # [B, D/2, H/2, W/2]
    return entropy
```

高熵区域对应：边界、小病灶、形状不规则的结构（与 EffiDec3D 中食管、胰腺、肾上腺的低 DICE 高度一致）。

### 可选增强

对于需要 calibrated uncertainty 的场景，可用 MC Dropout 替代：

```python
def mc_dropout_uncertainty(model, x, T=10):
    model.train()  # 保持 dropout 激活
    preds = torch.stack([model(x) for _ in range(T)], dim=0)
    uncertainty = preds.var(dim=0).mean(dim=1)  # [B, D/2, H/2, W/2]
    return uncertainty
```

**创新点**：本模块是完全轻量级的（仅熵计算，零参数），不增加推理负担；MC Dropout 版本提供更准确的不确定性估计，适合精度要求更高的任务。

---

## 4. 模块 2：Uncertainty-Guided MoE Decoder

### 4.1 设计动机

EffiDec3D 的消融实验（Table 5）表明：
- 48 通道：DICE 79.25%（选择的设计点）
- 88 通道：DICE 79.41%（+0.16%，但 +74% GFLOPs）

二者差距在平均 DICE 上微乎其微，但在**小结构精度**上会有实质差异。静态取 48 通道是全局折中，而非最优解。

### 4.2 Mixture-of-Experts 解码器

定义三个专家解码器，通道数不同：

| 专家 | 通道数 | 对应场景 |
|---|---|---|
| Expert-S | 32ch | 大器官、清晰边界，低不确定性 |
| Expert-M | 64ch | 中等复杂度结构 |
| Expert-L | 96ch | 小病灶、模糊边界，高不确定性 |

### 4.3 路由机制

路由网络同时接收全局特征和不确定性统计：

```python
class AdaptiveRouter(nn.Module):
    def __init__(self, feat_dim, n_experts=3):
        super().__init__()
        self.router = nn.Linear(feat_dim + 1, n_experts)  # +1 for mean uncertainty

    def forward(self, global_feat, uncertainty_map):
        u_stat = uncertainty_map.mean(dim=[1,2,3], keepdim=False)  # [B, 1]
        router_input = torch.cat([global_feat, u_stat], dim=-1)
        logits = self.router(router_input)  # [B, n_experts]
        return F.softmax(logits, dim=-1)   # soft routing weights
```

训练时使用 **soft routing**（加权求和输出），推理时可切换为 **hard routing**（argmax，避免多余的前向传播）。

### 4.4 与 EffiDec3D 集成

改动点仅在 `ModifiedUnetrUpBlock` / `MedNeXt_EffiDec3D` 的解码阶段：

```python
# EffiDec3D（静态）
dec = self.decoder_block(dec_feat)

# AdaDec3D（MoE）
weights = self.router(global_feat, uncertainty)  # [B, 3]
outputs = torch.stack([
    self.expert_s(dec_feat),
    self.expert_m(dec_feat),
    self.expert_l(dec_feat),
], dim=1)  # [B, 3, C, D, H, W]
dec = (weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * outputs).sum(dim=1)
```

**创新点**：将解码器优化从"选一个合适的通道数"提升为"根据解剖复杂度动态选择解码能力"，是 Conditional Computation 在 3D 分割解码器上的首次应用。

---

## 5. 模块 3：ROI-aware Progressive Refinement

### 5.1 设计动机

EffiDec3D 的核心 limitation：

> 以 D/2×H/2×W/2 输出代替 D×H×W，对大器官无影响（ΔDice ≈ 0），  
> 但对小结构有实质影响（如 Table 2：食管 74.31% vs 76.01%）。

全局降分辨率是 blunt instrument。正确做法是：**只对真正需要高分辨率的区域施加高分辨率解码**。

### 5.2 两阶段精化流程

**Stage 1 — 粗解码（全体积，低分辨率）**

```
Input: X ∈ [B, 1, D, H, W]
→ EffiDec3D Coarse Decoder
→ Coarse Pred: Y_coarse ∈ [B, C, D/2, H/2, W/2]
→ Uncertainty Map: U ∈ [B, D/2, H/2, W/2]
```

**Stage 2 — ROI 提取**

```python
def extract_roi(uncertainty, threshold=0.5, margin=4):
    # threshold 来自 uncertainty 分位数（自适应）
    high_unc = (uncertainty > uncertainty.quantile(threshold))
    # 膨胀 margin 体素，确保 ROI 覆盖完整边界
    roi_mask = F.max_pool3d(
        high_unc.float().unsqueeze(1),
        kernel_size=2*margin+1,
        stride=1,
        padding=margin
    ).squeeze(1).bool()
    return roi_mask
```

**Stage 3 — 局部高分辨率精化**

```
ROI 区域体素 (约 10–30% 全体积)
→ Expert-L Decoder（96ch, D 全分辨率）
→ Y_refined

合并：
Y_final = Y_coarse (上采样) + Y_refined * roi_mask
```

### 5.3 计算收益分析

粗略估算（设 ROI 占 20% 体积）：

| 方案 | 有效 FLOPs |
|---|---|
| EffiDec3D 全局 D/2 | 51.47 GFLOPs |
| 全局 D 全分辨率 | 404.4 GFLOPs |
| **AdaDec3D (80% D/2 + 20% D)** | **≈ 51.47 × 0.8 + 404.4 × 0.2 ≈ 122 GFLOPs** |

同时，精化路径仅作用于真正需要的区域，**小结构 DICE 预期改善 1–3%**。

**创新点**：将 EffiDec3D 的二选一分辨率决策，升级为"粗→精两阶段 + 不确定性驱动的局部精化"，是首个在 3D 分割解码器上实现 ROI-aware 计算分配的方案。

---

## 6. 训练策略

### 三阶段训练

**Stage 1 — Baseline 预训练**

- 训练：Encoder + EffiDec3D Coarse Decoder（即原始 EffiDec3D 训练流程）
- 目标：获取稳定的粗分割基线
- 损失：$\mathcal{L}_{Dice-CE}$

**Stage 2 — 不确定性与路由预训练**

- 冻结：Encoder + Coarse Decoder
- 训练：Uncertainty Head + Adaptive Router + Expert Decoders
- 目标：让 Router 学会将不确定性与解码能力建立正确关联
- 损失：$\mathcal{L}_{Dice-CE} + \lambda_2 \mathcal{L}_{resource} + \lambda_3 \mathcal{L}_{router}$

**Stage 3 — 端到端精调**

- 训练全部参数
- 加入 ROI Refinement Module
- 损失：$\mathcal{L} = \mathcal{L}_{Dice-CE} + \lambda_1 \mathcal{L}_{uncertainty} + \lambda_2 \mathcal{L}_{resource} + \lambda_3 \mathcal{L}_{router}$

### 损失函数定义

$$\mathcal{L} = \mathcal{L}_{Dice\text{-}CE} + \lambda_1 \mathcal{L}_{unc} + \lambda_2 \mathcal{L}_{res} + \lambda_3 \mathcal{L}_{router}$$

| 项 | 含义 |
|---|---|
| $\mathcal{L}_{Dice\text{-}CE}$ | 主分割损失（DiceCELoss，与 EffiDec3D 一致）|
| $\mathcal{L}_{unc}$ | 不确定性一致性损失：使高不确定区域预测熵与实际错误率对齐 |
| $\mathcal{L}_{res}$ | 资源惩罚项：鼓励在保证精度前提下选择更轻量的专家 |
| $\mathcal{L}_{router}$ | 路由正则化：防止所有 token 坍缩到同一专家（load balancing）|

---

## 7. 实验计划

### 7.1 数据集

| 数据集 | 内容 | 选择理由 |
|---|---|---|
| **FeTA 2021** | 胎儿脑 MRI，7 类结构 | 小目标（DGM、Vent.），验证不确定性路由对边界精度的提升 |
| **BTCV（Synapse）** | 腹部 CT，13 器官 | 与 EffiDec3D 直接对比的标准基准 |
| **MSD Task01–10** | 跨任务多器官 | 验证跨任务泛化，与 EffiDec3D Table 3 直接可比 |

### 7.2 评估指标

**分割精度**：Dice、HD95（organ-wise，尤其关注小结构）

**效率**：GFLOPs、参数量、推理延迟（ms）、GPU 显存峰值

**自适应行为分析**（AdaDec3D 专属）：
- 各样本实际激活的专家分布
- ROI Mask 覆盖率 vs 结构难度的相关性
- 不同器官的平均激活通道数

### 7.3 消融实验

| 消融维度 | 对比设置 |
|---|---|
| A1：基线对比 | EffiDec3D vs AdaDec3D |
| A2：不确定性路由的贡献 | 去除 Router，改为固定 Expert-M |
| A3：ROI 精化的贡献 | 去除 Refinement，仅保留 MoE |
| A4：MoE vs 单专家 | Expert-M only vs 三专家 MoE |
| A5：不确定性估计方式 | Entropy vs MC Dropout vs Deep Ensemble |
| A6：资源惩罚系数 | λ₂ ∈ {0.01, 0.1, 0.5, 1.0} |

---

## 8. 预期贡献

### Contribution 1
**首个不确定性引导的 3D 分割自适应解码器**：将解码器计算分配从"全局均匀"升级为"体素级难度感知"，直接解决 EffiDec3D 对小结构的精度缺陷。

### Contribution 2
**Mixture-of-Expert 解码架构**：将解码器优化从通道数调参（工程问题）转化为 Conditional Computation（研究问题），专家路由由样本本身的解剖复杂度决定。

### Contribution 3
**ROI-aware 两阶段精化框架**：以"全体积粗解码 + 不确定区域高分辨率精化"的范式替代 EffiDec3D 的二选一分辨率开关，在 FLOPs 可控的前提下显著提升边界精度。

### Contribution 4
**系统性扩展 EffiDec3D**：方法完全在 EffiDec3D 代码库上实现，核心改动仅涉及解码器模块，encoder 权重可直接复用，提供清晰的升级路径。

---

## 9. 创新性评估与发表潜力

| 维度 | EffiDec3D | AdaDec3D (原始 Proposal) | **AdaDec3D (本方案)** |
|---|---|---|---|
| 核心思想 | 静态通道压缩 + 静态分辨率裁剪 | 动态通道 + 动态分辨率 | 不确定性引导 MoE + ROI 精化 |
| 方法新颖性 | 工程优化 | 参数预测 | Conditional Computation |
| 对小结构的处理 | 有损失，未解决 | 动态分辨率缓解 | ROI 精化直接解决 |
| 创新评分 | — | 5.5/10 | **7.5–8.0/10** |
| 预期发表期刊 | CVPR | MICCAI Poster | **MICCAI / TMI / JBHI** |

---

## 10. 关键技术挑战

| 挑战 | 说明 | 应对策略 |
|---|---|---|
| Soft routing 梯度训练稳定性 | MoE soft routing 存在模式坍缩风险（所有样本走同一专家） | Load balancing loss（$\mathcal{L}_{router}$）+ 专家初始化多样性 |
| ROI 边界对齐 | 粗解码输出 D/2 分辨率，ROI mask 需上采样到 D 分辨率对齐 | 双线性上采样 mask + 形态膨胀保证边界覆盖 |
| 三阶段训练收敛 | 依次引入模块，若顺序不当可能导致梯度干扰 | Stage 2 冻结 backbone，Stage 3 分层 lr（backbone lr × 0.1）|
| 不确定性校准 | 预测熵与真实误差率未必对齐 | $\mathcal{L}_{unc}$ 项约束校准；评估 ECE（Expected Calibration Error）|

---

## 11. 下一步计划

1. **实现 Uncertainty Head**：在 EffiDec3D 粗输出后接熵计算，验证不确定性图与标注误差的相关性（可视化）
2. **实现 AdaptiveRouter + 三专家解码器**：基于 `MedNeXtV1_EffiDec3D.py` 或 `swin_unetr_effidec3d.py` 添加 MoE 分支
3. **实现 ROI Refinement Module**：设计 ROI 提取阈值策略（分位数自适应 vs 固定阈值消融）
4. **在 FeTA 和 BTCV 上初步实验**：对比 EffiDec3D baseline，记录 Dice / FLOPs / 专家激活分布
5. **撰写 MICCAI 2026 投稿**
