# EffiDec3D: An Optimized Decoder for High-Performance and Efficient 3D Medical Image Segmentation

**Md Mostafijur Rahman and Radu Marculescu**  
The University of Texas at Austin, Austin, Texas  
{mostafijur.rahman, radum}@utexas.edu

---

## Abstract

Recent 3D deep networks such as SwinUNETR, SwinUNETRv2, and 3D UX-Net have shown promising performance by leveraging self-attention and large-kernel convolutions to capture the volumetric context. However, their substantial computational requirements limit their use in real-time and resource-constrained environments. The high #FLOPs and #Params in these networks stem largely from complex decoder designs with high-resolution layers and excessive channel counts.

In this paper, we propose **EffiDec3D**, an optimized 3D decoder that employs a channel reduction strategy across all decoder stages, which sets the number of channels to the minimum needed for accurate feature representation. Additionally, EffiDec3D removes the high-resolution layers when their contribution to segmentation quality is minimal.

Our optimized EffiDec3D decoder achieves a **96.4% reduction in #Params** and a **93.0% reduction in #FLOPs** compared to the decoder of original 3D UX-Net. Similarly, for SwinUNETR and SwinUNETRv2 (which share an identical decoder), we observe reductions of **94.9% in #Params** and **86.2% in #FLOPs**. Our extensive experiments on 12 different medical imaging tasks confirm that EffiDec3D not only significantly reduces the computational demands, but also maintains a performance level comparable to original models, thus establishing a new standard for efficient 3D medical image segmentation.

Implementation: https://github.com/SLDGroup/EffiDec3D

---

## 1. Introduction

Medical image segmentation is a fundamental task in clinical practice, enabling precise delineation of anatomical structures for diagnostic and therapeutic purposes. Deep learning methods have become the gold standard for image segmentation. Among these, the U-shaped CNN architectures have been particularly influential due to their encoder-decoder structure with skip connections, allowing them to capture both global and local contextual information from 2D images.

Recently, vision transformers have shown promise in segmentation tasks by capturing long-range dependencies with self-attention (SA) mechanisms. Hierarchical transformers like Swin Transformer and PVT have advanced multi-resolution feature processing. However, 2D methods remain limited in capturing the full spatial relationships inherent in 3D volumetric data.

To overcome these limitations, researchers have developed 3D architectures that process volumetric data more effectively. These include transformer-based 3D models such as UNETR, nnFormer, TransBTS, SwinUNETR, and SwinUNETRv2, as well as CNN-based architectures like 3D UX-Net and MedNeXt that use large-kernel convolutions. These models make significant progress in 3D medical image segmentation; however, they substantially increase #Params and #FLOPs, limiting their applicability in real-time and resource-constrained environments.

Efficiency-focused architectures such as UNETR++, SegFormer3D, and SlimUNETR introduce efficiency improvements, but at the expense of suboptimal segmentation accuracy.

### Key Contributions

- **Complexity Analysis of SOTA Architectures:** In-depth analysis of 3D UX-Net, SwinUNETR, and SwinUNETRv2 to identify key bottlenecks in #FLOPs and #Params, highlighting the contribution of high-resolution layers and excessive channel counts.

- **Design of the Optimized EffiDec3D Decoder:** Achieves 96.4% reduction in #Params and 93.0% reduction in #FLOPs for 3D UX-Net, and similar reductions of 94.9%/#86.2% for SwinUNETR/SwinUNETRv2, without significantly compromising segmentation accuracy.

- **Adaptable Channel Reduction and Resolution Strategy:** EffiDec3D reduces #channels across decoder stages and selectively removes high-resolution layers, balancing computational efficiency with performance.

- **Extensive Multi-Model and Multi-Task Evaluation:** Validated across 3D UX-Net, SwinUNETR, and SwinUNETRv2 on 12 diverse medical imaging datasets.

- **Efficiency without Compromising Segmentation Performance:** Demonstrates that high-FLOP, high-resolution blocks can be minimized or removed with minimal impact on segmentation accuracy.

---

## 2. Related Work

### 2.1 2D Medical Image Segmentation Methods

CNNs have long been the cornerstone of medical image segmentation. The U-Net architecture, with its encoder-decoder structure and skip connections, was a pivotal advancement. Extensions such as UNet++ and ResUNet improve accuracy at the cost of increased computational complexity. nnU-Net automates U-Net configuration across datasets.

Vision transformers have recently shown promise in segmentation: TransUNet integrates CNNs for local feature extraction with transformers for global context; Swin-Unet utilizes Swin Transformer blocks in a U-shaped architecture. Other works such as PolypPVT, CASCADE, and EMCAD combine CNN-based decoders with transformer encoders. However, all these approaches have limitations in 3D segmentation where spatial context along all three dimensions is critical.

### 2.2 3D Medical Image Segmentation Methods

3D CNN-based models such as 3D U-Net and nnUNet3D adapt the U-shaped structure with 3D convolutions. Recent 3D transformer-based models (UNETR, nnFormer, SwinUNETR, SwinUNETRv2) use self-attention to capture long-range dependencies, achieving SOTA results on several benchmarks. 3D UX-Net and MedNeXt introduce large-kernel convolutions for similar gains while maintaining a CNN-based structure.

Our work focuses on the decoder bottleneck: in models like SwinUNETR, SwinUNETRv2, and 3D UX-Net, the decoder accounts for a substantial part of the computation due to repeated convolutional operations in high-resolution stages and large #channels across stages. Our approach leverages systematic channel reduction and selectively removes high-resolution layers when they provide minimal benefit.

---

## 3. Methodology

EffiDec3D is applicable across various backbone encoders, including CNN-based (e.g., 3D UX-Net) and transformer-based (e.g., SwinUNETR and SwinUNETRv2) encoders.

### 3.1 Complexity Analysis of SOTA Architectures

**A) High FLOPs in High-Resolution Decoder Stages**

Decoder stages operating at full input resolution (H×W×D) or half-resolution (H/2×W/2×D/2) are the primary contributors to high FLOPs. In 3D, spatial dimensions scale cubically, so upsampling followed by residual convolutions from H/2×W/2×D/2 to H×W×D alone can require several hundred GFLOPs.

Key empirical findings:
- On BTCV dataset: reducing output resolution to H/2×W/2×D/2 instead of H×W×D decreases DICE by only **0.3%** (79.7% → 79.4%) while saving **368.71 GFLOPs**.
- On FeTA dataset: this approach actually yields an accuracy **gain of 0.44%** (87.28% → 87.72%) with similarly high FLOPs savings.

**B) Excessive Channel Counts and Parameter Overhead**

The decoders in 3D UX-Net, SwinUNETR, and SwinUNETRv2 use large channel counts in residual convolution blocks. For example:
- A residual block with 768 channels contributes ~24.8M parameters in a single layer.
- A residual convolution (12.24M) + transposed convolution (2.36M) involving 768 input channels and 384 output channels adds 14.6M parameters.

These large channel sizes drastically increase #Params without proportional gains in accuracy.

### 3.2 Design of the EffiDec3D Decoder

EffiDec3D is a lightweight decoder maintaining a reduced channel count across stages and excluding the highest resolution layers. Given a 3D input volume **X ∈ R^(B×C×D×H×W)**, the encoder processes the input into hierarchical feature maps at varying resolutions.

#### 3.2.1 Optimization Strategy and Overall Design

**Channel Reduction Strategy:**

EffiDec3D employs a uniform channel reduction at each decoding stage via the `ChannelReductionResidualBlock`. All feature channels are reduced to a constant value C_reduced, calculated as:

```
C_reduced = min_{i=1,...,n} C_i                    (Eq. 1)
```

where C_i is the number of channels in encoder stage i (i.e., the minimum channel count across all encoder stages). Each encoder feature map F_i ∈ R^(B×Ci×Di×Hi×Wi) is processed as:

```
F'_i = ChannelReductionResidualBlock(F_i)           (Eq. 2)
```

where F'_i ∈ R^(B×C_reduced×Di×Hi×Wi).

**Resolution Restriction and Upsampling:**

EffiDec3D limits upsampling to D/2×H/2×W/2, avoiding computationally expensive full-resolution operations. The feature integration at each decoder stage:

```
Dec_i = ResidualUpBlock(F'_i, Dec_{i+1})            (Eq. 3)
```

where Dec_{i+1} is the feature map from the previous decoder stage, and Dec_n = F'_n (n is the last encoder stage).

**Final Output Prediction:**

A 1×1×1 conv maps the decoder output to C_out classes:

```
Y = Conv3D(Dec_i, K_{1×1×1})                        (Eq. 4)
```

Finally, 2× trilinear upsampling restores the output to (D, H, W) resolution.

#### 3.2.2 ChannelReductionResidualBlock

Given input X ∈ R^(B×Ci×D'×H'×W'), this block applies two successive convolutions:

```
Y1 = ReLU(IN(Conv3D(X, K, C_reduced)))              (Eq. 5)
Y2 = ReLU(IN(Conv3D(Y1, K, C_reduced)))             (Eq. 6)
```

where K is a 3×3×3 kernel and IN is Instance Normalization. A residual connection adjusts the input channel count if C_in ≠ C_reduced:

```
X_residual = IN(Conv3D(X, K_{1×1×1}, C_reduced))   (Eq. 7, if C_in ≠ C_reduced)
Y_reduced = ReLU(Y2 + X_residual)                   (Eq. 8)
```

#### 3.2.3 ResidualUpBlock

The lower-resolution decoder feature Dec_{i+1} is first upsampled:

```
U_i = TransposedConv(Dec_{i+1})
```

where U_i ∈ R^(B×C_reduced×Di×Hi×Wi). Then U_i is either added or concatenated to the skip feature F'_i:

```
Dec_i = U_i + F'_i    or    Dec_i = Concat(U_i, F'_i)    (Eq. 9)
```

Addition offers reduced complexity; concatenation doubles channel count before the next conv block. The resulting Dec_i is refined using the same residual conv block (Eqs. 5–8).

---

## 4. Experiments

### 4.1 FeTA 2021 Segmentation Results (Fetal Brain MRI)

**Table 1.** Fetal brain segmentation DICE scores (%) and HD95 on FeTA 2021 test set. FLOPs reported for 96×96×96 input.

| Architecture | MParams↓ | GFLOPs↓ | ECF | GM | WM | Vent. | Cereb. | DGM | BS | Avg.(%)↑ | HD95↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SlimUNETR | 1.79 | 20.17 | 87.44 | 79.29 | 92.10 | 87.76 | 79.46 | 86.59 | 86.87 | 85.64 | 2.83 |
| SegFormer3D | 4.50 | 5.03 | 88.16 | 79.63 | 92.59 | 89.17 | 83.50 | 85.82 | 87.10 | 86.57 | 2.70 |
| UNETR | 92.78 | 82.60 | 86.56 | 72.98 | 91.72 | 85.64 | 86.60 | 86.86 | 78.99 | 84.19 | 3.41 |
| nnUNet | 31.78 | 417.96 | 85.97 | 78.56 | 90.82 | 81.87 | 88.64 | 81.16 | 84.96 | 84.57 | 3.12 |
| nnFormer | 149.25 | 213.60 | 88.48 | 81.12 | 93.06 | 90.41 | 81.36 | 87.21 | 87.55 | 87.03 | 2.72 |
| TransBTS | 31.58 | 110.34 | 88.57 | 80.48 | 92.78 | 88.74 | 84.91 | 86.81 | 87.99 | 87.18 | 2.48 |
| UNETR++ | 42.62 | 53.99 | 88.61 | 81.42 | 93.03 | 88.87 | 80.60 | 86.78 | 87.71 | 86.72 | 2.87 |
| MedNeXt-M-K3 | 17.55 | 110.65 | 88.87 | 81.18 | 93.06 | 90.47 | 85.95 | 87.15 | 87.53 | 87.74 | 2.43 |
| **MedNeXt-M-K3 + EffiDec3D** | **5.77** | **49.73** | 88.46 | 81.45 | 93.03 | 90.05 | 83.46 | 88.10 | 88.23 | 87.54 | 2.48 |
| 3D UX-Net | 53.00 | 631.97 | 88.61 | 81.26 | 93.09 | 90.42 | 82.26 | 87.23 | 88.11 | 87.28 | 2.54 |
| **3D UX-Net + EffiDec3D** | **3.16** | **51.47** | 88.32 | 81.26 | 93.10 | 90.52 | **86.55** | 87.39 | **88.68** | **87.97** | **2.38** |
| SwinUNETR | 69.19 | 337.61 | 88.65 | 80.83 | 92.88 | 90.06 | 80.44 | 86.34 | 87.79 | 86.71 | 2.33 |
| **SwinUNETR + EffiDec3D** | **11.21** | **57.29** | 88.34 | 80.95 | 92.93 | 90.16 | 85.35 | 87.24 | 87.55 | 87.50 | 2.45 |
| SwinUNETRv2 | 83.19 | 353.61 | 88.86 | 81.22 | 93.02 | 90.29 | 81.63 | 87.73 | 88.25 | 87.29 | 2.44 |
| **SwinUNETRv2 + EffiDec3D** | **18.21** | **63.29** | 88.77 | 81.31 | **93.04** | **90.56** | 86.03 | 87.37 | 88.30 | 87.91 | 2.41 |

*ECF: External Cerebrospinal Fluid, GM: Grey Matter, WM: White Matter, Vent.: Ventricles, Cereb.: Cerebellum, DGM: Deep Grey Matter, BS: Brainstem.*

Key findings:
- 3D UX-Net + EffiDec3D: **−94% #Params** (53M → 3.16M), **−92% GFLOPs** (632 → 51.47), Avg. DICE **87.28% → 87.97%** (improved).
- Cerebellum DICE improves by **>4%** across all three optimized models.
- Minor trade-offs in small complex structures (DGM, ECF) due to reduced channel capacity.

### 4.2 BTCV 13-Organ Segmentation Results

**Table 2.** BTCV 13-organ segmentation validation DICE scores (%) and HD95.

| Architecture | MParams↓ | GFLOPs↓ | Spl | RKid | LKid | Gall | Eso | Liv | Sto | Aor | IVC | Veins | Pan | Rad | Lad | Avg.(%)↑ | HD95↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SlimUNETR | 1.79 | 20.17 | 84.88 | 81.79 | 83.05 | 64.63 | 65.39 | 94.61 | 76.13 | 87.34 | 80.08 | 58.96 | 57.37 | 49.34 | 46.45 | 71.54 | 12.34 |
| SegFormer3D | 4.50 | 5.03 | 87.42 | 82.72 | 84.92 | 70.82 | 69.42 | 93.83 | 79.05 | 87.82 | 81.30 | 63.18 | 64.47 | 52.76 | 48.75 | 74.34 | 13.41 |
| UNETR | 92.78 | 82.60 | 88.36 | 82.86 | 84.10 | 66.28 | 70.88 | 95.07 | 78.27 | 86.82 | 79.93 | 63.41 | 63.68 | 58.43 | 56.41 | 74.96 | 21.78 |
| nnUNet | 31.78 | 417.96 | 88.00 | 84.78 | 85.47 | 71.85 | 75.44 | 95.32 | 76.54 | 90.93 | 85.14 | 68.91 | 62.19 | 66.90 | 60.19 | 77.82 | 10.91 |
| nnFormer | 149.25 | 213.60 | 92.40 | 83.31 | 85.39 | 72.02 | 73.16 | 94.85 | 83.61 | 89.56 | 81.64 | 67.63 | 70.76 | 61.58 | 61.71 | 78.28 | 10.57 |
| TransBTS | 31.58 | 110.34 | 89.92 | 84.31 | 85.59 | 73.86 | 72.09 | 96.20 | 81.49 | 89.81 | 85.23 | 65.67 | 71.20 | 63.74 | 66.15 | 78.87 | 14.47 |
| UNETR++ | 42.62 | 53.99 | 94.69 | 85.99 | 86.90 | 78.83 | 73.79 | 96.22 | 83.27 | 91.20 | 87.02 | 72.40 | 71.82 | 67.64 | 62.19 | 80.92 | 9.59 |
| MedNeXt-M-K3 | 17.55 | 110.65 | 90.08 | 86.96 | 88.96 | 77.27 | 78.16 | 96.91 | 84.24 | 92.16 | 88.65 | 75.45 | 80.13 | 68.83 | 70.87 | **82.98** | **5.45** |
| **MedNeXt-M-K3 + EffiDec3D** | **5.77** | **49.73** | 90.81 | 86.49 | 86.60 | 79.90 | 76.68 | 96.87 | 84.17 | 91.75 | 88.40 | 73.05 | 79.35 | 70.42 | 68.61 | 82.55 | 6.41 |
| 3D UX-Net | 53.00 | 631.97 | 92.47 | 84.39 | 86.54 | 78.72 | 74.16 | 95.44 | 82.47 | 90.93 | 85.03 | 70.56 | 64.60 | 66.49 | 64.85 | 79.74 | 12.43 |
| **3D UX-Net + EffiDec3D** | **3.16** | **51.47** | 90.44 | 84.79 | 87.12 | 75.26 | 74.31 | 94.35 | 78.79 | 90.98 | 85.89 | 68.04 | 70.15 | 65.52 | 64.67 | 79.25 | 10.12 |
| SwinUNETR | 69.19 | 337.61 | 88.56 | 85.92 | 86.03 | 79.40 | 75.50 | 95.41 | 79.58 | 90.13 | 86.18 | 71.12 | 69.36 | 69.35 | 65.19 | 80.13 | 14.01 |
| **SwinUNETR + EffiDec3D** | **11.21** | **57.29** | 92.93 | 83.50 | 85.73 | 76.83 | 74.03 | 95.31 | 79.15 | 91.04 | 85.62 | 69.33 | 71.38 | 67.24 | 65.62 | 79.82 | 8.47 |
| SwinUNETRv2 | 83.19 | 353.61 | 90.78 | 86.29 | 85.62 | 79.20 | 75.90 | 95.21 | 78.90 | 90.00 | 86.25 | 72.61 | 74.50 | 71.44 | 69.66 | 81.26 | 12.86 |
| **SwinUNETRv2 + EffiDec3D** | **18.21** | **63.29** | 89.91 | 85.79 | 85.88 | 78.53 | 75.71 | 95.74 | 77.20 | 91.55 | 86.23 | 71.66 | 74.35 | 67.14 | 67.08 | 80.52 | 10.13 |

*Spl: spleen, RKid: right kidney, LKid: left kidney, Gall: gallbladder, Eso: esophagus, Liv: liver, Sto: stomach, Aor: aorta, IVC: inferior vena cava, Pan: pancreas, Rad/Lad: right/left adrenal glands.*

Key findings:
- Large organs (spleen, kidneys, liver) maintain high accuracy; small/thin structures (esophagus, veins) show slight drops from lower output resolution.
- 3D UX-Net + EffiDec3D sees *improvements* in IVC and pancreas, suggesting optimizations retain critical features for anatomically challenging structures.

### 4.3 MSD (10 Datasets) Segmentation Results

**Table 3.** MSD 10 dataset DICE scores (%) on validation set, excluding background.

| Architecture | Task01 BrainTumour (Avg.) | Task02 Heart | Task03 Liver (Avg.) | Task04 Hippocampus (Avg.) | Task05 Prostate (Avg.) | Task06 Lung | Task07 Pancreas (Avg.) | Task08 HepaticVessel (Avg.) | Task09 Spleen | Task10 Colon | **10-Task Avg.** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SlimUNETR | 72.66 | 90.42 | 67.12 | 86.29 | 49.65 | 67.66 | 42.76 | 41.20 | 95.42 | 25.13 | 63.83 |
| SegFormer3D | 73.85 | 91.64 | 65.06 | 86.46 | 65.75 | 55.07 | 31.49 | 33.18 | 95.68 | 33.09 | 63.13 |
| UNETR | 75.69 | 91.42 | 60.62 | 87.18 | 56.33 | 65.38 | 36.65 | 46.31 | 94.60 | 28.03 | 64.22 |
| nnFormer | 74.79 | 92.21 | 75.19 | 86.71 | 63.39 | 69.79 | 50.86 | 54.56 | 91.40 | 24.74 | 68.36 |
| TransBTS | 77.83 | 90.12 | 75.91 | 87.54 | 60.45 | 63.57 | 51.89 | 53.70 | 94.54 | 29.53 | 68.51 |
| UNETR++ | 77.16 | 92.39 | 74.97 | 87.76 | 53.74 | 76.09 | 52.01 | 51.63 | 96.49 | 40.82 | 70.31 |
| 3D UX-Net | 78.58 | 92.03 | 79.09 | 88.49 | 65.83 | 71.46 | 51.28 | 55.65 | 96.68 | 48.77 | 72.79 |
| **3D UX-Net + EffiDec3D** | 78.06 | 92.63 | 78.40 | 88.05 | **69.92** | **74.14** | 52.23 | 55.62 | 96.56 | 48.42 | 73.40 |
| SwinUNETR | 79.06 | 91.92 | 78.19 | 87.87 | 62.93 | 65.12 | 48.95 | 52.23 | 96.13 | 42.94 | 70.53 |
| **SwinUNETR + EffiDec3D** | 78.79 | 91.98 | 78.07 | 87.81 | 64.61 | 73.79 | 52.48 | 53.42 | **96.55** | 45.41 | 72.29 |
| SwinUNETRv2 | 78.78 | 91.96 | 76.59 | 87.67 | 66.67 | 73.52 | 57.59 | 59.83 | 96.77 | 48.87 | 73.83 |
| **SwinUNETRv2 + EffiDec3D** | 78.55 | 92.29 | **78.42** | 87.74 | 67.40 | 75.19 | **59.09** | 59.37 | **96.86** | **52.23** | **74.71** |

SwinUNETRv2 + EffiDec3D achieves the **best 10-task average DICE (74.71%)**, outperforming the original SwinUNETRv2 (73.83%).

---

## 5. Ablation Studies

### 5.1 Effect of Removing High-Resolution Stages

**Table 4.** Performance vs. computational complexity at different output resolutions (channel-optimized 3D UX-Net on BTCV).

| Output Resolution | MParams↓ | GFLOPs↓ | Avg. DICE (%)↑ |
|---|---|---|---|
| Original (D,H,W) — unoptimized | 53.00 | 632.19 | 79.74 |
| D, H, W | 3.55 | 404.40 | 79.44 |
| **D/2, H/2, W/2 (Ours)** | **3.15** | **51.47** | **79.25** |
| D/4, H/4, W/4 | 2.82 | 14.41 | 76.41 |
| D/8, H/8, W/8 | 2.41 | 8.85 | 70.77 |
| D/16, H/16, W/16 | 1.88 | 7.94 | 51.22 |

Key insight: **D/2, H/2, W/2 offers the optimal trade-off** — only 0.19% DICE drop vs. D,H,W but saves 352.93 GFLOPs. Going to D/4 causes a 2.84% DICE drop; D/8 and D/16 lead to severe degradation in small structures (pancreas, veins, adrenal glands).

### 5.2 Effect of Varying the Decoder #Channels

**Table 5.** Impact of decoder channel count on BTCV performance (channel-optimized 3D UX-Net).

| Decoder #Channels | MParams↓ | GFLOPs↓ | DICE (%)↑ |
|---|---|---|---|
| 16 | 1.73 | 14.83 | 77.25 |
| 24 | 2.01 | 21.19 | 78.69 |
| 32 | 2.34 | 29.47 | 78.47 |
| 40 | 2.72 | 39.65 | 78.73 |
| **48 (Ours)** | **3.15** | **51.47** | **79.25** |
| 56 | 3.52 | 53.73 | 79.46 |
| 64 | 3.93 | 56.23 | 78.67 |
| 72 | 4.84 | 61.94 | 78.35 |
| 80 | 4.37 | 58.97 | 78.77 |
| 88 | 5.35 | 65.14 | 79.41 |
| 96 | 5.88 | 68.45 | 79.21 |
| Original | 53.00 | 632.19 | 79.74 |

Key insight: **48 or 56 channels achieves the optimal balance** — near-original performance (79.25–79.46%) at a fraction of the cost. Beyond 56 channels, performance plateaus or slightly degrades (diminishing returns), while #Params and #FLOPs continue to rise.

---

## 6. Conclusion

EffiDec3D addresses computational inefficiencies in 3D medical image segmentation by:
1. Reducing channel counts uniformly to C_reduced (= minimum encoder channel count) across all decoder stages.
2. Eliminating the highest-resolution (D×H×W) decoder layers, limiting upsampling to D/2×H/2×W/2 with a final trilinear upsample.

Results: up to **93.0% FLOPs reduction** and **96.4% parameter reduction** (3D UX-Net decoder) with minimal to no decrease in segmentation accuracy across 12 tasks. EffiDec3D is suitable for integration with various backbone networks and offers a generalizable solution for refining computationally demanding decoders.

---

## References (Selected)

- [7] Isensee et al., *nnU-Net*, Nature Methods 2021
- [8] Hatamizadeh et al., *UNETR*, WACV 2022
- [9] He et al., *SwinUNETR-v2*, MICCAI 2023
- [12] Lee et al., *3D UX-Net*, ICLR 2023
- [13] Liu et al., *Swin Transformer*, ICCV 2021
- [16] Pang et al., *SlimUNETR*, IEEE TMI 2023
- [18] Perera et al., *SegFormer3D*, CVPRW 2024
- [24] Roy et al., *MedNeXt*, MICCAI 2023
- [25] Shaker et al., *UNETR++*, IEEE TMI 2024
- [26] Tang et al., *SwinUNETR*, CVPR 2022
