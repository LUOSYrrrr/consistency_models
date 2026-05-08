# 论文阅读日志 — Wavelet CT 项目

| Paper | Priority | Read date | Status |
| ---- | ---- | ---- | ---- |
| Feature Pyramid Networks for Object Detection (Lin et al., CVPR 2017) | ⭐⭐⭐ | 2026-05-08 | ✅ |
| LAPGAN — Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks (Denton et al., NeurIPS 2015) | ⭐⭐⭐ | 2026-05-08 | ✅ |
| MSG-GAN — Multi-Scale Gradients for Generative Adversarial Networks (Karnewar & Wang, CVPR 2020) | ⭐⭐⭐ | 2026-05-08 | ✅ |

---

# Feature Pyramid Networks for Object Detection

## 0. Metadata
- **Authors**: Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie
- **Venue & Year**: CVPR 2017（arXiv 1612.03144v2，2016 年 12 月首版）
- **arXiv / Link**: https://arxiv.org/abs/1612.03144
- **Code**: https://github.com/unsky/FPN
- **Read date**: 2026-05-08
- **Priority**: ⭐⭐⭐
- **Tags**: #multi-scale-supervision #top-down-architecture #lateral-connection #pyramid #shared-head #CV-backbone

## 1. TL;DR（一句话）
> 利用 ConvNet 自身的金字塔型 feature hierarchy，加一条 top-down pathway + lateral 1×1 connection，把"低分辨率高语义 + 高分辨率低语义"两条流融合成"**所有分辨率都同时高语义**"的特征金字塔，**在每个尺度独立做预测且 head 参数共享**，几乎零额外成本地把目标检测尤其是小目标检测大幅提升。

## 2. Problem & Motivation
- **要解决的问题**: 多尺度目标检测。同一张图里物体大小差异巨大（COCO 里 32² 到 512² 都有），单尺度特征做不好——尤其是小目标。
- **已有方法的局限**:
  - **Featurized image pyramid**（Fig.1a）：把图像缩放成多个尺度分别 forward → 慢 + 训练时显存爆炸（端到端不可行），只能在测试时用 → train/test 不一致。
  - **单一 feature map**（Fig.1b，Fast/Faster R-CNN）：快但小目标差。
  - **In-network feature hierarchy 直接用**（Fig.1c，SSD）：高分辨率层语义弱（C2 表征能力差），所以 SSD 干脆从高层（conv4_3）开始建塔，**主动放弃了高分辨率层**——错失了检测小目标的机会。
  - **U-Net / SharpMask / Stacked Hourglass 这类 top-down + skip 架构**（Fig.2 top）：虽然结构相似，但**只在最 fine 那一层做预测** → 仍是单尺度预测，依赖 image pyramid 才能多尺度。
- **本文 insight**:
  1. ConvNet 本身的 feature hierarchy 是天然的金字塔，不用重新建。
  2. 关键缺陷是"高分辨率层语义弱"——补救方法是 **top-down pathway 把深层语义传下来 + lateral connection 保留浅层精确定位**。
  3. **每个尺度都做独立预测**（不像之前的 top-down 工作只在最 fine 一层预测）。

## 3. Method（关键）

### 3.1 核心思想
把 backbone（ResNet）的多 stage 输出 {C2, C3, C4, C5}（stride 4/8/16/32）作为输入，分两个 pathway：

```
═══════════ Phase 1: Bottom-up（ResNet 前向传播） ═══════════

  Input image
       │  conv1 + pool
       ▼
      C2  (stride  4, 高分辨率 / 低语义)
       │
       ▼
      C3  (stride  8)
       │
       ▼
      C4  (stride 16)
       │
       ▼
      C5  (stride 32, 低分辨率 / 高语义)

═══════════ Phase 2: Top-down + Lateral（按 C5→C2 顺序构建塔） ═══════════

Step 1: 从 C5 起塔（无 lateral merge）
    C5 ── 1×1 conv (降维到 256-d) ──→ P5
                                       ├─→ 3×3 conv ─→ predict@P5
                                       └─→ 2× upsample (nearest) ──┐
                                                                    │
Step 2: P4 = lateral(C4) ⊕ upsample(P5)                             │
    C4 ── 1×1 conv (256-d, 无 ReLU) ─→ lateral_4                    │
                                          │                         │
                                         [⊕] ←─────────────────────┘
                                          │
                                          ↓ 3×3 conv (抗 aliasing)
                                          P4
                                          ├─→ predict@P4
                                          └─→ 2× upsample ──────────┐
                                                                    │
Step 3: P3 = lateral(C3) ⊕ upsample(P4)                             │
    C3 ── 1×1 conv ─→ lateral_3                                     │
                          │                                         │
                         [⊕] ←─────────────────────────────────────┘
                          │
                          ↓ 3×3 conv
                          P3
                          ├─→ predict@P3
                          └─→ 2× upsample ──────────────────────────┐
                                                                    │
Step 4: P2 = lateral(C2) ⊕ upsample(P3)                             │
    C2 ── 1×1 conv ─→ lateral_2                                     │
                          │                                         │
                         [⊕] ←─────────────────────────────────────┘
                          │
                          ↓ 3×3 conv
                          P2 ─→ predict@P2
```

**关键设计要点**:
- **Bottom-up pathway**：backbone 前向传播，每 stage 最后一个 residual block 输出作为 {C2..C5}。
- **Top-down pathway**：从 C5 开始，1×1 conv 降维到 d=256 → 2× nearest 上采样 → 与同尺度 lateral 输出做 **element-wise add (⊕)** → 3×3 conv 抗 aliasing → 得到 P_k。
- **Lateral connection**：每个 C_k 走一个 1×1 conv（**只为统一通道到 256，无 ReLU/BN**），与 top-down 流相加。
- **预测头**：每个 P_k 上挂一个 head（RPN: 3×3 conv + 两个 1×1 sibling），**4 个 head 参数完全共享**（因为所有 P_k 都是 256-d，且语义层级被 top-down 强制对齐）。
- **简洁性**：无 ReLU on lateral；nearest 上采样不学；lateral merge 选 add 而非 concat（作者说 concat 提升微小）。

### 3.2 Loss / 公式
FPN 本身**只是 backbone 改造**，没有引入新 loss。预测头沿用 RPN / Fast R-CNN 的 loss：每个尺度独立产生 anchor / RoI，每个 anchor 对应的 RPN loss（cls + reg）和 Fast R-CNN loss（cls + reg）按 anchor 累加：

$$
L = \sum_{k} \sum_{i \in \text{anchors}(P_k)} L_{\text{RPN}}(p_i, p_i^*) + L_{\text{reg}}(t_i, t_i^*)
$$

- **关键公式**：RoI 到 pyramid level 的分配（Eqn.1）
  $$k = \lfloor k_0 + \log_2(\sqrt{wh}/224) \rfloor, \quad k_0 = 4$$
  小 RoI 被分到 fine-resolution level（比如 P3），大 RoI 被分到 coarse-resolution level（比如 P5）。
- 与 baseline 的差异（公式层面）：sum 是对 anchors / RoIs 求和，而 anchors 现在分布在多个 P_k 上 → **梯度同时回传到多个分辨率的特征图**，浅层得到额外监督。

### 3.3 架构
- **多尺度**：✅ 4 个尺度同时输出 + 同时预测
- **级联**：⚠️ 不是严格的"先 P5 再 P4"级联预测，但 **top-down pathway 是级联的 feature flow**——P_{k+1} 的语义流到 P_k
- **参数共享**：✅ 所有尺度的 head 共享参数（这点很关键，作者明确做了 ablation 验证不共享和共享差不多 → 暗示所有 P_k 在语义层级上是"对齐"的）
- **特征维度统一**：✅ 所有 P_k 都是 d=256
- **lateral merge 方式**：**element-wise addition**（不是 concat），简洁
- **简洁性**：lateral 里没有 ReLU；用 nearest-neighbor 上采样；作者强调"试过更复杂的 connection module，提升微小，所以选最简版本"

## 4. Experiments
- **Datasets**: COCO（80 类，trainval35k 训练，5k minival 调，test-dev / test-std 评测）
- **Baselines**:
  - Faster R-CNN on C4 / C5 单尺度（重要的"controlled"对照）
  - SSD-style 仅用高层堆塔
  - Top-down without lateral
  - Bottom-up pyramid without top-down
  - 仅最 fine 的 P2
- **Key metrics**: COCO AP / AP@0.5 / AP_s / AP_m / AP_l；RPN 部分用 AR^100 / AR^1k

### 关键结果数字（最值得抄的几个）
**RPN proposals (Table 1, ResNet-50)**

| 方法 | AR^1k | AR^1k_s（小目标）|
|---|---|---|
| (a) baseline conv4 | 48.3 | 32.0 |
| (c) **FPN** | **56.3** (+8.0) | **44.9** (+12.9) |
| (d) bottom-up only (无 top-down) | 49.5 | 30.5 |
| (e) top-down only (无 lateral) | 46.1 | 26.5 |
| (f) only finest P2 | 51.3 | 35.1 |

**Faster R-CNN object detection (Table 3, ResNet-50)**

| 方法 | AP | AP_s |
|---|---|---|
| (a) baseline conv4 | 31.6 | 13.2 |
| (c) **FPN** | **33.9** (+2.3) | **17.8** (+4.6) |

### Ablations they did
- **Top-down 的重要性**：去掉 → 大跌（AR^1k 56.3 → 49.5）
- **Lateral 的重要性**：去掉 → 大跌 10 个点（56.3 → 46.1）
- **多尺度 vs 仅 finest 单尺度**：56.3 vs 51.3 → 多尺度赢（即使 P2 已经是 top-down 的产物）
- **head 是否共享参数**：共享和不共享精度相似 → "所有 P_k 在语义层级上对齐"
- **share features (RPN ↔ Fast R-CNN)**：略涨 0.4 AP，且 test 时间更短

## 5. ⚡ 与 Wavelet CT 的关系（最重要）

### 直接相关：是
**FPN 是"multi-scale supervision in CV"的最 canonical 引用**，本 idea 在结构上就是 FPN-style 的一种变体——在 U-Net backbone 上加 multi-resolution heads + 每个 head 一个独立 loss。

### 可借鉴的

**架构设计**（最 actionable）:
1. **Top-down + lateral**：CT 用的 U-Net 已经天然有这个结构（decoder = top-down，skip = lateral），所以**实现 multi-scale 输出几乎零成本**——只需要在 decoder 的每一层加一个输出 head（类似 FPN 的 3×3 conv → predict）。
2. **统一通道数 d=256**：所有尺度 head 共享通道数，简化网络设计。**我们也可以让所有 wavelet 系数预测 head 用统一 hidden dim**。
3. **简单 lateral merge（add 而不是 concat）**：U-Net 默认是 concat，但 FPN 选择 add 强调"简洁性"。**消融对比时可以尝试 add 替换 concat**——尤其当我们的 wavelet 多尺度需要在多个深度都加 head 时，add 更省参数。
4. **head 参数共享**：所有尺度的 prediction head 共享参数。**这点对 CT 非常有吸引力**——CT 已经是 (x, σ) → x_0 的单网络条件映射，加个 (..., scale) 的额外 condition 让单网络处理所有尺度，参数不爆炸。
5. **避坑：top-down 和 lateral 缺一不可**。如果以后要做 ablation，可以借鉴 Table 1 的 (d)(e) 对照。

**Loss 形式**:
- FPN 是"每个尺度独立 sliding-window loss + 直接相加"，**这是最朴素的 sum supervision 模板**——每个 P_k 上的 anchor loss 平均后等权相加。
- 暗示在 CT 上的初版做法：每个尺度算一份 L_CT，**等权相加**作为总 loss。复杂的 loss 加权（learned weight / 自适应）可以做 ablation 之一，但 FPN 证明**等权简单加就已经能 work**。

**实验设置**:
- "**仅最 fine level（P2）**" 也作为 baseline 来对照"完整 pyramid" → **我们应该跑：(a) 只在最高分辨率算 CT loss vs (b) 多尺度算 CT loss**，直接复用这个 ablation 思路证明"多尺度监督"的增益。
- "**top-down only / bottom-up only**" 这种结构性 ablation → 在我们的 setting 下可以转化成"是否需要 wavelet 多尺度间的级联 conditioning"。

### 与本 idea 的差异（delta）

**FPN 没做但我们要做的**:
1. **wavelet 分解的目标**：FPN 的多尺度都是同一个任务（预测同一组 bbox），我们的多尺度对应**不同的目标**（base + 3 个 wavelet residual），是真正的"频率分解"，不是"同一个 prediction 在不同尺度重复"。
2. **CT loss / Consistency 性质**：FPN 是 detection 监督学习，我们是无教师的 self-consistency loss。multi-scale supervision 在 CT 上的效果（是否降低 loss variance、是否加速收敛）需要重新验证。
3. **Residual vs sum supervision 的 design choice**：FPN 完全是 sum/parallel（每个尺度独立预测同一目标），**没回答 residual supervision 的问题**——这是我们的核心 ablation。
4. **生成任务 vs 检测任务**：检测可以容忍粗糙的多尺度对齐（最后还有 NMS），生成必须精确重构 → 我们对 wavelet 重构精度的要求远高于 FPN 对 anchor 对齐的要求。

**FPN 做了但我们要避免/改造的**:
1. **lateral 不加非线性**：FPN 故意只用 1×1 conv 不加 ReLU，是为了 detection 的稳定性。**但 CT 网络通常每层都有 SiLU + GroupNorm，照搬 U-Net 主干就可以，不必模仿 FPN 这个细节**。
2. **head 完全共享参数**：FPN 在同分布任务上共享。**我们不同尺度对应不同 wavelet 系数，分布差异可能更大**——共享 vs 独立 head 应作为 ablation。
3. **整数倍下采样**：FPN 用 2 的整数次幂分辨率（4/8/16/32）。MNIST 是 28×28，整除性差。**我们要么用 32×32 padding，要么改用 4/7/14/28 的非标准 wavelet 层级**。

### 如果对方做过类似 idea：novelty 还剩什么？
FPN 没做生成、没做 CT、没做 wavelet、没做 sum vs residual 对比 → **完全不冲突，反而是我们的"理论先例"和"架构模板"**。在 report / paper 里它的角色是：
> "We adopt a Feature-Pyramid-Network-style multi-resolution prediction head [Lin et al. 2017], which has been shown effective for multi-scale dense prediction in CV. Our contribution is to extend this design to (a) wavelet-decomposed targets and (b) consistency training objective in few-step generation."

## 6. 我的疑问 / 不确定点
1. FPN 的 head **完全共享参数** 在我们 wavelet setting 下能不能 work？wavelet 不同 sub-band 的统计差异显著（base 是低频 smooth，HH 子带是高频稀疏），共享 head 可能不够。**先按共享实现，failure mode 出现后再独立化**。
2. FPN 用 add 做 lateral merge，但 U-Net 用 concat。CT 现有 U-Net 是 concat 风格——我们要不要给 multi-scale output head **重新设计一个 add-style 分支**，还是直接复用 concat 的 decoder？倾向直接复用 concat decoder + 在每个 decoder 层挂额外 output head，最少改动。
3. FPN 在 detection 任务下证明"top-down pathway 必不可少"。**对应到我们 setting**：wavelet 高频子带能否在没有低频上下文的情况下被准确预测？大概率不能 → 必须保留级联条件（这其实就是我们 idea 本来就规定的"高频 condition on 低频"）。FPN 这条 ablation 间接给我们的"级联 conditioning"设计做了背书。
4. FPN 的 ablation Table 1(f) "仅最 fine level" 比 baseline 还高 5 个点 → 暗示**即使去掉多尺度预测，光是 top-down pathway 带来的 feature 增强本身就有用**。**对应到 CT**：这意味着我们的 baseline 必须严格控制——和 vanilla CT 比的时候，要保证 backbone 容量一致，否则 multi-scale supervision 的增益可能被"更深的 top-down 流"污染。

## 7. 关键引用（值得我也读的）
- [ ] **U-Net** [Ronneberger et al. 2015]（ref 31）— 已熟悉，跳过
- [ ] **SharpMask** [Pinheiro et al. 2016]（ref 28）— FPN 直接对比的 top-down + skip 工作，但只在 finest level 预测；看一眼为什么"只最 finest 预测"会输给"每个尺度都预测"
- [ ] **Stacked Hourglass Networks** [Newell et al. 2016]（ref 26）— 经典的 top-down 多尺度结构 + intermediate supervision，**和我们的"每个尺度独立 loss"思想最接近，必读**
- [ ] **Ghiasi & Fowlkes 2016 — Laplacian Pyramid for FCN segmentation**（ref 8）— **直接是 Laplacian pyramid（≈ wavelet 思想）+ 渐进 refinement，与本 idea 几乎同构**，只是任务是 segmentation。**强烈建议下一篇就读这个**
- [ ] **HED** [Xie & Tu 2015] — FPN 论文里没直接引（FPN 偏 detection），但是 deep supervision 鼻祖，沿着 ref 8 的脉络一起读

---

**下一步建议读**: Stacked Hourglass（ref 26）+ Ghiasi & Fowlkes Laplacian Pyramid for FCN（ref 8）。后者是 wavelet/Laplacian × dense prediction 最直接的先例。

---

# LAPGAN — Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks

## 0. Metadata
- **Authors**: Emily Denton, Soumith Chintala, Arthur Szlam, Rob Fergus
- **Venue & Year**: NeurIPS 2015（arXiv 1506.05751v1, 2015-06-18）
- **arXiv / Link**: https://arxiv.org/abs/1506.05751
- **Code**: http://soumith.ch/eyescream (Torch)
- **Read date**: 2026-05-08
- **Reading time**: ~45 min (10 pages + 全附录)
- **Priority**: ⭐⭐⭐
- **Tags**: #residual-supervision #Laplacian-pyramid #cascaded #coarse-to-fine #conditional-GAN #generative

## 1. TL;DR（一句话）
> 把图像生成分解为一系列"先粗后细"的级联：在 Laplacian pyramid 的每一层训练一个 conditional GAN，**每个 GAN 只负责生成当前尺度的 residual（差分图像 h_k）**，条件于已生成的低频图像 l_k = u(I_{k+1})；最终图像 = 各级 residual 累加。CIFAR10 样本从 ~10% 真伪混淆率提升到 ~40%。

## 2. Problem & Motivation
- **要解决的问题**: GAN 不能直接生成高分辨率自然图像（2015 年状态：原始 GAN 只有 MNIST/CIFAR10 等小图能 work）。
- **已有方法的局限**:
  - 原始 GAN [Goodfellow 2014]：直接在像素空间建模全局分布，生成大图时 fidelity 差
  - 当时已有的深度生成模型（RBM、Deep Boltzmann、VAE、Denoising AE）只能在 MNIST/NORB 上演 demo，无法扩展到真实场景
- **本文 insight**:
  1. **放弃"全局 fidelity"**：不试图让单个网络一步到位生成整张图，而是**把问题拆成多步 refinement**
  2. **Laplacian pyramid 是天然的多尺度解耦工具**：h_k = I_k - u(I_{k+1})，每层只包含一个特定 octave 的 band-pass 信息
  3. **每个 scale 独立训练**：不用联合 loss，每个 level 的 G 只需要"给定当前低频上下文，生成逼真的高频差分"——一个更简单的子问题

## 3. Method（关键）

### 3.1 核心思想
利用 Laplacian pyramid 的正交分解，把一张图拆成 K 层：K-1 层 band-pass residual h_k + 1 层 base h_K。每层训练一个独立 GAN（G_k + D_k），G_0..G_{K-1} 是 conditional GAN（条件变量 l_k = 当前低频图像），G_K 在最粗尺度不条件直接生成。

#### 采样流程（Fig.1，以 LSUN K=3, 64×64 为例）

```
═══════ Step 1: 最粗层 base 生成（无条件 GAN） ═══════

   z_3 ───→ G_3 ───→ Ĩ_3   (8×8, base = 低频图像本身)
                      │
                      ↓ u(·) 上采样到 16×16
                     l_2

═══════ Step 2: 16×16 残差生成（条件 GAN） ═══════

       z_2 ──┐
              ├──→ G_2 ──→ h̃_2  (16×16, residual)
       l_2 ──┘                    │
                                  │
   Ĩ_2 = l_2 + h̃_2  (16×16, 重构图)
                                  │
                                  ↓ u(·) 上采样到 32×32
                                 l_1

═══════ Step 3: 32×32 残差生成（条件 GAN） ═══════

       z_1 ──┐
              ├──→ G_1 ──→ h̃_1  (32×32, residual)
       l_1 ──┘                    │
                                  │
   Ĩ_1 = l_1 + h̃_1  (32×32)
                                  │
                                  ↓ u(·) 上采样到 64×64
                                 l_0

═══════ Step 4: 64×64 残差生成（条件 GAN，最终输出） ═══════

       z_0 ──┐
              ├──→ G_0 ──→ h̃_0  (64×64, residual)
       l_0 ──┘                    │
                                  │
   Ĩ_0 = l_0 + h̃_0  ── 最终生成图 (64×64)
```

**采样核心公式**（Eqn.5）: $\tilde{I}_k = u(\tilde{I}_{k+1}) + G_k(z_k, u(\tilde{I}_{k+1}))$，从 $\tilde{I}_{K+1} = 0$ 开始递归。

#### 训练流程（Fig.2，每层 k 独立训练，无跨层梯度）

```
对每张训练图 I 构建真实 Laplacian pyramid:
   I_0       = I
   I_{k+1}   = d(I_k)              (下采样)
   l_k       = u(I_{k+1})          (该层低频上下文)
   h_k       = I_k - l_k           (该层真实 residual = band-pass)

═══════ 训练第 k 层 (k = 0, 1, ..., K-1) ═══════

       ┌────────────────────────────────┐
       │  扔硬币 (50% 概率选真 / 假)   │
       └────────────────────────────────┘
              ↓                  ↓
       (真分支)              (假分支)
       h_k = I_k - l_k       h̃_k = G_k(z_k, l_k)
              │                  │
              └────────┬─────────┘
                       ↓
       把 l_k 显式加到 h_k / h̃_k 上
       (在 D 第一层卷积之前合并)
                       ↓
              D_k(input, l_k)  ──→  真 / 假
                       ↓
       per-scale GAN minimax loss 反传
                       ↓
       只更新 G_k, D_k     ⚠ 不动其他层

═══════ 训练最粗层 K (无条件 GAN) ═══════

       直接以 I_K 当真样本, z_K → G_K 生成假样本
       D_K 输入只有 h_K = I_K (没有 l_k)
```

- **关键 1：D_k 同时看 (h_k, l_k)** —— 不仅判断"残差像不像真实残差"，还判断"残差和低频上下文是否一致"，这是 cGAN 的结构性监督
- **关键 2：每层完全独立训练** —— 没有跨层联合 loss、没有跨层梯度。LSUN 上每层训几天。**这是 residual supervision 最极端的形式：连优化都解耦**
- **关键 3：噪声 z_k 拼接到 l_k 当作"第 4 个 color channel"** —— CIFAR10 上的简单做法
- **关键 4：最粗层 G_K 用全连接 MLP**（CIFAR10 上 8×8 尺度），其余层用 ConvNet

### 3.2 Loss / 公式（CGAN 的 minimax）

$$L_k = \min_{G_k} \max_{D_k} \mathbb{E}_{h_k, l_k}[\log D_k(h_k, l_k)] + \mathbb{E}_{z_k, l_k}[\log(1 - D_k(G_k(z_k, l_k), l_k))]$$

- **Laplacian 分解**：h_k = I_k - u(I_{k+1})（band-pass residual）；最后一层 h_K = I_K（low-frequency base）
- **采样重构**：Ĩ_k = u(Ĩ_{k+1}) + G_k(z_k, u(Ĩ_{k+1}))
- **与 vanilla GAN 的差异**：在每个 scale k，D_k 的输入是 (l_k, h_k) 的 pair，而不仅仅是 h_k → D 不只是判断"这张差分像不像真实差分"，还判断"这张差分和低频上下文是否一致"（**结构性的 conditioning 监督**）
- **class conditional 变体**：额外拼接 class embedding 到 G_k 和 D_k

### 3.3 架构设计
- **多尺度**：✅ K 层递归级联（CIFAR10: 8→14→28，3 层。STL: 8→16→32→64→96，5 层。LSUN: 4→8→16→32→64，5 层）
- **级联方向**：✅ 严格的 coarse→fine，每层**条件于上一层的输出**（Fig.1 的 l_k 横向传递）
- **参数共享**：❌ 每层是**完全独立的 G_k + D_k**（不共享参数）
- **独立训练**：✅ 每层单独训，没有跨层梯度——LSUN 上每层训几天。**这是 residual supervision 最极端的形式：连优化都是解耦的**
- **G_k 输入形式**：z_k 作为 l_k 的第 4 个"color channel"拼接（CIFAR10 上）——简单有效
- **D_k 输入形式**：l_k 显式加到 h_k/h̃_k 上才进 D 的第一层卷积 → D 同时看"差分 + 上下文"
- **最粗层 G_K**：全连接 MLP（8×8 尺度），不用卷积

## 4. Experiments
- **Datasets**: CIFAR10 (32×32)、STL (96×96，仅无标注部分)、LSUN (~10M 张，downsample 到 64×64，10 个场景类各训一个模型)
- **Baselines**: 原始 GAN [Goodfellow 2014]（同一作者的 reimplemented，加了数据增强）
- **Key metrics**: Parzen-window log-likelihood、human evaluation (% classified real)、qualitative visual

### 关键结果数字

| 方法 | CIFAR10 Log-Lik | Human eval (% classified as real) |
|---|---|---|
| GAN [Goodfellow 2014] | -3617 ± 353 | ~10% (≈ human baseline) |
| **LAPGAN** | **-1799** ± 826 (+1818) | ~30% |
| **CC-LAPGAN** (class cond.) | N/A | **~40%** |
| Real images | N/A | >90% |

**定性结果**（从文字描述和已知回顾）：
- LSUN 上生成了 64×64 令人信服的场景（塔楼/卧室/教堂），"no other generative model had been able to produce samples of this complexity"
- STL 上物体轮廓模糊但纹理锐利
- CIFAR10 类条件 LAPGAN 样本有清晰的物体边缘和结构

### Ablations they did
- 每个级别的 G_k/D_k 架构做了不同尝试（hidden units、layer depth、dropout、batch norm），选最佳 per dataset
- data augmentation (crop) → 提升 CIFAR10
- class conditioning → 显著提升（~30% → ~40%）
- LSUN：用 4×4 validation image 启动采样（类似 image-to-image 的 conditional 应用）

## 5. ⚡ 与 Wavelet CT 的关系（最重要）

### 直接相关：是——**这是 residual supervision 的鼻祖 & 最直接的架构模板**

LAPGAN 和本 idea 的相似度远高过 FPN：都是用频域/金字塔分解，都是 coarse-to-fine，每个尺度都有独立的监督信号。**几乎可以把 LAPGAN 看作"GAN 版的 Wavelet-CT"**。

### 可借鉴的

**架构设计（最 actionable 的部分）**:

1. **Laplacian pyramid ≈ wavelet 分解**：h_k = I_k - u(I_{k+1}) 这个 band-pass residual 在数学上就是离散 wavelet 的一阶差分近似。**我们的 wavelet 分解可以看作是 LAPGAN 的 linear decomposition 的自然推广**（从 Laplacian → 更一般的小波基，能更好地捕获方向性信息）。报告里可以直接写："Our wavelet decomposition naturally subsumes the Laplacian pyramid [Denton 2015] as a special case."

2. **Conditioning 输入端**：G_k 的输入是 (z_k, l_k)，其中 l_k = u(I_{k+1}) 是低频上采样。实现方式是"z_k 作为额外 channel 拼到 l_k 上"。**对应到 CT**：我们可以把 z（或 CT 里的噪声级 σ）和 low-pass context l_k 拼成多通道输入，或者直接利用 CT 已有的 "σ 注入机制"（FiLM/scale-shift），把 l_k 作为额外的 spatial condition 加进去。

3. **D_k 的上下文输入：D_k sees (l_k, h_k)**。D 不仅看差分长什么样，还看它和低频 context 是否一致 → **对应到 CT**：CT 是 self-consistency loss，没有 D。但我们多尺度的 CT loss 需要确保"每个尺度的 residual 和它的 context 是一致的"——这个一致性可以通过 wavelet 重构自然地 enforce：final image = base + sum residuals，如果某个 residual 和 base 不协调，重构的 final image 就假，CT loss 上去惩罚的就是整个 pipeline。**wavelet 重构天然代替了 D 的"全局判断"职能**。

4. **完全独立训练**：每层 G_k 独立训 → 没有跨层梯度流、没有 error propagation。**在 CT setting 下这是否可取？** 因为 CT 天然是 end-to-end 的蒸馏，断掉梯度意味着失去"让高频适配低频"的学习信号 → **我们大概率不能独立训**，但这是 LAPGAN 给的一个有趣的 ablation 灵感："Joint vs decoupled training in Wavelet-CT"。

5. **class conditioning 的简单实现**：1-hot class → linear layer → reshape 成 1 个 plane → 拼到第一层 feature maps。**对应到 CT**：不需要 class condition，但我们可能需要 scale embedding（类似 FPN 的 level k 信息），可以用同样的方式实现。

**Loss / 监督形式（直接定义我们的"residual supervision"原型）**:
- LAPGAN 的 loss 是 **per-scale GAN minimax**：每个 G_k 只负责"这一层的 residual 看起来像真实 residual"
- **对应到 CT residual supervision**：每个尺度 k 的 CT loss = loss(wavelet_coeff_pred[k], wavelet_coeff_target[k])；sum supervision 的版本 = loss(final_image_pred, final_image_target)。**LAPGAN 证明 residual 方式有效**（训练信号更 focused），但它是 adversarial loss 不是 consistency loss → **我们的 novelty 就是把 residual supervision 从 GAN loss 换成 CT loss**。
- 最粗的一层（base）不需要条件：G_K(z_K) 直接生成 → 对应我们 wavelet 里最粗的 base 也是无条件（或条件于 noise）

**训练策略**:
- 每层不同架构（粗尺度用 MLP，细尺度用 ConvNet）→ **对应到 CT**：我们的各尺度是否需要不同的 head capacity？Wavelet 粗尺度 base 频率低但全局结构重要，细尺度高频局部稀疏 → 可能 base 需要更宽的 head，residual 可以用轻量 head
- 每层训练的 monitor：Parzen-window log-lik → 与我们关系不大，我们 CT 直接监控 FID/loss

### 与本 idea 的差异（delta）

**LAPGAN 没做但我们要做的**:
1. **CT/consistency loss 代替 GAN adversarial loss**：LAPGAN 每层是 GAN loss + independent training；我们要做的是所有层同时用 CT loss 训，端到端。**这是最大的 delta**。
2. **Wavelet 代替 Laplacian pyramid**：Laplacian 是 isotropic band-pass（旋转等效），wavelet 是 direction-sensitive（水平/垂直/对角）。Wavelet 能提供更丰富的频率方向解耦 → 可能 better few-step quality
3. **单网络/共享 backbone**：LAPGAN 是 K 个独立 G_k → K 倍参数。我们要做的是单网络 + multi-head（类似 Matryoshka）→ 参数效率高很多
4. **Multi-scale CT loss 的 joint optimization**：LAPGAN 完全独立训 → 没有尺度间梯度协调。在 CT 下，joint training 可能更有利于 CT 的 self-consistency 性质（因为 CT 要求所有 σ 对同一 x_0 一致，而多尺度的 x_0 本身就是分解后的）
5. **Sum vs residual ablation**：LAPGAN 是 pure residual，没和 sum 做过对比 → 这是我们的核心 ablation

**LAPGAN 做了但我们要避免/改造的**:
1. **独立网络 → 改成共享骨干**：独立架构参数太多了。共享 backbone + multi-head + scale embedding 更合理（尤其是 CT 已经从 diffusion distillation 继承了这一范式）
2. **GAN loss → 改成 CT loss**：GAN 训练不稳定（比 CT 还不稳定），如果我们多层都用 GAN loss 会灾难性的不稳定。CT loss 在一致性上的 self-supervision 可能天然更稳
3. **Laplacian 的尺度数固定为 log2**：我们的 wavelet 可以更灵活——例如在 28×28 MNIST 上用 Haar 2-level (14×14 + 7×7) 而不是严格的 power-of-2

### 如果对方做过类似 idea：novelty 还剩什么？
LAPGAN 和我们的 idea 在"结构骨架"上高度相似（coarse-to-fine + frequency decomposition + per-scale supervision）。但关键 delta 是：
- **LAPGAN = 全独立网络 + GAN loss + Laplacian**
- **Wavelet-CT = (可能)共享 backbone + CT loss + Wavelet**

**Novelty 保住了**：因为 CT 的 self-consistency target 和 GAN 的 adversarial target 在动力学上完全不一样（CT 沿着 ODE 轨迹约束映射光滑，GAN 是在样本空间打 minimax game）。**把"LAPGAN 骨架"套上"CT 发动机"是成立的 novelty**。

在 report/paper 里的写法：
> "We build on the coarse-to-fine cascade paradigm introduced by Denton et al. (LAPGAN), but replace (a) independent per-scale GANs with a shared-backbone consistency-trained network, and (b) a fixed Laplacian pyramid with learnable multi-scale wavelet supervision. This combination inherits LAPGAN's proven coarse-to-fine architecture benefit while enabling few-step generation with CT-level stability."

## 6. 我的疑问 / 不确定点
1. **LAPGAN 的完全独立训练在 CT setting 下意味着什么？** 如果我们尝试"decoupled Wavelet-CT"（每层独立训 CT），粗尺度的错误会累积到细尺度 → 可能比 joint training 差。但 joint training 又可能让 loss 更难收敛（多个尺度的梯度互相干扰）。**建议 Day 4/5 实验里加一个 "joint vs decoupled" 的 ablation**。
2. **Laplacian pyramid 要求下采样到 2 的整数次幂**。MNIST 是 28×28 → 8×8 或 7×7？如果按 LAPGAN 的 8→14→28 拆分，那 base=7×7 是 8×7×7 的 patch。**实际上 Laplacian pyramid 可以在 CIFAR10 的 32 上完美工作，但 MNIST 28 需要处理**。
3. **z_k 作为额外 channel 拼接的条件注入方式 vs CT 的 FiLM**：CT 当前用的是 FiLM (scale+shift)，如果我们再加 l_k 条件 → 是拼到输入里还是也走 FiLM？倾向拼到输入（类似 LAPGAN）+ 保持 FiLM 做时间条件 → 两个条件用不同的注入路径。
4. **LAPGAN 的 LSUN 上用的是 4×4 validation image 启动**，暗示 LAPGAN 在 image-to-image 任务上特别强。**对应的 CT 实验方向**：除了 unconditional generation，我们可以考虑"给定 low-res base，生成 high-res residual"的 conditional setting → Day 6 后的扩展

## 7. 关键引用（值得我也读的）
- [ ] **Laplacian Pyramid as Compact Image Code** [Burt & Adelson 1983] — 原始 Laplacian pyramid 公式，写 wavelet 相关章节时的理论引用来源
- [x] **Portilla & Simoncelli 2000** — steerable pyramid wavelet + texture synthesis，LAPGAN ref [20] 说"similar to our use of a Laplacian pyramid" → **这是 wavelet × generation 的更早期工作，Block C 阅读时重点读**
- [ ] **Conditional GAN** [Mirza & Osindero 2014] — LAPGAN 的直接技术前身（conditioning 机制），理解 cGAN 的 l variable 如何注入 G/D
- [x] **Ghiasi & Fowlkes 2016 — Laplacian Pyramid for FCN** — 前面 FPN 笔记已经推荐过，**下一篇读这个**，把 Laplacian pyramid × dense prediction 的线补完

---

**下一步建议读**: Ghiasi & Fowlkes Laplacian Pyramid for FCN Segmentation（残差监督 × dense prediction）+ 然后转向 CV 里 deep supervision 的经典（HED / Stacked Hourglass）。或者如果你想先补完"残差 vs 和"的对比，先读 HED/UNet++（deep supervision = sum supervision 在 CV 里最典型的形式），然后再回来读 Laplacian × segmentation。

---

# MSG-GAN — Multi-Scale Gradients for Generative Adversarial Networks

## 0. Metadata
- **Authors**: Animesh Karnewar, Oliver Wang
- **Venue & Year**: CVPR 2020（arXiv 1903.06048v4, 2019-03, 最终 2020-06）
- **arXiv / Link**: https://arxiv.org/abs/1903.06048
- **Code**: https://github.com/akanimax/msg-stylegan-tf
- **Read date**: 2026-05-08
- **Reading time**: ~40 min (10 pages + 附录)
- **Priority**: ⭐⭐⭐
- **Tags**: #sum-supervision #multi-scale-gradients #stable-training #single-discriminator #shared-generator #generative

## 1. TL;DR（一句话）
> **单 G + 单 D，G 的每个中间层通过 1×1 conv 输出一个不同分辨率的图像，全部同时送进 D 的不同深度层判断真假**；梯度通过 D 的多尺度入口同时反传到 G 的所有层，替代了 progressive growing 且更稳定。

## 2. Problem & Motivation
- **要解决的问题**: GAN 训练不稳定。根因：fake/real 分布 support 不重叠时，D 传给 G 的梯度随机无信息。
- **已有方法的局限**:
  - **ProGAN（progressive growing）**：逐层训练，先低分辨率稳了再加高分辨率层。问题：额外超参数（每分辨率分多少 iteration、不同 lr、fade-in 节奏）、训练复杂、会产生 phase artifacts（固定空间位置的固定特征）
  - **VDB**：自适应变分 bottleneck，限制 D 只关注最区分性的特征 → 和 MSG-GAN 正交
  - **多 D 方案**（StackGAN、Pix2PixHD 等）：多个独立 D 做不同分辨率 → **参数爆炸**（每个 D 需要自己的下采样塔），不同 D 之间不能共享信息
  - **LAPGAN**：完全独立的 G_k + D_k 每层，**无跨尺度梯度流**
- **本文 insight**:
  1. **梯度同时到所有尺度**：让 D 不仅看 G 的最终输出，还看所有中间层的输出 → G 每一层都有自己的"直达梯度"，D→G 的信息流不再单薄
  2. **单 D 单 G，零额外超参数**：和多 D 方案比参数量线性增长（不是指数），和多阶段训练比不需要 per-resolution schedule
  3. **替代 progressive growing**：MSG-GAN 在所有数据集上用同一套固定超参数 → "out-of-the-box" 易用

## 3. Method（关键）

### 3.1 核心思想 — 单生成器多尺度输出 + 单判别器多尺度输入

```
═══════ Generator: 一个网络，多个 toRGB 出口 ═══════

   z (latent vector)
     │
     ▼
  ┌─────────────────┐
  │  g_block_4×4    │  → activation_4×4
  └────────┬────────┘
           ├────── 1×1 conv (toRGB) ──→ RGB_4    ─────────┐
           │                                              │
           ▼ upsample                                     │
  ┌─────────────────┐                                     │
  │  g_block_8×8    │  → activation_8×8                   │
  └────────┬────────┘                                     │
           ├────── 1×1 conv (toRGB) ──→ RGB_8     ────────┤
           │                                              │
           ▼ upsample                                     │
  ┌─────────────────┐                                     │
  │  g_block_16×16  │  → activation_16×16                 │
  └────────┬────────┘                                     │
           ├────── 1×1 conv (toRGB) ──→ RGB_16    ────────┤
           │                                              │
           ▼ upsample                                     │
       ... (省略中间块) ...                               │
           │                                              │
           ▼                                              │
  ┌─────────────────┐                                     │
  │ g_block_1024×1024│ → activation_final                 │
  └────────┬────────┘                                     │
           └────── 1×1 conv (toRGB) ──→ RGB_final  ───────┤
                                                          │
═══════ Discriminator: 一个网络，从最高分辨率开始顺次接收 ═══════
                                                          │
                  ┌───────────────────────────────────────┘
                  │
   RGB_final ──→ │
                  ▼
              ┌──────────────┐
              │ d_block_1024 │  (正常输入第一层)
              └──────┬───────┘
                     │ AvgPool 2×
                     ▼
   RGB_512 ── φ ──→ ┌──────────────┐
                    │ d_block_512  │
                    └──────┬───────┘
                           │ AvgPool
                           ▼
       ... (省略中间块) ...
                           ▼
   RGB_16  ── φ ──→ ┌──────────────┐
                    │ d_block_16×16│
                    └──────┬───────┘
                           │ AvgPool
                           ▼
   RGB_8   ── φ ──→ ┌──────────────┐
                    │ d_block_8×8  │
                    └──────┬───────┘
                           │ AvgPool
                           ▼
   RGB_4   ── φ ──→ ┌──────────────┐
                    │ d_block_4×4  │
                    └──────┬───────┘
                           ▼
                MinBatchStdDev → Conv → critic score (单标量)
```

**关键对应关系**：G 的 N 个 toRGB 出口 ↔ D 的 N 个 RGB 入口（分辨率一一对应、配对监督）。梯度从 D 的 critic score 同时回传到 G 的 N 个 g_block。

三个关键设计：
1. **G 的每个中间块后挂一个 `1×1 conv → RGB` 输出（r_i）**：把 activation volume 投影成 RGB 图像，不同分辨率 4×4, 8×8, ..., 1024×1024。**这个 1×1 当作 regularizer**——强制每层的 feature map 都能被直接投影到像素空间。
2. **D 的对应层接收多尺度 RGB**：D 是正常的卷积下采样塔（AvgPool → Conv → Conv），但在每个分辨率块入口，把 G 过来的 RGB 图像通过 combine function φ 拼到当前 activation volume 上。
3. **单 D 的一个 critic score**：D 最终只有一个 scalar 输出，但看的是所有尺度拼起来的信息。梯度 penalty 取所有尺度输入的平均。

**combine function φ 的三种变体**（Eq 11-13）：
- `φ_simple(x1, x2) = [x1; x2]` — 直接 channelwise concat（ProGAN 上用最好）
- `φ_lin_cat = [1×1 conv(x1); x2]` — 先投影再 concat
- `φ_cat_lin = 1×1 conv([x1; x2])` — 先 concat 再投影（StyleGAN 上用最好）

### 3.2 Loss / 公式

**没有新 loss**。MSG-GAN 只是架构改造，loss 完全沿用所使用的 baseline：ProGAN 用 WGAN-GP，StyleGAN 用 Non-saturating GAN loss + 1-sided GP。唯一的适配是把梯度 penalty **在所有尺度输入上取平均**。

### 3.3 架构设计
- **多尺度**：✅ G 在所有中间层同步输出多分辨率图像；D 在所有对应层同步接收
- **级联**：❌ 没有。所有尺度**并行输出**，D 并行接收 → **纯 sum/parallel 监督**
- **参数共享**：✅ 强烈的共享——单 G + 单 D，只是加了 1×1 toRGB 层和 concat 操作，参数量线性增长（vs 多 D 的指数增长）
- **训练方式**：**joint end-to-end**（和 LAPGAN 的独立训练完全相反；和 FPN 的 joint 相同，但 FPN 是 detection 不是 generation）

## 4. Experiments
- **Datasets**: CIFAR10 (32²), Oxford Flowers (256²), LSUN Churches (256²), Indian Celebs (256²), CelebA-HQ (1024²), FFHQ (1024²)
- **Baselines**: ProGAN [15], StyleGAN [16]（作者复现标 *）
- **Key metrics**: FID（越低越好）, IS（CIFAR10）; 训练时间、所需 GPUs、"# Real Images shown"（收敛所需的数据量）

### 关键结果数字

| Dataset | Method | #Real Imgs | FID |
|---|---|---|---|
| Oxford Flowers 256² | ProGAN* | 10M | 60.40 |
| | MSG-ProGAN | **1.7M** | **28.27** |
| | StyleGAN* | 7.2M | 64.70 |
| | MSG-StyleGAN | **1.6M** | **19.60** |
| LSUN Churches 256² | StyleGAN* | 25M | 6.58 |
| | MSG-StyleGAN | 24M | **5.2** |
| FFHQ 1024² | ProGAN* | 12M | 9.49 |
| | MSG-ProGAN | 6M | 8.36 |
| | StyleGAN | 25M | 4.40 |
| | MSG-StyleGAN | 9.6M | 5.8 |

**收敛更快**（需要更少的 real images），FID 在 mid-resolution 上全面碾压 baseline。1024² 上 FID 略输 StyleGAN（5.8 vs 4.40），但**训练极简**（零 per-resolution 超参数、零 fade-in）。

### Ablations they did
- **多尺度连接粒度**（最关键的 ablation，Table 4）：
  | 连接粒度 | FID(FFHQ) |
  |---|---|
  | DCGAN (无连接) | 14.20 |
  | Coarse only (4²+8²) | 10.84 |
  | Middle only (16²+32²) | **9.17** |
  | Fine only (64²+) | 9.74 |
  | **All (MSG-ProGAN)** | **8.36** |
  > 启示：**中间层连接贡献最大**（9.17 仅次于全连的 8.36），单靠粗或细都不够
- **combine function φ 三种实现**（Table 5）：φ_simple 在 ProGAN 上最好（8.36），φ_cat_lin 在 StyleGAN 上最好（5.80）
- **Learning rate 鲁棒性**（Table 3）：lr ∈ [0.001, 0.01] 全收敛，IS 在 7.92–8.63

### 训练稳定性（Fig.6）
MSE between consecutive epoch samples on fixed latent → MSG-ProGAN 所有分辨率同步快速收敛，ProGAN 只低分辨率收敛、高分辨率持续漂移。

## 5. ⚡ 与 Wavelet CT 的关系（最重要）

### 直接相关：是——**这是 sum supervision 在 generation 上的最干净实现**

MSG-GAN 和本 idea 的相似度：**结构骨架几乎完全相同**。MSG-GAN = 单网络多分辨率输出 + 所有分辨率同时被监督 + end-to-end joint training。把 MSG-GAN 的 GAN loss 换成 multi-scale CT loss，把 MSG-GAN 的多分辨率 RGB 输出换成 wavelet 分解的目标，几乎就是我们要做的事情。

### 可借鉴的

**架构设计（最高参考价值，比 FPN/LAPGAN 更 actionable）**:

1. **`1×1 conv → RGB/目标` 作为 toWavelet 层**：MSG-GAN 在每个 G 中间块后用 1×1 conv 把 activation 投到 RGB → 我们要做的就是在 decoder 每个分辨率层后挂 1×1 conv → wavelet 系数。**完全一致的实现模板**。
2. **单网络 + 多输出 + joint training**：MSG-GAN 证明不需要每层独立网络（like LAPGAN），也不需要 progressive training（like ProGAN），单网络一次训到底就能 work → **直接回应"Wavelet-CT 要不要每尺度独立网络"的问题——不需要**
3. **φ 函数（combine function）** 对应我们的 "low-res context 如何注入高频预测"：MSG-GAN 在 D 侧用 φ 把多尺度输入拼起来；我们在 G 侧需要把 low-res wavelet base 的 context 传给高分辨率 wavelet residual 的预测分支 → 可能同样需要设计 combine function（看是 concat / FiLM / cross-attn）
4. **中间层贡献最大**：MSG-GAN 的 coarse/middle/fine ablation 证明中间尺度连接贡献最大 → 暗示**不是分辨率越高监督越有用**，对我们选择 wavelet 分解层数有指导意义（可能 3 层就够了，不需要 4–5 层）
5. **超参数鲁棒**：MSG-GAN 证明多尺度梯度让训练对各种 lr 都很鲁棒 → 如果 Wavelet-CT 也获得类似的稳定效果，那将是对 vanilla CT 的重要提升
6. **MinBatchStdDev 多尺度扩展**：每层 D block 入口都加一个 MinBatchStdDev → 跨尺度 diversity 监督

### 与本 idea 的差异（delta）

**MSG-GAN 没做但我们要做的**:
1. **GAN loss → CT loss**：MSG-GAN 的 D 提供 adversarial 监督；我们要用 CT 的 self-consistency loss 做无教师监督。**这是最大的 delta**——multi-scale CT loss 在 generation 上没有任何已有工作
2. **Wavelet 分解的目标**：MSG-GAN 每个尺度输出同样的 RGB 图像（只是分辨率不同）；我们的每个尺度输出的是 wavelet 系数（base + HH/HL/LH residuals），目标数学性质完全不同
3. **Residual vs sum 的对比**：MSG-GAN 是纯 sum（所有尺度 independently 同时被监督），没有 residual version → 我们的核心 ablation 没被做过
4. **无 D 的设定**：MSG-GAN 依赖 D 的多尺度输入；CT 没有 D → "多尺度监督信号从哪里来"这个问题必须靠 wavelet 分解来回答（wavelet 系数本身就是多尺度目标）

**MSG-GAN 做了但我们要借鉴/改造的**:
1. **1×1 toRGB**：对 wavelet 高频系数，1×1 够不够？Wavelet HH 子带是稀疏的（大部分像素接近 0），1×1 可能可以；但 base 子带包含全局低频结构，可能需要更大的感受野 → **toWavelet 层的设计应作为 ablation**
2. **所有尺度等权监督**：MSG-GAN 每个尺度的输出都进 D 的 critic → 暗示初版 Wavelet-CT 可以所有尺度等权求和 CT loss
3. **没有 mixing regularization**：MSG-StyleGAN 不能用 mixing regularization → CT 没有这个问题

### 如果对方做过类似 idea：novelty 还剩什么？
MSG-GAN 用的是 GAN loss + 无分解的多分辨率监督。**我们 = MSG-GAN 的架构骨架 × CT loss × wavelet 分解**。三个组件都有先例，但**组合从未出现**：
- Multi-scale gradient architecture → MSG-GAN 证明了
- CT loss → Song 2023 定义了
- Wavelet decomposition → Burt & Adelson 1983 定义了
- **三者的结合 = Wavelet-CT** → 未被探索

## 6. 我的疑问 / 不确定点
1. **无 D 的 CT 里，"多尺度梯度"的等价物是什么？** MSG-GAN 的核心卖点是"梯度从 D 同时到 G 的所有层"；CT 没有 D，梯度来自 multi-scale CT loss 的监督信号 → **本质上是 multi-scale self-consistency 提供正则化**，和 MSG-GAN 的机制不同。需要实验验证"multi-scale CT loss 是否也能像 MSG-GAN 一样稳定训练"
2. **MSG-GAN 的 combine function φ 对应我们的什么？** 如果做 residual version（高频 condition on 低频），低频 context 如何注入高频分支 → 可以借鉴 φ_simple/φ_lin_cat/φ_cat_lin 的三种模式做 ablation
3. **MSG-GAN 在 1024² 上 FID 略输 StyleGAN**：作者归因于 mixing regularization 的缺失。**对我们的启示**：multi-scale supervision 不一定在所有 setting 下都赢单尺度，尤其是当单尺度 baseline 已经有很强的 regularization → vanilla CT 的 EMA + Pseudo-Huber 已经是一种强 regularization，multi-scale 可能增益不如 GAN 那么显著

## 7. 关键引用（值得我也读的）
- [ ] **ProGAN** [Karras et al. 2018] — MSG-GAN 的 baseline，progressive growing 要理解后才能明白 MSG-GAN "替代了什么"
- [ ] **StyleGAN** [Karras et al. 2019] — 另一条 baseline，"mixing regularization"是 MSG-GAN 丢失的东西
- [x] **LAPGAN** [Denton et al. 2015] — MSG-GAN 明确对比：LAPGAN 是多个独立网络 vs MSG-GAN 是单网络多输出
- [x] **StackGAN** [Zhang et al. 2017] — 多 D 方案的代表，MSG-GAN 避开了它的"参数爆炸"问题

---

# Day 1 文献调研报告 — Multi-scale Supervision 架构全景 + Wavelet-CT 思路整理

> 阅读日期：2026-05-08 | 论文数：3/8（Day 1 上午 D-0/D-1 完成）

## 1. 已读论文速览

| # | 论文 | 范式 | 核心架构 | 监督类型 | 对 Wavelet-CT 最大价值 |
|---|---|---|---|---|---|
| D0b | FPN (Lin 2017) | CV detection | 单 backbone + top-down lateral + 每尺度独立 head (共享参数) | **Sum** | "每尺度独立 head + 等权 sum loss"模板；lateral merge 设计 |
| D1 | LAPGAN (Denton 2015) | GAN generation | K 个独立 G_k + 级联条件 (粗→细) | **Residual** (cascaded) | "residual supervision 鼻祖"；Laplacian ≈ wavelet 原型；conditioning 输入方式 |
| D2 | MSG-GAN (Karnewar 2020) | GAN generation | **单 G + 单 D**，G 每层 1×1→RGB，D 每层 concat 接收 | **Sum** (parallel, joint) | **最直接架构模板**：单网络多输出 + joint training + 所有尺度同时监督 |

## 2. 三种范式的对比：我的 Wavelet-CT 该选哪个？

```
                        FPN                      LAPGAN                    MSG-GAN
                        │                        │                         │
监督类型:              Sum (parallel)          Residual (cascaded)      Sum (parallel)
多尺度输出:            同时                     逐级                       同时
尺度间依赖:            无（仅 top-down feature） 严格级联                    无
参数:                  共享 head                完全独立 G_k+D_k            单网络共享
训练:                  Joint end-to-end         完全解耦逐层训              Joint end-to-end
Loss 数:               N 个（每尺度 1 个）       N 个（每尺度 1 个 GAN）    1 个（D 的所有尺度汇总）
```

**我的初版设计选择**（基于今天阅读）：
- **骨架**: MSG-GAN 风格（单网络 + 多分辨率输出 head + joint training）——最简洁，参数效率最高
- **输出目标**: 把 MSG-GAN 的"多分辨率 RGB"换成"wavelet 分解系数"（base + residuals）
- **监督策略**: 两种都做对比——Sum version（每个 head 独立 CT loss 等权相加）vs Residual version（coarse-to-fine 级联条件 + 每个 residual head 自己的 CT loss）
- **FPN 的贡献**: head 共享参数、lateral connection 用 add 还是 concat
- **LAPGAN 的贡献**: residual 的 conditioning 方式（低频 context 作为输入拼到 z 旁边）、wavelet ≈ Laplacian 的理论连接
- **MSG-GAN 的贡献**: 1×1 toWavelet 层、joint training、multi-scale 无需 progressive

## 3. 我对 Wavelet 的理解（联系到今天读的论文）

**Wavelet 分解本质上是 Laplacian pyramid 的泛化**：
- Laplacian pyramid：h_k = I_k - u(I_{k+1})，每层是 isotropic band-pass（方向无关的频带差分）
- Wavelet：每层分解成 4 个子带（LL=base, LH=水平边缘, HL=垂直边缘, HH=对角细节），**保留了方向信息**

```
Laplacian pyramid (LAPGAN 用的):           Wavelet (我要用的):
                                           
I_0 (原图)                                  I_0 (原图)
  │                                          │
  ├─ h_0 (高频残差，无方向区分)               ├─ LL (低频 base)
  ├─ h_1                                  ├─ LH (水平高频)
  ├─ ...                                  ├─ HL (垂直高频)
  └─ I_K (最粗低频)                          └─ HH (对角高频)
                                              │
每个 h_k 是 isotropic 的                   对 LL 递归分解 → 多尺度
```

**Wavelet 相对 Laplacian 的优势（为什么不用 LAPGAN 的 Laplacian）**：
1. **方向分解**：LH/HL/HH 分别捕捉水平/垂直/对角结构 → 生成任务中方向信息很重要（边缘、纹理都有方向性）
2. **更稀疏**：高频系数大多接近 0 → 预测目标更简单（学习预测稀疏残差 vs 学习预测 full band-pass）
3. **正交性**：某些 wavelet（如 Haar、Daubechies）是正交变换，重构无信息损失且数值稳定
4. **Laplacian 是 wavelet 的特例**：Laplacian pyramid ≈ Haar wavelet 的退化版（只分两个频带，不分方向）→ 用 wavelet 可以宣称"generalizes LAPGAN"

## 4. 目前的论文搜索思路（Day 1 下午-晚上 + Day 2）

### 已覆盖（今天三篇 = D-0/D-1 完成）
- ✅ CV 通用多尺度监督 → FPN（sum, detection）
- ✅ GAN 多尺度监督 → LAPGAN（residual, cascaded）+ MSG-GAN（sum, joint）
- ⏳ 还需要读：D-2 Diffusion（Matryoshka, Cascaded Diffusion）— 最贴近 CT 的同期工作

### Day 1 剩余 + Day 2
1. **Matryoshka Diffusion** (D6, ⭐⭐⭐) — 单网络嵌套多分辨率，**是 diffusion 版的 MSG-GAN**，和 CT 最相关（CT 也是 diffusion 蒸馏来的）
2. **HED / UNet++** (D0c/D0d, ⭐⭐) — Deep supervision 的 CV 经典，适合写作时引用"multi-scale loss 在 CV 里已经被广泛验证"
3. **Ghiasi & Fowlkes — Laplacian Pyramid for FCN** (⭐⭐⭐) — Laplacian × dense prediction，**和本 idea 最同构的 CV 工作**
4. **Block B & C 快速扫描** — C4（GitHub wavelet × CT 搜索）是 novelty 判定关键；B 是写作时的理论引用

### 当前阶段判断
- 架构参考已经足够做出第一版设计（MSG-GAN skeleton + FPN-style heads + LAPGAN-style conditioning + wavelet targets）
- Day 2 重点转向 **novelty 验证**：确定没有人做过 wavelet × CT
- Block B（频率偏置证据）可以写 report 时再补

## 5. Wavelet-CT 初版网络草图

```
                           CT loss 在多尺度上各算一份
                             ↑    ↑    ↑    ↑
                           [L_ct0][L_ct1][L_ct2][L_ct3]    ← multi-scale supervision
                             ↑    ↑    ↑    ↑
      noise z, σ ──────────→|    |    |    |
                             |    |    |    |
   ┌───────────────────── U-Net (共享 backbone) ────────────────────┐
   │                                                                 │
   │  encoder: 28→14→7→4                                           │
   │  decoder:  4→7→14→28  (skip connection 保留 U-Net 原生)        │
   │              │    │    │    │                                    │
   │              ↓    ↓    ↓    ↓                                    │
   │          [head3][head2][head1][head0]     ← 1×1 conv toWavelet  │
   │              │    │    │    │                                    │
   │          4×4    7×7  14×14 28×28  wavelet coeff                 │
   │          base   res2  res1  res0   (各自对应 wavelet 子带)       │
   └─────────────────────────────────────────────────────────────────┘
                             ↓
                    final image = Σ(wavelet recon)
```

**设计选择（待实验决定）**：
- Head 共享 vs 独立 ← MSG-GAN 的 1×1 toRGB 是共享的，FPN 的 head 也是共享的 → 初版先共享
- Conditioning：residual version 需要 low-res context 注入高频 head → LAPGAN 的 style（拼到输入）+ FiLM（CT 已有的 σ 条件）
- Loss 加权：初版等权，后续 ablation 加 learned weight
- Wavelet 类型：初版用 Haar（最简单、和 Laplacian 最接近），后续试 Db2

---

**今天状态**: 三篇读完，架构模板清晰了。下一步读 Matryoshka Diffusion 把 diffusion 侧的 multi-scale 经验也拿进来，然后转向 novelty 验证。
