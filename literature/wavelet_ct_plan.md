# Wavelet CT — Literature Review & Project Plan

> **Project context**: 基于 Consistency Training (CT) 的 research idea，结合 wavelet 分解 + multi-scale supervision，**唯一目标**：探索能否提升 **Consistency Training (CT)** 的性能（few-step FID / 训练稳定性 / 收敛速度）。
>
> ⚠️ Scope 限定：本项目只针对 **CT**。MF / GAN 不是目标方法，仅可作为架构参考或外部对照（必要时）。
>


---

## 0. Idea 速览（用于 self-check 是否始终对齐）

- **Observation**: Diffusion 采样过程先生成低频、后生成高频；高频依赖低频。
- **Question**: 能否利用这个先验，显式地用 wavelet 分解 + 级联生成 + multi-scale supervision，**提升 Consistency Training 的性能**（更低 1-step / 2-step FID、更稳定的训练、更快收敛）？
- **Method**: 把图像分解为 4×4 / 8×8 / 16×16 / 32×32 四层（base + residuals），生成器从粗到细级联，每个高频层条件于已生成的低频层；最终输出为各层之和。整个 pipeline 用 **CT loss** 监督（multi-scale 版本）。
- **⚠️ 易混淆**：U-Net 的 skip connection 是"内部多尺度**特征融合**"，**不等于**本 idea 的"多尺度**监督**"。CT 的现有代码已经用了 U-Net，但只在最终输出算 1 个 loss；我们要做的是在多个分辨率上各算一个 loss。
- **Open design choice**: Sum supervision vs Residual supervision (teacher forcing) — Boss 留的开放问题，**实验设计的核心 ablation 之一**。
- **Success criterion**: 在同一数据集 + 同一 NFE 下，Wavelet-CT 的 FID 显著低于 vanilla CT / iCT baseline；或在更少训练步数下达到同等 FID。

---

## 1. Literature Review 范围（需要 review 的全部内容）

> 调研按"主题块"组织，每篇论文带 **关键词 / 我要从中带走的信息 / 优先级**。优先级：⭐⭐⭐ 必读 / ⭐⭐ 重要 / ⭐ 选读。
>
> **阅读顺序（重要，已根据"先看可迁移架构"调整）**:
> 1. **Block D — Multi-scale supervision 架构参考**（CV/GAN/Diffusion）← **先读，最 actionable**
> 2. Block C — Wavelet × 生成模型（novelty 判定）
> 3. Block B — Frequency bias（写作时的理论引用）
> 4. Block A — CT（已有基础，只速读 A2 抄 baseline 超参）

---

### Block A — Consistency Training 系列（已有基础，**只补关键 delta**）

> ✅ A1 (Consistency Models, Song 2023) **已读**，CT 基本概念、L_CT loss、CT vs CD 的区别已经掌握。
> 这个 block 只需要快速扫一下 A2/A3 的 **核心 trick** 和 **实验配置**，目的是后面写 baseline / 实验部分时知道行业标准。**不再精读**。

| # | 论文 | 关键词 | 优先级 | 只需要带走的信息 |
|---|---|---|---|---|
| A1 | **Consistency Models** (Song et al., ICML 2023) | `consistency models song 2023 arxiv 2303.01469` | ✅ 已读，跳过 | — |
| A2 | **Improved Techniques for Training CT (iCT)** (Song & Dhariwal, ICLR 2024) | `improved consistency training song 2024` | ⭐⭐ 速读 | 只看：噪声调度、Pseudo-Huber loss、EMA、超参表（用作我的 baseline 配置） |
| A3 | **Simplifying CT (sCT/sCD)** (Lu & Song, 2024) | `simplifying consistency training lu song` | ⭐ 扫一眼 | 只看：实验表格 + 是否提到 frequency / multi-scale，知道 SOTA 在哪即可 |
| A4 | **Easy Consistency Tuning (ECT)** (Geng et al., 2024) | `easy consistency tuning ECT geng` | ⭐ 可跳过 | 与本 idea 关系不大，确认一下不冲突即可 |

> ❌ **Mean Flow / few-step GAN 不在本项目目标范围内**，不需要读。

**速读时只关心两个问题**（不再深挖 CT 本身）:
- iCT 的标准训练 recipe 是什么？（直接抄作为我的 baseline 超参）
- A2/A3 里有没有提到 frequency / multi-scale / wavelet？（如果有，立刻深读那一段；如果没有，move on）

---

### Block B — Frequency Bias of Diffusion（idea 的理论基础，必读）

> ⚠️ 这个 block 的目的是**为 PDF 中"低频先于高频生成"的观察找到 ground truth 引用**，否则 report 里没法严肃地写"This observation is supported by..."。

| # | 论文 / 资源 | 关键词 | 优先级 | 带走的信息 |
|---|---|---|---|---|
| B1 | **Generative Modelling with Inverse Heat Dissipation** (Rissanen et al., ICLR 2023) | `inverse heat dissipation generative model rissanen` | ⭐⭐⭐ | 把 diffusion 看作热扩散，理论证明逆过程是从粗到细 |
| B2 | **Perception Prioritized Training of Diffusion Models** (Choi et al., CVPR 2022) | `perception prioritized training diffusion P2 choi` | ⭐⭐⭐ | 实证分析不同时间步对应的频率成分 |
| B3 | **Diffusion is spectral autoregression** (Dieleman, 2024 blog) | `diffusion spectral autoregression dieleman` | ⭐⭐⭐ | 一个广为流传的观点：diffusion 本质上是频域上的自回归 |
| B4 | **f-DM: Frequency Domain Diffusion** (Gu et al.) | `f-DM frequency domain diffusion model` | ⭐⭐ | 直接在频域做 diffusion 的代表工作 |
| B5 | Spectral bias / spectral analysis of diffusion (综合搜索) | `spectral bias diffusion model frequency analysis` | ⭐ | 找综述类文章 |

**核心带走问题**:
- "低频先于高频生成"有哪些理论 / 实证证据？
- 这个现象在 **few-step CT** 设定下是否仍成立？（关键，决定 idea 在 CT 上是否合理）
- 有没有量化分析（如 power spectral density 随时间的变化）？

---

### Block C — Wavelet + 生成模型（核心相关工作，重点）

> 这个 block 决定 **idea 的真正 novelty 在哪里**。如果有人做过 wavelet + CT，那 gap 就要重新定位。

| # | 论文 | 关键词 | 优先级 | 带走的信息 |
|---|---|---|---|---|
| C1 | **WaveDiff / Wavelet Diffusion Models** (Phung et al., NeurIPS 2023) | `wavelet diffusion model wavediff phung` | ⭐⭐⭐ | wavelet 在 diffusion 里最经典的用法 |
| C2 | **Wavelet Score-Based Generative Modeling** (Guth et al., NeurIPS 2022) | `wavelet score-based generative model guth` | ⭐⭐⭐ | 多尺度 score-based model，结构非常接近本 idea |
| C3 | **DiWa / Wavelet-aware Diffusion** (若存在) | `wavelet aware diffusion DiWa` | ⭐⭐ | 看是否有相关变体 |
| C4 | GitHub 搜索 `wavelet consistency model` / `multi-scale consistency training` | — | ⭐⭐⭐ | **必查**：有没有人已经做过 wavelet × CT |

**核心带走问题（决定 novelty）**:
- wavelet 在 diffusion 里已经被怎么用了？
- 有没有人做过 **wavelet × CT / wavelet × few-step**？
- 如果有，他们用的是 sum 还是 residual supervision？
- 我们的 idea 相对他们的 delta 是什么？

---

### Block D — Multi-scale Supervision 架构参考（CV + GAN + Diffusion）⭐ **第一优先级**

> Boss 原话："各种 CV 的任务都有 multi-scale supervision，GAN 的 work 里面以前也有过 multi-scale supervision 的，所以应该有一些类似的架构可以参考。"
>
> 🎯 **读这个 block 的目的**：直接抄架构。Wavelet 分解是已知的数学操作，CT loss 是已知的 loss，**真正需要设计的是"多尺度级联 + 多尺度监督"的网络结构和训练策略**。这些在 CV / GAN / Diffusion 里都已经被反复实践过，没必要重新发明。
>
> ⚠️ GAN / Diffusion 不是本项目的目标方法，**目标始终是 CT**，这里只搬架构。

#### D-0. 通用 CV 多尺度监督（最早的灵感来源）

> 🚨 **关键概念区分（容易踩坑）**：
> - **U-Net 的"多尺度特征融合"**：发生在网络**内部**（skip connection），但**只有 1 个最终输出 + 1 个 loss**。`consistency_mnist.py` 里的 U-Net 就是这种——梯度只从单一最终监督反传。
> - **Multi-scale supervision（本 idea 要做的）**：在**多个不同分辨率的输出**上**分别算 loss**，N 个监督信号联合反传。
> - 两者**完全是两件事**。U-Net 已经是 CT 的 backbone，但它的 skip connection 不等于本 idea 的多尺度监督；我们要做的是**给 U-Net 加多个分辨率的输出 head + 多个 CT loss**。

| # | 论文 | 关键词 | 优先级 | 带走的信息 |
|---|---|---|---|---|
| D0a | **U-Net** (Ronneberger et al., MICCAI 2015) | `u-net biomedical image segmentation` | ⭐ 已熟悉 | 只是 backbone 模板，**不算 multi-scale supervision**，仅在此提及避免概念混淆 |
| D0b | **FPN — Feature Pyramid Network** (Lin et al., CVPR 2017) | `feature pyramid network lin` | ⭐⭐⭐ | 多尺度特征 + **每个尺度独立 head + 独立 loss**，最经典的多尺度监督范式 |
| D0c | **HED — Holistically-Nested Edge Detection** (Xie & Tu, ICCV 2015) | `holistically nested edge detection HED` | ⭐⭐⭐ | "deep supervision"（每个 side output 都有 loss）的鼻祖，**直接对应 sum supervision**；U-Net + HED 的扩展几乎就是我们要做的事 |
| D0d | **UNet++ / Deep Supervision in Segmentation** (Zhou et al., 2018) | `unet++ deep supervision zhou` | ⭐⭐ | 在 U-Net 上加 deep supervision（每个 decoder 层都有 loss）的具体实现 + 训练稳定性技巧——**最贴近我们要在 CT 的 U-Net 上做的改造** |
| D0e | **PSPNet / DeepLab multi-scale** | `pyramid scene parsing deeplab multi-scale` | ⭐ | 选读，看 dense prediction 怎么做多尺度 |

#### D-1. GAN 阵营（Boss 重点提到）

| # | 论文 | 关键词 | 优先级 | 带走的信息 |
|---|---|---|---|---|
| D1 | **LAPGAN — Laplacian Pyramid GAN** (Denton et al., NeurIPS 2015) | `laplacian pyramid GAN denton` | ⭐⭐⭐ | **多尺度级联生成的鼻祖**，本 idea 最直接的前身。每层 G 生成 Laplacian residual，级联重构 → **residual supervision 模板** |
| D2 | **MSG-GAN** (Karnewar & Wang, CVPR 2020) | `MSG-GAN multi-scale gradients karnewar` | ⭐⭐⭐ | 单 G 同时输出多分辨率 + 多分辨率 D，**multi-scale supervision 模板** |
| D3 | **ProGAN** (Karras et al., ICLR 2018) | `progressive growing GAN karras` | ⭐⭐ | 渐进式训练（先低再高）→ 我的 warm-up 策略可能借鉴 |
| D4 | **StyleGAN-XL** | `stylegan-xl multi-resolution discriminator` | ⭐ | 大规模训练时的多分辨率 D |

#### D-2. Diffusion 阵营

| # | 论文 | 关键词 | 优先级 | 带走的信息 |
|---|---|---|---|---|
| D5 | **Cascaded Diffusion Models** (Ho et al., JMLR 2022) | `cascaded diffusion models ho` | ⭐⭐ | 多个独立 diffusion，每个负责一个分辨率 → 独立分支方案的参考 |
| D6 | **Matryoshka Diffusion** (Apple, ICLR 2024) | `matryoshka diffusion model apple` | ⭐⭐⭐ | **单网络嵌套多分辨率**，最现代且最贴近本 idea 的网络结构方案（共享参数 + scale embedding） |
| D7 | **Simple Diffusion** (Hoogeboom et al., ICML 2023) | `simple diffusion high resolution hoogeboom` | ⭐ | 高分辨率 diffusion 的训练技巧 |
| D8 | **RIN — Recurrent Interface Networks** (Jabri et al., ICML 2023) | `recurrent interface network diffusion jabri` | ⭐ | 多分辨率 token 处理 |

**核心带走问题（全部围绕"如何把这些经验搬到 CT 上"）**:
- 不同尺度的 loss **怎么加权**？固定 / 自适应 / 随训练变化？→ 我在 CT loss 上要怎么加权？（HED / FPN / MSG-GAN 都有现成方案）
- 不同尺度 **共享网络还是独立分支**？→ Matryoshka 给出共享方案；Cascaded / LAPGAN 给出独立分支方案；CT 通常是单网络 + time embedding，能否扩展为单网络 + (time, scale) embedding？
- 训练 **稳不稳定**？用了什么技巧（warm-up、渐进式、loss balancing）？→ ProGAN / UNet++ 有经验
- **residual vs sum supervision** 哪个更好？→ LAPGAN 是 residual，HED / MSG-GAN 是 sum，**直接看它们的对比**
- 有没有论文报告过 multi-scale supervision **降低了 loss variance** 或 **加速了收敛**？（与 CT 训练不稳定的痛点直接对应，是把这套搬到 CT 的核心动机）

---

### Block E — 实验设计参考

> 不是单独读，是在 Block A/C/D 论文中**附带提取**。

要带走的信息：
- **数据集**：CIFAR-10 / ImageNet-64 / CelebA / LSUN（CT 系列的标配是 CIFAR-10 + ImageNet-64）
- **指标**：FID（**1-step 和 2-step**，CT 的核心 metric）、Inception Score、采样步数 vs 质量曲线
- **Baseline 选择**（本项目锁定）：**vanilla CT / iCT** 作为主对照，sCT 作为参考上界（如果跑得起）
- **Ablation 惯例**：sum vs residual supervision、层数（2/3/4）、wavelet 类型、是否共享网络
- **训练稳定性 metric**：loss curve 平滑度、grad norm、EMA 收敛速度（CT 特有的关注点）

---

## 2. 两天文献调研排期

### Day 1 — "搬架构：multi-scale supervision 经验地图"

**目标**: 通读 Block D，建立 **multi-scale supervision 在 CV / GAN / Diffusion 里的全景图**，提炼可直接迁移到 CT 的架构和训练经验。
> ✅ Block A 已有基础，本日完全不读 CT。

**时间分配**（约 6–7 小时）：

| 时段 | 任务 | 输出 |
|---|---|---|
| 上午 (2h) | **Block D-0 通用 CV**：FPN / HED / UNet++ / U-Net 多尺度监督传统 | "Deep supervision / multi-scale loss"机制对比表 |
| 上午-下午 (2.5h) | **Block D-1 GAN**：LAPGAN（residual）/ MSG-GAN（sum）/ ProGAN（渐进式） | GAN 多尺度架构对比表 + sum vs residual 初步对比 |
| 下午 (2h) | **Block D-2 Diffusion**：Matryoshka（共享网络嵌套）/ Cascaded（独立分支） | Diffusion 多尺度架构对比表 + 共享 vs 独立 |
| 傍晚 (1h) | 写 Day 1 Report：架构经验清单 + 初步 CT 适配草图 | 提交给 Boss |

**Day 1 结束时必须能回答**:
- ✅ Multi-scale supervision 的几大主流范式分别长什么样？（deep supervision / cascaded residual / shared backbone 多输出）
- ✅ 不同尺度的 loss **加权**通常怎么做？（等权 / 高分辨率重 / 自适应）
- ✅ **共享网络 vs 独立分支** 的取舍是什么？哪个适合 CT 这种 single-step distillation setting？
- ✅ **sum vs residual supervision** 在 GAN/CV 里有没有过对比？哪个赢了？
- ✅ 哪些训练稳定性技巧（warm-up / progressive / loss balancing）可直接迁移到 CT？
- ✅ 我的 Wavelet-CT 网络结构初稿应该长什么样？（一句话描述 + 简笔画）

---

### Day 2 — "定位 gap + 设计实验"

**目标**: 基于 Day 1 的架构地图，**判定 novelty + 收集理论引用 + 锁定 CT 实验方案**。
**时间分配**（约 6–7 小时）：

| 时段 | 任务 | 输出 |
|---|---|---|
| 上午 (2.5h) | **Block C — Wavelet × 生成模型**（C1/C2/C3 + **必查 C4 GitHub: wavelet × CT**） | wavelet 用法对比表 + novelty 判定 |
| 下午 (2h) | **Block B — Frequency bias**（B1/B2/B3，写作时引用） | 频率偏置证据汇总（重点：CT/few-step 是否仍成立） |
| 傍晚 (1h) | **Block A 速读**（A2 抄 iCT 超参，A3 扫一眼） + Block E 横向整理 | iCT baseline cheatsheet + 实验配置表 |
| 晚上 (1.5h) | 写 Day 2 Report：novelty + CT 实验方案 + 网络草图 | 提交给 Boss |

**Day 2 结束时必须能回答**（全部围绕 CT）:
- ✅ **有没有人做过 wavelet × CT？**（最关键 novelty 判定）
- ✅ 已有的 wavelet × diffusion 工作用的是 sum 还是 residual？我的设计相对它们 delta 是什么？
- ✅ "低频先于高频"在 CT/few-step setting 下证据如何？
- ✅ 主 baseline 锁定哪个 CT 变体？（建议 iCT）
- ✅ Ablation 怎么设计？（sum vs residual、层数、wavelet 类型、loss 加权、共享 vs 独立）
- ✅ 跑通 CT baseline + 我的方法大概需要多少 GPU·小时？

---

## 3. Day 1 / Day 2 Report 模板

### Day 1 Report 模板

```markdown
# Day 1 Literature Review Report — Wavelet CT (Multi-scale architecture map)

## 1. CV 通用多尺度监督范式
- FPN / HED / UNet++ 各自怎么做 multi-scale loss
- Deep supervision 的 loss 加权惯例

## 2. GAN 多尺度监督
- LAPGAN（residual / cascaded）怎么做
- MSG-GAN（sum / shared backbone）怎么做
- ProGAN 渐进式训练经验

## 3. Diffusion 多尺度监督
- Matryoshka（共享网络嵌套）vs Cascaded（独立分支）
- scale embedding / multi-resolution head 的实现细节

## 4. 横向对比：sum vs residual / shared vs independent
- 表格化对比已有工作的选择和效果

## 5. Wavelet-CT 网络结构初稿
- 一句话描述 + 简笔画
- 待 Day 2 验证的设计选择：__

## 6. Day 2 计划
- 重点：Block C novelty 判定 + Block B 理论引用 + 锁定 baseline
```

### Day 2 Report 模板

```markdown
# Day 2 Literature Review Report — Wavelet CT (Novelty + Experiment plan)

## 1. Wavelet 在生成模型中的现有应用（Block C）
- 主要工作 + 各自做法（C1/C2/...）
- 监督方式（sum vs residual）有没有被对比过
- ⚠️ **有没有人做过 wavelet + CT 的组合** ← 关键

## 2. 频率偏置证据（Block B，写作时引用）
- 低频先于高频生成的理论 / 实证支持（B1/B2/B3）
- 在 few-step CT 中是否已有类似观察

## 3. Novelty 定位
- 真正的 novelty gap：__
- 与最相关工作的 delta：__

## 4. 实验设计（**CT only**）
- Dataset: __（建议 CIFAR-10 起步；MNIST 做 sanity check）
- Baseline: **vanilla CT + iCT**（同一仓库、同一超参，只换 supervision 形式）
- Method variants（Wavelet-CT）:
  - Sum supervision
  - Residual supervision (teacher forcing)
  - 不同层数（2 / 3 / 4 层）
  - 不同 conditioning 方式（concat / FiLM / cross-attn）
  - 不同 wavelet 类型（Haar / Db2 / average pooling）
- Metrics: **1-step FID / 2-step FID**（核心）, IS, NFE-quality curve, loss curve 平滑度
- 估算 GPU·小时: __

## 5. Day 3+ 行动计划
- __
```

---

## 4. 后续完整计划（Day 3 之后）

> 待 Boss 确认总工期后调整；以下为 5–7 天版本草案。

### Day 3 — Code Setup + Wavelet 分解 baseline

- Fork & 熟悉 `consistency_mnist.py` 仓库结构
- 实现 wavelet 分解 / 重构 pipeline（先用 Haar 或简单 average pooling 验证）
- 在 MNIST 上跑通"分解 → 直接重构"的 sanity check（无生成模型）
- **如果 Boss 同意上 CIFAR-10**：准备数据 pipeline
- **Day 3 报告**：baseline 跑通截图 + 训练日志

### Day 4 — Sum Supervision 实现（CT 上）

- 实现级联生成器（多层 network，conditioning 链路）
- 实现 **multi-scale CT loss**（sum supervision 版本）
- 跑第一组实验：**vanilla CT baseline vs Wavelet-CT (sum sup)**，同数据集 / 同 NFE
- **Day 4 报告**：第一组初步结果（即使 FID 还没收敛也提交，重点关注 loss 曲线和稳定性）

### Day 5 — Residual Supervision 实现 + 对比实验

- 实现 residual supervision（teacher forcing 风格）
- 跑第二组实验：sum vs residual
- 开始 ablation：层数（2 / 3 / 4）
- **Day 5 报告**：sum vs residual 对比 + 初步 ablation

### Day 6 — Ablation 完整化 + Few-step 性能曲线

- NFE-quality 曲线（1-step / 2-step / 4-step / 8-step），**全部基于 CT**
- 与 vanilla CT / iCT 在相同 NFE 下对比
- 训练稳定性指标对比（loss variance、grad norm、达到目标 FID 所需 step 数）
- 收集 sample 可视化
- **Day 6 报告**：完整实验结果，能否回答"wavelet + multi-scale supervision 是否真的提升了 CT"

### Day 7 — Writeup + 收尾

- 整理最终 report（method + experiments + analysis + limitations）
- 代码整理 push 到 GitHub
- 提交最终交付物

---

## 5. 风险与待澄清问题（要问 Boss）

1. **总预期工期是几天？**（已问，等回复）
2. **数据集**：MNIST 起步还是直接上 CIFAR-10？MNIST 上 wavelet 高频信息很少，可能看不出效果。
3. **Wavelet 类型**：Haar / Daubechies / 还是直接用 average pooling 当作粗糙近似？
4. **GPU 预算**：本地 RTX 4070 Ti Super 16GB 足够，还是需要走 Spartan？
5. **交付物形式**：只要 report + code，还是需要 slides / 论文风格的 writeup？
6. **每日报告的颗粒度**：bullet 形式即可，还是需要 formal writeup？

---

## 6. 操作小贴士

- 每篇论文不超过 30 分钟，先看 abstract + figure 1 + method 概览 + experiments 表格，决定要不要深读
- 用 Notion / markdown 表格做笔记，列：标题 / 一句话方法 / 数据集 / 关键 metric / 与本 idea 的关系
- 每天结束前留 1 小时写 daily report，不要拖到次日
- GitHub 检索关键词：`wavelet consistency model` / `multi-scale consistency training` / `frequency consistency model` — 必查未发表的相关代码

---

**Last updated**: Day 0 (planning) — scope locked to **CT only**；**阅读顺序重排：Block D（架构）→ C（novelty）→ B（理论）→ A（baseline 速读）**
**Next action**: Day 1 上午 — 开始 Block D-0（FPN / HED / UNet++）通用 CV 多尺度监督
