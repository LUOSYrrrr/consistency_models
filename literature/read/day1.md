# Day 1 文献调研报告 

**Date**: 2026-05-08
**Today's reading**: FPN / LAPGAN / MSG-GAN

**报告结构**：

1. 我对 Wavelet-CT 这个 idea 的理解
2. 我对 Wavelet transform 的理解
3. 三篇论文综述
4. 下一步阅读计划

---

## 1. 对 Wavelet-CT  idea 的理解

### 1.1 理解

把 wavelet 分解作为**多尺度监督信号的来源**，让 consistency training 同时在多个 wavelet 子带上获得一致性监督，而不是只在最终像素空间的单一一致性目标上做监督。



CT 的原始 loss 只在像素空间最终输出上约束 $f_\theta(x_t, t)$ 沿 ODE 轨迹的一致性。这是一种"全局到全局"的监督——网络要同时学好低频结构（构图、颜色块）和高频细节（边缘、纹理），但 loss 没有区分这两类目标的难度。few-step 生成里常见的"细节漂移"现象，本质上就是这种**单一监督信号下容量分配不均衡**的表现。

Wavelet 分解恰好提供了一种把"低频结构"和"高频细节"显式拆开的语言：把目标 $x_0$ 分解为 $\{LL, LH, HL, HH\}$ 多个子带，分别施加一致性损失。这样：
- 网络的不同 head/branch 学习真正不同的目标分布；
- 每个监督信号的"语义层级"明确（base 学全局结构、HH 学对角细节）；
- 低频子带容易学的部分先稳住，再约束高频细节，符合直觉上的 coarse-to-fine。

### 1.2 idea 在文献谱系中的位置

把 idea 拆成三个原子组件：

```
                        Wavelet-CT
                  ─────────────────────
                          │
          ┌───────────────┼───────────────┐
          │               │               │
   架构骨架       监督信号来源        监督目标
          │               │               │
   单网络多输出   多尺度频域分解       CT loss
   joint training        │          (self-consistency)
          │               │               │
       MSG-GAN     LAPGAN: Laplacian       │
       (2020)         pyramid (2015)    Song 2023
                          ↑             (CT 原始论文)
                  Wavelet-CT 的升级:
                  方向敏感 + 正交临界采样
                  的 wavelet 分解
```

LAPGAN 给的不是 wavelet 本身（它用的是 Laplacian pyramid），而是更上一层的"**用多尺度频域分解作为生成模型的监督信号来源**"这个思路；Wavelet-CT 把它的 Laplacian pyramid 替换为方向敏感、正交临界采样的 wavelet。架构骨架借鉴 MSG-GAN 的单网络多输出 + joint training 范式，监督目标继承 Song 2023 的 self-consistency loss。

**三个组件各自有先例，但三者的组合在公开文献中尚未找到**

### 1.3 assumption

为了让后续实验能分开测、分开归因，我把 idea 拆成三个独立命题：

**假设 A（频域分解）**：在 CT 训练中，把像素空间一致性损失替换为多个 wavelet 子带上的一致性损失，**信号承载能力更强**——因为不同子带的预测难度被显式拆开，网络的容量分配更合理。

**假设 B（多尺度正则化）**：CT 的 self-consistency 损失原本只在最终输出上计算，加多尺度监督可能起到类似 CV 中 deep supervision 的正则化作用，**缓解 few-step 生成中的高频漂移**。

**假设 C（Sum vs Residual）**：Sum 型监督（每个子带独立预测目标值）和 Residual 型监督（高频 condition on 已生成低频）在 CT 训练动力学下哪个更优？现有文献里 GAN 路径上 LAPGAN 选 residual、MSG-GAN 选 sum，**但没有任何工作在 CT 上做过这个对比**。

### 1.4 research question

1. **多尺度梯度的稳定性机制能否迁移到无 D 的 CT？** MSG-GAN 的稳定性来自 D 直接在多尺度上约束 G；CT 没有 D，等价机制是什么？需要读 Matryoshka Diffusion（Diffusion 版的 MSG-GAN）来补上这个环节。
2. **CT 已有 EMA + Pseudo-Huber 两道正则化，多尺度监督还能不能带来边际收益？** 这是 idea 是否成立的最终判据。MSG-GAN 在 1024² 上 FID 略输 StyleGAN，作者归因于 mixing regularization 的缺失——暗示当 baseline 已有强正则时，多尺度的增益会被压缩。
3. **极端稀疏的高频子带（HH）会不会引发训练病态？** 稀疏目标信息密度高，但若 90%+ 像素接近 0，loss 可能被少数大值主导，其余位置梯度饥饿。

---

## 2. 对 Wavelet transform 的理解

### 2.1 数学骨架

二维 wavelet transform 把图像分解为**正交的多分辨率、多方向子带**。一次 2D DWT 把 H×W 图变成四个 H/2 × W/2 的子带：

| 子带 | 行操作 | 列操作 | 物理含义 |
|---|---|---|---|
| **LL** | 平均 | 平均 | 低频近似（base，类似平滑下采样） |
| **LH** | 做差 | 平均 | 水平边缘 |
| **HL** | 平均 | 做差 | 垂直边缘 |
| **HH** | 做差 | 做差 | 对角细节 / 角点 |

对 LL 子带递归 DWT → 多尺度金字塔。Haar 是最简形式（2×2 mean/diff），Daubechies-4/6 是更高阶变体（基函数支撑域更大、平滑性更好）。

正交性意味着 **逆变换 (IDWT) 完美重构**，零信息损失——这一点对 CT 训练很关键，因为损失里"分解 → 子带预测 → 重构"的链路必须无损，否则一致性约束会被分解-重构本身的误差污染。

### 2.2 与 Laplacian pyramid 的关系

LAPGAN 用的 Laplacian pyramid $h_k = I_k - u(I_{k+1})$ 是 **isotropic band-pass**：每层只有一个无方向区分的频带差分。Wavelet 把同一频带进一步拆成三个方向：

```
Laplacian pyramid (LAPGAN):              2D Wavelet:

I_0 (原图)                                I_0 (原图)
  │                                         │
  ├─ h_0 (高频残差，无方向区分)             ├─ LH_0, HL_0, HH_0  (3 个方向高频)
  ├─ h_1                                    └─ LL_0
  ├─ ...                                       │
  └─ I_K (最粗低频)                            ├─ LH_1, HL_1, HH_1
                                               └─ LL_1
                                                  │
                                                  └─ ...
```

两者的核心差异：

| | Laplacian pyramid | 2D Wavelet (Haar / Daubechies) |
|---|---|---|
| 每级子带数 | 1 个 (无方向区分的 isotropic band-pass) | 4 个 (LL + LH + HL + HH，方向敏感) |
| 表示是否冗余 | **过完备**：各层 $h_k$ 都和原图同大小，总系数数 > 原像素数 (约 4/3 倍) | **正交临界采样**：总系数数 = 原像素数 |
| 重构 | 完美但需存所有 $h_k$ | 完美且无冗余 |
| 方向信息 | 无 | 有 (水平/垂直/对角) |

**Laplacian pyramid 严格意义上不是 wavelet 的特例**——最关键的差别是"过完备 vs 正交临界采样"。两者的共同点是都来自 multiresolution analysis 这个数学框架，本质都是把图像分解到不同频带后分别处理。所以更准确的写作角度是：

> Wavelet-CT 沿用 LAPGAN 的"多尺度频域分解 + per-scale supervision"思路，但把 Laplacian pyramid 替换为方向敏感、正交临界采样的 wavelet 分解，使监督信号同时具备方向解耦与无冗余两个优势。

而不是"严格 ≥"。

### 2.3 在生成任务上的潜在优势

1. **方向解耦**：自然图像中边缘、纹理本身有强方向性。把它们拆到 LH/HL/HH 三个独立子带，每个子带的统计分布更纯粹，目标更易拟合。
2. **稀疏性**：高频子带大部分像素接近 0（自然图像能量集中在低频）→ 监督信号信息密度高、噪声梯度小。
3. **正交性 + 完美重构**：Haar/Daubechies 是正交变换，逆变换零信息损失，数值稳定。
4. **可微分实现成熟**：`pytorch_wavelets` 等库已把 forward/inverse DWT 实现为可微算子，工程门槛极低。

---

## 3. 三篇论文

### 3.1 FPN — Feature Pyramid Networks for Object Detection (Lin et al., CVPR 2017)

<img src="/Users/siyuanluo/STUDY/robot/consistency_models/literature/read/assets/image-20260508232545103.png" alt="image-20260508232545103" style="zoom:50%;" />

> **图（FPN paper Fig.3）**：FPN 与已有 top-down 架构的对比。
>
> **上半部分**是 "top-down + skip connection"，虽然有完整的 top-down pathway，但**只在最 fine 那一层做预测**（右上方仅一个 predict box），仍然是单尺度输出。
>
> **下半部分**是 FPN 本身：top-down 之后**每个分辨率都独立挂一个 head 同时预测**（右侧三个 predict box）——这是 FPN 与已有 top-down 工作的核心差异。论文的关键 ablation 就是验证"多个尺度同时预测 vs 仅最 fine 一层预测"的差距，证明多尺度预测显著优于单尺度。

**问题**：目标检测里同一张图的物体大小差异巨大（COCO 中 32² 到 512² 都有），单尺度特征对小目标不友好。已有方法要么用 image pyramid（慢、显存爆炸、train/test 不一致），要么直接用 ConvNet 自带的 hierarchy（高分辨率层语义弱），要么只在 top-down 后的最 fine 一层做预测（仍然是单尺度预测）。

**核心思想**：ConvNet 自身的 feature hierarchy 就是天然的金字塔，关键缺陷是"高分辨率层语义弱"。**补救方法是加一条 top-down pathway 把深层语义传下来 + lateral connection 保留浅层精确定位**。给定 ResNet 各 stage 输出 {C2, C3, C4, C5}（stride 4/8/16/32），从 C5 开始，1×1 conv 降维到 d=256 → 2× nearest 上采样 → 与同尺度的 lateral 输出 element-wise add → 3×3 conv 抗 aliasing → 得到 {P2, P3, P4, P5}。**每个 P_k 上独立挂一个 head 做预测，head 跨尺度参数共享**。

**关键架构特性**：
- 多尺度输出 + 同时预测，sum 型监督（每个尺度独立 RPN/Fast R-CNN loss 直接求和）
- Lateral merge 用 add 而非 concat，无 ReLU/BN
- Head 跨尺度共享参数（作者明确做了 ablation 验证不共享与共享精度相近 → 暗示所有 P_k 在语义上对齐）

**关键结果**（COCO，ResNet-50）：RPN AR^1k 从 baseline 48.3 提升到 56.3（小目标 AR^1k_s 从 32.0 提升到 44.9，+12.9），Faster R-CNN AP 从 31.6 提升到 33.9（小目标 AP_s 从 13.2 提升到 17.8，+4.6）。Ablation 表明 top-down 和 lateral 缺一不可。

**对 Wavelet-CT 的启发**：FPN 是"sum 型多尺度监督在共享网络上"的最简模板。它的等权 sum loss、共享 head、lateral merge 设计，提供了一个可借鉴的工程模板。它的局限是所有尺度预测的是同一目标（bbox）的不同分辨率视角，**而 Wavelet-CT 想做的是让每个尺度预测真正不同的频带目标**——这超出了 FPN 的范式。

---

### 3.2 LAPGAN — Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks (Denton et al., NeurIPS 2015)

![image-20260508232805638](/Users/siyuanluo/STUDY/robot/consistency_models/literature/read/assets/image-20260508232805638.png)

> **图（LAPGAN paper Fig.1）**：采样流程，从右往左读。
>
> 从最粗层开始：噪声 $z_3 \to G_3 \to \tilde{I}_3$（8×8 base，无条件生成）。然后迭代地把已生成图作为低频上下文送入下一层：
>
> 1. 上采样 $\tilde{I}_3$ → $l_2$（绿色箭头）
> 2. $G_2$ 接收 $(z_2, l_2)$ → 生成残差 $\tilde{h}_2$
> 3. $\tilde{I}_2 = l_2 + \tilde{h}_2$（紫色 + 号）
> 4. 重复 → $\tilde{I}_1$ → $\tilde{I}_0$（64×64 最终图）
>
> 严格的 coarse→fine 级联：每一层 $G_k$ 只生成"在已知低频之上的差分"，把全局生成拆成多个"已知低频，补高频"的简单子问题——这是 **residual supervision** 的核心思想。

![image-20260508232813910](/Users/siyuanluo/STUDY/robot/consistency_models/literature/read/assets/image-20260508232813910.png)

> **图（LAPGAN paper Fig.2）**：训练流程，64×64 输入图像 I 为例。
>
> 训练在每一层 k 上独立进行，最关键的机制是**扔硬币**：以 50% 概率走两条分支之一——
>
> - **真分支**（蓝色箭头）：用真实低频上下文 $l_k = u(I_{k+1})$ 算出真实残差 $h_k = I_k - l_k$，送进 $D_k$；
> - **假分支**（品红箭头）：让 $G_k$ 接收 $(z_k, l_k)$ 生成假残差 $\tilde{h}_k$，也送进 $D_k$。
>
> 不论真假，$l_k$ 都通过橙色箭头额外送进 $D_k$ 第一层卷积之前——这是 LAPGAN **结构性 conditioning** 的核心：$D_k$ 同时看（残差，低频上下文），不仅判断"残差像不像真"，还判断"残差与 $l_k$ 是否一致"。
>
> 每一层 $(G_k, D_k)$ 完全独立训练，**没有跨层梯度**——这是 residual supervision 最极端的形式。最粗的 level 3 是 8×8，简单到可以直接用无条件 GAN（$G_3, D_3$ 不接收 $l_k$）。

**问题**：原始 GAN（Goodfellow 2014）无法生成高分辨率自然图像——直接在像素空间建模全局分布时，G 的梯度信号太弱，生成大图保真度差。

**核心思想**：放弃"全局 fidelity"的目标，**把生成问题拆解成多步 coarse-to-fine refinement**。利用 Laplacian pyramid 的正交分解，把图像拆成 K-1 层 band-pass residual $h_k = I_k - u(I_{k+1})$ + 1 层 base $h_K = I_K$。每层训练一个独立的 conditional GAN $G_k$：除最粗层 $G_K$ 不条件直接生成 base 外，$G_0..G_{K-1}$ 都条件于 $l_k = u(\tilde{I}_{k+1})$（已生成的低频上采样图）和噪声 $z_k$，仅生成"在低频上下文之上的高频差分"。

**关键架构特性**：
- 严格 coarse→fine 级联（采样时 $\tilde{I}_k = u(\tilde{I}_{k+1}) + G_k(z_k, u(\tilde{I}_{k+1}))$）
- 每层是独立的 $G_k + D_k$，**完全独立训练，无跨层梯度**——这是 residual supervision 最极端的形式
- $D_k$ 同时看 $(h_k, l_k)$：不仅判断"残差像不像真实残差"，还判断"残差与低频上下文是否一致"，提供结构性 conditioning 监督
- 最粗层 $G_K$ 用全连接 MLP（CIFAR10 上 8×8 尺度），其余层用 ConvNet

**关键结果**：CIFAR10 上 Parzen-window log-likelihood 从原始 GAN 的 -3617 提升到 -1799（+1818）；human evaluation 中"被误认为真实图像"的比例从 ~10% 提升到 ~30%（class-conditional 版本 ~40%）。LSUN 64×64 上生成了当时其他生成模型无法企及的复杂场景（教堂/卧室/塔楼）。

**对 Wavelet-CT 的启发**：LAPGAN 是 **residual supervision 的鼻祖**，与 Wavelet-CT 在结构骨架上高度相似（频域分解 + coarse-to-fine + per-scale supervision）。它给的最重要启示是：**让每个尺度预测真正不同的目标**（band-pass residual）这件事本身是 work 的，把生成拆成多个简单子问题比一步到位生成更稳定。但它"完全独立网络 + 完全独立训练"的极端做法在 CT 下不可取——CT 是 end-to-end 的蒸馏，断掉跨层梯度等于自废武功。

---

### 3.3 MSG-GAN — Multi-Scale Gradients for Generative Adversarial Networks (Karnewar & Wang, CVPR 2020)

![image-20260508233049964](/Users/siyuanluo/STUDY/robot/consistency_models/literature/read/assets/image-20260508233049964.png)

> **图（MSG-GAN paper Fig.2）**：MSG-GAN 完整架构（基于 ProGAN 的 base model 画的），可拆成三块看。
>
> **左侧 — Generator**：从 latent vector 开始，逐级上采样产生 4×4 → 8×8 → 16×16 → ... → 高分辨率的 activation（粉色长方体 g_gen / g^1 / g^2 / ...）。每个分辨率块下面挂一个红色 `r` 块——这就是 **toRGB 层（1×1 conv）**，把每一级 activation 投影成对应分辨率的 RGB 图像（图最下面的 4×4 / 8×8 / 16×16 图块）。一次前向同时产出多分辨率 RGB。
>
> **底部连线 — 多尺度 RGB 喂给 D**：G 产出的每个分辨率 RGB 都单独连一根线到 D 的对应深度入口（4×4 → d^k，8×8 → d^k-1，16×16 → d^k-2 ……）。**真实图也下采样到对应分辨率**（图右上方"Real Images downsampled to various resolutions"），实/假以同样多尺度的方式进 D。
>
> **右侧 — Discriminator**：从最高分辨率开始进入第一个 d block，每个 d block 入口处有一个**黄色 combine function $\varphi$**——把当前 activation 和"从 G 进来的同分辨率 RGB"合并（concat 或先投影再 concat）。然后 3×3 Conv（紫色）→ 2×2 AvgPool（蓝色）→ 进下一个 d block。最末端的 4×4 d block 经过 MinibatchStd（浅绿）→ 4×4 Conv（黑）→ FC（橙）→ 输出**单个 critic score**。
>
> **核心机制**：D 最后只产出一个标量，但反传时这一个 critic score 的梯度会**通过 N 个 RGB 入口同时流回 G 的 N 个 g block**——这就是论文标题里说的"multi-scale gradients"。每个 g block 都有"直达监督"，不像普通 GAN 那样只有最后一层有强梯度。这套机制让 MSG-GAN 不需要 progressive growing 的复杂 schedule 就能稳定训练。

**问题**：GAN 训练不稳定的根因是 fake/real 分布 support 不重叠时 D 传给 G 的梯度噪声大、信号弱。已有解法 ProGAN（progressive growing）需要复杂的 per-resolution 超参（每分辨率训多少 iteration、不同 lr、fade-in 节奏），训练复杂且会产生 phase artifacts；多 D 方案（StackGAN、Pix2PixHD）参数爆炸；LAPGAN 完全独立网络，无跨尺度梯度流。

**核心思想**：**单 G + 单 D，G 的每个中间层通过 1×1 conv 输出对应分辨率的 RGB 图像，D 的每个深度层接收对应 RGB**。梯度从 D 的 critic score 通过多尺度入口同时回传到 G 的所有层 → G 每一层都有"直达监督"，D→G 的信息流不再单薄。**替代了 progressive growing**，所有数据集用同一套固定超参数。

**关键架构特性**：
- G：每个分辨率块后挂 `1×1 conv → RGB`（toRGB 层），把 activation 投影成像素空间图像
- D：每个分辨率块入口通过 combine function $\varphi$ 把对应尺度 RGB 图像拼到当前 activation 上（$\varphi$ 有三种变体：直接 concat / 先 1×1 投影再 concat / 先 concat 再 1×1 投影）
- 单 critic score：D 最终只输出一个 scalar，但已综合所有尺度的信息
- 全部 joint end-to-end 训练

**关键结果**：FFHQ 1024² 上 MSG-StyleGAN 用 9.6M images 达到 FID 5.8，对比 StyleGAN 25M images 的 FID 4.4——**收敛快约 2.6×**。LSUN Churches 256² 上从 6.58 → 5.2。Ablation（FFHQ）揭示：**中间分辨率连接贡献最大**（middle-only FID 9.17 vs full 8.36 vs DCGAN 14.20），单靠 coarse 或 fine 都不够。Learning rate 鲁棒性极强（lr ∈ [0.001, 0.01] 全部收敛）。

**对 Wavelet-CT 的启发**：MSG-GAN 的架构骨架与 Wavelet-CT 几乎完全同构——单网络多输出 + joint training。它证明了**不需要每尺度独立网络也能得到 LAPGAN 同等的多尺度监督收益**，参数效率高得多，训练也更稳。它的 "中间尺度贡献最大" 这一 ablation 结论直接告诉我们：Wavelet-CT 不必把 wavelet 分解到极端深度，3 层左右可能就够。它的 combine function $\varphi$ 也对应了"低频 context 如何注入高频分支"这个 Wavelet-CT residual 版本必须回答的问题。**唯一的差距是 MSG-GAN 是 GAN loss、Wavelet-CT 是 CT loss——多尺度梯度的稳定性机制是否能脱离 D 而存在，是开放问题。**

---

## 4. 下一步阅读计划

按优先级：

| 优先级 | 论文 | 为什么读 |
|---|---|---|
| 高 | **Matryoshka Diffusion** | Diffusion 版的 MSG-GAN，与 CT 同源（CT 是 diffusion 蒸馏），是回答"多尺度梯度机制能否迁移到无 D setting"的最关键文献 |
| 高 | **Ghiasi & Fowlkes 2016 — Laplacian Pyramid for FCN segmentation** | 频域分解 × 稠密预测的早期工作，与 Wavelet-CT 在 CV 侧最同构 |
| 中 | **HED / UNet++** | CV 中 deep supervision 的经典模板，写作时引用"多尺度监督在 CV 中已被广泛验证" |

 重点：(1) 读 Matryoshka Diffusion 把 diffusion 侧多尺度脉络补完
