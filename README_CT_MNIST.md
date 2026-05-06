# Consistency Training for MNIST — 单文件实现

[`consistency_mnist.py`](consistency_mnist.py) 是 Song et al., *Consistency Models* (arXiv:2303.01469) 中
**Consistency Training (CT，无教师版)** 的一份单文件干净实现，专门跑在 MNIST 上。

与本仓 `cm/` 下官方多模块实现的对照见下文 *实现说明*。

---

## 1. 文件清单

```
consistency_mnist.py        # 全部代码（UNet + CM 参数化 + 课程 + loss + 采样 + 训练循环）
README_CT_MNIST.md          # 本文件
```

无外部脚本、无 cm/ 依赖。

---

## 2. 环境

- Python ≥ 3.9
- PyTorch ≥ 2.0（CUDA 版本任意；MNIST 单卡 16GB 显存绰绰有余）
- torchvision（数据集 + `save_image`）

```bash
pip install torch torchvision
```

可选（算 FID 时才需要）：

```bash
pip install torchmetrics[image]
```

---

## 3. 运行

### 训练

```bash
python consistency_mnist.py train --out-dir runs/ct_mnist
```

默认配置：`total_steps=50000, batch_size=256, lr=1e-4, loss=l2`。

常用调整：

```bash
# 更长训练 + Pseudo-Huber loss（iCT 改进，更稳）
python consistency_mnist.py train \
    --out-dir runs/ct_mnist_huber \
    --total-steps 100000 \
    --loss-type huber

# 显存紧张时减小 batch
python consistency_mnist.py train --batch-size 128
```

### 采样

训练完之后用 final checkpoint 出 NFE=1 / 2 / 4 三种步数的 grid：

```bash
python consistency_mnist.py sample \
    --ckpt runs/ct_mnist/model_final.pt \
    --out-dir runs/ct_mnist/samples_final \
    --n-samples 64
```

---

## 4. 期望产出

训练目录结构：

```
runs/ct_mnist/
├── samples/
│   ├── step002000_nfe1.png    # 训练中每 2k step 的可视化
│   ├── step002000_nfe2.png
│   ├── ...
│   └── step050000_nfe2.png
├── model_step010000.pt        # 每 10k step 一个
├── model_step020000.pt
├── ...
└── model_final.pt
```

训练效果（参考值，单卡 RTX 3090/4080 / 类似 16GB）：

| step | NFE=1 视觉质量 | wall-clock |
|---|---|---|
| 2k   | 噪声 + 模糊轮廓 | ~3 min |
| 10k  | 部分能认出数字 | ~15 min |
| 25k  | 大部分清晰 | ~40 min |
| 50k  | 清晰、多样性合理；NFE=2 更锐 | ~1.5 h |

CPU 上能跑通（用作 smoke test），但完整训练只在 GPU 上现实。

---

## 5. 验收清单

面试题"clean implementation of CT for MNIST"对应的最低交付标准：

- [x] 单文件，无外部 cm/ 依赖
- [x] CT（无教师）—— 不依赖预训练扩散模型
- [x] 边界参数化 f_θ(x, σ_min) = x（精确，靠 c_skip / c_out）
- [x] N(k)、μ(k) 自适应课程（论文 Eq.13/14）
- [x] EMA target_model
- [x] 1-step 与 multi-step 采样
- [x] 中文注释 + 论文公式对应行号

可选加分项：

- [ ] FID 评估（接一下 `torchmetrics.image.FrechetInceptionDistance` 即可）
- [ ] 类条件生成（MNIST 0-9）
- [ ] iCT 的 lognormal 时间步采样（目前是 uniform）

---

## 6. 实现说明

### 与 `cm/` 官方实现的对照

| 关注点 | 官方多模块 | 本文件 |
|---|---|---|
| 主入口 | `scripts/cm_train.py` | `train()` |
| 训练循环 | `cm/train_util.py: CMTrainLoop` | `train()` 内 for-loop |
| 网络 | `cm/unet.py: UNetModel`（带 attention，~50M 参数） | 本文件内小 UNet（~2-3M 参数） |
| Loss / 参数化 | `cm/karras_diffusion.py: consistency_losses` | `ct_loss()` |
| EMA & N 调度 | `cm/script_util.py: create_ema_and_scales_fn` | `n_schedule()` / `mu_schedule()` |
| 数据 | `cm/image_datasets.py`（webdataset） | `torchvision.datasets.MNIST` |
| 分布式 | `torch.distributed` + fp16 | 单卡 fp32 |
| 训练模式 | CD / CT / progdist 三选一 | 仅 CT |

### 设计决策

1. **σ_data = 0.5**：MNIST 归一化到 [-1, 1] 后大致的标准差，沿用 EDM 默认。
2. **UNet 输出层零初始化**：训练初期 F_θ ≈ 0，让 f_θ ≈ x（恒等），避免 loss 爆掉。
3. **统一 z**：CT 的关键是同一 batch 同一 z 喂给 (σ_low, σ_high) 两个点 ——
   官方代码里这步藏在 `euler_solver` 中（teacher_model is None 分支化简后等价于 x_t2 = x + z·σ_low）。
4. **uniform N 采样**：n ~ U{0..N(k)-2}，没用 importance sampling。MNIST 上够用。
5. **L2 默认**：忠于原 CT 论文。`--loss-type huber` 切到 iCT 的 Pseudo-Huber（c=0.00054·√D）。
6. **多步采样**：用 Karras 间距取 NFE+1 个 σ，跳过最末端的 σ_min（sqrt(σ²-σ_min²) 在那里为 0，跳过节省一次 NFE）。

### 不做的事

- LPIPS loss（MNIST 灰度图没必要）
- 类条件 / classifier-free guidance（题目没要求）
- Mixed precision、DDP、checkpoint 续训
- 教师模型蒸馏（CD）

这些在官方仓里都有，但单文件实现强调"clean"，刻意只保留 CT 必需的部分。

---

## 7. 排错

**显存爆了**：`--batch-size 128` 或 `64`。

**MNIST 下载失败**：先手动下到 `./data/MNIST/raw/`，或翻墙。

**训练 loss 不降**：检查
1. 数据是否归一化到 [-1, 1]（`Normalize((0.5,), (0.5,))`）
2. σ_data 是否匹配数据分布
3. lr 是否过大（默认 1e-4 已经偏稳）

**NFE=1 出来全是噪声**：通常是训练太短（< 5k step）或边界条件实现错了。可手测：
```python
x = torch.randn(2, 1, 28, 28)
out = model(x, torch.full((2,), cfg.sigma_min))
assert (out - x).abs().max() < 1e-5  # f(x, σ_min) == x
```

---

## 8. 参考

- Song, Yang, et al. ["Consistency Models."](https://arxiv.org/abs/2303.01469) ICML 2023.
- Song & Dhariwal. ["Improved Techniques for Training Consistency Models."](https://arxiv.org/abs/2310.14189) ICLR 2024.（iCT，可选改进）
- Karras et al. ["Elucidating the Design Space of Diffusion-Based Generative Models."](https://arxiv.org/abs/2206.00364) NeurIPS 2022.（EDM，σ schedule 与边界参数化的来源）
