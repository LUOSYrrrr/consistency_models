"""
Consistency Training (CT) for MNIST — single-file clean implementation.

参考：Song et al., "Consistency Models", arXiv:2303.01469（原版 CT，无教师）。

用法
----
训练（默认 50k step，~1-2h on a 16GB GPU）：
    python consistency_mnist.py train --out-dir runs/ct_mnist

采样（NFE=1 / NFE=2 / NFE=4 各保存一张 8x8 grid）：
    python consistency_mnist.py sample --ckpt runs/ct_mnist/model_final.pt --out-dir runs/ct_mnist/samples

训练目标
--------
学一个 f_θ(x, σ) 满足 self-consistency: 在概率流 ODE 同一条轨迹上,
任何 σ 都映射到同一起点 x_0。在 σ_min 处通过参数化保证 f_θ(x, σ_min) = x。

CT loss（论文 Alg. 3，无教师）：
    n  ~ Uniform{0, ..., N(k)-2}
    z  ~ N(0, I)
    L  = || f_θ(x + σ_{n+1} z, σ_{n+1})  -  f_{θ⁻}(x + σ_n z, σ_n) ||²
其中 σ 由 Karras schedule 生成（σ_min < σ_n < σ_{n+1} < σ_max），
θ⁻ 是 θ 的 EMA 拷贝（target_model）。
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


# ============================================================
# 配置
# ============================================================
@dataclass(frozen=True)
class Config:
    # data
    image_size: int = 28
    in_channels: int = 1
    data_dir: str = "./data"

    # σ schedule (Karras EDM)
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5  # MNIST 归一化到 [-1, 1] 后大致的标准差
    rho: float = 7.0

    # CT 课程（论文 Eq.13/14）
    s_0: int = 2          # 起点离散化数 N(0)
    s_1: int = 150        # 终点离散化数 N(K)
    mu_0: float = 0.95    # μ 计算用的参考 EMA 衰减
    loss_type: str = "l2"  # "l2" or "huber"

    # optimization
    total_steps: int = 50_000
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # network
    base_channels: int = 64
    channel_mults: tuple = (1, 2, 2)
    num_res_blocks: int = 2

    # logging
    log_interval: int = 100
    sample_interval: int = 2_000
    ckpt_interval: int = 10_000
    out_dir: str = "./runs/ct_mnist"
    seed: int = 42


# ============================================================
# UNet
# ============================================================
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """正弦位置编码——把连续 σ（经 c_noise 缩放后）映射到 dim 维向量。"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """带 FiLM 时间条件的 GroupNorm + Conv 残差块。"""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int, groups: int = 8) -> None:
        super().__init__()
        g = min(groups, in_ch)
        self.norm1 = nn.GroupNorm(g, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.t_proj = nn.Linear(t_dim, 2 * out_ch)  # FiLM: scale + shift
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.t_proj(F.silu(t_emb)).chunk(2, dim=-1)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """28x28 MNIST 用的小 UNet。默认 ~6.5M 参数，16GB 显存非常宽裕。"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        ch = cfg.base_channels
        t_dim = ch * 4
        self.t_in_dim = ch
        self.t_mlp = nn.Sequential(
            nn.Linear(ch, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        mults = cfg.channel_mults
        nrb = cfg.num_res_blocks

        self.in_conv = nn.Conv2d(cfg.in_channels, ch, kernel_size=3, padding=1)

        # encoder
        self.downs = nn.ModuleList()
        skip_chs: list[int] = [ch]
        cur = ch
        for i, m in enumerate(mults):
            out = ch * m
            for _ in range(nrb):
                self.downs.append(ResBlock(cur, out, t_dim))
                cur = out
                skip_chs.append(cur)
            if i < len(mults) - 1:
                self.downs.append(Downsample(cur))
                skip_chs.append(cur)

        # middle
        self.mid1 = ResBlock(cur, cur, t_dim)
        self.mid2 = ResBlock(cur, cur, t_dim)

        # decoder（每层 nrb+1 个 ResBlock，用于消化所有 skip）
        self.ups = nn.ModuleList()
        for i, m in enumerate(reversed(mults)):
            out = ch * m
            for _ in range(nrb + 1):
                self.ups.append(ResBlock(cur + skip_chs.pop(), out, t_dim))
                cur = out
            if i < len(mults) - 1:
                self.ups.append(Upsample(cur))

        self.out_norm = nn.GroupNorm(8, cur)
        self.out_conv = nn.Conv2d(cur, cfg.in_channels, kernel_size=3, padding=1)
        # 输出层零初始化：训练初期 F_θ ≈ 0，配合 c_skip/c_out 让 f_θ(x, σ) ≈ x，更稳。
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        t_emb = timestep_embedding(c_noise, self.t_in_dim)
        t_emb = self.t_mlp(t_emb)

        h = self.in_conv(x)
        skips = [h]
        for layer in self.downs:
            h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
            skips.append(h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for layer in self.ups:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, skips.pop()], dim=1)
                h = layer(h, t_emb)
            else:
                h = layer(h)

        return self.out_conv(F.silu(self.out_norm(h)))


# ============================================================
# Consistency 参数化
# ============================================================
class ConsistencyModel(nn.Module):
    """
    给 UNet 包一层 EDM/CM 的边界参数化：
        f_θ(x, σ) = c_skip(σ) · x + c_out(σ) · F_θ(c_in(σ)·x, c_noise(σ))

    精确满足 f_θ(x, σ_min) = x（CM 论文要求的边界条件）：
        c_skip(σ_min) = 1, c_out(σ_min) = 0
    """

    def __init__(self, unet: UNet, cfg: Config) -> None:
        super().__init__()
        self.unet = unet
        self.sigma_min = cfg.sigma_min
        self.sigma_data = cfg.sigma_data

    def _scalings(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sd = self.sigma_data
        smin = self.sigma_min
        c_skip = sd**2 / ((sigma - smin) ** 2 + sd**2)
        c_out = (sigma - smin) * sd / torch.sqrt(sigma**2 + sd**2)
        c_in = 1.0 / torch.sqrt(sigma**2 + sd**2)
        c_noise = 0.25 * torch.log(sigma + 1e-44)
        return c_skip, c_out, c_in, c_noise

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        c_skip, c_out, c_in, c_noise = self._scalings(sigma)
        # 把 (B,) 形状的标量广播到 (B, 1, 1, 1)
        cs, co, ci = (s[:, None, None, None] for s in (c_skip, c_out, c_in))
        F_out = self.unet(ci * x, c_noise)
        return cs * x + co * F_out


# ============================================================
# Karras σ 调度 + CT 课程（N(k), μ(k)）
# ============================================================
def karras_sigmas(n: int, sigma_min: float, sigma_max: float, rho: float, device: torch.device) -> torch.Tensor:
    """生成 n 个 Karras 间距的 σ，升序：σ[0]=σ_min, σ[n-1]=σ_max。"""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    # 上式得到的是降序（max→min），翻转成升序方便按"低 idx → 低 σ"思考
    return sigmas.flip(0)


def n_schedule(step: int, total_steps: int, s0: int, s1: int) -> int:
    """论文 Eq.13：N(k) 从 s0 平滑增长到 s1+1（取整后离散化数）。"""
    progress = step / max(total_steps - 1, 1)
    n = math.ceil(math.sqrt(progress * ((s1 + 1) ** 2 - s0**2) + s0**2) - 1) + 1
    return max(n, s0)


def mu_schedule(n_k: int, mu_0: float, s_0: int) -> float:
    """论文 Eq.14：μ(k) = exp(s_0 · log(μ_0) / N(k))。N 越大 μ 越大（更慢的 EMA）。"""
    return math.exp(s_0 * math.log(mu_0) / n_k)


# ============================================================
# Consistency Training loss（无教师版）
# ============================================================
def ct_loss(
    model: ConsistencyModel,
    target_model: ConsistencyModel,
    x: torch.Tensor,
    n_k: int,
    cfg: Config,
) -> torch.Tensor:
    """
    论文 Alg.3 的 loss：
        sigmas = karras(N_k); n ~ U{0..N_k-2}; z ~ N(0,I)
        loss = d( f_θ(x + σ_{n+1}·z, σ_{n+1}),  f_θ⁻(x + σ_n·z, σ_n).detach() )
    f_θ 在更高噪声端被监督，target 在更低噪声端给出"更接近 x"的目标。
    """
    device = x.device
    b = x.shape[0]
    sigmas = karras_sigmas(n_k, cfg.sigma_min, cfg.sigma_max, cfg.rho, device)

    # n 在 [0, N_k - 2]。t_low = σ[n], t_high = σ[n+1]，t_high > t_low。
    n_idx = torch.randint(0, n_k - 1, (b,), device=device)
    t_low = sigmas[n_idx]
    t_high = sigmas[n_idx + 1]

    z = torch.randn_like(x)
    x_high = x + t_high[:, None, None, None] * z
    x_low = x + t_low[:, None, None, None] * z  # 同一个 z！这是 CT 无教师的关键。

    pred = model(x_high, t_high)
    with torch.no_grad():
        target = target_model(x_low, t_low)

    if cfg.loss_type == "l2":
        loss = (pred - target).pow(2).mean()
    elif cfg.loss_type == "huber":
        # iCT 的 Pseudo-Huber: c ≈ 0.00054 · sqrt(D)
        d = float(x[0].numel())
        c = 0.00054 * math.sqrt(d)
        loss = (torch.sqrt((pred - target).pow(2) + c**2) - c).mean()
    else:
        raise ValueError(f"unknown loss_type: {cfg.loss_type}")
    return loss


# ============================================================
# 采样（论文 Alg.1）
# ============================================================
@torch.no_grad()
def sample(model: ConsistencyModel, n_steps: int, batch_size: int, cfg: Config, device: torch.device) -> torch.Tensor:
    """
    NFE = n_steps。
    n_steps=1: 纯单步 — z·σ_max → f_θ(·, σ_max)。
    n_steps>1: 在中间 σ 处反复 "去噪 → 重新加噪"，质量随步数提升。
    """
    shape = (batch_size, cfg.in_channels, cfg.image_size, cfg.image_size)

    # 生成 n_steps+1 个 σ：σ[0]=σ_max（初始）, σ[n_steps]=σ_min（边界，仅作为 sqrt 里的减项）
    ramp = torch.linspace(0, 1, n_steps + 1, device=device)
    min_inv_rho = cfg.sigma_min ** (1.0 / cfg.rho)
    max_inv_rho = cfg.sigma_max ** (1.0 / cfg.rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** cfg.rho  # 降序

    # 初始：x_T ~ N(0, σ_max²·I)；先做一次单步去噪
    x = torch.randn(shape, device=device) * sigmas[0]
    sigma_b = sigmas[0].expand(batch_size)
    x = model(x, sigma_b)

    # 中间步（不含最末的 σ_min）：n_steps-1 次再加噪→再去噪
    for i in range(1, n_steps):
        sigma = sigmas[i]
        z = torch.randn_like(x)
        x = x + math.sqrt(max(sigma.item() ** 2 - cfg.sigma_min**2, 0.0)) * z
        x = model(x, sigma.expand(batch_size))

    return x


# ============================================================
# EMA 更新
# ============================================================
@torch.no_grad()
def ema_update(target: nn.Module, source: nn.Module, mu: float) -> None:
    """θ⁻ ← μ·θ⁻ + (1-μ)·θ。mu 越接近 1，target 跟得越慢。"""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(mu).add_(sp.data, alpha=1.0 - mu)
    for tb, sb in zip(target.buffers(), source.buffers()):
        tb.data.copy_(sb.data)


# ============================================================
# 训练
# ============================================================
def get_dataloader(cfg: Config) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # → [-1, 1]
        ]
    )
    dataset = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=transform)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )


def infinite(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def train(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    out = Path(cfg.out_dir)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    # 模型 + EMA target（θ⁻ 初始化为 θ）
    online = ConsistencyModel(UNet(cfg), cfg).to(device)
    target = ConsistencyModel(UNet(cfg), cfg).to(device)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in online.parameters())
    print(f"[init] model params: {n_params/1e6:.2f}M, device: {device}")

    optimizer = torch.optim.RAdam(online.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    data_iter = infinite(get_dataloader(cfg))

    losses: list[float] = []
    for step in range(1, cfg.total_steps + 1):
        x, _ = next(data_iter)
        x = x.to(device, non_blocking=True)

        # 当前的 N(k) 与 μ(k)
        n_k = n_schedule(step - 1, cfg.total_steps, cfg.s_0, cfg.s_1)
        mu_k = mu_schedule(n_k, cfg.mu_0, cfg.s_0)

        loss = ct_loss(online, target, x, n_k, cfg)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(online.parameters(), cfg.grad_clip)
        optimizer.step()

        ema_update(target, online, mu_k)

        losses.append(loss.item())
        if step % cfg.log_interval == 0:
            avg = sum(losses[-cfg.log_interval:]) / cfg.log_interval
            print(f"[step {step:>6d}/{cfg.total_steps}] loss={avg:.4f}  N(k)={n_k}  μ(k)={mu_k:.4f}")

        if step % cfg.sample_interval == 0 or step == cfg.total_steps:
            online.eval()
            for n_steps in (1, 2):
                samples = sample(online, n_steps=n_steps, batch_size=64, cfg=cfg, device=device)
                samples = (samples.clamp(-1, 1) + 1) / 2.0
                save_image(samples, out / "samples" / f"step{step:06d}_nfe{n_steps}.png", nrow=8)
            online.train()

        if step % cfg.ckpt_interval == 0 or step == cfg.total_steps:
            ckpt = {
                "step": step,
                "online": online.state_dict(),
                "target": target.state_dict(),
                "cfg": cfg.__dict__,
            }
            torch.save(ckpt, out / ("model_final.pt" if step == cfg.total_steps else f"model_step{step:06d}.pt"))

    print(f"[done] checkpoints in {out}")


# ============================================================
# 推理（独立 sample 子命令）
# ============================================================
def sample_from_ckpt(ckpt_path: str, out_dir: str, n_samples: int = 64) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location=device)
    cfg = Config(**state["cfg"])

    online = ConsistencyModel(UNet(cfg), cfg).to(device)
    online.load_state_dict(state["online"])
    online.eval()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for n_steps in (1, 2, 4):
        samples = sample(online, n_steps=n_steps, batch_size=n_samples, cfg=cfg, device=device)
        samples = (samples.clamp(-1, 1) + 1) / 2.0
        save_image(samples, out / f"final_nfe{n_steps}.png", nrow=int(math.sqrt(n_samples)))
        print(f"[sample] NFE={n_steps} -> {out / f'final_nfe{n_steps}.png'}")


# ============================================================
# main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--out-dir", default="./runs/ct_mnist")
    p_train.add_argument("--data-dir", default="./data")
    p_train.add_argument("--total-steps", type=int, default=50_000)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--loss-type", default="l2", choices=["l2", "huber"])
    p_train.add_argument("--seed", type=int, default=42)

    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--ckpt", required=True)
    p_sample.add_argument("--out-dir", default="./runs/ct_mnist/samples_final")
    p_sample.add_argument("--n-samples", type=int, default=64)

    args = parser.parse_args()

    if args.cmd == "train":
        cfg = Config(
            out_dir=args.out_dir,
            data_dir=args.data_dir,
            total_steps=args.total_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            loss_type=args.loss_type,
            seed=args.seed,
        )
        train(cfg)
    elif args.cmd == "sample":
        sample_from_ckpt(args.ckpt, args.out_dir, args.n_samples)


if __name__ == "__main__":
    main()
