"""
Train a diffusion model on images.

中文说明
========
这是 OpenAI 官方 Consistency Models 仓库的训练入口脚本。
一份脚本同时支持三种训练模式（由 --training_mode 切换）：

    1. consistency_distillation (CD)  : 从一个预训练好的 EDM/Karras 教师模型蒸馏出 CM
    2. consistency_training     (CT)  : 不依赖教师模型，从零开始训练 CM
    3. progdist                       : Progressive Distillation（渐进式蒸馏，对照基线）

核心组件：
    - model         : 在线网络  f_θ(x, t)        —— 真正在被优化
    - target_model  : 目标网络  f_{θ⁻}(x, t)     —— θ 的 EMA，提供 self-consistency 监督信号
    - teacher_model : 教师网络（仅 CD 用）       —— 提供 ODE 一步推演的真值
    - ema_scale_fn  : 给定当前训练步 k，返回当下应该使用的
                        (target_ema_decay μ(k), num_scales N(k))
                      也就是论文里 Eq.13 / Eq.14 描述的离散化与 EMA curriculum

外部入口：scripts/launch.sh 里有各数据集的具体启动命令。
"""

import argparse

from cm import dist_util, logger
from cm.image_datasets import load_data
from cm.resample import create_named_schedule_sampler
from cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    cm_train_defaults,
    args_to_dict,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cm.train_util import CMTrainLoop
import torch.distributed as dist
import copy


def main():
    # 1) 解析命令行参数（默认值合并自三处：本文件 + model_and_diffusion_defaults + cm_train_defaults）
    args = create_argparser().parse_args()

    # 2) 初始化分布式环境（torch.distributed），并配置 logger（写到 OPENAI_LOGDIR）
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")

    # 3) 构造 (μ(k), N(k)) 调度函数。
    #    - target_ema_mode: "fixed" 或 "adaptive"
    #        adaptive 时 μ(k) 会随 N(k) 自适应（论文 Eq.14），CT 训练时常用
    #    - scale_mode    : "fixed" / "progressive" / "progdist"
    #        progressive 时 N(k) 从 start_scales 增长到 end_scales（论文 Eq.13 离散化课程）
    #    - 返回的 ema_scale_fn(step) -> (target_ema, num_scales)
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )

    # 4) 决定是否走 "蒸馏式" 的网络分支。
    #    consistency_* (CD/CT) 都需要 distillation=True 的网络结构（带边界参数化 c_skip / c_out 等）；
    #    progdist 走传统 ε-prediction 头，distillation=False。
    if args.training_mode == "progdist":
        distillation = False
    elif "consistency" in args.training_mode:
        distillation = True
    else:
        raise ValueError(f"unknown training mode {args.training_mode}")

    # 5) 构造模型与 diffusion 包装器。
    #    create_model_and_diffusion 内部会创建：
    #      - UNetModel：主干网络
    #      - KarrasDenoiser：负责 σ 调度、c_skip/c_out 参数化、loss 计算等所有 CM 数学细节
    model_and_diffusion_kwargs = args_to_dict(
        args, model_and_diffusion_defaults().keys()
    )
    model_and_diffusion_kwargs["distillation"] = distillation
    model, diffusion = create_model_and_diffusion(**model_and_diffusion_kwargs)
    model.to(dist_util.dev())
    model.train()
    if args.use_fp16:
        # 用混合精度训练。注意 fp16 的 loss scale 由 fp16_scale_growth 动态调节
        model.convert_to_fp16()

    # 6) 时间步采样器：决定每个 batch 在 [1, N(k)-1] 中怎么采 n。
    #    "uniform" 即论文里默认的均匀采样。
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    # 7) 处理 batch_size。
    #    - 若 batch_size=-1，则用 global_batch_size 在所有 rank 上均分
    #    - 否则就是每个 rank 的本地 batch_size
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size

    # 8) 构造数据迭代器（无限循环 yield）。
    #    class_cond=True 时每个 batch 还会带上类别标签，用于条件生成。
    data = load_data(
        data_dir=args.data_dir,
        batch_size=batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    # 9) 仅 CD（蒸馏）模式才会走这里：加载已经训好的 EDM 教师模型。
    #    教师模型用来在 PF-ODE 上从 t_{n+1} 走一步到 t_n（Heun solver），给出"真值终点"。
    #    并且会用教师权重去初始化 student（model）—— 这是论文里的标准 trick。
    if len(args.teacher_model_path) > 0:  # path to the teacher score model.
        logger.log(f"loading the teacher model from {args.teacher_model_path}")
        teacher_model_and_diffusion_kwargs = copy.deepcopy(model_and_diffusion_kwargs)
        teacher_model_and_diffusion_kwargs["dropout"] = args.teacher_dropout
        teacher_model_and_diffusion_kwargs["distillation"] = False  # 教师是普通扩散模型
        teacher_model, teacher_diffusion = create_model_and_diffusion(
            **teacher_model_and_diffusion_kwargs,
        )

        teacher_model.load_state_dict(
            dist_util.load_state_dict(args.teacher_model_path, map_location="cpu"),
        )

        teacher_model.to(dist_util.dev())
        teacher_model.eval()  # 教师不训练

        # 用教师权重初始化 student：CM 主干和 EDM 主干结构兼容，可直接 copy
        for dst, src in zip(model.parameters(), teacher_model.parameters()):
            dst.data.copy_(src.data)

        if args.use_fp16:
            teacher_model.convert_to_fp16()

    else:
        # CT（从零训练）或 progdist：没有教师
        teacher_model = None
        teacher_diffusion = None

    # load the target model for distillation, if path specified.

    # 10) 构造 target_model（θ⁻）。它是 model（θ）的 EMA 拷贝，提供 self-consistency 监督。
    #     在 train loop 里每个 step 之后，会用 μ(k) 去 EMA 更新它的参数。
    logger.log("creating the target model")
    target_model, _ = create_model_and_diffusion(
        **model_and_diffusion_kwargs,
    )

    target_model.to(dist_util.dev())
    target_model.train()

    # 多卡时同步初始权重，保证所有 rank 的 target_model 起点一致
    dist_util.sync_params(target_model.parameters())
    dist_util.sync_params(target_model.buffers())

    # 用当前 model 权重初始化 target_model，确保 θ⁻(0) = θ(0)
    for dst, src in zip(target_model.parameters(), model.parameters()):
        dst.data.copy_(src.data)

    if args.use_fp16:
        target_model.convert_to_fp16()

    logger.log("training...")

    # 11) 构造训练循环并运行。
    #     CMTrainLoop 内部每个 step 做：
    #       a. 拿 batch (x, [y])
    #       b. 在 [1, N(k)-1] 采 n，计算 t_n 和 t_{n+1}
    #       c. 用 teacher（CD）或 ODE 直接采样（CT）得到 (x_{t_n}, x_{t_{n+1}}) 这对相邻轨迹点
    #       d. f_θ(x_{t_{n+1}}, t_{n+1}) vs f_{θ⁻}(x_{t_n}, t_n)，算 L2 / LPIPS / Pseudo-Huber loss
    #       e. 反传更新 θ；按 μ(k) EMA 更新 θ⁻
    #       f. 每 save_interval 步保存 ckpt
    CMTrainLoop(
        model=model,
        target_model=target_model,
        teacher_model=teacher_model,
        teacher_diffusion=teacher_diffusion,
        training_mode=args.training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=args.total_training_steps,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,  # 注意：这是"用于推理的 EMA"，与 target_model 的 μ(k) 是两回事
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    """
    构造 argparse。所有默认值会被三层叠加：
        defaults（本函数）           —— 优化器、batch、log/ckpt 相关
        model_and_diffusion_defaults —— 网络结构、σ 范围、image_size 等
        cm_train_defaults            —— CM 专属参数（training_mode、ema/scale 调度、教师路径等）
    """
    defaults = dict(
        data_dir="",                     # 训练数据目录（webdataset 或图片文件夹）
        schedule_sampler="uniform",      # n 的采样策略
        lr=1e-4,                         # 学习率（Adam/RAdam）
        weight_decay=0.0,
        lr_anneal_steps=0,               # 0 表示不退火
        global_batch_size=2048,          # 跨所有 GPU 的总 batch
        batch_size=-1,                   # 单卡 batch；-1 表示用 global_batch_size 自动均分
        microbatch=-1,                   # 梯度累积单元；-1 表示一次跑完整 batch
        ema_rate="0.9999",               # 推理用 EMA 的衰减率（可以多个，逗号分隔）
        log_interval=10,                 # 每多少 step 打一次日志
        save_interval=10000,             # 每多少 step 存一次 ckpt
        resume_checkpoint="",            # 续训用的 ckpt 路径
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cm_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
