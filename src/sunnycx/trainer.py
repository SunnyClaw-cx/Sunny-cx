"""sunny-cx 训练器

西瓜书（周志华《机器学习》2016）第 5 章理论映射表：

  ┌──────────────────────┬──────────────────────────────────────────────┐
  │ 西瓜书 5.3 误差逆传播 │ fwd → loss → backward → step，由 autograd 执行 │
  │ 西瓜书 5.3 标准BP      │ batch_size 小批量（标准BP 是单样本）           │
  │ 西瓜书 5.3 累积BP      │ accumulation_steps 累积梯度后再更新            │
  │ 西瓜书 5.3 早停法      │ EarlyStopper：验证误差升则停                   │
  │ 西瓜书 5.3 正则化      │ weight_decay：目标函数加权平方和 λ‖w‖²        │
  │ 西瓜书 5.4 全局/局部   │ 余弦退火 + 预热 + 梯度裁剪，跳出局部极小       │
  │ 西瓜书 2.3 性能度量    │ 交叉熵损失                                     │
  └──────────────────────┴──────────────────────────────────────────────┘

工程增强（西瓜书未涵盖，另据文献）：
  · AMP fp16 混合精度 + GradScaler（Micikevicius et al. 2018）
  · 8-bit AdamW 优化器（Dettmers et al. 2022）—— 使 0.5B 可容于 6GB 显存
  · 梯度检查点（Chen et al. 2016）
"""
from __future__ import annotations

import gc
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .config import SunnyConfig
from .model import SunnyForCausalLM


# ======================================================================
# 西瓜书 5.3 早停法
# ======================================================================
class EarlyStopper:
    """早停法（西瓜书 5.3 节「过拟合的缓解策略」）。

    原文：「若训练集误差降低但验证集误差升高，则停止训练，同时返回具有
          最小验证集误差的连接权和阈值。」

    实现要点：**按验证次数计数，不按训练步数计数**。
    若用 step 差与 patience 比较，在 val_interval>1 时量纲错乱，
    patience 会形同虚设（永远只等于 1 次）。
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val = float("inf")
        self.best_step = 0
        self.bad_count = 0          # 连续未改善的**验证次数**
        self.stopped = False

    def update(self, val_loss: float, step: int) -> bool:
        """返回 True 表示本次是最优（应保存权重）。"""
        if val_loss < self.best_val - self.min_delta:
            self.best_val = val_loss
            self.best_step = step
            self.bad_count = 0
            return True
        self.bad_count += 1
        if self.bad_count >= self.patience:
            self.stopped = True
        return False

    def state_dict(self) -> dict:
        return {
            "best_val": self.best_val,
            "best_step": self.best_step,
            "bad_count": self.bad_count,
        }

    def load_state_dict(self, d: dict) -> None:
        self.best_val = d.get("best_val", float("inf"))
        self.best_step = d.get("best_step", 0)
        self.bad_count = d.get("bad_count", 0)

    def save_sidecar(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.state_dict(), ensure_ascii=False, indent=2),
                     encoding="utf-8")

    def load_sidecar(self, path: str | Path) -> bool:
        p = Path(path)
        if p.exists():
            self.load_state_dict(json.loads(p.read_text(encoding="utf-8")))
            return True
        return False


# ======================================================================
# 梯度裁剪（西瓜书 5.4 节：抑制梯度爆炸使迭代稳定）
# ======================================================================
def clip_grad_norm_safe(params, max_norm: float) -> torch.Tensor:
    """fp16 安全版梯度裁剪。

    torch.nn.utils.clip_grad_norm_ 默认在梯度自身精度下求范数。
    当梯度是 fp16 时，grad.pow(2) 在 |grad|>~256 时即上溢为 inf，
    总范数随之变成 inf，裁剪系数退化为 0，**梯度被整体清零**——
    表现为 loss 突然卡住不动，且没有任何报错。故统一在 fp32 下累加。
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    total = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    for g in grads:
        total += g.detach().float().pow(2).sum()
    total_norm = total.sqrt()
    scale = max_norm / (total_norm + 1e-6)
    if scale < 1.0:
        for g in grads:
            g.mul_(scale.to(g.dtype))
    return total_norm


# ======================================================================
# 静态损失缩放器
# ======================================================================
class StaticScaler:
    """fp16 权重下的固定系数损失缩放（替代 torch GradScaler）。

    为什么需要它：
      标准 AMP 流程是「权重 fp32 + autocast 计算 fp16 + GradScaler」。
      但 0.5B 要在 6GB 卡上跑，权重必须降到 fp16 驻留（省 1.05GB），
      此时梯度也是 fp16，而 torch 的 GradScaler 在 unscale_ 时会直接抛
      ValueError("Attempting to unscale FP16 gradients.")。

    原理（与 GradScaler 相同，只是系数固定不动态调整）：
      前向  loss' = loss · S
      反向  grad' = grad · S        （把 fp16 下溢的小梯度抬到可表示范围）
      更新  grad  = grad' / S       （还原真实量级后再 clip / step）

    S 取 2 的幂（如 1024）以避免除法引入舍入误差。
    """

    def __init__(self, scale: float = 1024.0):
        if scale <= 0:
            raise ValueError("loss_scale 必须为正")
        self.scale_value = float(scale)
        self._enabled = True

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale_value

    @torch.no_grad()
    def unscale_(self, optimizer) -> None:
        inv = 1.0 / self.scale_value
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.mul_(inv)

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        pass

    def is_enabled(self) -> bool:
        return self._enabled

    def state_dict(self) -> dict:
        return {"scale": self.scale_value, "kind": "static"}

    def load_state_dict(self, d: dict) -> None:
        if d and "scale" in d:
            self.scale_value = float(d["scale"])


# ======================================================================
# 学习率调度（西瓜书 5.4 节：策略性跳出局部极小）
# ======================================================================
def lr_schedule(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 200,
    min_ratio: float = 0.1,
) -> float:
    """线性预热 + 余弦退火到 min_ratio·base_lr。

    预热：避免训练初期梯度方向不稳定时步子过大。
    余弦退火：逐步减小学习率使收敛更精细，同时在初期保持较大步长
             以降低陷入局部极小的风险（西瓜书 5.4 节）。
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1 - min_ratio) * cosine)


# ======================================================================
# 训练配置
# ======================================================================
@dataclass
class TrainArgs:
    stage: str = "pretrain"                 # pretrain | sft
    data_path: str = ""
    val_data_path: str = ""
    tokenizer_path: str = ""
    save_dir: str = "checkpoints"
    save_weight: str = "sunnycx"

    epochs: int = 1
    batch_size: int = 4
    accumulation_steps: int = 8             # 西瓜书 5.3 累积BP
    learning_rate: float = 3e-4
    weight_decay: float = 0.1               # 西瓜书 5.3 正则化
    max_seq_len: int = 512
    dtype: str = "float16"

    val_interval: int = 200                 # 每多少步做一次验证
    val_samples: int = 200
    early_stop_patience: int = 5            # 西瓜书 5.3 早停
    grad_clip: float = 1.0                  # 西瓜书 5.4 梯度裁剪

    warmup_steps: int = 200
    log_interval: int = 20
    save_interval: int = 500
    num_workers: int = 0
    max_steps: int = 0                      # >0 时提前停止，用于冒烟/调试

    use_8bit_optim: bool = True             # 8-bit AdamW，6GB 卡训 0.5B 的关键
    model_dtype: str = "auto"               # auto|fp32|fp16|bf16 —— 权重驻留精度
    loss_scale: float = 1024.0              # fp16 权重驻留时的静态损失缩放系数
    gradient_checkpointing: bool = False
    from_resume: bool = False
    from_weight: str = ""                   # SFT 时载入预训练权重
    seed: int = 42


# ======================================================================
# 优化器构造
# ======================================================================
def build_optimizer(model: nn.Module, args: TrainArgs):
    """构造 AdamW 优化器。

    use_8bit_optim=True 时使用 bitsandbytes 的 8-bit AdamW：
    优化器状态从 16 bytes/参数（fp32 m,v）降到 2 bytes/参数，
    使 524M 参数的训练态显存从约 10GB 降至约 3.9GB，可容纳于 GTX 1060 6GB。

    西瓜书 5.3 节的正则化项 λ‖w‖² 对应 weight_decay；
    注意 bias 与归一化权重不施加权重衰减。
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if args.use_8bit_optim:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(groups, lr=args.learning_rate, betas=(0.9, 0.95))
            print("[optim] 使用 bitsandbytes 8-bit AdamW（优化器状态 2 bytes/参数）")
            return opt
        except ImportError:
            print("[optim] 警告：bitsandbytes 不可用，回退到 fp32 AdamW")

    print("[optim] 使用 torch fp32 AdamW")
    return torch.optim.AdamW(groups, lr=args.learning_rate, betas=(0.9, 0.95))


# ======================================================================
# 验证（西瓜书 2.2.1 留出法 + 2.3 性能度量）
# ======================================================================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: Optional[int] = None,
) -> float:
    """在留出验证集上计算平均交叉熵损失（西瓜书 2.3 节性能度量）。"""
    model.eval()
    total_loss, n = 0.0, 0
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=(device.type == "cuda")):
            out = model(ids, attention_mask=mask, labels=labels)
        total_loss += out["loss"].item()
        n += 1
    model.train()
    gc.collect()
    return total_loss / max(n, 1)


# ======================================================================
# 主训练循环
# ======================================================================
def train(model: SunnyForCausalLM, args: TrainArgs, train_loader, val_loader,
          device: torch.device, total_steps: int, resumer=None, on_step=None,
          thermal=None, rest=None) -> dict:
    """执行训练。resumer/on_step/thermal/rest 为可选钩子（断点续训、状态回调、温度熔断、节律休息）。"""
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if device.type == "cuda" and args.dtype == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            print("[warn] 本 GPU 不支持 bfloat16，自动降级为 float16")
            dtype = torch.float16

    # ------------------------------------------------------------------
    # 权重驻留精度：决定 0.5B 能否容于 GTX 1060 6GB 的关键一步
    # ------------------------------------------------------------------
    #  fp32 驻留  → 权重 4 B/参数。524M × 4 = 2.10 GB，梯度同量级 = 4.2 GB
    #  fp16 驻留  → 权重 2 B/参数。524M × 2 = 1.05 GB，梯度同量级 = 2.10 GB
    #  该项是 num_params × 4 vs × 2 的差别，直接决定 OOM 与否。
    #  注意：track_running_stats 的 BatchNorm 类模块不支持 fp16，本模型
    #  全部为 RMSNorm（无可学习统计量），转 fp16 是安全的。
    model_dtype = None
    if device.type == "cuda":
        if args.model_dtype == "auto":
            model_dtype = torch.float16          # sm_61 无 bf16，用 fp16 驻留
        elif args.model_dtype == "fp16":
            model_dtype = torch.float16
        elif args.model_dtype == "bf16":
            model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif args.model_dtype == "fp32":
            model_dtype = torch.float32
    else:
        model_dtype = torch.float32              # CPU 上 fp16 极慢，保持 fp32

    model.to(device=device, dtype=model_dtype)
    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(True)
        print("[model] 已启用梯度检查点（用时间换激活显存）")

    if device.type == "cuda":
        n = model.num_parameters
        bpp = 2 if model_dtype == torch.float16 else 4
        print(f"[mem] 权重驻留 {str(model_dtype).replace('torch.','')} "
              f"≈ {n*bpp/1024**3:.2f} GiB（{bpp} B/参数 × {n:,}）")
        print(f"[mem] 权重+梯度  ≈ {n*bpp*2/1024**3:.2f} GiB，"
              f"8bit Adam 状态 ≈ {n*2/1024**3:.2f} GiB，"
              f"合计基线 ≈ {n*(bpp*2+2)/1024**3:.2f} GiB")
        torch.cuda.reset_peak_memory_stats()

    optimizer = build_optimizer(model, args)

    # ------------------------------------------------------------------
    # 损失缩放器：fp16 梯度下溢保护
    # ------------------------------------------------------------------
    #  · 权重 fp32（传统 AMP）→ 用 torch GradScaler（它只接受 fp32 梯度）
    #  · 权重 fp16（6GB 卡训 0.5B 的必要条件）→ 梯度也是 fp16，
    #    GradScaler.unscale_ 会抛 "Attempting to unscale FP16 gradients."
    #    故改用静态缩放：loss*S → backward → grad/S → clip → step
    if model_dtype == torch.float16:
        scaler = StaticScaler(scale=args.loss_scale)
        print(f"[amp] 权重 fp16 → 启用静态损失缩放 S={args.loss_scale:g}"
              f"（GradScaler 不支持 fp16 梯度）")
    else:
        scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    stopper = EarlyStopper(patience=args.early_stop_patience)

    start_epoch, start_step = 0, 0
    if args.from_resume and resumer is not None:
        loaded = resumer.load(model, optimizer, scaler)
        if loaded:
            start_epoch = loaded.get("epoch", 0)
            start_step = loaded.get("step", 0)
            st = loaded.get("early_stop", {})
            if st:
                stopper.load_state_dict(st)
                print(f"[resume] 恢复早停状态：best_val={stopper.best_val:.4f} "
                      f"@ step {stopper.best_step}, bad={stopper.bad_count}")
            print(f"[resume] 从 epoch {start_epoch+1} / step {start_step+1} 继续")

    history: List[dict] = []
    global_step = start_step
    t0 = time.time()
    stop_flag = False

    # 最近一次真实验证结果 (val_loss, 测量步号)。
    # 训练状态每一步都要带 val_loss 供顶栏持续显示，但必须同时带上它
    # **测于哪一步**（val_at_step）——大屏据此区分"真实验证事件"与"沿用值"，
    # 否则会把每个 step 都记成一次验证（"VAL ×N" 虚高的根因）。
    _last_val: List = [None, None]

    for epoch in range(start_epoch, args.epochs):
        for step_in_epoch, batch in enumerate(train_loader):
            if step_in_epoch < start_step and epoch == start_epoch:
                continue

            ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)

            # ---- 西瓜书 5.3 误差逆传播：前向 ----
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=(device.type == "cuda")):
                out = model(ids, attention_mask=mask, labels=labels)
                loss = out["loss"] / args.accumulation_steps

            # ---- 反向传播（autograd 执行 BP 的链式求导）----
            scaler.scale(loss).backward()

            accum_done = (step_in_epoch + 1) % args.accumulation_steps == 0

            if accum_done:
                # 西瓜书 5.4：梯度裁剪，抑制梯度爆炸使迭代稳定
                scaler.unscale_(optimizer)
                gnorm = clip_grad_norm_safe(
                    [p for p in model.parameters() if p.grad is not None],
                    args.grad_clip,
                )
                # 学习率调度
                lr = lr_schedule(global_step, total_steps, args.learning_rate,
                                 args.warmup_steps)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                # 参数更新
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # ---- GPU 温度熔断（在 step 边界检查：中断点干净，恢复不丢状态）----
                # 只在 optimizer step 之后查，天然是"可安全暂停"的位置：
                # 梯度已清零、参数已更新，暂停期间不持有中间状态。
                if thermal is not None:
                    ev = thermal.guard()
                    if ev and ev.get("paused"):
                        _ts = time.time()
                        print(f"[thermal] 训练已暂停，等待 GPU 降温…"
                              f"（已运行 {global_step} step）", flush=True)
                        thermal.wait_cool()
                        # 恢复后把冷却时长从"训练耗时"里扣除，
                        # 否则 elapsed 会把等待时间算成训练时间，进度估算全错
                        t0 += time.time() - _ts
                        print(f"[thermal] 已恢复训练，继续 step {global_step}", flush=True)

                # ---- 节律休息（主动：连续训练满 N 小时 → 休息 M 分钟）----
                # 与温度熔断同一位置：optimizer.step() 之后，梯度已清零、
                # 参数已落地，暂停不持有中间状态，恢复可直接继续。
                if rest is not None:
                    rv = rest.guard()
                    if rv and rv.get("rest"):
                        _rs = time.time()
                        print(f"[rest] 训练已暂停，休息中…（已运行 {global_step} step）",
                              flush=True)
                        rest.wait_rest()
                        # 休息时长必须从"训练耗时"里扣除，否则 elapsed 虚高、
                        # ETA 整体偏长；且累计训练时长会把休息算进去 → 休息越来越频繁
                        t0 += time.time() - _rs
                        print(f"[rest] 已恢复训练，继续 step {global_step}", flush=True)

                # ---- 日志 ----
                if global_step % args.log_interval == 0:
                    elapsed = time.time() - t0
                    mem = ""
                    if device.type == "cuda":
                        peak = torch.cuda.max_memory_allocated() / 1024**3
                        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        mem = f", peak: {peak:.2f}/{total:.2f}GB"
                    print(f"Epoch:[{epoch+1}/{args.epochs}]"
                          f"({global_step}/{total_steps}), "
                          f"loss: {loss.item()*args.accumulation_steps:.4f}, "
                          f"lr: {lr:.8f}, gnorm: {gnorm.item():.3f}, "
                          f"elapsed: {elapsed/60:.1f}min{mem}")

                # ---- 验证（西瓜书 2.2.1 留出法）----
                is_val = (val_loader is not None
                          and global_step % args.val_interval == 0)
                if is_val:
                    val_loss = evaluate(model, val_loader, device, dtype)
                    train_loss = loss.item() * args.accumulation_steps
                    improved = stopper.update(val_loss, global_step)
                    _last_val[0] = val_loss
                    _last_val[1] = global_step          # 真实测量步
                    print(f"[VAL] step {global_step}/{total_steps}  "
                          f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
                          f"gap: {train_loss - val_loss:+.4f}  best: {stopper.best_val:.4f}")
                    history.append({
                        "step": global_step, "epoch": epoch + 1,
                        "train_loss": train_loss, "val_loss": val_loss,
                        "lr": lr, "kind": "val",
                    })
                    if improved:
                        _save_checkpoint(model, optimizer, scaler, args, epoch,
                                         global_step, stopper, device,
                                         suffix="best", history=history)
                        print(f"[early-stop] 新最优 val={val_loss:.4f}，已保存 best 权重")
                else:
                    history.append({
                        "step": global_step, "epoch": epoch + 1,
                        "train_loss": loss.item() * args.accumulation_steps,
                        "lr": lr, "kind": "train",
                    })

                if on_step is not None:
                    # val_loss 每步都下发（供顶栏持续显示"最近一次验证值"），
                    # 但必须同时标明它**测于哪一步**——否则大屏无法区分
                    # "真实验证事件"与"沿用值"，会把每个 step 都误记成一次验证。
                    _st = {
                        "step": global_step, "epoch": epoch + 1,
                        "total_steps": total_steps,
                        "loss": loss.item() * args.accumulation_steps,
                        "lr": lr,
                        "val_loss": _last_val[0],
                        "val_at_step": _last_val[1],
                        "grad_norm": gnorm.item(),
                        "elapsed_sec": time.time() - t0,
                    }
                    # 温度读数（真实 nvidia-smi 采样）随状态一起下发
                    if thermal is not None:
                        _st.update(thermal.snapshot())
                    # 节律休息状态随状态一起下发（大屏显示"休息中/已休 N 次"）
                    if rest is not None:
                        _st.update(rest.snapshot())
                    on_step(_st)

                if global_step % args.save_interval == 0:
                    _save_checkpoint(model, optimizer, scaler, args, epoch,
                                     global_step, stopper, device,
                                     suffix="latest", history=history)

                # ---- 西瓜书 5.3 早停：验证误差连续不降则停止 ----
                if stopper.stopped:
                    print(f"[early-stop] 验证损失连续 {stopper.patience} 次未改善，"
                          f"停止训练（最优 val={stopper.best_val:.4f} @ step {stopper.best_step}）")
                    stop_flag = True
                    break

                if args.max_steps and global_step >= args.max_steps:
                    print(f"[train] 达到 --max_steps={args.max_steps}，提前结束（调试用）")
                    stop_flag = True
                    break

        start_step = 0
        if stop_flag:
            break

    _save_checkpoint(model, optimizer, scaler, args, epoch, global_step,
                     stopper, device, suffix="final", history=history)
    return {
        "final_step": global_step,
        "best_val": stopper.best_val,
        "best_step": stopper.best_step,
        "stopped_early": stopper.stopped,
        "history": history,
    }


# ======================================================================
# 权重保存
# ======================================================================
def _save_checkpoint(model, optimizer, scaler, args: TrainArgs, epoch: int,
                     step: int, stopper: EarlyStopper, device, suffix: str,
                     history: List[dict]) -> None:
    """保存权重。

    suffix="best"  → 只存权重 + 早停 sidecar（体积小，供最终交付）
    suffix="latest"/"final" → 存完整训练态（model+optimizer+scaler），供续训
    """
    sd = Path(args.save_dir)
    sd.mkdir(parents=True, exist_ok=True)
    tag = f"{args.save_weight}_{suffix}"

    # 权重（半精度，方便分发；fp16 由 sm_61 硬件支持）
    if suffix == "best":
        state = {k: v.half().cpu() for k, v in model.state_dict().items()}
        torch.save(state, sd / f"{tag}.pth")
        stopper.save_sidecar(sd / f"{tag}.json")
        return

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": epoch,
            "step": step,
            "early_stop": stopper.state_dict(),
            "args": vars(args),
            "history": history,
        },
        sd / f"{tag}.pt",
    )
    meta = {
        "epoch": epoch, "step": step,
        "best_val": stopper.best_val, "best_step": stopper.best_step,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (sd / f"{tag}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class Resumer:
    """断点续训：从 latest 检查点恢复模型/优化器/早停状态。"""

    def __init__(self, save_dir: str, save_weight: str):
        self.path = Path(save_dir) / f"{save_weight}_latest.pt"
        # None=未尝试 / True=已恢复 / False=显存不足已跳过
        self.optimizer_restored: Optional[bool] = None

    def exists(self) -> bool:
        return self.path.exists()

    def load(self, model, optimizer=None, scaler=None) -> Optional[dict]:
        if not self.path.exists():
            print(f"[resume] 未找到检查点 {self.path}，从头开始")
            return None
        ckpt = torch.load(self.path, map_location="cpu", weights_only=False)
        # 权重精度对齐：检查点存的是 fp16，模型此刻可能是 fp32（或反之）。
        # load_state_dict 要求 dtype 一致，故按模型当前 dtype 转换后再载入。
        target_dtype = next(model.parameters()).dtype
        state = ckpt["model"]
        state = {k: (v.to(target_dtype) if v.is_floating_point() else v)
                 for k, v in state.items()}
        model.load_state_dict(state)
        # state 与 ckpt["model"] 是两份等大的 CPU 副本；优化器状态搬运前先放掉，
        # 避免与 bnb 的逐参数 cast 抢内存（CPU 侧也影响页表与 GC 抖动）。
        del state
        ckpt["model"] = None
        if optimizer is not None and ckpt.get("optimizer"):
            # bnb 8-bit 优化器恢复 = 逐参数把状态张量搬上卡（几百次小额分配）。
            # 6GB 卡上极易撞分配器碎片，表现为「还有 3.5GB 空闲却分配不出 2MB」。
            # 先清缓存 + 触发 GC，把可复用的块还给分配器。
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            optimizer_restored = False
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                optimizer_restored = True
            except (ValueError, KeyError) as e:
                print(f"[resume] 优化器状态恢复失败（可能换了优化器）：{e}")
            except RuntimeError as e:
                # bnb 8-bit 恢复会临时把 int8 状态升精度，峰值远超常驻的 0.98GiB。
                # 6GB 卡上偶发 OOM（表现为「还有 3.5GB 空闲却分配不出 2MB」）。
                # 这里只吞 OOM，别的 RuntimeError 照常抛。
                if "out of memory" not in str(e).lower():
                    raise
                print("[resume] ⚠ 显存不足，无法恢复 8-bit 优化器状态 —— 已跳过")
                print("        模型权重 / step / 早停状态均已恢复，训练可继续；")
                print("        代价是 Adam 一阶二阶矩重置，接下来数百步 loss 会有一次小幅回升。")
                print("        要避免此降级：先关闭占显存的程序再续训。")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            if optimizer is not None:
                self.optimizer_restored = optimizer_restored
        if scaler is not None and ckpt.get("scaler") and scaler.is_enabled():
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                print(f"[resume] scaler 状态恢复失败：{e}")
        return {
            "epoch": ckpt.get("epoch", 0),
            "step": ckpt.get("step", 0),
            "early_stop": ckpt.get("early_stop", {}),
            "optimizer_restored": self.optimizer_restored,
        }


def load_pretrained_weights(model: SunnyForCausalLM, weight_path: str,
                            device: torch.device) -> bool:
    """载入初始权重（SFT 参数迁移 / 预训练热启动）。

    注意：目标权重常以 fp16 存盘，而模型此刻可能是 fp32（CPU 构建阶段）。
    load_state_dict 虽能跨 dtype copy_，但显式对齐更稳，且避免同 storage
    的 tied 权重（embed_tokens / lm_head）被拆成两份数值。
    """
    p = Path(weight_path)
    if not p.exists():
        print(f"[weights] 未找到权重文件 {p}")
        return False
    state = torch.load(p, map_location=device, weights_only=True)
    target_dtype = next(model.parameters()).dtype
    state = {k: (v.to(target_dtype) if v.is_floating_point() else v)
             for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[weights] 缺失 {len(missing)} 项（前 3）：{missing[:3]}")
    if unexpected:
        print(f"[weights] 多余 {len(unexpected)} 项（前 3）：{unexpected[:3]}")
    print(f"[weights] 已载入 {p}  ({len(state)} 张量，dtype={target_dtype})")
    return True