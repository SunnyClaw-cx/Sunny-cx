"""SFT 分桶真机基准（GTX 1060 6GB / sm_61）

在**同一个进程**里跑两种 padding 策略，避免重复初始化 0.5B（>1 分钟）：
    fixed  ：每条样本补到 max_seq_len=512（原状）
    bucket ：动态 pad 到 batch 最长 + 块内按长度降序（分桶）

测量口径与正式训练一致：fp16 权重驻留 + 梯度检查点 + 8bit AdamW + 静态损失缩放。
每 batch 都做一次 optimizer step（accumulation_steps=1），所以
"s/batch" 即 "s/optimizer-step"，可直接乘总步数折算单轮天数。

用法：
    python scripts/bench_sft_bucket.py                     # 默认各 20 个 batch
    python scripts/bench_sft_bucket.py --n 30 --batch_size 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sunnycx.config import SunnyConfig                       # noqa: E402
from sunnycx.dataset import SFTDataset, collate, collate_pad  # noqa: E402
from sunnycx.model import SunnyForCausalLM                   # noqa: E402
from sunnycx.trainer import (                                # noqa: E402
    StaticScaler, build_optimizer, clip_grad_norm_safe, TrainArgs,
    load_pretrained_weights,
)


def make_loader(tok, path, max_seq_len, batch_size, mode, chunk):
    dyn = mode != "fixed"
    ds = SFTDataset(path, tok, max_seq_len=max_seq_len, shuffle_buffer=2000, seed=42,
                    dynamic_pad=dyn, bucket_chunk=(chunk if dyn else 0))
    return DataLoader(ds, batch_size=batch_size, num_workers=0, drop_last=True,
                      collate_fn=(collate_pad if dyn else collate))


@torch.no_grad()
def _warm():
    torch.cuda.synchronize()


def run_mode(model, loader, device, dtype, accum, opt, scaler, n, warmup, stride=1,
             log_interval=1):
    """跑 n 个 **optimizer step**，返回 (秒/step, 有效 token/s, 计算位置/s)。

    与正式训练完全同构：每个 optimizer step 内做 accumulation_steps 个
    micro-batch 的前向/反向，再统一 unscale → 裁剪 → step。

    ⚠ 采样必须**跨块等距**：分桶后块内按长度降序，若只取开头几个 batch，
    拿到的全是该块最长的样本 → 会把加速比算高（首次实测正是踩了这个坑）。
    """
    it = iter(loader)

    def next_batch():
        while True:
            try:
                return next(it)
            except StopIteration:      # 数据太短时重新开始，避免基准中断
                return next(iter(loader))

    def one_step():
        """一个完整 optimizer step，返回本 step 的有效/计算 token 数。"""
        real = comp = 0
        for _ in range(accum):
            batch = next_batch()
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["attention_mask"].to(device)
            real += int(mask.sum())
            comp += batch["input_ids"].numel()
            with torch.autocast("cuda", dtype=dtype):
                out = model(ids, attention_mask=mask, labels=labels)
            scaler.scale(out["loss"] / accum).backward()
        scaler.unscale_(opt)
        clip_grad_norm_safe([p for p in model.parameters() if p.grad is not None], 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        return real, comp

    # 预热（首次 step 含 cudaMalloc 与 cuBLAS 句柄创建，必须排除）
    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()

    real_tok = comp_pos = 0
    timed = 0
    dt_total = 0.0
    while timed < n:
        for _ in range(max(0, stride - 1)):     # 跳过若干 step，只为跨块
            one_step()
        torch.cuda.synchronize()
        t0 = time.time()
        r, c = one_step()
        torch.cuda.synchronize()
        dt_total += time.time() - t0
        real_tok += r
        comp_pos += c
        timed += 1
        if timed % max(1, log_interval) == 0:
            print(f"    · {timed}/{n} step，均 {dt_total/timed:.2f} s/step，"
                  f"有效 {real_tok/dt_total:.0f} tok/s", flush=True)

    sp = dt_total / max(timed, 1)
    return sp, real_tok / dt_total, comp_pos / dt_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "tcm-llm" / "data" /
                                          "processed_v2" / "tcm_classics_qa.train.jsonl"))
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--accumulation_steps", type=int, default=1)
    ap.add_argument("--n", type=int, default=20, help="每种模式计时 batch 数")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--stride", type=int, default=0,
                    help="计时 step 之间跳过多少 step（0=自动=chunk/每step样本数）。"
                         "取 1 且 n=chunk/每step样本数 时覆盖整个块，最无偏")
    ap.add_argument("--from_weight",
                    default=str(ROOT / "checkpoints" / "sunnycx_0.5b_pretrain_warm5600.pth"))
    ap.add_argument("--n_samples", type=int, default=1_081_549,
                    help="SFT 训练集条数（v2 实测 1,081,549），用于折算单轮步数/天数")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    print(f"[env] {torch.cuda.get_device_name(0)}  {device}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT.parent / "tcm-llm" / "model_tcm_tokenizer"))

    cfg = SunnyConfig.preset("0.5b")
    model = SunnyForCausalLM(cfg)
    print(f"[model] {model.num_parameters:,}")

    if a.from_weight and Path(a.from_weight).exists():
        load_pretrained_weights(model, a.from_weight, device)
    model.to(device=device, dtype=torch.float16)
    model.gradient_checkpointing_enable(True)
    model.train()

    targs = TrainArgs(use_8bit_optim=True, learning_rate=2e-4)
    opt = build_optimizer(model, targs)
    scaler = StaticScaler(scale=1024.0)

    rows = []
    per_step = a.batch_size * a.accumulation_steps
    stride = a.stride if a.stride > 0 else max(1, a.chunk // per_step)
    for mode in ("fixed", "bucket"):
        loader = make_loader(tok, a.data, a.max_seq_len, a.batch_size, mode, a.chunk)
        torch.cuda.reset_peak_memory_stats()
        sp, tps, pps = run_mode(model, loader, device, dtype, a.accumulation_steps,
                                opt, scaler, a.n, a.warmup, stride)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        rows.append((mode, sp, tps, pps, peak))
        print(f"[{mode:6s}] {sp:6.2f} s/step   有效 {tps:6.1f} tok/s   "
              f"计算 {pps:6.1f} 位置/s   peak {peak:.2f} GB", flush=True)

    base = rows[0][1]
    samples_per_step = a.batch_size * a.accumulation_steps
    total_steps = a.n_samples // samples_per_step
    print("\n" + "=" * 80)
    print(f"{'模式':<10}{'s/step':>10}{'有效tok/s':>12}{'计算位置/s':>12}"
          f"{'peak GB':>10}{'加速':>8}{'单轮天':>10}")
    for mode, sp, tps, pps, peak in rows:
        days = total_steps * sp / 86400
        print(f"{mode:<10}{sp:>10.2f}{tps:>12.1f}{pps:>12.1f}{peak:>10.2f}"
              f"{base/sp:>7.2f}x{days:>9.2f}")
    print("=" * 80)
    print(f"折算口径：{a.n_samples:,} 条 ÷ ({a.batch_size}×{a.accumulation_steps})"
          f" = {total_steps:,} 个 optimizer step；等距抽样 stride={stride} 个 batch")
    print("⚠ 单轮天数只按训练 step 线性折算，不含验证与休息")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
