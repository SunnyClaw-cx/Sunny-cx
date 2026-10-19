"""吞吐基准：为 0.5B 找最优 batch/accum 组合。

思路：每 optimizer step 的 token 数 = batch × seq × accum。
若增大 batch 而 GPU 吞吐（tok/s）提升，则总训练时间下降；
若 tok/s 不随 batch 提升，说明 GPU 已饱和，加大 batch 只减少 step 数、
不减少时间（且会改变有效 batch size，需重调 lr）。

⚠ 历史缺陷（2026-10-17 修正）
  旧版此脚本**从未真正调用 gradient_checkpointing_enable**，
  但日志里硬编码打印了 "梯度检查点" 字样 → 把「未开 gck 的 5.04GB 峰值」
  误当成正式训练口径。正式训练实际开了 gck，峰值仅 3.17GB。
  这正是 "batch=4 需 >6GB" 结论失真之源。故：
    - 加 --no-gck 显式开关，默认 ON（对齐正式训练）
    - 打印状态改为读取真实变量，不再硬编码

用法：
  python scripts/bench_throughput.py                 # 0.5b + gck（对齐正式训练）
  python scripts/bench_throughput.py --no-gck        # 0.5b 不开 gck
  python scripts/bench_throughput.py --cfg tiny      # 快速自检（不抢训练显存）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sunnycx.config import SunnyConfig
from sunnycx.model import SunnyForCausalLM
from sunnycx.trainer import TrainArgs, build_optimizer, StaticScaler, clip_grad_norm_safe

SEQ = 512
# 前三组 tok/step 相同（8192），用于比较"纯吞吐"；最后一组 tok/step 翻倍
CFGS = [
    (2, 8),
    (4, 4),
    (8, 2),
    (4, 8),
]
WARMUP, MEASURE = 2, 4


def _argv(flag: str, default=None):
    """极简参数读取，避免与训练脚本的 argparse 习惯冲突。"""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def bench(model, opt, scaler, batch: int, accum: int):
    ids = torch.randint(0, 6400, (batch, SEQ), device="cuda")
    mask = torch.ones_like(ids)
    model.train()

    def one_micro():
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(ids, attention_mask=mask, labels=ids)
            loss = out["loss"] / accum
        scaler.scale(loss).backward()

    # 预热
    for _ in range(WARMUP):
        for _ in range(accum):
            one_micro()
        scaler.unscale_(opt)
        clip_grad_norm_safe(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(MEASURE):
        for _ in range(accum):
            one_micro()
        scaler.unscale_(opt)
        clip_grad_norm_safe(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / MEASURE

    step_tok = batch * SEQ * accum
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return dt, step_tok / dt, peak


def main():
    cfg_name = _argv("--cfg", "0.5b")
    gck = "--no-gck" not in sys.argv

    cfg = SunnyConfig.preset(cfg_name)
    torch.manual_seed(0)
    model = SunnyForCausalLM(cfg).to(device="cuda", dtype=torch.float16)
    n_par = sum(p.numel() for p in model.parameters())
    if gck:
        # 与 trainer.train() 一致：必须在建优化器之前开启
        model.gradient_checkpointing_enable(True)

    args = TrainArgs(use_8bit_optim=True, learning_rate=1e-4)
    opt = build_optimizer(model, args)
    scaler = StaticScaler(scale=1024.0)

    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"cap={torch.cuda.get_device_capability(0)}  total={total:.2f}GB")
    print(f"cfg={cfg_name}  params={n_par:,}  seq_len={SEQ}  权重 fp16  "
          f"8bit AdamW  gradient_checkpointing={'ON' if gck else 'OFF'}")
    print()
    print(f"{'batch':>6}{'accum':>7}{'tok/step':>10}{'s/step':>9}{'tok/s':>9}{'峰值GB':>9}  判定")

    results = []
    for b, a in CFGS:
        try:
            dt, tps, peak = bench(model, opt, scaler, b, a)
            ok = "OK" if peak < total * 0.92 else "接近上限"
            results.append((b, a, dt, tps, peak))
            print(f"{b:>6}{a:>7}{b*SEQ*a:>10}{dt:>9.1f}{tps:>9.0f}{peak:>9.2f}  {ok}")
        except RuntimeError as e:
            print(f"{b:>6}{a:>7}{'-':>10}{'-':>9}{'-':>9}{'-':>9}  OOM/失败: "
                  f"{str(e)[:40]}")
            torch.cuda.empty_cache()
            break
        finally:
            opt.zero_grad(set_to_none=True)

    if results:
        best = max(results, key=lambda r: r[3])
        print(f"\n[最优吞吐] batch={best[0]} accum={best[1]}  "
              f"{best[3]:.0f} tok/s  峰值 {best[4]:.2f}GB")
        lo, hi = min(r[3] for r in results), max(r[3] for r in results)
        verdict = ("GPU 已饱和 → 加大 batch 只省 step 数、不省时间"
                   if (hi - lo) / lo < 0.10 else
                   "加大 batch 有实际吞吐收益")
        print(f"[吞吐极差] {lo:.0f} ~ {hi:.0f} tok/s  (差异 {(hi-lo)/lo*100:.1f}%)  → {verdict}")
        total_tok = 97_900_000  # 打包后实测语料 token（0.979 亿）
        for (b, a, dt, tps, peak) in results:
            print(f"  batch{b}×accum{a:<3} 1 epoch = {total_tok/tps/3600:6.1f} h "
                  f"({total_tok/tps/86400:5.2f} 天)  2 epoch = {2*total_tok/tps/86400:5.2f} 天")


if __name__ == "__main__":
    main()