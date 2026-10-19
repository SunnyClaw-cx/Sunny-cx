# -*- coding: utf-8 -*-
"""从 latest.pt 提取纯模型权重，作为预训练热启动源。

背景：latest@5600 在大样本(1200条)上 ppl 31.21，优于 best@4600 的 34.02。
但 latest.pt 是包装格式（model+optimizer+scaler+step），不能直接喂给
load_pretrained_weights（它用 weights_only=True 读纯 state_dict）。

关键：config.tie_word_embeddings=True → lm_head.weight 与 embed_tokens.weight
是同一个 Parameter。提取时**绝不能 clone**，否则破坏 storage 共享，
文件多出一份 7.37M 参数（+14MB），且埋下 tied 语义被拆开的地雷。
"""
import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8")

CKPT = Path(__file__).resolve().parent.parent / "checkpoints"
src = CKPT / "sunnycx_0.5b_pretrain_latest.pt"
dst = CKPT / "sunnycx_0.5b_pretrain_warm5600.pth"

ck = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
model = ck["model"]
print(f"[read] {src.name}  step={ck.get('step')}  epoch={ck.get('epoch')}  张量={len(model)}")
print(f"[read] early_stop={ck.get('early_stop')}")

# 已是 fp16 → .to() 返回自身，storage 共享得以保留
state = {k: v.to(torch.float16) for k, v in model.items()}
torch.save(state, dst)

uniq = sum(v.numel() for v in {v.untyped_storage().data_ptr(): v
                               for v in state.values()}.values())
print(f"[write] {dst.name}  {dst.stat().st_size / 1024**2:.2f} MB  独立参数={uniq:,}")
print(f"[check] 与目标 524,133,504 一致: {uniq == 524_133_504}")

# 回读验证共享
back = torch.load(dst, map_location="cpu", mmap=True, weights_only=True)
a = back["model.embed_tokens.weight"]
b = back["lm_head.weight"]
print(f"[verify] tied 共享保持: {a.data_ptr() == b.data_ptr()}  dtype={a.dtype}")
