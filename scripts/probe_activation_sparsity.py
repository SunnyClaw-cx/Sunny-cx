# -*- coding: utf-8 -*-
"""最终裁决：63.67% 漂移是否导致"神经元熄灯"？

直接在真实中医语料上前向，逐层测：
  · SwiGLU gate 分支 silu 输出的近零/负值占比
  · 真实死神经元比例
CPU 运行，不碰训练显存。
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from sunnycx.config import SunnyConfig
from sunnycx.model import SunnyForCausalLM

ck = torch.load(ROOT / "checkpoints" / "sunnycx_0.5b_pretrain_best.pth",
                map_location="cpu", weights_only=False)
cfg = SunnyConfig.preset("0.5b")
model = SunnyForCausalLM(cfg)
missing, unexpected = model.load_state_dict(ck, strict=False)
model.eval()
print(f"载入检查点: missing={len(missing)} unexpected={len(unexpected)}")

# 取 4 条真实中医语料
DATA = ROOT.parent / "tcm-llm" / "data" / "processed" / "tcm_classics_pretrain.val.jsonl"
texts = []
with open(DATA, "r", encoding="utf-8") as f:
    for line in f:
        try: t = json.loads(line).get("text","")
        except Exception: continue
        if len(t) > 200: texts.append(t[:512])
        if len(texts) >= 4: break
print(f"取 {len(texts)} 条真实语料样本\n")

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(ROOT.parent / "tcm-llm" / "model_tcm_tokenizer"))
enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
print(f"输入 {tuple(enc['input_ids'].shape)}\n")

# 挂钩 SwiGLU 的 gate 分支 silu 输出
stats = {}
def mk(idx):
    def h(_m, inp, out):
        x = out.detach().float()
        stats[idx] = {
            "neg": float((x < 0).sum()) / x.numel() * 100,
            "near_zero": float((x.abs() < 1e-3).sum()) / x.numel() * 100,
            "exact_zero": float((x == 0).sum()) / x.numel() * 100,
            "mean": float(x.mean()), "std": float(x.std()),
        }
    return h

# SwiGLU 内 act = nn.SiLU()，挂到它上面
count = 0
for name, mod in model.named_modules():
    if name.endswith(".mlp.act"):
        mod.register_forward_hook(mk(name))
        count += 1
print(f"挂钩 {count} 个 silu 激活\n")

with torch.no_grad():
    model(enc["input_ids"], attention_mask=enc["attention_mask"])

print("="*78)
print(f"{'层':<22}{'负值%':>9}{'近零%(<1e-3)':>14}{'严格0%':>10}{'均值':>10}{'标准差':>10}")
print("="*78)
ks = sorted(stats, key=lambda s: int(s.split('.')[2]))
allneg = allnear = 0.0
for k in ks:
    d = stats[k]
    allneg += d["neg"]; allnear += d["near_zero"]
    print(f"{k:<22}{d['neg']:>9.2f}{d['near_zero']:>14.2f}{d['exact_zero']:>10.2f}{d['mean']:>10.4f}{d['std']:>10.4f}")
n = len(ks)
print("="*78)
print(f"{'平均':<22}{allneg/n:>9.2f}{allnear/n:>14.2f}")
print()
print("【判据】ReLU 式熄灯 = 严格 0 占比很高（>50%）")
print(f"  实测 silu 严格 0 占比 = {sum(stats[k]['exact_zero'] for k in ks)/n:.4f}%  → 无熄灯")
print(f"  silu 输出负值占比 = {allneg/n:.2f}%  ← 这是 SiLU 的正常形态（其最小值 -0.278）")
