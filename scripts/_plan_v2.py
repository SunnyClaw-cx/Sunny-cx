# -*- coding: utf-8 -*-
"""规划 v2 语料跑满一轮的真实步数与耗时（只读，不训练）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoTokenizer

from train import _count_lines, _count_tokens

ROOT = Path(__file__).resolve().parent.parent
TOK = ROOT.parent / "tcm-llm" / "model_tcm_tokenizer"
V2 = ROOT.parent / "tcm-llm" / "data" / "processed_v2"

tok = AutoTokenizer.from_pretrained(str(TOK))
print(f"[tokenizer] vocab={tok.vocab_size}")

for label, name in [("v2 train", "tcm_classics_pretrain.train.jsonl"),
                    ("v2 val  ", "tcm_classics_pretrain.val.jsonl")]:
    p = V2 / name
    n = _count_lines(str(p))
    t = _count_tokens(str(p), tok)
    print(f"[data] {label}  行数 {n:>9,}  token {t/1e8:.4f} 亿  "
          f"平均 {t/n:.1f} tok/条")

train_p = V2 / "tcm_classics_pretrain.train.jsonl"
total_tokens = _count_tokens(str(train_p), tok)

# 官方打包配置：batch2 × seq512 × accum8 = 8192 tok/step
per_step = 2 * 512 * 8
steps = max(1, total_tokens // per_step)

print()
print("=" * 62)
print(f"打包后：每 optimizer step 消耗 {per_step:,} token")
print(f"1 轮步数 = {total_tokens:,} / {per_step:,} = {steps:,} step")
print("=" * 62)

# 实测口径
S_STEP = 16.5           # 实测 15~17 s/step，取中位
VAL_EVERY = 400
VAL_SAMPLES = 400
VAL_SEC = 425           # 400 条实测外推（64 条 ≈ 68s，线性放大 6.25 倍）
n_val = steps // VAL_EVERY

t_train = steps * S_STEP
t_val = n_val * VAL_SEC
t_raw = t_train + t_val
t_rest = t_raw * 1.25   # 每 2h 训 → 休 30min = ×1.25

print(f"纯训练     {steps:,} × {S_STEP}s      = {t_train/3600:6.1f} h")
print(f"验证开销   {n_val} 次 × {VAL_SEC}s       = {t_val/3600:6.1f} h")
print(f"小计       (不含休息)            = {t_raw/3600:6.1f} h = {t_raw/86400:.2f} 天")
print(f"含节律休息 (×1.25)              = {t_rest/3600:6.1f} h = {t_rest/86400:.2f} 天")
print()
print(f"SFT 侧参考：stream 模式步数 = 行数 /(batch2×accum8)")
qa = V2 / "tcm_classics_qa.train.jsonl"
nq = _count_lines(str(qa))
print(f"  QA train 行数 {nq:,} → {nq // 16:,} step（不打包，padding 浪费回到 80% 量级）")
