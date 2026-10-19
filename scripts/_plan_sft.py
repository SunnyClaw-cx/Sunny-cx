# -*- coding: utf-8 -*-
"""SFT 阶段真实成本评估（只读）。

预训练用序列打包，SFT 不能打包（要保持对话结构），所以每条样本都 pad 到
固定 max_seq_len —— 这是 SFT 慢的根因。本脚本量化：
  1. 真实长度分布（P50/P90/P99、截断率）
  2. 固定 pad 的算力浪费比例
  3. 按长度分桶（length bucketing）能省多少
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
QA = ROOT.parent / "tcm-llm" / "data" / "processed_v2" / "tcm_classics_qa.train.jsonl"
tok = AutoTokenizer.from_pretrained(str(ROOT.parent / "tcm-llm" / "model_tcm_tokenizer"))

N = 3000
random.seed(7)
# 蓄水池抽样，避免只读文件头
reservoir = []
n_seen = 0
with open(QA, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        n_seen += 1
        if len(reservoir) < N:
            reservoir.append(line)
        else:
            j = random.randint(0, n_seen - 1)
            if j < N:
                reservoir[j] = line

print(f"[sample] 文件总行 {n_seen:,}，蓄水池抽样 {len(reservoir)}")

lens = []
n_turn = []
n_trunc = {256: 0, 384: 0, 512: 0, 768: 0}
for line in reservoir:
    obj = json.loads(line)
    convs = obj.get("conversations", [])
    n_turn.append(len(convs))
    total = 0
    for t in convs:
        c = t.get("content", "")
        if c:
            total += len(tok.encode(c, add_special_tokens=False))
    total += 1  # eos
    lens.append(total)
    for cap in n_trunc:
        if total > cap - 1:
            n_trunc[cap] += 1

lens.sort()
n = len(lens)


def pct(p):
    return lens[min(n - 1, int(n * p))]


print(f"\n=== 样本 token 长度分布（{n} 条）===")
print(f"  平均 {sum(lens)/n:.1f}   P50 {pct(.5)}   P90 {pct(.9)}   "
      f"P95 {pct(.95)}   P99 {pct(.99)}   最大 {lens[-1]}")
print(f"  轮次：平均 {sum(n_turn)/n:.1f} 轮/条")

print(f"\n=== 固定 pad 到 max_seq_len 的浪费 ===")
for cap in (256, 384, 512, 768):
    used = sum(min(L, cap) for L in lens)
    total = cap * n
    print(f"  cap={cap:>3}  有效 {used/total*100:5.1f}%  "
          f"填充 {100-used/total*100:5.1f}%  截断 {n_trunc[cap]/n*100:5.1f}%")

# 长度分桶：按 32 的倍数向上取整，桶内 batch 只 pad 到桶上界
print(f"\n=== 按长度分桶（向上取整到 32 倍数）===")
bucketed = [((L + 31) // 32) * 32 for L in lens]
print(f"  分桶后平均计算长度 {sum(bucketed)/n:.1f}  "
      f"vs 固定 512 浪费 {100 - sum(bucketed)/(512*n)*100:.1f}%")
speed = 512 * n / sum(bucketed)
print(f"  相对固定 512 的加速比：{speed:.2f}x")

STEPS_FIXED = 1081549 // 16
print(f"\n=== 时间估算（实测 16.5 s/optimizer step）===")
print(f"  固定 pad 512：{STEPS_FIXED:,} step × 16.5s = {STEPS_FIXED*16.5/3600/24:.1f} 天")
print(f"  分桶 pad    ：约 {STEPS_FIXED*16.5/3600/24/speed:.1f} 天（同一步数，仅省算力）")
print(f"  ⚠ 分桶需改 collate 为动态 pad，属代码改动，未实施")
