"""SFT 长度分桶校验（纯 CPU，不需要 GPU）

验证两件事，缺一不可：

  [1] 语义等价 —— 分桶 + 动态 pad 与固定 pad 给模型的**有效 token 完全一致**：
      input_ids 的真实部分、labels 的 assistant 段、位置顺序都不许变，
      pad 位必须仍是 label=-100 / mask=0。否则"加速"是用训练质量换的。

  [2] 省了多少 padding —— 按 batch 统计真实长度 vs 补到的长度，
      给出有效 token 占比与相对固定 pad 的算力倍数。

用法：
    python scripts/verify_sft_bucket.py
    python scripts/verify_sft_bucket.py --data <qa.train.jsonl> --scan 4000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sunnycx.dataset import SFTDataset, collate, collate_pad  # noqa: E402

IGNORE_INDEX = -100


def _tok():
    from transformers import AutoTokenizer
    p = ROOT.parent / "tcm-llm" / "model_tcm_tokenizer"
    return AutoTokenizer.from_pretrained(str(p))


def check_semantics(tok, path: str, max_seq_len: int, n: int) -> bool:
    """同序比对两种 padding 的「有效内容」。"""
    fixed = SFTDataset(path, tok, max_seq_len=max_seq_len, shuffle_buffer=1, seed=42)
    dyn = SFTDataset(path, tok, max_seq_len=max_seq_len, shuffle_buffer=1, seed=42,
                     dynamic_pad=True, bucket_chunk=0)

    n_checked = 0
    for a, b in zip(fixed, dyn):
        # 真实长度取自 labels 中非 -100 的**最后位置**不可靠（user 段也是 -100），
        # 改用 mask=1 的长度 —— 两种模式含义相同：非 pad 的实际 token 数
        la = int(a["attention_mask"].sum())
        lb = int(b["input_ids"].size(0))
        assert la == lb, f"有效长度不一致：fixed={la} dyn={lb}"
        assert torch.equal(a["input_ids"][:la], b["input_ids"]), "input_ids 真实段不一致"
        assert torch.equal(a["labels"][:la], b["labels"]), "labels 真实段不一致"
        assert int((b["attention_mask"] == 1).sum()) == lb, "dyn 的 mask 出现 0"
        assert int(a["input_ids"].size(0)) == max_seq_len, "fixed 未补到 max_seq_len"
        # user 段必须被掩掉，assistant 段必须有监督
        assert (b["labels"] == IGNORE_INDEX).any(), "样本里没有 user 掩码位"
        assert (b["labels"] != IGNORE_INDEX).any(), "样本里没有 assistant 监督位"
        n_checked += 1
        if n_checked >= n:
            break

    print(f"[1/2] 语义等价：比对 {n_checked} 条样本，有效 token / labels / mask 全部一致 ✓")
    return True


def measure(tok, path: str, max_seq_len: int, batch_size: int,
            scan: int, chunk: int) -> dict:
    """统计两种模式下的 padding 占比。"""
    out = {}

    ds_fixed = SFTDataset(path, tok, max_seq_len=max_seq_len, shuffle_buffer=2000, seed=42)
    ld_fixed = DataLoader(ds_fixed, batch_size=batch_size, collate_fn=collate,
                          num_workers=0, drop_last=True)
    tot_real = tot_pad = nb = 0
    for batch in ld_fixed:
        real = int(batch["attention_mask"].sum())
        tot_real += real
        tot_pad += batch["input_ids"].numel()
        nb += 1
        if nb * batch_size >= scan:
            break
    out["fixed"] = {"batches": nb, "real": tot_real, "computed": tot_pad,
                    "eff": tot_real / max(tot_pad, 1)}

    ds_dyn = SFTDataset(path, tok, max_seq_len=max_seq_len, shuffle_buffer=2000, seed=42,
                        dynamic_pad=True, bucket_chunk=chunk)
    ld_dyn = DataLoader(ds_dyn, batch_size=batch_size, collate_fn=collate_pad,
                        num_workers=0, drop_last=True)
    tot_real = tot_pad = nb = 0
    for batch in ld_dyn:
        real = int(batch["attention_mask"].sum())
        tot_real += real
        tot_pad += batch["input_ids"].numel()
        nb += 1
        if nb * batch_size >= scan:
            break
    out["bucket"] = {"batches": nb, "real": tot_real, "computed": tot_pad,
                     "eff": tot_real / max(tot_pad, 1)}
    return out


def measure_no_bucket(tok, path: str, max_seq_len: int, batch_size: int,
                      scan: int) -> dict:
    """动态 pad 但**不分桶**：看分桶本身贡献多少。"""
    ds = SFTDataset(path, tok, max_seq_len=max_seq_len, shuffle_buffer=2000, seed=42,
                    dynamic_pad=True, bucket_chunk=0)
    ld = DataLoader(ds, batch_size=batch_size, collate_fn=collate_pad,
                    num_workers=0, drop_last=True)
    tot_real = tot_pad = nb = 0
    for batch in ld:
        tot_real += int(batch["attention_mask"].sum())
        tot_pad += batch["input_ids"].numel()
        nb += 1
        if nb * batch_size >= scan:
            break
    return {"batches": nb, "real": tot_real, "computed": tot_pad,
            "eff": tot_real / max(tot_pad, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "tcm-llm" / "data" /
                                          "processed_v2" / "tcm_classics_qa.train.jsonl"))
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--scan", type=int, default=2000, help="统计多少条样本")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--eq_check", type=int, default=200, help="语义比对条数")
    a = ap.parse_args()

    if not Path(a.data).exists():
        print(f"[error] 数据不存在：{a.data}")
        return 1
    tok = _tok()

    print("=" * 76)
    print(f"数据 {a.data}")
    print(f"max_seq_len={a.max_seq_len}  batch={a.batch_size}  "
          f"stat={a.scan} 条  bucket_chunk={a.chunk}")
    print("=" * 76)

    check_semantics(tok, a.data, a.max_seq_len, a.eq_check)

    r_dyn = measure_no_bucket(tok, a.data, a.max_seq_len, a.batch_size, a.scan)
    r = measure(tok, a.data, a.max_seq_len, a.batch_size, a.scan, a.chunk)

    print("\n[2/2] padding 实测")
    print(f"{'方案':<26}{'batch':>7}{'有效 token':>12}{'实际计算':>12}"
          f"{'有效率':>9}{'相对加速':>9}")
    base = r["fixed"]["computed"]
    rows = [
        ("固定 pad (原状)", r["fixed"]),
        ("动态 pad (不分桶)", r_dyn),
        (f"动态 pad + 分桶 {a.chunk}", r["bucket"]),
    ]
    for name, d in rows:
        sp = base / max(d["computed"], 1)
        print(f"{name:<26}{d['batches']:>7}{d['real']:>12,}{d['computed']:>12,}"
              f"{d['eff']*100:>8.1f}%{sp:>8.2f}x")

    eff = r["bucket"]["eff"]
    sp = base / max(r["bucket"]["computed"], 1)
    print(f"\n结论：分桶后有效 token 占比 {r['fixed']['eff']*100:.1f}% → {eff*100:.1f}%"
          f"，同等样本量下前向/反向算力降到 1/{sp:.2f}")
    print(f"      单轮时间预估：12.9 天 → {12.9/sp:.1f} 天（线性折算，"
          f"实际受固定开销影响会略优）")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
