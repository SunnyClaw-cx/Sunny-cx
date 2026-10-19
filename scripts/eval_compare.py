"""早停可信度核验：在「大样本 + 官方 64 条」两种口径上对比 best 与 latest。

动机：训练时 --val_samples 64，而 ValSubset 每 2000 行取 1 条，val 文件仅 53,705 行
→ 只抽到 27 条，余下 37 条由「顺序补齐」从文件头取 → 验证集 60% 是文件开头。
需验证：best@4600 是否真的优于 latest@5600，还是早停踩了偏样本。

用法：python -u scripts/eval_compare.py [--n_big 1200]
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import List

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sunnycx.config import SunnyConfig  # noqa: E402
from sunnycx.model import SunnyForCausalLM  # noqa: E402

IGNORE_INDEX = -100
MAX_SEQ = 512
VAL_PATH = ROOT.parent / "tcm-llm" / "data" / "processed" / "tcm_classics_pretrain.val.jsonl"
TOK_PATH = ROOT.parent / "tcm-llm" / "model_tcm_tokenizer"


def load_texts(path: Path) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue
            if t:
                out.append(t)
    return out


def build_sets(texts: List[str], n_big: int):
    total = len(texts)
    # ① 均匀抽样：跨整个文件，代表全量验证分布
    step = max(1, total // n_big)
    big = texts[::step][:n_big]
    # ② 官方口径复现：每 2000 条取 1（≤64），不足则从文件头顺序补齐
    official = texts[::2000][:64]
    if len(official) < 64:
        need = 64 - len(official)
        official = official + texts[:need]
    return big, official


def encode(tok, text: str):
    ids = tok.encode(text, add_special_tokens=False)[: MAX_SEQ - 1]
    eos = tok.eos_token_id if tok.eos_token_id is not None else 2
    ids = ids + [eos]
    pad = MAX_SEQ - len(ids)
    return (
        ids + [0] * pad,
        ids + [IGNORE_INDEX] * pad,
        [1] * len(ids) + [0] * pad,
    )


@torch.no_grad()
def eval_set(model, tok, texts: List[str], device, dtype, bs: int, tag: str) -> tuple:
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    t0 = time.time()
    for i in range(0, len(texts), bs):
        chunk = texts[i: i + bs]
        rows = [encode(tok, t) for t in chunk]
        ids = torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)
        lab = torch.tensor([r[1] for r in rows], dtype=torch.long, device=device)
        msk = torch.tensor([r[2] for r in rows], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            out = model(ids, attention_mask=msk, labels=lab)
        n = int((lab != IGNORE_INDEX).sum().item())
        tot_loss += float(out["loss"].item()) * n
        tot_tok += n
        if (i // bs) % 20 == 0:
            print(f"    [{tag}] {i + len(chunk)}/{len(texts)} 累计 {time.time() - t0:.0f}s", flush=True)
    return tot_loss / max(tot_tok, 1), tot_tok


def load_model(ckpt: Path, cfg, device):
    model = SunnyForCausalLM(cfg)
    ck = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    del sd, ck
    gc.collect()
    dtype = torch.float16
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, dtype


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_big", type=int, default=1200)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] {device}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOK_PATH))
    print(f"[tok] vocab={tok.vocab_size}", flush=True)

    texts = load_texts(VAL_PATH)
    big, official = build_sets(texts, args.n_big)
    print(f"[data] val 全量 {len(texts)} 条 → 大样本 {len(big)} / 官方口径 {len(official)}", flush=True)

    cfg = SunnyConfig.preset("0.5b")
    ckpts = [
        ("best@4600", ROOT / "checkpoints" / "sunnycx_0.5b_pretrain_best.pth"),
        ("latest@5600", ROOT / "checkpoints" / "sunnycx_0.5b_pretrain_latest.pt"),
    ]

    results = {}
    for name, p in ckpts:
        if not p.exists():
            print(f"[skip] {p} 不存在", flush=True)
            continue
        print(f"\n===== {name}  ({p.stat().st_size / 1024**3:.2f} GB) =====", flush=True)
        t0 = time.time()
        model, dtype = load_model(p, cfg, device)
        print(f"  [load] {time.time() - t0:.1f}s", flush=True)
        l_big, n_big = eval_set(model, tok, big, device, dtype, args.bs, "big")
        l_off, n_off = eval_set(model, tok, official, device, dtype, args.bs, "official")
        results[name] = dict(big=l_big, big_tok=n_big, official=l_off, official_tok=n_off)
        print(f"  ▶ {name}: 大样本 loss={l_big:.4f}（{n_big} tok） | 官方64条 loss={l_off:.4f}（{n_off} tok）", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 64)
    print("口径            大样本(val全分布)    官方64条(偏文件头)")
    for k, v in results.items():
        print(f"{k:<14}  {v['big']:.4f}            {v['official']:.4f}")
    if len(results) == 2:
        a, b = list(results.values())
        d_big = b["big"] - a["big"]
        d_off = b["official"] - a["official"]
        print("-" * 64)
        print(f"latest - best:  大样本 {d_big:+.4f}   官方口径 {d_off:+.4f}")
        print("判读：大样本为正 → best 确实更优（早停成立）；为负 → 早停误触发")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
