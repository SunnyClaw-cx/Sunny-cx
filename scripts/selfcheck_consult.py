# -*- coding: utf-8 -*-
"""白话问诊正确性自检（可量化 / 可回归）

三块，缺一不可：

A. **判定准确率**：golden_set.jsonl 上跑 top-1 / top-3 / 拒答正确率
B. **闸门有效性**：故意造 8 条"错答案"喂给 check_answer，必须全部被拦
   —— 不测这块，等于没装闸门（闸门自己坏了不会有人发现）
C. **真机可复算**：断言 35 证型全部可被查到方剂、每方组成非空、无剂量字样

退出码：0 全过；1 有闸门漏检或断言失败（准确率低只报告，不阻断——
准确率是模型/词表水平问题，闸门漏检是安全问题）。

用法：
  python scripts/selfcheck_consult.py
  python scripts/selfcheck_consult.py --verbose
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sunnycx.consult import (  # noqa: E402
    DISCLAIM, ConsultEngine, check_answer, parse_base,
)

GOLDEN = ROOT / "data" / "consult" / "golden_set.jsonl"
HOLDOUT = ROOT / "data" / "consult" / "holdout_set.jsonl"


def part_a(eng: ConsultEngine, verbose: bool, path: Path = GOLDEN, tag: str = "A") -> dict:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(rows)
    top1 = top3 = dec_ok = refuse_ok = misjudge = 0
    fails = []
    for r in rows:
        res = eng.consult(r["text"])
        cands = [c["syndrome"] for c in res["candidates"]]
        exp = r["expect"]
        if r["expect_decision"] == "refuse":
            ok = res["decision"] == "refuse"
            refuse_ok += ok
            if not ok:
                fails.append((r["id"], "该拒答却给了结论", res["decision"], cands[:1]))
        else:
            if cands and cands[0] in exp:
                top1 += 1
            if any(c in exp for c in cands[:3]):
                top3 += 1
            if r["expect_decision"] == "weak":
                ok = res["decision"] in ("weak", "refuse", "ok")
            else:
                ok = res["decision"] in ("ok", "differential")
            dec_ok += ok
            if res["decision"] == "ok" and cands and cands[0] not in exp:
                misjudge += 1
                fails.append((r["id"], "判错且给了结论", cands[0], exp))
            elif verbose:
                print(f"  {r['id']} {res['decision']:12s} {cands[:1]} 期望{exp}")
    return {"n": n, "top1": top1, "top3": top3, "dec_ok": dec_ok,
            "refuse_ok": refuse_ok, "misjudge": misjudge, "fails": fails}


def part_b(eng: ConsultEngine) -> list[str]:
    """故意造错答案。每条都必须被拦下。"""
    good = f"\n{DISCLAIM}"
    poison = {
        "编造方名": f"【辨证方向】气虚血瘀证\n【参考方剂】益气活血通脉汤（《自制方》）{good}",
        "编造药味": f"【参考药味】黄芪、当归尾、天山雪莲、龙涎香{good}",
        "编造证型": f"【辨证方向】肝郁脾虚湿阻证\n【参考方剂】逍遥散（《局方》）{good}",
        "给出剂量": f"【参考方剂】四妙丸\n【用法】苍术9克，黄柏12克，水煎服{good}",
        "绝对化断言": f"【辨证方向】湿热下注证，确诊为前列腺炎{good}",
        "缺免责声明": "【辨证方向】湿热下注证\n【参考方剂】四妙丸",
        # 真方但表里没收录 —— 比编造方名更危险，因为人看不出错
        "表外真方未收录": f"【参考方剂】当归四逆汤（《伤寒论》）{good}",
        "把治法当方名": f"【参考方剂】疏风散寒解表（无出处）{good}",
    }
    leaks = []
    for name, txt in poison.items():
        c = check_answer(txt, eng)
        if c["pass"]:
            leaks.append(name)
            print(f"  ✗ 漏检：{name}")
        else:
            print(f"  ✓ 拦截：{name:12s} → {c['violations'][0]}")
    return leaks


def part_c(eng: ConsultEngine) -> list[str]:
    bad = []
    for s in eng.syndromes:
        nm = s["syndrome"]
        pos, nat = parse_base(s["base"])
        if not pos or not nat:
            bad.append(f"{nm} base 解析失败: {s['base']!r}")
        if not s.get("formula"):
            bad.append(f"{nm} 无首选方")
        if not s.get("composition"):
            bad.append(f"{nm} 组成空")
        if not s.get("source"):
            bad.append(f"{nm} 无出处")
        for h in s["composition"]:
            if re.search(r"\d|克|钱|两|克数", h):
                bad.append(f"{nm} 组成里混入剂量: {h}")
    # 词表里的别名不能互相包含导致误判（长别名优先已处理，这里只查完全重复）
    terms = [e["term"] for e in eng.lex["symptoms"]]
    if len(terms) != len(set(terms)):
        bad.append("词表 term 有重复")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    eng = ConsultEngine.load()
    print(f"[load] 证型 {len(eng.syndromes)} 类 / 症状词条 {len(eng.lex['symptoms'])} 条"
          f" / 别名 {len(eng._alias)} 个 / 白名单方药 {len(eng.formulas)} 方 {len(eng.herbs)} 味")
    if eng.orphan_tokens:
        print(f"[warn] 证型表里 {len(eng.orphan_tokens)} 个 token 拿不到证据: {eng.orphan_tokens}")

    print("\n=== A 判定准确率（调参集，词表是照着它调的 → 只能当上限） ===")
    A = part_a(eng, a.verbose)
    print(f"  样本 {A['n']}   top-1 {A['top1']}/{A['n']} = {A['top1']/A['n']:.1%}"
          f"   top-3 {A['top3']}/{A['n']} = {A['top3']/A['n']:.1%}")
    print(f"  拒答正确 {A['refuse_ok']} 例   **判错且给结论 {A['misjudge']} 例**（安全指标，应尽量为 0）")
    for f in A["fails"]:
        print(f"  ✗ {f[0]} {f[1]}：得 {f[2]} / 期望 {f[3]}")

    print("\n=== A2 留出集（未参与调词，跑一次报一次，不许再调） ===")
    A2 = None
    if HOLDOUT.exists():
        A2 = part_a(eng, a.verbose, HOLDOUT, "A2")
        print(f"  样本 {A2['n']}   top-1 {A2['top1']}/{A2['n']} = {A2['top1']/A2['n']:.1%}"
              f"   top-3 {A2['top3']}/{A2['n']} = {A2['top3']/A2['n']:.1%}")
        print(f"  拒答正确 {A2['refuse_ok']} 例   判错且给结论 {A2['misjudge']} 例")
        for f in A2["fails"]:
            print(f"  ✗ {f[0]} {f[1]}：得 {f[2]} / 期望 {f[3]}")
        print("  ⚠ 留出集一旦被用来改词表就不再是留出集；要调请另写一份新的")
    else:
        print("  （缺 holdout_set.jsonl，跳过）")

    print("\n=== B 闸门有效性（每条都必须被拦） ===")
    leaks = part_b(eng)

    print("\n=== C 表内一致性断言 ===")
    bad = part_c(eng)
    if bad:
        for b in bad[:10]:
            print(f"  ✗ {b}")
    else:
        print("  ✓ 35 证型 base 可解析 / 方剂齐备 / 组成无剂量字样")

    print("\n=== 结论 ===")
    print(f"  闸门漏检 {len(leaks)} 项（必须 0）  表内异常 {len(bad)} 项（必须 0）")
    print(f"  调参集 判错且给结论 {A['misjudge']} 例"
          + (f" / 留出集 {A2['misjudge']} 例" if A2 else "")
          + " —— 这是「正确性」的直接反面指标，必须为 0")
    if leaks or bad:
        print("  ❌ 不通过")
        return 1
    print("  ✅ 闸门与表内一致性通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
