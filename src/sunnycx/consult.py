# -*- coding: utf-8 -*-
"""白话问诊 · 受控判定引擎（正确性优先，不做自由生成）

## 这个模块为什么长这样

0.5B 模型在古籍上预训练 + SFT，**会续写、不保证答对**。让它在自由生成里
「辨证 + 开方」，必然出现三类硬错误：

1. **证型乱判** —— 把「乏力 + 舌紫暗」判成阴虚火旺
2. **方名臆造** —— 生成「益气活血汤」这类不存在的方
3. **药味错配 / 剂量幻觉** —— 往方里加不该有的药，或编出「三钱」

所以本引擎把**医学结论从生成器手里拿走**：

```
白话主诉 → ① 规范症状（受控词表） → ② 病位/病性证据 → ③ 35 证型打分
        → ④ 方剂查表（只读 syndrome_table.json，绝不生成）
        → ⑤ 模板渲染白话（槽位全部来自 ③④） → ⑥ 越界校验 → ⑦ 输出/拒答
```

模型（若接入）**只允许改写 ⑥ 已通过校验的文本的措辞**，不得引入任何新的
证型名、方名、药味。这是「正确」在工程上唯一能兑现的形态。

## 可检查的「正确」定义

| 指标 | 要求 | 谁来保证 |
|---|---|---|
| 证型在 35 类内 | 100% | 打分器只在表内排序 |
| 方剂与证型一致 | 100% | 查表，不生成 |
| 药味属于该方 | 100% | 组成字段直出 |
| 不出现剂量 | 100% | 词表层面禁用 |
| 不出现绝对化断言 | 100% | `check_answer()` 拦截 |
| 证型 top-1 准确率 | 需评测集实测 | 见 scripts/selfcheck_consult.py |
| 信息不足时拒答 | 必须有 | 覆盖率阈值 |

## 用法

```python
from sunnycx.consult import ConsultEngine
eng = ConsultEngine.load()
r = eng.consult("我最近老觉得累，一动就出汗，舌头发紫，胸口时不时刺痛")
print(r["answer"])
```

命令行：
  python -m sunnycx.consult --text "..."        # 单条
  python -m sunnycx.consult --check answer.txt  # 只做越界校验
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]          # sunny-cx/
DEFAULT_LEXICON = ROOT / "data" / "consult" / "symptom_lexicon.json"
# 仓库内自带一份，clone 下来即可跑（数据自包含）
DEFAULT_TABLE = ROOT / "data" / "consult" / "syndrome_table.json"
# 开发机历史路径：仓库副本缺失时回退，仅为向后兼容
LEGACY_TABLE = ROOT.parent / "tcm-llm" / "data" / "syndrome_table.json"

# 病位 token 的标称权重低于病性：单说"胃"不能证明是脾胃不和
W_POS = 0.6
W_NAT = 1.0
TOKEN_CAP = 1.2        # 同一 token 被多条症状命中时封顶，避免"反复说累"刷分

# 判定阈值（宁可拒答，不可错答）
RATIO_OK = 0.75        # 覆盖率达标才给结论
RATIO_WEAK = 0.50      # 仅够给"方向"
TIE_EPS = 0.04         # 前二差距小于此值 → 视为鉴别不清，不给单一结论

# 输出层合规黑名单
DOSE_PAT = re.compile(
    r"\d+(\.\d+)?\s*(g|G|克|钱|两|分|毫升|ml|mL|mg|毫克|片|粒|剂|包)\b"
    r"|[一二三四五六七八九十]钱|[一二三四五六七八九十]两|适量|少许")
ABS_PAT = re.compile(
    r"一定是|肯定是|确诊|包治|百分之百|100%|保证治好|根治|绝对|必然|必定")
FORMULA_PAT = re.compile(r"([\u4e00-\u9fa5]{2,10}(?:汤|丸|散|饮|煎|丹))")
SYN_PAT = re.compile(r"([\u4e00-\u9fa5]{2,8}证)")
# 这些词里的"证"不是证型名
SYN_STOP = ("建议", "保证", "验证", "辨证", "论证", "证实", "证据", "症证",
            "属", "是", "为", "即", "乃", "凡", "此")
LABEL_PAT = re.compile(r"【[^】]*】")
FORMULA_TAIL = "汤丸散饮煎丹"


def _runs(text: str):
    """把文本切成"可能承载实体名的中文片段"。

    先剥掉【标签】（否则「参考方剂」会被当方名的一部分），
    再按标点/数字切成中文长串，最后按「合/与/和/加」拆开复合方名。
    """
    for line in text.splitlines():
        line = LABEL_PAT.sub(" ", line)
        for run in re.findall(r"[\u4e00-\u9fa5]+", line):
            for part in re.split(r"[合与和加]", run):
                if len(part) >= 2:
                    yield part


def formula_mentions(text: str) -> list[str]:
    """抽方名候选。必须**以汤/丸/散/饮/煎/丹结尾**——
    「疏风散寒」这种治法不能算方名，否则会把正确输出拦下来。"""
    return [p for p in _runs(text) if p[-1] in FORMULA_TAIL]


def syndrome_mentions(text: str) -> list[str]:
    return [p for p in _runs(text)
            if p.endswith("证") and 3 <= len(p) <= 8
            and not any(k in p for k in SYN_STOP)]
DISCLAIM = "以上仅为中医知识性说明，不构成诊疗建议；具体请面诊执业中医师。"


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def parse_base(base: str) -> tuple[list[str], list[str]]:
    """ '全身·气虚+血瘀' → (['全身'], ['气虚','血瘀']) """
    if not base:
        return [], []
    left, _, right = base.partition("·")
    pos = [left] if left else []
    nat = [t for t in re.split(r"[+＋]", right) if t]
    return pos, nat


class ConsultEngine:
    def __init__(self, lexicon: dict, syndromes: list[dict]):
        self.lex = lexicon
        self.syndromes = syndromes
        self.axes = lexicon["axes"]
        # 白话别名 → 规范条目（长别名优先，避免"怕冷"被"冷"抢占）
        self._alias: list[tuple[str, dict]] = []
        for e in lexicon["symptoms"]:
            for a in [e["term"]] + list(e.get("alias", [])):
                if a:
                    self._alias.append((a, e))
        self._alias.sort(key=lambda x: -len(x[0]))
        self._verify_consistency()
        self._build_index()

    # ---------- 构造 ----------
    @classmethod
    def load(cls, lexicon_path=DEFAULT_LEXICON, table_path=None) -> "ConsultEngine":
        lex = json.loads(Path(lexicon_path).read_text(encoding="utf-8"))
        tp = Path(table_path) if table_path else DEFAULT_TABLE
        if not tp.exists():
            tp = LEGACY_TABLE
        tab = json.loads(tp.read_text(encoding="utf-8"))
        return cls(lex, tab["syndromes"])

    def _verify_consistency(self) -> None:
        """词表 vs 证型表 的轴 token 必须完全对齐，否则静默错配。"""
        known = {("病位", t) for t in self.axes["病位"]} | {("病性", t) for t in self.axes["病性"]}
        bad = set()
        for e in self.lex["symptoms"]:
            for ax, tok in e["axes"]:
                if (ax, tok) not in known:
                    bad.add(f"{e['term']}:{ax}.{tok}")
        if bad:
            raise SystemExit(f"[fatal] 词表里有词表外 token（会静默失效）: {sorted(bad)[:8]}")

        # 证型表的 token 必须都在词表轴里，否则该类永远拿不到证据
        orphan = []
        for s in self.syndromes:
            pos, nat = parse_base(s["base"])
            for t in pos:
                if ("病位", t) not in known:
                    orphan.append(f"{s['syndrome']}:病位.{t}")
            for t in nat:
                if ("病性", t) not in known:
                    orphan.append(f"{s['syndrome']}:病性.{t}")
        self.orphan_tokens = sorted(set(orphan))

    def _build_index(self) -> None:
        """结论位实体白名单 + 编者注里合法引用的表外实体。"""
        self.syn_names = {s["syndrome"] for s in self.syndromes}
        self.formulas = set()
        self.herbs = set()
        for s in self.syndromes:
            for nm in [s["formula"]] + list(s.get("alt") or []):
                # alt 里常见「杞菊地黄丸（阴虚为本者）」这种带注的写法，
                # 白名单必须同时收"原文"和"去掉括号注的名字"，否则自己的正确答案会被自己拦下
                self.formulas.add(nm)
                clean = re.sub(r"[（(].*?[)）]", "", nm).strip()
                self.formulas.add(clean)
                # 合方「四君子汤合桃红四物汤」在文本里会被拆成两个，
                # 单方也必须进白名单，否则合方永远过不了校验
                for part in re.split(r"[合与和加]", clean):
                    if len(part) >= 2:
                        self.formulas.add(part)
            self.herbs.update(s["composition"])
        # 证型表的 note 是人写的编者注，里面会提到表外方名（如「湿重用三仁汤」）。
        # 这些属于"有据引用"，不算越界，但必须当 warning 暴露出来——
        # 既不误伤，也不让它们混进"已核验"的结论位。
        self.note_extra_formula, self.note_extra_syn = set(), set()
        for s in self.syndromes:
            note = s.get("note") or ""
            for cand in formula_mentions(note):
                if cand not in self.formulas:
                    self.note_extra_formula.add(cand)
            for cand in syndrome_mentions(note):
                if cand not in self.syn_names:
                    self.note_extra_syn.add(cand)

    # ---------- ① 规范化 ----------
    def normalize(self, text: str) -> list[dict]:
        t = _norm(text)
        hits, used = [], []
        for alias, e in self._alias:
            if alias in t:
                span = t.find(alias)
                if any(span < b and span + len(alias) > a for a, b in used):
                    continue                      # 已被更长的别名覆盖
                used.append((span, span + len(alias)))
                if e["term"] not in [h["term"] for h in hits]:
                    hits.append({"term": e["term"], "surface": alias,
                                 "axes": e["axes"], "w": e["w"]})
        return hits

    # ---------- ②③ 打分 ----------
    def score(self, hits: list[dict]) -> list[dict]:
        # token → 累计命中权重
        ev: dict[tuple[str, str], float] = {}
        for h in hits:
            for ax, tok in h["axes"]:
                ev[(ax, tok)] = ev.get((ax, tok), 0.0) + h["w"]

        out = []
        for s in self.syndromes:
            pos, nat = parse_base(s["base"])
            # 「全身」不是可定位的病位，主诉说不出"我全身"，把它当约束会把
            # 全身性证型（气虚血瘀、阴虚火旺…）永远压在 100% 以下
            pos = [t for t in pos if t != "全身"]
            tokens = [("病位", t) for t in pos] + [("病性", t) for t in nat]
            nominal = sum(W_POS if ax == "病位" else W_NAT for ax, _ in tokens)
            got, matched, missing = 0.0, [], []
            for ax, tok in tokens:
                cap = W_POS if ax == "病位" else W_NAT
                if ev.get((ax, tok), 0.0) > 0:
                    # 单 token 的贡献不得超其标称权重，否则覆盖率会 >100%、
                    # 反复描述同一个症状就能刷到第一
                    got += min(cap, TOKEN_CAP, ev[(ax, tok)])
                    matched.append(tok)
                else:
                    missing.append(tok)
            ratio = min(1.0, got / nominal) if nominal else 0.0
            n_nat_hit = sum(1 for t in matched if t in nat)
            # 只靠病位命中没有辨证意义
            score = ratio * (1 + 0.15 * len(matched)) if n_nat_hit else 0.0
            out.append({
                "syndrome": s["syndrome"], "base": s["base"],
                "ratio": round(ratio, 3), "score": round(score, 3),
                "matched": matched, "missing": missing,
                "n_nat_hit": n_nat_hit,
                "_row": s,
            })
        out.sort(key=lambda x: (-x["score"], x["syndrome"]))
        return out

    # ---------- ④ 查表 ----------
    @staticmethod
    def fetch_formula(row: dict) -> dict:
        """方剂只从证型表取，不做任何生成/改写。"""
        return {"formula": row["formula"], "composition": list(row["composition"]),
                "source": row["source"], "alt": list(row.get("alt") or []),
                "zhi_fa": row["zhi_fa"], "note": row.get("note", "")}

    # ---------- ⑤⑥ 渲染 + 自检 ----------
    def consult(self, text: str, topk: int = 3) -> dict:
        hits = self.normalize(text)
        ranked = self.score(hits)
        best = ranked[0] if ranked else None
        second = ranked[1] if len(ranked) > 1 else None

        if not hits:
            decision = "refuse"
            reason = "没有识别到任何可辨证的症状线索"
            # 一个症状都没识别到时，全表 35 个证型的分数都是 0。
            # 若照常返回 ranked[:topk]，接口就会输出「心脉痹阻证 0%」这类
            # 纯字母序候选——读者会误以为它被判定过。宁可为空。
            ranked = []
        elif best is None or best["score"] <= 0.0:
            # 全零时 ranked[0] 只是字母序第一，展示它等于凭空造了一个假候选
            decision = "refuse"
            reason = "没有任何证型获得有效证据（症状未命中任何病位/病性要素）"
            ranked = []
        elif best["ratio"] < RATIO_WEAK or best["n_nat_hit"] == 0:
            decision = "refuse"
            reason = "症状线索不足以支撑任何证型判定"
        elif best["ratio"] < RATIO_OK:
            decision = "weak"
            reason = f"仅覆盖 {best['ratio']:.0%} 的辨证要素，只能给方向，不能给结论"
        elif second and (best["score"] - second["score"]) < TIE_EPS:
            decision = "differential"
            reason = f"与「{second['syndrome']}」证据接近，需鉴别，不给单一结论"
        else:
            decision = "ok"
            reason = f"覆盖 {best['ratio']:.0%} 辨证要素，证据充分"

        res = {"input": text, "symptoms": hits,
               "candidates": [{k: v for k, v in c.items() if k != "_row"} for c in ranked[:topk]],
               "decision": decision, "reason": reason}

        if decision in ("ok", "differential"):
            row = best["_row"]
            res["syndrome"] = best["syndrome"]
            res["formula"] = self.fetch_formula(row)
            if decision == "differential":
                res["differential_with"] = second["syndrome"]
        elif decision == "weak":
            res["syndrome_hint"] = best["syndrome"]
            res["missing"] = best["missing"]

        res["answer"] = self.render(res)
        chk = check_answer(res["answer"], self)
        res["selfcheck"] = chk
        if not chk["pass"]:
            # 渲染层出问题就不许输出：退回安全模板
            res["answer"] = self.render_refusal(res, f"内部校验未通过：{chk['violations'][:2]}")
            res["decision"] = "refuse"
        return res

    def render(self, r: dict) -> str:
        L = []
        heard = "、".join(h["term"] for h in r["symptoms"]) or "（无）"
        L.append(f"【我理解到的不适】{heard}")
        L.append("")

        if r["decision"] == "ok":
            f = r["formula"]
            L.append(f"【辨证方向】{r['syndrome']}（依据：{'、'.join(r['candidates'][0]['matched'])}）")
            L.append(f"【治法】{f['zhi_fa']}")
            L.append(f"【参考方剂】{f['formula']}（{f['source']}）")
            L.append(f"【参考药味】{'、'.join(f['composition'])}")
            if f["alt"]:
                L.append(f"【备选方】{'、'.join(f['alt'])}")
            if f["note"]:
                L.append(f"【用药要点·编者注】{f['note']}")
            L.append("【关于剂量】按约定不给出具体克数。中药剂量须由医师依体质、"
                     "年龄、兼证加减，照搬网络剂量有风险。")
        elif r["decision"] == "differential":
            f = r["formula"]
            L.append(f"【辨证方向】倾向 {r['syndrome']}，但需与 {r.get('differential_with','')} 鉴别。")
            L.append(f"【治法】{f['zhi_fa']}")
            L.append(f"【参考方剂】{f['formula']}（{f['source']}）")
            L.append(f"【参考药味】{'、'.join(f['composition'])}")
            L.append("【重要】两者治法不同，请勿自行选方，需医师面诊后定夺。")
        elif r["decision"] == "weak":
            L.append(f"【目前只能给方向】接近 {r.get('syndrome_hint','')}，"
                     f"但还缺 {('、'.join(r.get('missing') or [])) or '关键信息'}。")
            L.append("【建议补充】请说明起病多久、什么情况下加重、饮食睡眠二便情况、舌苔颜色厚薄。")
        else:
            L.append("【无法辨证】目前的信息不足以做出证型判断。")
            L.append("【建议补充】请描述：哪里不舒服、持续多久、什么情况下加重或缓解、"
                     "饮食睡眠二便、舌苔颜色与厚薄、是否怕冷怕热。")

        L.append("")
        L.append(DISCLAIM)
        return "\n".join(L)

    def render_refusal(self, r: dict, why: str) -> str:
        return ("【无法给出结论】本次回答未通过内置安全校验："
                f"{why}。\n为避免给出错误的中医判断，此处不作辨证与方剂建议。\n\n{DISCLAIM}")


def check_answer(text: str, eng: ConsultEngine | None = None) -> dict:
    """对任意文本做越界校验。返回 {pass, violations[], warnings[], stats}。

    可对**模型生成的文本**调用，是"白话答案正确性"的最终闸门。
    violations 非空 → 不许输出；warnings 非空 → 可以输出，但必须如实告知。
    """
    v: list[str] = []
    w: list[str] = []
    if eng is not None:
        # R1 证型名越界 / R2 方名越界
        # 编者注里的表外方名（如「湿重用三仁汤」）只允许出现在注段内，
        # 挪到别处一律算越界——否则模型可以把编造的方名塞进注段蒙混过关
        for line in text.splitlines():
            in_note = line.lstrip().startswith("【用药要点")
            for cand in syndrome_mentions(line):
                if cand in eng.syn_names:
                    continue
                if in_note and cand in eng.note_extra_syn:
                    w.append("编者注引用的表外证型：" + cand)
                else:
                    v.append("证型名不在 35 类内：" + cand)
            for cand in formula_mentions(line):
                if cand in eng.formulas:
                    continue
                if in_note and cand in eng.note_extra_formula:
                    w.append("编者注引用的表外方名：" + cand)
                else:
                    v.append("方名不在表内：" + cand)
        # R7 正向断言：结论位必须至少含一个可核验方名
        # （黑名单挡不住「把治法写进【参考方剂】」这类，必须正向查）
        for seg in re.findall(r"【(?:参考方剂|备选方)】([^\n]*)", text):
            if not any(f in eng.formulas for f in formula_mentions(seg)):
                v.append(f"结论位无可核验方名：{seg.strip()[:20]!r}")
        # R3 药味越界：只看【参考药味】段，避免误伤通用词
        for seg in re.findall(r"【参考药味】([^\n]+)", text):
            for h in re.split(r"[、,，]", seg):
                h = h.strip()
                if h and h not in eng.herbs:
                    v.append(f"药味不在任何方剂组成内：{h}")
    # R4 剂量
    m = DOSE_PAT.search(text)
    if m:
        v.append(f"出现剂量表述：{m.group(0)!r}")
    # R5 绝对化
    m = ABS_PAT.search(text)
    if m:
        v.append(f"出现绝对化断言：{m.group(0)!r}")
    # R6 免责声明
    if DISCLAIM not in text:
        v.append("缺少免责声明")
    return {"pass": not v, "violations": v, "warnings": w,
            "stats": {"chars": len(text)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="白话主诉，直接判定")
    ap.add_argument("--check", help="对文件内容只做越界校验")
    ap.add_argument("--json", action="store_true", help="输出完整 JSON")
    a = ap.parse_args()
    eng = ConsultEngine.load()
    if eng.orphan_tokens:
        print(f"[warn] 证型表里有 {len(eng.orphan_tokens)} 个 token 不在词表轴内"
              f"（该类永远拿不到证据）: {eng.orphan_tokens[:6]}")

    if a.check:
        txt = Path(a.check).read_text(encoding="utf-8")
        c = check_answer(txt, eng)
        print(json.dumps(c, ensure_ascii=False, indent=1))
        return 0 if c["pass"] else 1

    if not a.text:
        ap.error("需要 --text 或 --check")
    r = eng.consult(a.text)
    if a.json:
        print(json.dumps({k: v for k, v in r.items() if k != "symptoms"}, ensure_ascii=False, indent=1))
    else:
        print(r["answer"])
        print(f"\n--- decision={r['decision']}  {r['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
