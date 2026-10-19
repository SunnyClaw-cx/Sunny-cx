# -*- coding: utf-8 -*-
"""TMC-Tongue → 多标签分类清单（按真实 json 自动建表）

TMC-Tongue（Gao Longfei, Dryad 10.5061/dryad.1c59zw48r，CC0）原始标注是
**目标检测框**（COCO），一张舌图常带 1~5 个标签（既有舌色、又有苔色、还有脏腑区），
本质是**多标签分类**而非单标签。

## 为什么不再写死类别表（2026-10-18 实测修正）

先前版本按 README 猜的类别名，与真实 json 里 `categories[].name` 有 5 处不一致
（`shoushe` 不是 `shouxiaoshe`；`shenquao/gandanao/piweiao/xinfeiao` 是"凹"不是"凹陷"）。
本版**一律从 json 的 categories 读**，类名不匹配就报警而不是静默错位。

## 四类"凸起"必须从输出层删掉（全量标注计数为证）

| 类 | 全量标注 | 处置 |
|---|---|---|
| shenqutu 肾区凸起 | 2 | 删 |
| gandantu 肝胆区凸起 | 9 | 删 |
| piweitu 脾胃区凸起 | 0 | 删 |
| xinfeitu 心肺区凸起 | 2 | 删 |

合计 13 条标注 / 6,719 张图 = 0.19‰，任何模型都学不出 → 留在输出层只会污染 micro-F1。
删后 **17 类**（16 病理特征 + 1 健康舌）。

## test 集有 7 条越界标注

`category_id=21` 而 categories 只有 0~20（均在 image_id 423）→ 显式计数并丢弃，不静默。

## 输出

  manifests/tmc_tongue_{split}.jsonl   每行 {path, labels:[17 维 0/1], boxes, names}
  manifests/tmc_tongue_stats.json      类别分布 / 每图标签数 / 删除与越界计数

用法：
  python prepare_tmc_tongue.py --root <解压目录>/tmc_tongue
  python prepare_tmc_tongue.py --root ... --check     # 只体检不写文件
  python prepare_tmc_tongue.py --root ... --keep_empty  # 保留无标签图（默认丢弃）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# 保留的 17 类：官方 id 顺序，剔除四个"凸起"类
# (official_name, 中文, 属性维度)
KEEP = [
    ("jiankangshe", "未见异常", "base"),
    ("botaishe",    "剥苔",     "coating"),
    ("hongshe",     "舌红",     "color"),
    ("zishe",       "舌紫",     "color"),
    ("pangdashe",   "舌体胖大", "shape"),
    ("shoushe",     "舌体瘦小", "shape"),
    ("hongdianshe", "舌有红点", "shape"),
    ("liewenshe",   "舌有裂纹", "shape"),
    ("chihenshe",   "舌边齿痕", "shape"),
    ("baitaishe",   "苔白",     "coating"),
    ("huangtaishe", "苔黄",     "coating"),
    ("heitaishe",   "苔黑",     "coating"),
    ("huataishe",   "苔滑",     "coating"),
    ("shenquao",    "肾区凹陷", "zone"),
    ("gandanao",    "肝胆区凹陷", "zone"),
    ("piweiao",     "脾胃区凹陷", "zone"),
    ("xinfeiao",    "心肺区凹陷", "zone"),
]
KEYS = [k for k, _, _ in KEEP]
ZH = {k: z for k, z, _ in KEEP}
SLOT = {k: i for i, k in enumerate(KEYS)}
N_CLASS = len(KEEP)

# 标注数过少、无法训练 → 直接移出输出层
DROP = {
    "shenqutu": "肾区凸起",
    "gandantu": "肝胆区凸起",
    "piweitu":  "脾胃区凸起",
    "xinfeitu": "心肺区凸起",
}


def find_coco(root: Path) -> dict:
    """在 root 下找出每个 split 的 COCO json。"""
    found = defaultdict(list)
    for p in root.rglob("*.json"):
        low = p.name.lower()
        if "instance" not in low and "coco" not in low and low not in (
                "train.json", "val.json", "test.json"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not (isinstance(d, dict) and "images" in d and "annotations" in d):
            continue
        rel = str(p.parent).lower()
        split = None
        for k in ("train", "valid", "val", "test"):
            if k in rel:
                split = "train" if k == "train" else ("valid" if k in ("valid", "val") else "test")
                break
        found[split or "unknown"].append(p)
    return dict(found)


def load_split(coco_path: Path) -> dict:
    d = json.loads(coco_path.read_text(encoding="utf-8"))
    id2name = {c["id"]: c.get("name", "") for c in d.get("categories", [])}

    unknown_names = sorted({n for n in id2name.values()
                            if n not in SLOT and n not in DROP})
    drop_cnt = Counter()
    oob_cnt = Counter()
    img2labels = defaultdict(set)
    img2nbox = Counter()

    for ann in d["annotations"]:
        iid = ann["image_id"]
        img2nbox[iid] += 1
        cid = ann["category_id"]
        name = id2name.get(cid)
        if name is None:                      # category_id 超出 categories 范围
            oob_cnt[cid] += 1
            continue
        if name in DROP:
            drop_cnt[name] += 1
            continue
        if name not in SLOT:                  # 未预期类名：不猜，记为 unknown
            oob_cnt[f"name:{name}"] += 1
            continue
        img2labels[iid].add(name)

    rows, missing, empty = [], 0, 0
    for im in d["images"]:
        iid = im["id"]
        fn = im.get("file_name") or ""
        cand = (coco_path.parent.parent / "images" / fn,   # {split}/annotations/../images
                coco_path.parent / fn)
        path = next((c for c in cand if c.exists()), None)
        if path is None:
            base = os.path.basename(fn)
            path = next(iter(list(coco_path.parent.parent.rglob(base))[:1]), None)
        if path is None:
            missing += 1
            continue
        names = img2labels.get(iid, set())
        if not names:
            empty += 1
        labels = [0] * N_CLASS
        for n in names:
            labels[SLOT[n]] = 1
        rows.append({"path": str(path.resolve()), "labels": labels,
                     "boxes": img2nbox.get(iid, 0), "names": sorted(names)})
    return {"rows": rows, "missing": missing, "empty": empty,
            "drop": drop_cnt, "oob": oob_cnt, "unknown_names": unknown_names,
            "id2name": id2name}


def main() -> int:
    ap = argparse.ArgumentParser()
    default_data = Path(__file__).resolve().parents[3] / "tongue-data"
    ap.add_argument("--root", default=str(default_data / "tmc_tongue"))
    ap.add_argument("--out", default=str(default_data / "manifests"))
    ap.add_argument("--check", action="store_true", help="只体检，不写清单")
    ap.add_argument("--keep_empty", action="store_true",
                    help="保留无任何标注的图（默认丢弃：会把'没标注'误教成'健康'）")
    a = ap.parse_args()

    root = Path(a.root)
    if not root.exists():
        print(f"[fatal] 目录不存在: {root}")
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    found = find_coco(root)
    print(f"[scan] {root}")
    for sp in sorted(found):
        print(f"   {sp}: {[str(p.relative_to(root)) for p in found[sp]]}")
    if not found:
        print("[fatal] 没找到任何 COCO 标注 json")
        return 2

    stats = {"kept_classes": KEYS, "zh_names": ZH, "dropped_classes": DROP,
             "n_class": N_CLASS, "per_split": {}}
    total = 0
    for sp in sorted(found):
        coco = max(found[sp], key=lambda p: p.stat().st_size)   # 同 split 多份只取最大
        r = load_split(coco)
        rows = r["rows"]
        if not a.keep_empty:
            kept = [x for x in rows if sum(x["labels"]) > 0]
            dropped_empty = len(rows) - len(kept)
            rows = kept
        else:
            dropped_empty = 0

        label_cnt = [0] * N_CLASS
        hist = Counter()
        for x in rows:
            k = sum(x["labels"])
            hist[k] += 1
            for i, v in enumerate(x["labels"]):
                if v:
                    label_cnt[i] += 1

        print(f"\n[split] {sp}  保留图 {len(rows)}  "
              f"丢无标签 {dropped_empty}  缺文件 {r['missing']}  "
              f"删凸起类标注 {sum(r['drop'].values())}  越界 {sum(r['oob'].values())}")
        if r["oob"]:
            print(f"   越界明细: {dict(r['oob'])}")
        if r["unknown_names"]:
            print(f"   ⚠ 未预期类名（未计入）: {r['unknown_names']}")
        print(f"   每图标签数: {dict(sorted(hist.items()))}")
        mx = max(label_cnt) or 1
        for i in range(N_CLASS):
            bar = "█" * min(36, int(label_cnt[i] / mx * 36))
            flag = "  ⚠样本过少" if label_cnt[i] < 30 else ""
            print(f"     {i:2d} {KEYS[i]:12s} {ZH[KEYS[i]]:8s} {label_cnt[i]:6d} {bar}{flag}")

        stats["per_split"][sp] = {
            "source": str(coco.relative_to(root)),
            "images_kept": len(rows),
            "images_dropped_no_label": dropped_empty,
            "images_missing_file": r["missing"],
            "annotations_dropped_bulge": dict(r["drop"]),
            "annotations_out_of_range": {str(k): v for k, v in r["oob"].items()},
            "per_class": {KEYS[i]: label_cnt[i] for i in range(N_CLASS)},
            "labels_per_image": dict(sorted(hist.items())),
        }
        total += len(rows)

        if not a.check:
            fp = out / f"tmc_tongue_{sp}.jsonl"
            with fp.open("w", encoding="utf-8") as f:
                for x in rows:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
            print(f"   → {fp}")

    if not a.check:
        (out / "tmc_tongue_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[wrote] {out}/tmc_tongue_stats.json  合计 {total} 条 / {N_CLASS} 类")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
