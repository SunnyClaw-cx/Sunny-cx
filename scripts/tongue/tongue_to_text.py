"""舌象特征 → 文字描述 + 证候倾向提示。

这是「视觉 → 文本」的转换层：分类器只输出概率，本脚本把概率翻成人能读、
也能喂给语言模型的舌象描述。

设计原则：
  ① 只描述**看得见的**（舌色/舌形/苔色/苔质），不下诊断。
     说「舌淡胖、边有齿痕、苔白腻」而不是「脾虚湿盛」——
     后者是辨证结论，必须由辨证模型综合症状后给出。
  ② 证候倾向只作**提示**，明确标注「供参考，需结合症状」。
  ③ 置信度低（0.35~0.5）的特征标「疑似」，不让模型当成确定事实。

用法：
    # 单张图
    python scripts/tongue/tongue_to_text.py --ckpt checkpoints/tongue_cls.pth --image x.jpg
    # 只看规则（不需模型）
    python scripts/tongue/tongue_to_text.py --demo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# ----------------------------------------------------------------------
# 舌象特征 → 中文描述
# ----------------------------------------------------------------------
# 类别键沿用 TCM-Tongue 的英文代号；不同数据集可在 ckpt 的 classes 字段里覆盖
FEATURE_ZH = {
    # ---- 舌色 ----
    "jiankangshe": ("舌色", "正常"),
    "hongshe": ("舌色", "偏红"),
    "zishe": ("舌色", "偏紫暗"),
    # ---- 苔色 ----
    "baitaishe": ("苔色", "白"),
    "huangtaishe": ("苔色", "黄"),
    "heitaishe": ("苔色", "黑"),
    # ---- 苔质 ----
    "botaishe": ("舌苔", "剥落少苔"),
    "huataishe": ("苔质", "滑"),
    # ---- 舌形 ----
    "pangdashe": ("舌形", "胖大"),
    "shouxiaoshe": ("舌形", "瘦小"),
    "chihenshe": ("舌形", "边有齿痕"),
    "liewenshe": ("舌形", "有裂纹"),
    "hongdianshe": ("舌形", "有红点"),
    # ---- 舌面脏腑分区（TMC-Tongue 独有，舌尖心肺 / 舌中脾胃 / 舌边肝胆 / 舌根肾）----
    "shenquaxian": ("舌面分区", "肾区凹陷"),
    "shenqutuxi": ("舌面分区", "肾区凸起"),
    "gandanquaxian": ("舌面分区", "肝胆区凹陷"),
    "gandanqutuxi": ("舌面分区", "肝胆区凸起"),
    "piweiquaxian": ("舌面分区", "脾胃区凹陷"),
    "xinfeiquaxian": ("舌面分区", "心肺区凹陷"),
    "xinfeiqutuxi": ("舌面分区", "心肺区凸起"),
}

# 兼容早期草稿里的旧键名，避免旧 ckpt 直接失效
_ALIAS = {"shoushe": "shouxiaoshe",
          "shenquao": "shenquaxian", "shenqutu": "shenqutuxi",
          "gandanao": "gandanquaxian", "gandantu": "gandanqutuxi",
          "piweiao": "piweiquaxian",
          "xinfeiao": "xinfeiquaxian", "xinfeitu": "xinfeiqutuxi"}

# ----------------------------------------------------------------------
# 舌象 → 证候倾向（中医诊断学基础对应关系，只作提示）
# ----------------------------------------------------------------------
# 每条：(必须命中的特征集合, 倾向证候, 说明)
RULES = [
    ({"pangdashe", "chihenshe"}, "脾虚湿盛", "舌胖大兼齿痕，为脾虚水湿不化之象"),
    ({"pangdashe"}, "水湿内停", "舌体胖大，多为水湿痰饮内停"),
    ({"shouxiaoshe"}, "阴血不足", "舌体瘦小，常见于阴血亏虚"),
    ({"liewenshe"}, "阴液不足", "舌有裂纹，多提示阴伤津亏"),
    ({"hongshe", "huangtaishe"}, "里热", "舌红苔黄，属热象"),
    ({"hongshe"}, "有热", "舌质偏红，提示有热"),
    ({"zishe"}, "血瘀", "舌质紫暗，多与血行瘀滞有关"),
    ({"baitaishe", "huataishe"}, "寒湿", "苔白而滑，多为寒湿"),
    ({"heitaishe"}, "重证", "苔黑多属重证，须结合全身情况判断"),
    ({"botaishe"}, "胃气阴不足", "苔剥落，提示胃之气阴受损"),
    ({"hongdianshe"}, "热盛", "舌生红点，多见于热盛"),
    # 舌面分区：按「舌尖心肺 / 舌中脾胃 / 舌边肝胆 / 舌根肾」定位。
    # 凹陷多主虚、凸起多主实（积聚），属望诊经验，把握度低于舌色舌苔，只作提示。
    ({"shenquaxian"}, "肾气不足倾向", "舌根（肾区）见凹陷，多提示肾气不足"),
    ({"shenqutuxi"}, "下焦积聚倾向", "舌根（肾区）见凸起，须留意下焦实邪或积聚"),
    ({"piweiquaxian"}, "脾胃气虚倾向", "舌中（脾胃区）见凹陷，多提示脾胃气虚"),
    ({"gandanquaxian"}, "肝血不足倾向", "舌边（肝胆区）见凹陷，多提示肝血不足"),
    ({"gandanqutuxi"}, "肝郁气滞倾向", "舌边（肝胆区）见凸起，多提示肝郁气滞"),
    ({"xinfeiquaxian"}, "心肺气虚倾向", "舌尖（心肺区）见凹陷，多提示心肺气虚"),
    ({"xinfeiqutuxi"}, "心肺热盛倾向", "舌尖（心肺区）见凸起，多提示心肺有热"),
]

DISCLAIMER = "以上舌象判断仅为图像特征描述，不构成诊断。"


def features_from_probs(classes: list[str], probs: list[float],
                        thr: float = 0.5, low: float = 0.35) -> dict:
    """按置信度把概率转成 {特征分组: [描述]}。"""
    groups: dict[str, list[str]] = {}
    weak: list[str] = []
    hits: set[str] = set()
    unknown: list[str] = []

    for c, p in zip(classes, probs):
        key_c = _ALIAS.get(c, c)
        key, zh = FEATURE_ZH.get(key_c, (None, None))
        if key is None:
            unknown.append(c)
            continue
        if p >= thr:
            groups.setdefault(key, []).append(zh)
            hits.add(key_c)
        elif p >= low:
            weak.append(f"{zh}(疑似{p:.2f})")

    return {"groups": groups, "weak": weak, "hits": hits, "unknown": unknown}


def render_text(feat: dict) -> str:
    """把分组结果拼成一句通顺的舌象描述。"""
    g = feat["groups"]
    parts = []
    if "舌色" in g:
        parts.append("舌色" + "、".join(g["舌色"]))
    if "舌形" in g:
        parts.append("舌形" + "、".join(g["舌形"]))
    if "苔色" in g or "苔质" in g:
        color = "".join(g.get("苔色", []))          # 白 / 黄 / 黑
        qual = "".join(g.get("苔质", []))           # 滑
        if color:
            parts.append(f"苔{color}{qual}")        # 苔白滑
        else:
            parts.append(f"苔{qual or '异常'}")
    if "舌苔" in g:
        parts.append("、".join(g["舌苔"]))
    if "舌面分区" in g:
        parts.append("舌面见" + "、".join(g["舌面分区"]))

    if not parts:
        s = "未见明显异常舌象"
    else:
        s = "，".join(parts)

    if feat["weak"]:
        s += "；另疑有：" + "、".join(feat["weak"])
    return s


def tendencies(hits: set[str]) -> list[tuple[str, str]]:
    """规则命中 → [(证候倾向, 说明)]"""
    out = []
    for need, syn, why in RULES:
        if need <= hits:
            out.append((syn, why))
    return out


def describe(classes: list[str], probs: list[float]) -> dict:
    feat = features_from_probs(classes, probs)
    text = render_text(feat)
    tend = tendencies(feat["hits"])
    return {
        "text": text,
        "tendency": [{"syndrome": s, "reason": w} for s, w in tend],
        "unknown_classes": feat["unknown"],
        "disclaimer": DISCLAIMER,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "tongue_cls.pth"))
    ap.add_argument("--image", default="")
    ap.add_argument("--demo", action="store_true", help="不加载模型，用假概率演示规则")
    args = ap.parse_args()

    if args.demo:
        # 用 TMC-Tongue 全部 21 类做一次规则覆盖自检，防止「类别键写错 → 静默跳过」
        all_cls = ["jiankangshe", "botaishe", "hongshe", "zishe", "pangdashe",
                   "shouxiaoshe", "hongdianshe", "liewenshe", "chihenshe",
                   "baitaishe", "huangtaishe", "heitaishe", "huataishe",
                   "shenquaxian", "shenqutuxi", "gandanquaxian", "gandanqutuxi",
                   "piweiquaxian", "xinfeiqutuxi", "xinfeiquaxian"]
        print(f"[check] 数据集类别 {len(all_cls)} 个，规则表覆盖 {len(FEATURE_ZH)} 个")
        miss = [c for c in all_cls if c not in FEATURE_ZH]
        print("[check] 未覆盖:", miss if miss else "无 ✅")

        # 构造一个「舌胖大 + 齿痕 + 苔白滑 + 脾胃区凹陷」的典型脾虚湿盛舌象
        classes = ["pangdashe", "chihenshe", "baitaishe", "huataishe",
                   "piweiquaxian", "hongshe"]
        probs = [0.91, 0.77, 0.83, 0.61, 0.58, 0.12]
        r = describe(classes, probs)
        print("\n[demo] 舌象描述：", r["text"])
        print("[demo] 证候倾向：")
        for t in r["tendency"]:
            print(f"  - {t['syndrome']}：{t['reason']}")
        print("[demo]", r["disclaimer"])
        return 0

    if not args.image:
        print("[error] 需要 --image 或 --demo")
        return 2
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[error] 检查点不存在: {ckpt_path}")
        print("        先用 train_tongue_classifier.py 训练，或加 --demo 看规则")
        return 2

    import torchvision.transforms as T
    from PIL import Image
    from torchvision import models

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    arch = ck.get("arch", "resnet18")
    size = ck.get("image_size", 224)

    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, len(classes))
    else:
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, len(classes))
    m.load_state_dict(ck["model"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = m.to(dev).eval()

    tf = T.Compose([
        T.Resize((size, size)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        x = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(dev)
        probs = torch.sigmoid(m(x)).cpu().squeeze(0).tolist()

    r = describe(classes, probs)
    print("舌象描述：", r["text"])
    print("\n证候倾向（仅供参考，须结合症状）：")
    for t in r["tendency"]:
        print(f"  - {t['syndrome']}：{t['reason']}")
    print("\n各类概率：")
    for c, p in sorted(zip(classes, probs), key=lambda t: -t[1]):
        print(f"  {c:14s} {p:.3f}")
    print("\n" + r["disclaimer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
