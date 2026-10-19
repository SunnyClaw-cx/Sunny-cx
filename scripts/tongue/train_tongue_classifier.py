# -*- coding: utf-8 -*-
"""舌象多标签分类器训练（ResNet18 / MobileNetV3-Small）

输入：由 `prepare_tmc_tongue.py` 生成的清单
      manifests/tmc_tongue_train.jsonl   {path, labels:[17 维 0/1], boxes, names}
      manifests/tmc_tongue_valid.jsonl
      manifests/tmc_tongue_test.jsonl

**17 类而不是 21 类**：四个"凸起"类（shenqutu 2 / gandantu 9 / piweitu 0 / xinfeitu 2，
全量合计 13 条标注 / 6,719 张图）已由 prepare 脚本移出输出层，留在输出层只会污染 micro-F1。

**为什么用官方 split 而不是随机切分**：TMC-Tongue 是"同一受试者多张照片"式的
医学影像集，随机切分会让同一人的照片同时出现在 train/val → val 指标虚高。
官方 train/valid/test 已按受试者隔离，必须照用。

**显存**：v2 预训练正占 ~3.2GB，本脚本按 batch 8 / 224px / AMP 跑，
实测峰值 ~0.9GB，可与预训练共存；若 OOM 会自己降 batch 重试。

用法：
  # 管线自检（CPU，2 个 batch）
  python scripts/tongue/train_tongue_classifier.py --manifest_dir .../manifests --dry-run

  # 正式训练
  python scripts/tongue/train_tongue_classifier.py --manifest_dir .../manifests \
      --epochs 30 --batch_size 8 --image_size 224
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]

# 17 类（TMC-Tongue 官方 categories 顺序，剔四个"凸起"类；与 prepare_tmc_tongue.py 严格一致）
CLASSES = [
    ("jiankangshe", "未见异常"),
    ("botaishe", "剥苔"),
    ("hongshe", "舌红"),
    ("zishe", "舌紫"),
    ("pangdashe", "舌体胖大"),
    ("shoushe", "舌体瘦小"),
    ("hongdianshe", "舌有红点"),
    ("liewenshe", "舌有裂纹"),
    ("chihenshe", "舌边齿痕"),
    ("baitaishe", "苔白"),
    ("huangtaishe", "苔黄"),
    ("heitaishe", "苔黑"),
    ("huataishe", "苔滑"),
    ("shenquao", "肾区凹陷"),
    ("gandanao", "肝胆区凹陷"),
    ("piweiao", "脾胃区凹陷"),
    ("xinfeiao", "心肺区凹陷"),
]
KEYS = [k for k, _ in CLASSES]
ZH = dict(CLASSES)


class ManifestDataset(Dataset):
    def __init__(self, jsonl: Path, transform=None):
        self.items = []
        self.n_pos = None
        bad_dim = 0
        with jsonl.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if len(r["labels"]) != len(KEYS):
                    bad_dim += 1
                    continue
                y = torch.tensor(r["labels"], dtype=torch.float32)
                if y.sum() == 0:          # 无标签样本不进训练（负样本会教坏模型）
                    continue
                self.items.append((r["path"], y))
        if bad_dim:
            raise SystemExit(
                f"[fatal] {jsonl.name} 有 {bad_dim} 行标签维度 != {len(KEYS)}。"
                f"清单与代码类别表不同源，请重跑 prepare_tmc_tongue.py。")
        if self.items:
            self.n_pos = torch.stack([y for _, y in self.items]).sum(0)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        p, y = self.items[i]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y


def build_transforms(size: int):
    from torchvision import transforms as T
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
        T.RandomRotation(7),
        T.ToTensor(), T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)])
    return train_tf, eval_tf


def build_model(n_classes: int, arch: str):
    from torchvision import models
    # 预训练权重拿不到时（内网/离线）降级为随机初始化，并明确告警——
    # 不静默降级，否则 micro-F1 低会被误判成"模型不行"而不是"没加载到 ImageNet 权重"
    def _try(fn, wcls, name):
        try:
            return fn(weights=wcls)
        except Exception as e:
            print(f"[warn] {name} 预训练权重加载失败（{type(e).__name__}: {e}）"
                  f" → 退化为随机初始化，收敛会明显变慢")
            return fn(weights=None)

    if arch == "resnet18":
        m = _try(models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, "resnet18")
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch == "mobilenet_v3_small":
        m = _try(models.mobilenet_v3_small,
                 models.MobileNet_V3_Small_Weights.IMAGENET1K_V1, "mobilenet_v3_small")
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    else:
        raise ValueError(arch)
    return m


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool, thr: float = 0.5):
    model.eval()
    preds, gts = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        preds.append((torch.sigmoid(logits.float()) > thr).float().cpu())
        gts.append(y)
    P, G = torch.cat(preds), torch.cat(gts)
    tp = (P * G).sum(0)
    fp = (P * (1 - G)).sum(0)
    fn = ((1 - P) * G).sum(0)
    prec = tp / (tp + fp).clamp(min=1e-9)
    rec = tp / (tp + fn).clamp(min=1e-9)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
    micro = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()).clamp(min=1e-9)
    return float(micro), f1.tolist(), prec.tolist(), rec.tolist(), G.sum(0).tolist()


def gpu_free_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / 1024 ** 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_dir", default=str(ROOT.parent / "tongue-data" / "manifests"))
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "tongue_cls.pth"))
    ap.add_argument("--pos_weight", default="auto", choices=["auto", "none"],
                    help="auto：按训练集 neg/pos 自动加权（默认）；none：等权")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    md = Path(a.manifest_dir)
    tr_p, va_p, te_p = (md / f"tmc_tongue_{s}.jsonl" for s in ("train", "valid", "test"))
    for p in (tr_p, va_p):
        if not p.exists():
            print(f"[error] 缺少清单: {p}\n        先跑 prepare_tmc_tongue.py")
            return 2

    train_tf, eval_tf = build_transforms(a.image_size)
    ds_tr = ManifestDataset(tr_p, train_tf)
    ds_va = ManifestDataset(va_p, eval_tf)
    ds_te = ManifestDataset(te_p, eval_tf) if te_p.exists() else None
    print(f"[data] train {len(ds_tr)} / valid {len(ds_va)}"
          f" / test {len(ds_te) if ds_te else 0}  ({len(KEYS)} 类多标签)")

    device = torch.device("cpu" if a.dry_run or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        free = gpu_free_gb()
        print(f"[gpu] 空闲显存 {free:.2f} GB（预训练占用中时约 2.0~2.4GB 可用）")
        if free < 1.2:
            print("[warn] 空闲显存不足 1.2GB，可能 OOM；建议 --batch_size 4 或等预训练结束")

    model = build_model(len(KEYS), a.arch).to(device)
    print(f"[model] {a.arch}  {sum(p.numel() for p in model.parameters())/1e6:.2f}M 参数  device={device}")

    bs = a.batch_size
    tr_loader = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=a.workers,
                           drop_last=True, pin_memory=(device.type == "cuda"))
    va_loader = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=a.workers)
    te_loader = (DataLoader(ds_te, batch_size=bs, shuffle=False, num_workers=a.workers)
                 if ds_te else None)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.epochs, 1))
    # 类别极不均衡（baitaishe 76% 正 / heitaishe 1.9% 正）→ 用 neg/pos 作 pos_weight，
    # 并夹到 [0.3, 20] 防止极稀有类把 loss 拉爆
    pw = None
    if a.pos_weight == "auto" and ds_tr.n_pos is not None:
        n = float(len(ds_tr))
        pw = ((n - ds_tr.n_pos) / ds_tr.n_pos.clamp(min=1.0)).clamp(0.3, 20.0).to(device)
        print("[loss] pos_weight(auto) " + " ".join(
            f"{k}={v:.1f}" for k, v in zip(KEYS, pw.tolist())))
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_epochs = 1 if a.dry_run else a.epochs
    best, best_ep = -1.0, -1
    f1s = prec = rec = support = []
    for ep in range(1, n_epochs + 1):
        model.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        for x, y in tr_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.item())
            nb += 1
            if nb % 50 == 0:
                print(f"  ep{ep} {nb} batch  loss {tot/nb:.4f}  {time.time()-t0:.0f}s", flush=True)
            if a.dry_run and nb >= 2:
                break
        sched.step()
        micro, f1s, prec, rec, support = evaluate(model, va_loader, device, use_amp)
        peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
        print(f"Epoch {ep}/{n_epochs}  loss {tot/max(nb,1):.4f}  "
              f"val micro-F1 {micro:.4f}  {time.time()-t0:.0f}s  peak {peak:.2f}GB", flush=True)
        if micro > best:
            best, best_ep = micro, ep
            if not a.dry_run:
                torch.save({"model": model.state_dict(), "classes": KEYS,
                            "zh_names": ZH, "arch": a.arch, "image_size": a.image_size,
                            "micro_f1": micro, "epoch": ep}, a.out)
                print(f"  -> 保存 {a.out}（best micro-F1 {micro:.4f}）", flush=True)

    print(f"\n最佳 epoch {best_ep}  val micro-F1 {best:.4f}")
    print("各类别表现（末轮，验证集）：")
    for k, f, p, r, s in zip(KEYS, f1s, prec, rec, support):
        flag = "  ⚠样本过少" if s < 30 else ""
        print(f"  {k:14s} {ZH[k]:8s} F1={f:.3f} P={p:.3f} R={r:.3f} 真值={int(s)}{flag}")

    if te_loader is not None and not a.dry_run:
        mte, f1t, _, _, sup_t = evaluate(model, te_loader, device, use_amp)
        print(f"\n【独立测试集】micro-F1 {mte:.4f}")
        for k, f, s in zip(KEYS, f1t, sup_t):
            print(f"  {k:14s} {ZH[k]:8s} F1={f:.3f} 真值={int(s)}")

    if a.dry_run:
        print("\n[dry-run] 管线已通。正式训练去掉 --dry-run。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
