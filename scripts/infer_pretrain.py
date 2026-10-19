"""基座推理探针：加载 pretrain 权重，做续写 / 问答对照测试。

用途：确认「第一次训练」的权重仍在且可用，同时量化它的能力边界。

⚠ 这是**纯基座**（只做过语言建模，从未做过 SFT）：
   - 续写：应该能接上古文语感
   - 问答：不要期待——它没有学过「被提问就作答」这种格式

用法：
    python -u scripts/infer_pretrain.py --ckpt v1_best --device cpu
    python -u scripts/infer_pretrain.py --ckpt warm5600 --device cuda --max_new_tokens 60
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sunnycx.config import SunnyConfig  # noqa: E402
from sunnycx.model import SunnyForCausalLM  # noqa: E402

TOK_PATH = ROOT.parent / "tcm-llm" / "model_tcm_tokenizer"

CKPTS = {
    # v1 = 第一次训练（早停于 step 5600，best 在 4600）
    "v1_best": ROOT / "checkpoints" / "sunnycx_0.5b_v1_best4600.pth",
    "v1_last": ROOT / "checkpoints" / "sunnycx_0.5b_pretrain_warm5600.pth",
    # v2 = 第二次训练（从 v1 的 step5600 热启动）；训练中随时可能更新
    "v2_latest": ROOT / "checkpoints" / "sunnycx_0.5b_pretrain_v2_latest.pth",
    "v2_best": ROOT / "checkpoints" / "sunnycx_0.5b_pretrain_v2_best.pth",
}

# 探针设计：前两条考「续写」，后两条考「问答」
# 如果基座正常，续写应出古文；问答则大概率跑偏（这是预期，不是 bug）
PROBES = [
    ("续写", "伤寒之为病，"),
    ("续写", "凡用药之法，当先察其"),
    ("问答", "问：《伤寒论》的作者是谁？答："),
    ("问答", "请解释什么是气血两虚。"),
]


def load_model(ckpt: Path, device: torch.device, dtype: torch.dtype):
    cfg = SunnyConfig.preset("0.5b")
    model = SunnyForCausalLM(cfg)
    t0 = time.time()
    ck = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] {ckpt.name}  {ckpt.stat().st_size / 1024**2:.0f} MB  "
          f"{len(sd)} 张量  missing={len(missing)} unexpected={len(unexpected)}  "
          f"{time.time() - t0:.1f}s")
    if missing:
        print(f"[warn] 缺失键示例: {missing[:3]}")
    if unexpected:
        print(f"[warn] 多余键示例: {unexpected[:3]}")
    del sd, ck
    gc.collect()

    model = model.to(device=device, dtype=dtype)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[load] 参数量 {n:,}  dtype={dtype}  device={device}")
    return model


@torch.no_grad()
def run_probe(model, tok, label: str, prompt: str, device, n_new: int,
              temperature: float, top_k: int):
    ids = tok.encode(prompt, add_special_tokens=False)
    inp = torch.tensor([ids], dtype=torch.long, device=device)

    t0 = time.time()
    out = model.generate(inp, max_new_tokens=n_new,
                         temperature=temperature, top_k=top_k)
    dt = time.time() - t0

    new_ids = out[0, len(ids):].tolist()
    text = tok.decode(new_ids, skip_special_tokens=True)

    # 首 token 的置信度：判断模型是"有把握"还是"在乱猜"
    with torch.autocast(device_type=device.type,
                        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                        enabled=False):
        logits = model(inp)["logits"][0, -1, :].float()
    prob = torch.softmax(logits, dim=-1)
    top_p, top_i = prob.topk(3)

    print(f"\n--- [{label}] {dt / n_new:.2f}s/token ---")
    print(f"输入：{prompt}")
    print(f"续写：{text}")
    print("首token top3：" + "  ".join(
        f"{tok.decode([int(i)])!r}={p:.3f}" for p, i in zip(top_p.tolist(), top_i.tolist())
    ), flush=True)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="v1_best", help=f"预设名 {list(CKPTS)} 或 .pth 路径")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--prompt", default="", help="只测单条自定义输入")
    args = ap.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt not in CKPTS else CKPTS[args.ckpt]
    if not ckpt.exists():
        print(f"[error] 检查点不存在: {ckpt}")
        print("        现有: " + ", ".join(f"{k}={v.name}" for k, v in CKPTS.items()))
        return 2

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"[env] {device}  torch {torch.__version__}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOK_PATH))
    print(f"[tok] vocab={tok.vocab_size}  eos={tok.eos_token_id}")

    model = load_model(ckpt, device, dtype)

    probes = [("自定义", args.prompt)] if args.prompt else PROBES
    for label, p in probes:
        try:
            run_probe(model, tok, label, p, device,
                      args.max_new_tokens, args.temperature, args.top_k)
        except Exception as e:                        # 单条失败不影响其余
            print(f"\n--- [{label}] 失败: {type(e).__name__}: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
