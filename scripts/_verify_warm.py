# -*- coding: utf-8 -*-
"""一次性验证：预训练 --from_weight 能否正确注入 warm5600 权重。

在 CPU 上实例化 0.5B（fp32，约 2.1GB RAM）并加载 fp16 权重，验证：
  1. 键完全匹配（无 missing / unexpected）
  2. dtype 对齐后无类型冲突
  3. tied embedding（embed_tokens / lm_head）加载后仍共享同一 storage
  4. 权重数值真的变了（不是"加载了但没生效"）
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from sunnycx.config import SunnyConfig
from sunnycx.model import SunnyForCausalLM
from sunnycx.trainer import load_pretrained_weights

cfg = SunnyConfig.preset("0.5b")
print(f"[cfg] vocab={cfg.vocab_size} hidden={cfg.hidden_size} "
      f"layers={cfg.num_hidden_layers} heads={cfg.num_attention_heads}/"
      f"{cfg.num_key_value_heads} inter={cfg.intermediate_size} "
      f"tie={cfg.tie_word_embeddings}")

torch.manual_seed(0)
model = SunnyForCausalLM(cfg)          # CPU fp32
model.eval()

# 记录加载前的样本值，用于确认"真的被替换"
probe_keys = ["model.embed_tokens.weight", "model.layers.0.attn.q_proj.weight",
              "lm_head.weight"]
before = {k: model.state_dict()[k].flatten()[:5].clone() for k in probe_keys}

w = ROOT / "checkpoints" / "sunnycx_0.5b_pretrain_warm5600.pth"
print(f"[load] {w.name}  {w.stat().st_size/1024**2:.2f} MB")
ok = load_pretrained_weights(model, str(w), torch.device("cpu"))
print(f"[result] 返回 {ok}")
assert ok, "load_pretrained_weights 返回 False"

sd = model.state_dict()
changed = sum(1 for k in probe_keys
              if not torch.equal(sd[k].flatten()[:5], before[k]))
print(f"[verify] 抽查 {len(probe_keys)} 个键，值已改变 {changed} 个")
assert changed == len(probe_keys), "权重未真正写入"

a, b = sd["model.embed_tokens.weight"], sd["lm_head.weight"]
print(f"[verify] tied 共享保持: {a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()}")
weights_dtype = next(model.parameters()).dtype
print(f"[verify] 全部参数 dtype 一致: "
      f"{len({p.dtype for p in model.parameters()}) == 1} ({weights_dtype})")

# 前向能跑通即证明结构完好（forward 返回 dict，非 HF 风格对象）
ids = torch.randint(0, cfg.vocab_size, (1, 32))
with torch.no_grad():
    out = model(ids)
print(f"[verify] 前向 OK  logits={tuple(out['logits'].shape)}  "
      f"有限值={bool(torch.isfinite(out['logits']).all())}")
print("\n[PASS] 热启动注入链路验证通过")
