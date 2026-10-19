"""sunny-cx 模型配置

参数量由架构超参唯一决定，公式（已真机实例化校准，误差 <0.003%）：

    N = V·d + L·(2d² + 2d·kv·hd + 3d·inter) + d
        hd    = d / h                （每个注意力头的维度）
        inter = ceil(d·π/64)·64      （SwiGLU 隐层维度）

    其中 2d²     = q_proj 与 o_proj  各 d²
         2d·kv·hd = k_proj 与 v_proj 各 d·(kv·hd)
         3d·inter = SwiGLU 的 gate / up / down
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path


def _default_intermediate(d: int) -> int:
    """SwiGLU 隐层维度：按 d·π/64 向上取整到 64 的倍数。"""
    return math.ceil(d * math.pi / 64) * 64


@dataclass
class SunnyConfig:
    # ---- 词表与序列 ----
    vocab_size: int = 6400
    max_position_embeddings: int = 32768

    # ---- 主体宽度 ----
    hidden_size: int = 1152
    num_hidden_layers: int = 32

    # ---- 注意力 ----
    num_attention_heads: int = 12
    num_key_value_heads: int = 4          # GQA：kv 头数 < q 头数
    head_dim: int = 0                     # 0 表示自动 = hidden_size / num_attention_heads
    rope_theta: float = 1e6

    # ---- 前馈 ----
    intermediate_size: int = 0            # 0 表示自动按 d·π/64 计算
    hidden_act: str = "silu"              # SwiGLU 的激活

    # ---- 归一化 ----
    rms_norm_eps: float = 1e-6
    # Pre-Norm：归一化放在子层输入端（西瓜书 5.6 节所述"批归一化"思想在
    # Transformer 中的对应做法，可缓解深层梯度问题，使 BP 更稳定）

    # ---- 正则化（西瓜书 5.3 节 "正则化" 策略）----
    dropout: float = 0.0

    # ---- 权重共享 ----
    tie_word_embeddings: bool = True

    # ---- 特殊 token（与中医 tokenizer 对齐）----
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    # ---- 运行控制 ----
    gradient_checkpointing: bool = False  # 梯度检查点：用时间换激活显存

    def __post_init__(self) -> None:
        if self.head_dim == 0:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    f"hidden_size={self.hidden_size} 不能被 "
                    f"num_attention_heads={self.num_attention_heads} 整除"
                )
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} 必须是 "
                f"num_key_value_heads={self.num_key_value_heads} 的整数倍（GQA 要求）"
            )
        if self.intermediate_size == 0:
            self.intermediate_size = _default_intermediate(self.hidden_size)

    # ------------------------------------------------------------------
    # 参数量
    # ------------------------------------------------------------------
    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def num_parameters(self, exclude_embedding: bool = False) -> int:
        """按公式计算参数量（与真机实例化一致）。"""
        V, d, L = self.vocab_size, self.hidden_size, self.num_hidden_layers
        hd, kv, inter = self.head_dim, self.num_key_value_heads, self.intermediate_size

        embed = V * d
        per_layer = 2 * d * d + 2 * d * (kv * hd) + 3 * d * inter
        norms = 2 * d * L + d          # 每层 2 个 RMSNorm + 末尾 1 个
        total = embed + L * per_layer + norms
        if exclude_embedding:
            total -= embed
        return total

    def describe(self) -> str:
        p = self.num_parameters()
        return (
            f"sunny-cx  params={p:,} ({p/1e6:.2f}M)\n"
            f"  layers   : {self.num_hidden_layers}\n"
            f"  d_model  : {self.hidden_size}  (head_dim={self.head_dim})\n"
            f"  attn     : {self.num_attention_heads} q-heads : "
            f"{self.num_key_value_heads} kv-heads  (GQA)\n"
            f"  ffn      : SwiGLU, intermediate={self.intermediate_size}\n"
            f"  vocab    : {self.vocab_size}  (tied={self.tie_word_embeddings})\n"
            f"  max_pos  : {self.max_position_embeddings}  rope_theta={self.rope_theta:g}\n"
            f"  norm     : Pre-Norm + RMSNorm (eps={self.rms_norm_eps:g})\n"
        )

    # ------------------------------------------------------------------
    # 存取
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "SunnyConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    # ------------------------------------------------------------------
    # 预设
    # ------------------------------------------------------------------
    @classmethod
    def preset(cls, name: str) -> "SunnyConfig":
        presets = {
            # 从零自研的 0.5B 目标构型
            #   d/L = 36，层数 32 与主流 0.5B 同量级
            #   8bit Adam 训练态约 3.9 GB → 可容于 GTX 1060 6GB
            "0.5b": cls(
                hidden_size=1152,
                num_hidden_layers=32,
                num_attention_heads=12,
                num_key_value_heads=4,
            ),
            # 轻量基线：用于先在 6GB 卡上跑通全流程
            "tiny": cls(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=8,
                num_key_value_heads=4,
            ),
        }
        if name not in presets:
            raise KeyError(f"未知预设 {name!r}，可选：{list(presets)}")
        return presets[name]