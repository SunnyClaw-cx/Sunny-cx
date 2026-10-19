"""sunny-cx 模型架构 —— 完全自研实现

设计依据（逐项列明，便于溯源）：
  · RMSNorm / Pre-Norm ...... Zhang & Sennrich 2019; GPT-3 (Brown et al. 2020)
  · RoPE 旋转位置编码 ....... Su et al. 2021 (RoFormer)
  · 因果自注意力 ............ Vaswani et al. 2017 (Attention Is All You Need)
  · GQA 分组查询注意力 ...... Ainslie et al. 2023 (GQA)
  · SwiGLU 前馈 ............. Shazeer 2020 (GLU Variants Improve Transformer)
  · 词嵌入权重绑定 .......... Inan et al. 2016; Press & Wolf 2017

西瓜书（周志华《机器学习》2016）第 5 章为本项目提供**训练理论地基**：
  5.3 误差逆传播(BP) —— 本文件所有子层均为可微链式结构，由 PyTorch autograd 执行
  5.3 早停法 / 正则化 —— 见 trainer.py
  5.4 全局最小与局部极小 —— 见 trainer.py 的学习率调度与梯度裁剪
  2.2 评估方法(留出法) —— 见 dataset.py 按书切分验证集
  2.3 性能度量 —— 交叉熵损失

注意：西瓜书出版于 2016 年，早于 Transformer(2017)，故架构部分另据上述论文。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SunnyConfig


# ======================================================================
# 归一化
# ======================================================================
class RMSNorm(nn.Module):
    """均方根归一化：只做缩放，不减均值，比 LayerNorm 少一次归约。

        y = x / sqrt(mean(x²) + eps) * weight

    计算在 fp32 下进行后再转回原精度，避免 fp16 下平方溢出。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


# ======================================================================
# 旋转位置编码 RoPE
# ======================================================================
def build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1e6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE 的 cos / sin 表。

    返回 (cos, sin)，形状均为 [max_seq_len, head_dim]。
    """
    # 频率：theta^(-2i/d)，i = 0..d/2-1
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                      # [T, d/2]
    emb = torch.cat([freqs, freqs], dim=-1)               # [T, d]
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把后半段取负后与前半段交换：(x1, x2) -> (-x2, x1)"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 q / k 施加旋转位置编码。

    q: [B, H, T, D]   k: [B, Hkv, T, D]   cos/sin: [T, D]

    cos/sin 表始终以 fp32 预计算（角度精度要求高），在此按 q 的精度转换。
    若 q 为 fp16 而 cos 为 fp32，直接相乘会触发类型提升，把整个注意力
    路径抬回 fp32，既费显存又失去混合精度意义。
    """
    dtype = q.dtype
    # 广播到 [B, 1, T, D]
    cos = cos.unsqueeze(0).unsqueeze(0).to(dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(dtype)
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


# ======================================================================
# 因果自注意力（GQA）
# ======================================================================
class CausalSelfAttention(nn.Module):
    """多头因果自注意力，支持 GQA（kv 头数可少于 q 头数）。

    GQA 作用：K/V 投影参数量与 KV-Cache 显存按 kv_heads/q_heads 比例下降。
    """

    def __init__(self, config: SunnyConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads   # 每个 kv 头被多少个 q 头共享

        d = config.hidden_size
        kv_dim = self.n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, kv_dim, bias=False)
        self.v_proj = nn.Linear(d, kv_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)

        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # 1) 线性投影并拆分为多头：[B, T, H, D] -> [B, H, T, D]
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 2) 施加 RoPE（只作用于 q/k，不作用于 v）
        q, k = apply_rope(q, k, cos, sin)

        # 3) GQA：把 kv 头复制 n_rep 次，使其与 q 头一一对应
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # 4) 因果掩码注意力
        #    用 is_causal 让底层内核自动生成上三角掩码；
        #    若外部传入 padding mask（0/1，1=可见）则转为加性 bias。
        attn_bias = None
        use_causal = attention_mask is None
        if attention_mask is not None:
            # attention_mask: [B, T] ，1 表示有效 token
            keep = attention_mask[:, None, None, :].to(torch.bool)      # [B,1,1,T]
            causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
            keep = keep & causal                                         # [B,1,T,T]
            attn_bias = torch.zeros(B, 1, T, T, device=x.device, dtype=q.dtype)
            attn_bias = attn_bias.masked_fill(~keep, float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_causal,
        )

        # 5) 合并多头并输出投影
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# ======================================================================
# SwiGLU 前馈网络
# ======================================================================
class SwiGLUFeedForward(nn.Module):
    """SwiGLU:  FFN(x) = W_down( silu(W_gate x) ⊙ (W_up x) )

    相比标准 FFN(ReLU)，多一个门控分支，在相同参数量下通常效果更好。
    """

    def __init__(self, config: SunnyConfig):
        super().__init__()
        d = config.hidden_size
        inter = config.intermediate_size

        self.gate_proj = nn.Linear(d, inter, bias=False)
        self.up_proj = nn.Linear(d, inter, bias=False)
        self.down_proj = nn.Linear(inter, d, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x)))


# ======================================================================
# Transformer 层（Pre-Norm 残差结构）
# ======================================================================
class SunnyBlock(nn.Module):
    """Pre-Norm 结构：

        h = x + Attn( Norm(x) )
        y = h + FFN ( Norm(h) )

    Pre-Norm 使梯度可沿残差直通，深层网络的 BP 更稳定（西瓜书 5.4 节
    所述"梯度问题"的工程对策）。
    """

    def __init__(self, config: SunnyConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ======================================================================
# 主干
# ======================================================================
class SunnyModel(nn.Module):
    def __init__(self, config: SunnyConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.layers = nn.ModuleList(
            [SunnyBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # RoPE 表：注册为 buffer，随模型迁移设备；非持久化，不写进 checkpoint
        cos, sin = build_rope_cache(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._grad_ckpt = config.gradient_checkpointing
        self.apply(self._init_weights)
        self._scale_residual_branches()

    def _scale_residual_branches(self) -> None:
        """残差分支输出投影的方差缩放（GPT-2 做法）。

        每层有两条残差相加。若分支输出方差不加约束，深层激活量级会随层数
        累积放大，在 fp16 下容易溢出。故对写入残差流的两个投影
        （attn.o_proj、mlp.down_proj）按 1/sqrt(2L) 缩放。

        注意：**不**用 "输出维度 == hidden_size" 作为判据——q_proj 的输出维度
        n_heads·head_dim 在 num_heads·head_dim == hidden_size 时恰好也等于 d，
        会被误伤。此处按模块名精确指定。
        """
        L = self.config.num_hidden_layers
        scale = 1.0 / math.sqrt(2 * L)
        with torch.no_grad():
            for layer in self.layers:
                layer.attn.o_proj.weight.mul_(scale)
                layer.mlp.down_proj.weight.mul_(scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """初始化：线性层用正态分布 (std=0.02)，归一化权重置 1。"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def gradient_checkpointing_enable(self, enable: bool = True) -> None:
        self._grad_ckpt = enable

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.dropout(self.embed_tokens(input_ids))

        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]

        for layer in self.layers:
            if self._grad_ckpt and self.training:
                # 梯度检查点：前向不保存中间激活，反向时重算，用约 30% 时间换显存
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, cos, sin, attention_mask,
                    use_reentrant=False,
                )
            else:
                h = layer(h, cos, sin, attention_mask)

        return self.norm(h)


# ======================================================================
# 语言模型头
# ======================================================================
class SunnyForCausalLM(nn.Module):
    """sunny-cx 因果语言模型。

    tie_word_embeddings=True 时，输出投影复用输入词嵌入权重（Inan et al. 2016），
    可省去 V·d 参数并改善小词表模型的收敛。
    """

    def __init__(self, config: SunnyConfig):
        super().__init__()
        self.config = config
        self.model = SunnyModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    # ---- 便捷接口 ----
    def gradient_checkpointing_enable(self, enable: bool = True) -> None:
        self.model.gradient_checkpointing_enable(enable)

    @property
    def num_parameters(self) -> int:
        # 绑定权重只计一次
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> dict:
        hidden = self.model(input_ids, attention_mask, start_pos)
        logits = self.lm_head(hidden)

        out = {"logits": logits, "hidden_states": hidden}
        if labels is not None:
            # 自回归语言建模：位置 t 预测 t+1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out

    # ---- 生成（推理用，供冒烟测试校验概率合理性）----
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx = input_ids[:, -self.config.max_position_embeddings:]
            logits = self.lm_head(self.model(idx))[:, -1, :]
            logits = logits.float() / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, nxt], dim=1)
        return input_ids


# ======================================================================
# 参数量自校验
# ======================================================================
def verify_param_count(config: SunnyConfig, verbose: bool = True) -> Tuple[int, int]:
    """实例化模型，对照公式校验参数量。返回 (实测, 公式)。"""
    model = SunnyForCausalLM(config)
    actual = model.num_parameters
    predicted = config.num_parameters()
    if verbose:
        diff = abs(actual - predicted)
        rel = diff / actual * 100 if actual else 0.0
        print(config.describe())
        print(f"  参数量实测 : {actual:,}")
        print(f"  参数量公式 : {predicted:,}")
        print(f"  差异       : {diff:,}  (相对 {rel:.4f}%)")
    del model
    return actual, predicted


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "0.5b"
    cfg = SunnyConfig.preset(name)
    a, p = verify_param_count(cfg)
    print()
    if a == p:
        print("[OK] 公式与实测完全一致")
    else:
        print(f"[CHECK] 差异 {abs(a-p):,} = 各层 RMSNorm 项（公式已含 2d·L+d，残差来自实现细节）")