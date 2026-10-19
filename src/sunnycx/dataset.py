"""数据管线 —— 流式读取，不在内存里装 400MB JSONL。

依据西瓜书 2.2.1「留出法」：验证集必须与训练集**完全独立**。
本项目已验证集由 split_val.py 按「整本书」切分（ratio=0.05，seed=42），
验证集里的书训练时完全看不到，故此处只负责读取，不再二次切分。
"""
from __future__ import annotations

import json
import random
from typing import Iterator, List, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

IGNORE_INDEX = -100


# ======================================================================
# 预训练语料 · 序列打包（sequence packing）
# ======================================================================
class PackedPretrainDataset(IterableDataset):
    """序列打包预训练数据集 —— 解决短样本的填充浪费。

    为什么必须打包（本项目实测数据支撑）：
      预训练样本长度 **中位数仅 60 token、均值 92 token**
      （P90=183, P99=521, P100=917）。若按单样本 pad 到 max_seq_len=512：
          填充利用率仅 **17.6%** → **82% 的算力在算 padding**
          → 单 epoch 需 11.5 天（实测 15.6 s/step × 63663 step）

      打包后把短样本首尾相接填满序列，利用率接近 100%，
      算力效率提升约 **5.7 倍**，单 epoch 降到约 2 天。

    依据：GPT 系列 / LLaMA 等大模型的标准预训练做法
      （Krell et al. 2021, Packing Input Sequences 已证明不损伤效果）。

    实现：以 eos 作为文档分隔符把 token 流拼接，攒满 max_seq_len 即切出。
      跨文档注意力在预训练中被允许——模型以 eos 为文档边界信号。
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_seq_len: int = 512,
        shuffle_buffer: int = 20000,
        seed: int = 42,
    ):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    def _iter_raw(self) -> Iterator[str]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                if text:
                    yield text

    def _emit(self, chunk: List[int]) -> dict:
        """chunk 长度恒为 max_seq_len：无 padding，mask 全 1。"""
        ids = torch.tensor(chunk, dtype=torch.long)
        return {
            "input_ids": ids,
            "labels": ids.clone(),          # 预训练：全部 token 参与损失
            "attention_mask": torch.ones(len(chunk), dtype=torch.long),
        }

    def __iter__(self) -> Iterator[dict]:
        worker = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (worker.id if worker else 0))

        token_buf: List[int] = []
        text_buf: List[str] = []

        def flush_texts():
            """把缓冲的文档洗牌后逐个 tokenize，灌入 token 流。"""
            rng.shuffle(text_buf)
            for t in text_buf:
                ids = self.tokenizer.encode(t, add_special_tokens=False)
                if not ids:
                    continue
                token_buf.extend(ids)
                token_buf.append(self.eos_id)   # 文档分隔符
                while len(token_buf) >= self.max_seq_len:
                    yield self._emit(token_buf[: self.max_seq_len])
                    del token_buf[: self.max_seq_len]
            text_buf.clear()

        for text in self._iter_raw():
            text_buf.append(text)
            if len(text_buf) >= self.shuffle_buffer:
                yield from flush_texts()
        if text_buf:
            yield from flush_texts()
        # 尾部不足一条的残余 token 丢弃（≤ max_seq_len-1，可忽略）


# ======================================================================
# 预训练语料（自回归语言建模，单样本 pad）
# ======================================================================
class PretrainDataset(IterableDataset):
    """流式预训练数据集。

    每条样本：{"text": "..."}  →  tokenize → 截断/补齐到 max_seq_len
    标签 = 输入本身（下一 token 预测），pad 位置置 -100 不参与损失。
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_seq_len: int = 512,
        shuffle_buffer: int = 10000,
        seed: int = 42,
    ):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    def _iter_raw(self) -> Iterator[str]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                if text:
                    yield text

    def _encode(self, text: str) -> List[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        # 留 1 位给 eos
        ids = ids[: self.max_seq_len - 1]
        ids = ids + [self.eos_id]
        return ids

    def __iter__(self) -> Iterator[dict]:
        worker = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (worker.id if worker else 0))

        buf: List[str] = []
        for text in self._iter_raw():
            buf.append(text)
            if len(buf) >= self.shuffle_buffer:
                rng.shuffle(buf)
                for t in buf:
                    yield self._pack(t)
                buf = []
        if buf:
            rng.shuffle(buf)
            for t in buf:
                yield self._pack(t)

    def _pack(self, text: str) -> dict:
        ids = self._encode(text)
        pad_len = self.max_seq_len - len(ids)
        input_ids = ids + [0] * pad_len
        labels = ids + [IGNORE_INDEX] * pad_len
        attn = [1] * len(ids) + [0] * pad_len
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# ======================================================================
# SFT 问答语料
# ======================================================================
class SFTDataset(IterableDataset):
    """SFT 数据集：只对 assistant 回答部分计算损失。

    输入格式：{"conversations": [{"role":"user","content":...},
                                {"role":"assistant","content":...}, ...]}
    损失掩码策略：user 段与特殊 token 位置置 -100，只学 assistant 的输出。

    两种 padding 策略（实测见 scripts/bench_sft_bucket.py）：

    · dynamic_pad=False（原状）：每条样本各自补到 max_seq_len。
      SFT 样本长度 均 92.2 / P50 77 / P99 249 token，补到 512 后
      **有效 token 仅 18%，82% 算力在算 padding** → 单轮 12.9 天。

    · dynamic_pad=True：样本本身不补，由 collate_pad 补到**本 batch 最长样本**，
      配合 bucket_chunk 把长度相近的样本排到一起（分桶）。
      语义与固定 pad 完全一致（pad 位 label=-100、mask=0，不进损失也不进注意力），
      只是不再为短样本算无意义的 padding。

    ⚠ SFT **不能序列打包**（会串掉多轮对话结构），但**可以分桶**——这正是本开关的边界。
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_seq_len: int = 768,
        shuffle_buffer: int = 5000,
        seed: int = 42,
        dynamic_pad: bool = False,
        bucket_chunk: int = 0,
    ):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.dynamic_pad = dynamic_pad
        # >0 时按块内长度降序排列，使相邻 batch 长度接近（分桶）
        self.bucket_chunk = max(0, int(bucket_chunk))
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    def _iter_raw(self) -> Iterator[dict]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("conversations"):
                    yield obj

    def _build(self, obj: dict) -> Optional[dict]:
        convs = obj.get("conversations", [])
        input_ids, labels = [], []
        for turn in convs:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if not content:
                continue
            ids = self.tokenizer.encode(content, add_special_tokens=False)
            input_ids.extend(ids)
            # 只对 assistant 的内容回传梯度
            if role == "assistant":
                labels.extend(ids)
            else:
                labels.extend([IGNORE_INDEX] * len(ids))

        if not input_ids:
            return None

        input_ids = input_ids[: self.max_seq_len - 1] + [self.eos_id]
        labels = labels[: self.max_seq_len - 1] + [self.eos_id]

        # 整个序列都是 user 段（无 assistant）则丢弃，避免纯 -100 样本
        if all(l == IGNORE_INDEX for l in labels):
            return None

        if self.dynamic_pad:
            # 不在这里补：交给 collate_pad 按本 batch 最长样本补，省掉短样本的 padding 算力
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            }

        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids += [0] * pad_len
            labels += [IGNORE_INDEX] * pad_len
            attn = [1] * (self.max_seq_len - pad_len) + [0] * pad_len
        else:
            attn = [1] * self.max_seq_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    def _emit_block(self, buf: List[dict], rng: random.Random) -> Iterator[dict]:
        """洗牌 → 构建样本 → （可选）按块内长度降序排列。"""
        rng.shuffle(buf)
        items: List[dict] = []
        for o in buf:
            item = self._build(o)
            if item is not None:
                items.append(item)
        if self.bucket_chunk > 0 and len(items) > 1:
            for i in range(0, len(items), self.bucket_chunk):
                chunk = items[i: i + self.bucket_chunk]
                # 降序：让长度接近的样本落进同一个 batch，动态 pad 的浪费才压得住
                chunk.sort(key=lambda d: d["input_ids"].size(0), reverse=True)
                yield from chunk
        else:
            yield from items

    def __iter__(self) -> Iterator[dict]:
        worker = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (worker.id if worker else 0))

        buf: List[dict] = []
        for obj in self._iter_raw():
            buf.append(obj)
            if len(buf) >= self.shuffle_buffer:
                yield from self._emit_block(buf, rng)
                buf = []
        if buf:
            yield from self._emit_block(buf, rng)


# ======================================================================
# 验证集（抽样定长，保证每次验证可比）
# ======================================================================
def _count_lines(path: str) -> int:
    """快速统计 jsonl 行数（只读字节，不做 json 解析）。"""
    n = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    n += 1
    except OSError:
        return 0
    return n


class ValSubset(Dataset):
    """固定验证子集：从 val 文件等间隔抽 N 条，保证每轮验证用同一批数据。

    西瓜书 5.3 节早停法要求「验证集误差」可比，故验证样本必须固定。
    """

    def __init__(self, base: IterableDataset, num_samples: int = 200, seed: int = 42):
        self.samples: List[dict] = []
        want = num_samples
        # 步长按「验证集总条数」自适应，不能用写死的 2000：
        # 实测 val 仅 53,705 行，写死 2000 只能抽到 27 条，其余靠「顺序补齐」
        # 从文件头取 → 验证集 60% 集中在开头几本书，早停判据失真。
        stride = 1
        path = getattr(base, "path", None)
        if path:
            n_total = _count_lines(path)
            if n_total > 0:
                stride = max(1, n_total // want)
        for i, item in enumerate(base):
            if i % stride == 0:
                self.samples.append(item)
                if len(self.samples) >= want:
                    break
        if len(self.samples) < want:
            # 兜底：不足则顺序补齐（正常情况下 stride 自适应后不会走到这里）
            for item in base:
                self.samples.append(item)
                if len(self.samples) >= want:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        return self.samples[i]


def collate(batch: List[dict]) -> dict:
    """把样本列表堆叠成 batch 张量（要求长度已对齐）。"""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }


def collate_pad(batch: List[dict], multiple: int = 8) -> dict:
    """变长样本 → 补到「本 batch 最长」，而不是补到全局 max_seq_len。

    padding 位与固定 pad 完全同义：input_ids=0、labels=IGNORE_INDEX(-100)、mask=0，
    既不进交叉熵也不进注意力。区别只在**长度**——短样本不再陪跑到 512。

    `multiple` 把长度向上取整到 8 的倍数，避免张量尺寸过于零碎（nvidia 对齐习惯）。
    """
    max_len = max(b["input_ids"].size(0) for b in batch)
    if multiple > 1:
        max_len = (max_len + multiple - 1) // multiple * multiple

    bsz = len(batch)
    input_ids = torch.zeros(bsz, max_len, dtype=torch.long)
    labels = torch.full((bsz, max_len), IGNORE_INDEX, dtype=torch.long)
    attn = torch.zeros(bsz, max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].size(0)
        input_ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
        attn[i, :n] = b["attention_mask"]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def build_dataloaders(
    stage: str,
    tokenizer,
    data_path: str,
    val_path: Optional[str],
    max_seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    val_samples: int = 200,
    pack: bool = True,
    sft_dynamic_pad: bool = False,
    sft_bucket_chunk: int = 512,
) -> tuple:
    """按阶段（pretrain / sft）构造训练与验证 DataLoader。

    pack=True（默认）时，预训练启用**序列打包**：
    实测样本中位数仅 60 token，按单样本 pad 到 512 的填充利用率只有 17.6%，
    打包后接近 100%，算力效率提升约 5.7 倍。SFT 因需保留多轮对话结构，
    不做打包。

    sft_dynamic_pad=True 时，SFT 训练集改为**长度分桶 + 动态 pad**
    （见 SFTDataset 文档）。验证集始终用固定 pad —— 各次 val_loss 必须同口径。

    num_workers 默认 0：Windows 无 fork()，每个 worker 需重新 import torch 栈
    （约 300~500MB），且句柄回收滞后易造成系统 commit 内存枯竭。
    在本机 6GB 显存 / 16GB 内存条件下，单进程读取 + 流式管线更稳。
    """
    train_collate = collate
    if stage == "pretrain" and pack:
        train_ds = PackedPretrainDataset(data_path, tokenizer, max_seq_len=max_seq_len)
        # 验证集保持单样本 pad：使各次 val_loss 口径完全一致、可比（西瓜书 5.3）
        val_cls = PretrainDataset
    elif stage == "pretrain":
        train_ds = PretrainDataset(data_path, tokenizer, max_seq_len=max_seq_len)
        val_cls = PretrainDataset
    else:
        train_ds = SFTDataset(data_path, tokenizer, max_seq_len=max_seq_len,
                              dynamic_pad=sft_dynamic_pad,
                              bucket_chunk=sft_bucket_chunk if sft_dynamic_pad else 0)
        val_cls = SFTDataset
        if sft_dynamic_pad:
            train_collate = collate_pad

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=train_collate,
        drop_last=True,
    )

    val_loader = None
    if val_path:
        val_base = val_cls(val_path, tokenizer, max_seq_len=max_seq_len, shuffle_buffer=1)
        val_ds = ValSubset(val_base, num_samples=val_samples)
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=0,          # 验证常驻主进程，避免反复 spawn
            pin_memory=False,
            collate_fn=collate,
            drop_last=False,
        )
    return train_loader, val_loader