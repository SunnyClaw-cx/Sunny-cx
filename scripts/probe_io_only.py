"""只测数据加载耗时（完全不碰 GPU，可与训练进程并存）。"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transformers import AutoTokenizer
from sunnycx.dataset import build_dataloaders

tok = AutoTokenizer.from_pretrained(str(ROOT.parent / "tcm-llm" / "model_tcm_tokenizer"))
base = ROOT.parent / "tcm-llm" / "data" / "processed"
DATA = str(base / "tcm_classics_pretrain.train.jsonl")

def bench(num_workers, bs=2, N=25, warm=3):
    loader, _ = build_dataloaders(
        stage="pretrain", tokenizer=tok, data_path=DATA, val_path=None,
        max_seq_len=512, batch_size=bs, num_workers=num_workers,
        val_samples=64, pack=True,
    )
    it = iter(loader)
    for _ in range(warm):
        next(it)
    ts = time.time()
    for _ in range(N):
        next(it)
    dt = (time.time() - ts) / N * 1000
    return dt

print("=== 数据加载单 batch 耗时（batch=2, seq=512, 打包）===")
for nw in (0, 2, 4):
    try:
        dt = bench(nw)
        print(f"  num_workers={nw}: {dt:7.0f} ms/batch   （每 step 需 8 batch = {dt*8/1000:.1f} s）")
    except Exception as e:
        print(f"  num_workers={nw}: 失败 -> {type(e).__name__}: {e}")
