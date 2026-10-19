"""量化每步固定开销：nvidia-smi 子进程 + 推送序列化。"""
import subprocess, time, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

# 1) nvidia-smi 单次调用延迟（thermal.guard 每步调用）
ts = time.time()
for _ in range(10):
    subprocess.run(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader"],
                   capture_output=True, text=True)
t_smi = (time.time()-ts)/10
print(f"nvidia-smi 单次调用 : {t_smi*1000:7.0f} ms")

# 2) torch.cuda 内存查询（push 回调每步调用）
import torch
torch.zeros(1, device="cuda")
ts = time.time()
for _ in range(100):
    a = torch.cuda.max_memory_allocated(); b = torch.cuda.memory_allocated()
t_mem = (time.time()-ts)/100
print(f"torch.cuda 显存查询  : {t_mem*1000:7.2f} ms")
print()
print(f"合计每步固定开销 ≈ {(t_smi+t_mem)*1000:.0f} ms")
print(f"对比 17.4 s/step     ≈ {(t_smi+t_mem)/17.4*100:.1f}%")
