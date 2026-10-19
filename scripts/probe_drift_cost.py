"""微基准：精确量化 weight_drift() 每步的 CPU 往返开销。

不加载模型，只复刻 drift 的数据搬运与算术规模：
  · 抽样 50 个张量，fp16 合计约 366MB（快照上限口径）
  · 每步执行 p.detach().to("cpu", float32) → (w - w0).abs() → pow(2).sum()
"""
import torch, time

# 按 0.5B 实际抽样结构近似：attn/mlp/norm 混合，合计约 366MB fp16
SIZES = []
tot = 0
for i in range(50):
    n = 3_800_000 if i % 3 == 1 else 1_000_000   # mlp 大、attn/norm 小
    SIZES.append(n); tot += n
print(f"模拟 {len(SIZES)} 个张量，合计 {tot*2/1024**2:.0f} MB (fp16)")

gpu = [torch.randn(n, device="cuda", dtype=torch.float16) for n in SIZES]
w0  = [torch.randn(n, dtype=torch.float32) for n in SIZES]   # 快照常驻 CPU/fp32

def drift_once():
    tot_d = 0.0; sq_d = 0.0; sq_w = 0.0
    for g, f in zip(gpu, w0):
        w = g.detach().to("cpu", torch.float32)
        d = (w - f).abs()
        tot_d += float(d.sum()); sq_d += float(d.pow(2).sum()); sq_w += float(w.pow(2).sum())
    return tot_d, sq_d, sq_w

# 分阶段计时
torch.cuda.synchronize()
ts = time.time(); _ = [g.to("cpu", torch.float32) for g in gpu]; t_move = time.time() - ts
print(f"  仅 GPU→CPU 搬运+转fp32 : {t_move*1000:7.0f} ms")

R = 10
for _ in range(2): drift_once()
torch.cuda.synchronize()
ts = time.time()
for _ in range(R): drift_once()
torch.cuda.synchronize()
t_all = (time.time() - ts) / R
print(f"  完整 drift_once()      : {t_all*1000:7.0f} ms")
print()
print(f"实测训练 17.4 s/step ；基准 13.0 s/step ；差 4.4 s")
print(f"drift 每步占训练时间   : {t_all/17.4*100:.1f}%")
