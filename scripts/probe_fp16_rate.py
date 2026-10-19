"""实测 GTX 1060 的 fp16 vs fp32 矩阵乘法吞吐（cuBLAS 路径，非 shader 通路）。"""
import torch, time
torch.backends.cuda.matmul.allow_tf32 = False

def bench(dtype, n=2048, R=50):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(5):
        c = a @ b
    torch.cuda.synchronize()
    ts = time.time()
    for _ in range(R):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.time() - ts) / R
    flops = 2 * n**3 / dt
    return dt, flops

print("GTX 1060 cuBLAS 实测（2048x2048 方阵）")
for name, dt_ in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16) if torch.cuda.is_bf16_supported() else ("bf16","skip")):
    if dt_ == "skip":
        print(f"  {name}: 不支持"); continue
    try:
        dt, f = bench(dt_)
        print(f"  {name}: {dt*1000:.2f} ms/次   {f/1e12:.3f} TFLOPS")
    except Exception as e:
        print(f"  {name}: 失败 {type(e).__name__}: {str(e)[:120]}")

print()
print("理论值（GP106 规格）：FP32 4.375 TFLOPS · FP16 68.36 GFLOPS (1:64)")
