# -*- coding: utf-8 -*-
"""漂移趋势：逐个加载、及时释放，避免 segfault。"""
import sys, random, re, gc
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from sunnycx.config import SunnyConfig
from sunnycx.model import SunnyForCausalLM, verify_param_count

random.seed(42); torch.manual_seed(42)
cfg = SunnyConfig.preset("0.5b")
verify_param_count(cfg)
model = SunnyForCausalLM(cfg)
w0 = {n: p.detach().clone() for n, p in model.named_parameters()}
del model; gc.collect()

SD = re.compile(r"layers\.(\d+)\.")
ck_names = list(torch.load(ROOT/"checkpoints"/"sunnycx_0.5b_pretrain_best.pth",
                           map_location="cpu", weights_only=False).keys())
def sample(ns):
    groups, plain = {}, []
    for n in ns:
        m = SD.search(n)
        if m: groups.setdefault(int(m.group(1)), []).append(n)
        else: plain.append(n)
    keys = sorted(groups)
    if len(keys) > 16:
        idx = sorted({round(i*(len(keys)-1)/15) for i in range(16)})
        keys = [keys[i] for i in idx]
    picked = list(plain)
    for li in keys:
        for cat in ("attn","mlp","norm"):
            hit = next((g for g in groups[li] if cat in g), None)
            if hit: picked.append(hit)
    return picked

names = [n for n in w0 if n in ck_names]
picked = sample(names)
print(f"抽样 {len(picked)} 张量", flush=True)

def metrics(sd):
    sq_d=sq_w=0.0; tot_d=tot=0; moved=cnt=0; per={}
    for n in picked:
        w = sd[n].float(); f = w0[n].float()
        d = (w-f).abs()
        sq_d += float(d.pow(2).sum()); sq_w += float(w.pow(2).sum())
        tot_d += float(d.sum()); tot += d.numel()
        if float(d.mean())>1e-6: moved+=1
        cnt+=1
        cat = "mlp" if ".mlp." in n else ("attn" if ".attn." in n else "norm")
        c = per.get(cat,(0.0,0.0)); per[cat]=(c[0]+float(d.pow(2).sum()), c[1]+float(w.pow(2).sum()))
    return (sq_d/sq_w)**0.5*100, moved/cnt*100, tot_d/tot, per

res = {}
for tag, f, st, key in [("best", "sunnycx_0.5b_pretrain_best.pth", 4600, None),
                        ("latest", "sunnycx_0.5b_pretrain_latest.pt", 4800, "model")]:
    print(f"载入 {tag} …", flush=True)
    obj = torch.load(ROOT/"checkpoints"/f, map_location="cpu", weights_only=False)
    sd = obj[key] if key else obj
    short = {n: sd[n] for n in picked}
    del obj, sd; gc.collect()
    res[tag] = (st,) + metrics(short)
    del short; gc.collect()

print()
print("="*68)
print(f"{'检查点':<10}{'step':>7}{'rms_pct':>12}{'moved_pct':>12}{'mean_abs':>14}")
print("="*68)
for tag in ("best","latest"):
    st, rms, mv, ma, _ = res[tag]
    print(f"{tag:<10}{st:>7}{rms:>11.2f}%{mv:>11.1f}%{ma:>14.8f}")
print("="*68)

rms_b, rms_l = res["best"][1], res["latest"][1]
d = (rms_l-rms_b)/rms_b*100
print(f"\n【趋势】4600→4800（+200步）rms_pct: {rms_b:.2f}% → {rms_l:.2f}%  ({d:+.2f}%)")
print(f"【判据】随机游走 √step 预期 +2.2% ；线性预期 +4.3%")
print(f"  实测 {d:+.2f}% → {'符合随机游走（正常）' if abs(d)<3 else '偏快，需关注'}")

print("\n【归因】")
print(f"{'类别':<8}{'best@4600':>12}{'latest@4800':>14}{'增量':>10}")
for cat in sorted(res["latest"][4]):
    rb=(res["best"][4][cat][0]/res["best"][4][cat][1])**0.5*100
    rl=(res["latest"][4][cat][0]/res["latest"][4][cat][1])**0.5*100
    print(f"{cat:<8}{rb:>11.2f}%{rl:>13.2f}%{rl-rb:>+9.2f}")
