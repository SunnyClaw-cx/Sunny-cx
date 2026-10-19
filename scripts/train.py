"""sunny-cx 训练入口

用法：
    python scripts/train.py --stage pretrain --config 0.5b --dry-run
    python scripts/train.py --stage pretrain --config 0.5b
    python scripts/train.py --stage sft --config 0.5b --from-weight checkpoints/sunnycx_pretrain_best.pth
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 行缓冲：stdout 重定向到文件时 Python 默认用 4~8KB 块缓冲，会导致
# 日志长时间不落盘（看似"训练无输出"）。长跑训练的进度必须实时可见。
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

from sunnycx.config import SunnyConfig
from sunnycx.dataset import build_dataloaders
from sunnycx.model import SunnyForCausalLM, verify_param_count
from sunnycx.trainer import (
    Resumer, TrainArgs, evaluate, load_pretrained_weights, train,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("sunny-cx trainer")
    p.add_argument("--stage", choices=["pretrain", "sft"], default="pretrain")
    p.add_argument("--config", default="0.5b", help="架构预设名，或指向 config json")

    # 架构超参（覆盖预设）
    p.add_argument("--hidden_size", type=int, default=None)
    p.add_argument("--num_hidden_layers", type=int, default=None)
    p.add_argument("--num_attention_heads", type=int, default=None)
    p.add_argument("--num_key_value_heads", type=int, default=None)

    # 数据
    p.add_argument("--data_path", default="")
    p.add_argument("--val_data_path", default="")
    p.add_argument("--data_dir", default=str(ROOT.parent / "tcm-llm" / "data" / "processed_v2"),
                   help="未指定 --data_path 时使用的数据目录（默认 processed_v2，"
                        "即修完碎片 bug 并重跑的书级切分语料）")
    p.add_argument("--tokenizer_path", default="")
    p.add_argument("--save_dir", default=str(ROOT / "checkpoints"))
    p.add_argument("--save_weight", default="")
    p.add_argument("--from_weight", default="")

    # 超参
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--model_dtype", default="auto",
                   choices=["auto", "fp32", "fp16", "bf16"],
                   help="权重驻留精度；auto=CUDA 上 fp16、CPU 上 fp32（0.5B 容于 6GB 的关键）")
    p.add_argument("--loss_scale", type=float, default=1024.0,
                   help="fp16 权重驻留时的静态损失缩放系数（须为 2 的幂）")
    p.add_argument("--val_interval", type=int, default=200)
    p.add_argument("--val_samples", type=int, default=200)
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=0,
                   help=">0 时提前停止（冒烟/调试用）")
    p.add_argument("--no_pack", action="store_true",
                   help="禁用序列打包（预训练默认启用；打包可提升约 5.7 倍算力效率）")
    p.add_argument("--no_sft_bucket", action="store_true",
                   help="SFT 关闭长度分桶（默认开启：动态 pad 到 batch 最长，"
                        "实测可省约 4~5 倍 padding 算力）")
    p.add_argument("--sft_bucket_chunk", type=int, default=512,
                   help="SFT 分桶块大小，块内按长度降序排列（默认 512）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # 开关
    p.add_argument("--no_8bit_optim", action="store_true",
                   help="禁用 8-bit AdamW（显存充足时可用 fp32）")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="启用梯度检查点（省激活显存，慢约 30%%）")
    p.add_argument("--from_resume", action="store_true", help="从 latest 检查点续训")
    p.add_argument("--dry-run", action="store_true",
                   help="只校验模型/数据形状与显存，不真正训练")
    p.add_argument("--no_push", action="store_true",
                   help="关闭云端大屏推送（默认开启推送；离线训练时加此项）")
    p.add_argument("--push_stream", default="",
                   help="云端数据流名，默认取 cloud_config.json 的 stream（sunny-cx）")
    # ---- GPU 温度熔断 ----
    p.add_argument("--no_thermal", action="store_true",
                   help="关闭 GPU 温度熔断（默认开启：超温暂停、降温自恢复）")
    p.add_argument("--thermal_pause_c", type=float, default=82.0,
                   help="达到该温度并持续 hold 秒后暂停训练（默认 82℃）")
    p.add_argument("--thermal_resume_c", type=float, default=74.0,
                   help="降到此温度以下后自动恢复训练（默认 74℃，须低于暂停阈值）")
    p.add_argument("--thermal_hold_sec", type=float, default=20.0,
                   help="超温需持续多久才真正熔断，抵御瞬时尖峰（默认 20s）")
    p.add_argument("--thermal_poll_sec", type=float, default=5.0,
                   help="熔断后温度轮询间隔（默认 5s）")
    # ---- 节律休息（主动：到点就休，与被动测温熔断互补）----
    p.add_argument("--no_rest", action="store_true",
                   help="关闭节律休息（默认开启：连续训练满 2h 休息 30min 再继续）")
    p.add_argument("--rest_work_hours", type=float, default=2.0,
                   help="连续训练满多少小时触发休息（默认 2.0）")
    p.add_argument("--rest_minutes", type=float, default=30.0,
                   help="每次休息多少分钟（默认 30.0）")
    return p.parse_args()


def resolve_config(args) -> SunnyConfig:
    """解析架构配置：支持预设名、json 文件路径、以及命令行覆盖。"""
    src = args.config
    if Path(src).exists():
        cfg = SunnyConfig.load(src)
        print(f"[config] 从 {src} 载入")
    else:
        cfg = SunnyConfig.preset(src)
        print(f"[config] 使用预设 {src!r}")

    for k in ("hidden_size", "num_hidden_layers",
              "num_attention_heads", "num_key_value_heads"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg


def main() -> int:
    args = parse_args()

    # ---- 随机种子（西瓜书 5.4：多组初始化以降低局部极小风险）----
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")
    if device.type == "cuda":
        print(f"[env] GPU={torch.cuda.get_device_name(0)}  "
              f"cap={torch.cuda.get_device_capability(0)}  "
              f"显存={torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
        torch.backends.cuda.matmul.allow_tf32 = False

    cfg = resolve_config(args)
    if not args.save_weight:
        args.save_weight = f"sunnycx_{args.stage}"

    # ---- 参数量校验 ----
    print("=" * 72)
    verify_param_count(cfg)
    print("=" * 72)

    model = SunnyForCausalLM(cfg)
    n_params = model.num_parameters
    print(f"[model] 实例化完成，参数量 {n_params:,} ({n_params/1e6:.2f}M)")

    # ---- tokenizer ----
    from transformers import AutoTokenizer
    if not args.tokenizer_path:
        args.tokenizer_path = str(ROOT.parent / "tcm-llm" / "model_tcm_tokenizer")
    if not Path(args.tokenizer_path).exists():
        print(f"[error] tokenizer 路径不存在：{args.tokenizer_path}")
        return 1
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"[tokenizer] vocab={tokenizer.vocab_size}  "
          f"eos={tokenizer.eos_token_id}  path={args.tokenizer_path}")

    # ---- 数据 ----
    if not args.data_path:
        base = Path(args.data_dir)
        if args.stage == "pretrain":
            args.data_path = str(base / "tcm_classics_pretrain.train.jsonl")
            args.val_data_path = str(base / "tcm_classics_pretrain.val.jsonl")
        else:
            args.data_path = str(base / "tcm_classics_qa.train.jsonl")
            args.val_data_path = str(base / "tcm_classics_qa.val.jsonl")
        print(f"[data] 未指定路径，用默认数据目录 {base}")
    print(f"[data] train={args.data_path}")
    print(f"[data] val  ={args.val_data_path}")

    train_loader, val_loader = build_dataloaders(
        stage=args.stage,
        tokenizer=tokenizer,
        data_path=args.data_path,
        val_path=args.val_data_path or None,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_samples=args.val_samples,
        pack=(not args.no_pack),
        sft_dynamic_pad=(args.stage == "sft" and not args.no_sft_bucket),
        sft_bucket_chunk=args.sft_bucket_chunk,
    )
    if args.stage == "pretrain" and not args.no_pack:
        print("[data] 预训练启用序列打包（中位样本仅 60 tok，打包可省 ~82% padding 算力）")
    if args.stage == "sft":
        if args.no_sft_bucket:
            print(f"[data] SFT 固定 pad 到 {args.max_seq_len}（未分桶）"
                  f"· 有效 token 仅约 18%")
        else:
            print(f"[data] SFT 长度分桶 + 动态 pad（块 {args.sft_bucket_chunk}，"
                  f"上限 {args.max_seq_len}）· 验证集仍用固定 pad 保持口径一致")

    # ---- 训练参数 ----
    targs = TrainArgs(
        stage=args.stage,
        data_path=args.data_path,
        val_data_path=args.val_data_path,
        tokenizer_path=args.tokenizer_path,
        save_dir=args.save_dir,
        save_weight=args.save_weight,
        epochs=int(args.epochs),
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_seq_len=args.max_seq_len,
        dtype=args.dtype,
        val_interval=args.val_interval,
        val_samples=args.val_samples,
        early_stop_patience=args.early_stop_patience,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        num_workers=args.num_workers,
        max_steps=args.max_steps,
        use_8bit_optim=not args.no_8bit_optim,
        model_dtype=args.model_dtype,
        loss_scale=args.loss_scale,
        gradient_checkpointing=args.gradient_checkpointing,
        from_resume=args.from_resume,
        from_weight=args.from_weight,
        seed=args.seed,
    )
    Path(targs.save_dir).mkdir(parents=True, exist_ok=True)
    cfg.save(Path(targs.save_dir) / f"{targs.save_weight}_config.json")

    # ---- 冒烟测试 ----
    if args.dry_run:
        return _dry_run(model, train_loader, val_loader, targs, device)

    # ---- 初始权重注入：SFT 参数迁移 / 预训练热启动 ----
    # 预训练也允许 from_weight：用于「已训步数不再重跑、但要完整覆盖新语料」的场景。
    # 与 --from_resume 的区别：from_resume 会继承 step/epoch/early_stop 并**跳过**已训步数；
    # from_weight 只搬参数，优化器/步数/早停计数全部从零开始 → 新语料 100% 覆盖。
    if targs.from_weight:
        if load_pretrained_weights(model, targs.from_weight, device) and targs.stage == "pretrain":
            print("[pretrain] 热启动：权重已注入；优化器 / 步数 / 早停计数从零开始")

    # ---- 估算总步数 ----
    if args.stage == "pretrain" and not args.no_pack:
        # 打包后：每 step 消耗 batch_size × max_seq_len 个 token（利用率≈100%）
        total_tokens = _count_tokens(args.data_path, tokenizer)
        steps_per_epoch = max(1, total_tokens // (
            targs.batch_size * targs.max_seq_len * targs.accumulation_steps))
        print(f"[plan] 语料 {total_tokens/1e8:.3f} 亿 token（打包）  "
              f"每 epoch {steps_per_epoch:,} optimizer step  "
              f"合计 {steps_per_epoch*targs.epochs:,}")
    else:
        # 流式数据集无 len()，用文件行数估算
        n_lines = _count_lines(targs.data_path)
        steps_per_epoch = max(1, n_lines // (targs.batch_size * targs.accumulation_steps))
        print(f"[plan] 训练样本≈{n_lines:,} 条  每 epoch "
              f"{steps_per_epoch:,} optimizer step  合计 {steps_per_epoch*targs.epochs:,}")
    total_steps = steps_per_epoch * targs.epochs

    resumer = Resumer(targs.save_dir, targs.save_weight) if targs.from_resume else None

    # ---- 云端大屏推送（尽力而为，绝不打断训练）----
    on_step = _build_push_callback(model, tokenizer, targs, args, total_steps)

    # ---- GPU 温度熔断（超温暂停 → 降温自动恢复）----
    thermal = _build_thermal(args)

    # ---- 节律休息（连续训练 2h → 休息 30min → 继续）----
    rest = _build_rest(args, targs)

    result = train(model, targs, train_loader, val_loader, device,
                   total_steps, resumer=resumer, on_step=on_step,
                   thermal=thermal, rest=rest)

    print("=" * 72)
    print(f"[done] final_step={result['final_step']}  "
          f"best_val={result['best_val']:.4f} @ step {result['best_step']}  "
          f"early_stopped={result['stopped_early']}")
    if thermal is not None:
        thermal.close()
    if rest is not None:
        rest.close()
    if on_step is not None:
        _finish_push(model, targs, result)
    return 0


def _build_thermal(args):
    """装配 GPU 温度熔断。返回 None 表示未启用/不可用。

    与云端推送一样，它挂在训练循环的 **step 边界**：
    optimizer.step() 之后梯度已清零、参数已落地，此处暂停不持有中间状态，
    恢复时可直接继续，不需要重放任何一个 step。
    """
    if getattr(args, "no_thermal", False):
        print("[thermal] 已按 --no_thermal 关闭温度熔断")
        return None
    try:
        from sunnycx.thermal import ThermalArgs, ThermalGuard
    except Exception as e:                                     # noqa: BLE001
        print(f"[thermal] 模块不可用：{e}")
        return None

    if args.thermal_resume_c >= args.thermal_pause_c:
        print(f"[thermal] ⚠ 恢复阈值({args.thermal_resume_c:.0f}℃) 不低于 "
              f"暂停阈值({args.thermal_pause_c:.0f}℃)，会造成反复启停；"
              f"已自动下调为 {args.thermal_pause_c - 8:.0f}℃")
        args.thermal_resume_c = args.thermal_pause_c - 8

    targs = ThermalArgs(
        enable=True,
        pause_c=args.thermal_pause_c,
        resume_c=args.thermal_resume_c,
        hold_sec=args.thermal_hold_sec,
        poll_sec=args.thermal_poll_sec,
    )
    return ThermalGuard(targs)


def _build_rest(args, targs: TrainArgs):
    """装配节律休息。返回 None 表示未启用/不可用。

    与温度熔断一样挂在训练循环的 **step 边界**：optimizer.step() 之后
    梯度已清零、参数已落地，此处暂停不持有中间状态，恢复时可直接继续，
    不需要重放任何一个 step。

    累计状态落盘到 checkpoints/<save_weight>_reststate.json ——
    续训/重启后不会"重新获得一个完整的 2 小时"，否则反复重启即可绕过。
    """
    if getattr(args, "no_rest", False):
        print("[rest] 已按 --no_rest 关闭节律休息")
        return None
    try:
        from sunnycx.rest import RestArgs, RestGuard
    except Exception as e:                                     # noqa: BLE001
        print(f"[rest] 模块不可用：{e}")
        return None

    if args.rest_work_hours <= 0 or args.rest_minutes <= 0:
        print(f"[rest] ⚠ 参数非正（work={args.rest_work_hours}h / "
              f"rest={args.rest_minutes}min），已停用节律休息")
        return None

    state_path = str(Path(targs.save_dir) / f"{targs.save_weight}_reststate.json")
    rargs = RestArgs(
        enable=True,
        work_sec=args.rest_work_hours * 3600.0,
        rest_sec=args.rest_minutes * 60.0,
        state_path=state_path,
    )
    return RestGuard(rargs)


def _build_push_callback(model, tokenizer, targs: TrainArgs, args, total_steps: int):
    """装配每步推送回调。返回 None 表示推送未启用。

    这条链路是**结构化**的：训练器直接把真实张量级指标交给回调，
    没有"解析日志文本"这一层——这正是它能适配自研栈、而
    早期派生栈的 tcm_hook（按另一套日志格式做正则）适配不了的原因。
    """
    if getattr(args, "no_push", False):
        print("[cloud] 已按 --no_push 关闭大屏推送")
        return None
    # 流名须在导入前设好：cloud 模块在 import 时即确定 STREAM
    if getattr(args, "push_stream", ""):
        os.environ["SUNNYCX_PUSH_STREAM"] = args.push_stream
    try:
        from sunnycx import cloud
    except Exception as e:                                     # noqa: BLE001
        print(f"[cloud] 推送模块不可用：{e}")
        return None
    if not cloud.ENABLED:
        print(f"[cloud] 未启用：{cloud.status_line()}")
        return None

    n = cloud.install_activation_probe(model)
    if n:
        print(f"[cloud] 真实激活探针已挂载：{n} 层 × {cloud.ACT_BINS} 光点")
    else:
        print("[cloud] ⚠ 未能挂载激活探针（大屏圆盘将退回示意数据）")
    facts = cloud.collect_model_facts(model, tokenizer)
    cloud.snapshot_weights(model)
    if facts:
        print(f"[cloud] 真实参数：{facts['params']:,} = {facts['params_m']}M"
              f"（层 {facts.get('layers')} · hidden {facts.get('hidden')} · "
              f"GQA {facts.get('heads')}:{facts.get('kv_heads')} · "
              f"vocab {facts.get('vocab')}）")
    print(f"[cloud] {cloud.status_line()}")

    # 启动即推一条 initializing，让大屏立刻切到"实时"而不是停在旧数据
    cloud.push_state({
        "status": "initializing", "stage": targs.stage,
        "loss": None, "global_step": 0, "total_steps": total_steps,
        "vocab": facts.get("vocab") if facts else None,
        "params_m": facts.get("params_m") if facts else None,
        "note": "sunny-cx 自研栈启动",
    }, attach_activations=False)

    def _cb(st: dict) -> None:
        try:
            drift = cloud.weight_drift(model)
            payload = {
                "status": "running", "stage": targs.stage,
                "step": st.get("step"),
                "iters": total_steps,          # 单 epoch 内的口径由大屏按 total 处理
                "global_step": st.get("step"),
                "total_steps": st.get("total_steps") or total_steps,
                "epoch": st.get("epoch"),
                "epochs": targs.epochs,
                "loss": st.get("loss"),
                "val_loss": st.get("val_loss"),
                "val_at_step": st.get("val_at_step"),
                "lr": st.get("lr"),
                "grad_norm": st.get("grad_norm"),
                "elapsed_sec": st.get("elapsed_sec"),
                "progress": (st.get("step") or 0) / max(total_steps, 1) * 100,
                "vocab": facts.get("vocab") if facts else None,
                "params_m": facts.get("params_m") if facts else None,
                "timestamp": time.strftime("%H:%M:%S"),
            }
            if drift:
                payload["drift"] = drift
            # 真实显存读数（大屏可据此显"峰值/上限"）
            if torch.cuda.is_available():
                payload["mem_peak_gb"] = round(
                    torch.cuda.max_memory_allocated() / 1024**3, 2)
                payload["mem_total_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            # GPU 温度熔断状态（真实 nvidia-smi 采样）→ 大屏可直接显示温度与暂停态
            for tk in ("gpu_temp", "gpu_peak_temp", "thermal_paused",
                       "thermal_pause_count", "thermal_pause_sec",
                       "thermal_pause_c", "thermal_resume_c", "thermal_enabled"):
                if st.get(tk) is not None:
                    payload[tk] = st.get(tk)
            # 节律休息状态（主动：连续训练到点休息）→ 大屏可显示"休息中/已休 N 次"
            for rk in ("resting", "rest_enabled", "rest_count", "rest_total_sec",
                       "rest_active_sec", "rest_work_sec", "rest_sec"):
                if st.get(rk) is not None:
                    payload[rk] = st.get(rk)
            cloud.push_state(payload)
        except Exception:                                      # noqa: BLE001
            pass                                                   # 推送失败不影响训练

    return _cb


def _finish_push(model, targs: TrainArgs, result: dict) -> None:
    """训练收尾：推一条 finished 并等待队列排空。"""
    try:
        from sunnycx import cloud
        cloud.push_state({
            "status": "finished", "stage": targs.stage,
            "loss": None,
            "global_step": result.get("final_step"),
            "best_val": result.get("best_val"),
            "best_step": result.get("best_step"),
            "stopped_early": result.get("stopped_early"),
            "note": "sunny-cx 自研栈训练结束",
        }, attach_activations=False)
        cloud.flush(timeout=25.0)
        print(f"[cloud] 末态已落库 · {cloud.status_line()}")
    except Exception:                                          # noqa: BLE001
        pass


def _dry_run(model, train_loader, val_loader, targs: TrainArgs, device) -> int:
    """不训练，只验证：数据形状、前向、单步反向、显存峰值。"""
    print("=" * 72)
    print("[dry-run] 开始冒烟测试（不更新参数）")
    dtype = torch.float16

    print("\n[1/4] 数据管线形状检查")
    for i, batch in enumerate(train_loader):
        print(f"  input_ids      {tuple(batch['input_ids'].shape)} {batch['input_ids'].dtype}")
        print(f"  labels         {tuple(batch['labels'].shape)}")
        print(f"  attention_mask {tuple(batch['attention_mask'].shape)}")
        n_ignore = (batch["labels"] == -100).sum().item()
        print(f"  被掩码(-100)比例: {n_ignore/batch['labels'].numel()*100:.1f}%")
        sample = batch["input_ids"][0][:24].tolist()
        print(f"  首条前24 token : {sample}")
        break

    print("\n[2/4] 前向传播")
    # 权重驻留精度与正式训练保持一致（CUDA 上 fp16），否则测出的显存偏大、
    # 会误判为 OOM —— 这是最容易踩的坑。
    mdt = torch.float16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=mdt)
    model.train()
    batch = next(iter(train_loader))
    ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    mask = batch["attention_mask"].to(device)
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None

    with torch.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
        out = model(ids, attention_mask=mask, labels=labels)
    print(f"  logits {tuple(out['logits'].shape)}  loss {out['loss'].item():.4f}")

    print("\n[3/4] 反向传播 + 优化器单步")
    import bitsandbytes as bnb
    from sunnycx.trainer import StaticScaler, clip_grad_norm_safe
    opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    # 与正式训练一致：fp16 权重用静态缩放（GradScaler 不支持 fp16 梯度）
    if mdt == torch.float16:
        scaler = StaticScaler(scale=1024.0)
    else:
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    scaler.scale(out["loss"]).backward()
    scaler.unscale_(opt)
    gn = clip_grad_norm_safe(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    print(f"  grad_norm {gn.item():.4f}   单步完成 OK")

    print("\n[4/4] 显存统计")
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  当前分配 : {alloc:.2f} GB")
        print(f"  峰值分配 : {peak:.2f} GB")
        print(f"  卡总显存 : {total:.2f} GB")
        print(f"  余量     : {total - peak:.2f} GB")
        if peak > total * 0.92:
            print("  [警告] 显存占用接近上限，建议启用 --gradient_checkpointing")
        else:
            print("  [OK] 显存余量充足")
    print("=" * 72)
    return 0


def _count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def _count_tokens(path: str, tokenizer) -> int:
    """统计语料 token 总量（用于打包后的步数估算）。

    抽样 2000 行估算平均 tok/条，再乘总行数——全量 tokenize 需数分钟，
    估算误差 <1%，对步数规划足够。结果会缓存到同目录 .tokcount 文件。
    """
    import json as _json
    cached = Path(path).with_suffix(Path(path).suffix + ".tokcount")
    n_lines = _count_lines(path)
    if cached.exists():
        try:
            v = int(cached.read_text().strip())
            if v > 0:
                return v
        except ValueError:
            pass

    step = max(1, n_lines // 2000)
    tot, cnt = 0, 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % step:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                t = _json.loads(line).get("text", "")
            except _json.JSONDecodeError:
                continue
            if t:
                tot += len(tokenizer.encode(t, add_special_tokens=False))
                cnt += 1
    est = int(tot / max(cnt, 1) * n_lines)
    try:
        cached.write_text(str(est))
    except OSError:
        pass
    return est


if __name__ == "__main__":
    raise SystemExit(main())