# -*- coding: utf-8 -*-
"""训练状态解析 —— 只读本地文件，不依赖任何云端服务。

数据源（全部在本机仓库内）：
  logs/<run>.log                     进度行 / 验证行 / 休息行
  checkpoints/<run>_latest.json      最新 step、best_val、存盘时间
  checkpoints/<run>_reststate.json   累计休息次数与时长

设计约束：
  - 纯标准库；不 import torch、不调用 nvidia-smi
  - 任何解析失败都降级为空值，绝不抛异常打断服务
  - 只读日志尾部若干 KB，日志涨到几百 MB 也不拖慢
  - 不区分训练框架：只认 stdout 里的事实行
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
CKPT = ROOT / "checkpoints"

TAIL_BYTES = 512 * 1024
# 日志多久没更新就判为「已停止」。休息期最长 30min，其间每 5min 有一次 tick，
# 所以 5min 阈值不会把正常休息误判成停摆。
STOP_AFTER_SEC = 300

# Epoch:[1/1](3220/13212), loss: 3.1147, lr: 0.00017713, gnorm: 0.622,
#   elapsed: 857.6min, peak: 3.17/6.00GB
RE_STEP = re.compile(
    r"Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\),\s*"
    r"loss:\s*([0-9.]+),\s*"
    r"lr:\s*([0-9.eE+-]+),\s*"
    r"gnorm:\s*([0-9.]+),\s*"
    r"elapsed:\s*([0-9.]+)min,\s*"
    r"peak:\s*([0-9.]+)/([0-9.]+)GB"
)

# [VAL] step 3200/13212  train_loss: 2.2316  val_loss: 2.9653  gap: -0.7337  best: 2.9653
RE_VAL = re.compile(
    r"\[VAL\]\s*step\s*(\d+)/(\d+)\s+"
    r"train_loss:\s*([0-9.]+)\s+"
    r"val_loss:\s*([0-9.]+)\s+"
    r"gap:\s*(-?[0-9.]+)\s+"
    r"best:\s*([0-9.]+)"
)

RE_REST_BEGIN = re.compile(r"\[rest\].*已连续训练")
RE_REST_TICK = re.compile(r"\[rest\]\s*休息中\s*([0-9.]+)/([0-9.]+)min")
RE_REST_END = re.compile(r"\[rest\].*(?:休息结束|已恢复训练)")
RE_BEST = re.compile(r"\[early-stop\]\s*新最优\s*val=([0-9.]+)")


def _runs() -> list[str]:
    """可用的 run 名：日志与检查点两边的并集。"""
    names: set[str] = set()
    if LOGS.is_dir():
        names.update(p.stem for p in LOGS.glob("*.log"))
    if CKPT.is_dir():
        for p in CKPT.glob("*_latest.json"):
            names.add(p.name[: -len("_latest.json")])
    return sorted(names)


def _tokens(name: str) -> set[str]:
    return {t for t in name.replace("-", "_").split("_") if t}


def _resolve(dir_: Path, run: str, suffix: str) -> Path | None:
    """把 run 名映射到实际文件。

    启动脚本里日志名与权重名前缀并不一致 —— `--save_weight
    sunnycx_0.5b_pretrain_v2` 配的是 `logs/pretrain_0.5b_v2.log`。
    因此先试同名，再按词元重合度取最高分（至少重合 2 个词元，
    否则宁可返回 None 也不张冠李戴）。
    """
    exact = dir_ / f"{run}{suffix}"
    if exact.is_file():
        return exact

    base = _tokens(run)
    best, best_score = None, 1
    for cand in dir_.glob(f"*{suffix}"):
        stem = cand.name[: -len(suffix)] if suffix else cand.stem
        score = len(base & _tokens(stem))
        if score > best_score:
            best, best_score = cand, score
    return best


def _tail_lines(path: Path) -> list[str]:
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > TAIL_BYTES:
            f.seek(size - TAIL_BYTES)
            f.readline()                      # 从半行处开始，丢掉
        raw = f.read()
    return raw.decode("utf-8", errors="replace").splitlines()


def _read_json(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_run(run: str) -> dict:
    """把一个 run 的全部可观测状态整理成一个扁平 dict，供前端直接渲染。"""
    log_path = _resolve(LOGS, run, ".log")
    latest = _read_json(_resolve(CKPT, run, "_latest.json"))
    reststate = _read_json(_resolve(CKPT, run, "_reststate.json"))

    out: dict = {
        "run": run,
        "log_file": log_path.name if log_path else None,
        "ckpt_file": (_resolve(CKPT, run, "_latest.json").name
                      if _resolve(CKPT, run, "_latest.json") else None),
        "log_exists": bool(log_path),
        "log_age_sec": None,
        "log_bytes": None,
        "state": "unknown",          # training | resting | stopped | unknown
        "rest_timer": None,          # [已休分钟, 总分钟]
        "epoch": None,
        "epochs": None,
        "step": None,
        "total_steps": None,
        "progress": None,            # 0~1
        "loss": None,
        "lr": None,
        "gnorm": None,
        "elapsed_min": None,
        "peak_gb": None,
        "peak_total_gb": None,
        "sec_per_step": None,
        "eta_hours": None,
        "best_val": latest.get("best_val"),
        "best_step": latest.get("best_step"),
        "saved_at": latest.get("saved_at"),
        "rest_count": reststate.get("rest_count"),
        "rest_total_hours": (
            round(reststate["rest_total_sec"] / 3600, 2)
            if isinstance(reststate.get("rest_total_sec"), (int, float)) else None
        ),
        "curve": [],                 # [{step, loss, lr}]
        "vals": [],                  # [{step, train, val, gap, best}]
    }

    if log_path is None:
        return out

    out["log_bytes"] = log_path.stat().st_size
    age = max(0.0, time.time() - log_path.stat().st_mtime)
    out["log_age_sec"] = round(age, 1)

    resting = False
    for line in _tail_lines(log_path):
        m = RE_STEP.search(line)
        if m:
            epoch, epochs, step, total = (int(x) for x in m.groups()[:4])
            out.update(
                epoch=epoch, epochs=epochs, step=step, total_steps=total,
                loss=float(m.group(5)), lr=float(m.group(6)),
                gnorm=float(m.group(7)), elapsed_min=float(m.group(8)),
                peak_gb=float(m.group(9)), peak_total_gb=float(m.group(10)),
            )
            if len(out["curve"]) < 2000:
                out["curve"].append({"step": step, "loss": float(m.group(5))})
            continue

        m = RE_VAL.search(line)
        if m:
            out["vals"].append({
                "step": int(m.group(1)),
                "train": float(m.group(3)),
                "val": float(m.group(4)),
                "gap": float(m.group(5)),
                "best": float(m.group(6)),
            })
            continue

        if RE_REST_BEGIN.search(line):
            resting, out["rest_timer"] = True, None
            continue

        m = RE_REST_TICK.search(line)
        if m:
            resting = True
            out["rest_timer"] = [float(m.group(1)), float(m.group(2))]
            continue

        if RE_REST_END.search(line):
            resting, out["rest_timer"] = False, None
            continue

        m = RE_BEST.search(line)
        if m:                                   # 兜底：从 best 行补 best_val
            out["best_val"] = float(m.group(1))

    if resting:
        out["state"] = "resting"
    elif age <= STOP_AFTER_SEC:
        out["state"] = "training"
    else:
        out["state"] = "stopped"

    step, total, elapsed = out["step"], out["total_steps"], out["elapsed_min"]
    if step and total:
        out["progress"] = round(step / total, 5)
        if elapsed and step > 0:
            # elapsed 是墙钟时间（含休息），因此 ETA 天然把未来的休息算进去了
            sec_per_step = elapsed * 60 / step
            out["sec_per_step"] = round(sec_per_step, 2)
            out["eta_hours"] = round((total - step) * sec_per_step / 3600, 2)
    return out


def latest_run() -> str | None:
    """默认展示最近写过的那个 run。"""
    candidates = []
    for name in _runs():
        p = LOGS / f"{name}.log"
        if p.is_file():
            candidates.append((p.stat().st_mtime, name))
    if candidates:
        return max(candidates)[1]
    names = _runs()
    return names[-1] if names else None
