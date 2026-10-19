# -*- coding: utf-8 -*-
"""节律休息（duty-cycle rest guard）—— 自研栈 sunny-cx

目标
----
长时间连续训练（本项目 0.5B 需 4~6 天）对 6GB 的 GTX 1060 是持续高负载。
与 `thermal.py` 的**被动**熔断（等温度真的超限才停）不同，本模块是
**主动**节律保护：每连续训练满 `work_sec`（默认 2 小时），主动停下来
休息 `rest_sec`（默认 30 分钟）再继续。

两者的分工很清楚：
  · thermal  —— 异常保护。温度真的越界才触发，是"兜底"。
  · rest     —— 节律保护。无论温度多正常，到点就休，是"轮换"。
  短时 bench 实测 100% 利用率下 73~74℃，但 **24.8h 长跑稳态峰值达 81℃**
  （距 82℃ 熔断线仅 1℃）→ 机箱热堆积是小时级过程，短采样测不到。
  本轮长跑 thermal 仍未触发（熔断 0 次），但当室温升高或散热积灰后，
  它是有可能真正介入的 —— 两个机制**都需要**，不是二选一。

设计原则（与 thermal 保持同一套工程约守）
----------------------------------------
1. **零依赖训练核心**：只做时间算术、只请示"现在该不该休息"，
   不碰模型/优化器/数据。真正决定停与不停的是 train loop 的调用方。
2. **不阻塞主线程的状态机**：每个 optimizer step 调用 `guard()`
   （纯算术，无 subprocess，开销可忽略）；到点返回 rest=True，
   调用方调 `wait_rest()` 阻塞休息，休满自动恢复。
3. **失败安全**：任何异常都不拦训练 —— 休息是"锦上添花"，
   绝不能因为计时器出问题让 4 天的训练停摆。
4. **全程可观测**：每次休息/恢复都写日志 + 通过 on_event 下发（供大屏显示），
   并把累计休息时长与次数记入 state。
5. **跨进程可续**：累计训练时长落盘（见 state_path），
   进程重启/续训后不会"重新获得一个完整 2 小时" ——
   否则反复重启就能绕过节律保护，失去意义。

⚠ 与 thermal 一致的**关键正确性要求**：休息时长绝不能被计入训练时间。
   调用方在 `wait_rest()` 返回后必须把 elapsed 计时基准向后平移
   （trainer.py 里 `t0 += 实际休息秒数`），否则：
     · 日志 elapsed 虚高
     · ETA 估算整体偏长
     · 且累计训练时长会把休息算进去 → 休息越来越频繁（正反馈失控）
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass
class RestArgs:
    """节律休息参数。"""
    enable: bool = True
    work_sec: float = 2 * 3600.0     # 连续训练满这么久 → 休息（默认 2 小时）
    rest_sec: float = 30 * 60.0      # 每次休息时长（默认 30 分钟）
    max_tick_sec: float = 120.0      # 单次累积上限：防系统挂起被误算成训练
    state_path: str = ""             # 累计状态持久化路径（空=仅进程内）
    log_every_sec: float = 300.0     # 休息期间日志间隔（默认每 5 分钟报一次）
    quiet: bool = False


@dataclass
class _RestState:
    active_accum: float = 0.0        # 自上次休息以来累计的**训练**秒数
    rest_count: int = 0              # 已休息次数
    rest_total_sec: float = 0.0      # 累计休息总秒数
    resting: bool = False
    last_rest_at: Optional[float] = None


class RestGuard:
    """节律休息状态机。

    用法（训练循环内，与 thermal 完全同构）::

        rest = RestGuard(RestArgs(), on_event=cb)
        ...
        while True:
            r = rest.guard()            # 每个 optimizer step 调一次（纯算术）
            if r and r.get("rest"):
                _ts = time.time()
                rest.wait_rest()        # 阻塞休息 rest_sec 秒
                t0 += time.time() - _ts  # ⚠ 把休息时间从训练耗时里扣除
            ...继续训练...
    """

    def __init__(self, args: Optional[RestArgs] = None,
                 on_event: Optional[Callable[[dict], None]] = None,
                 log: Optional[Callable[[str], None]] = None):
        self.args = args or RestArgs()
        self.on_event = on_event
        self._log = log or (lambda s: print(s, flush=True))
        self.s = _RestState()
        self._last_tick: Optional[float] = None
        if self.args.enable:
            self._load_state()
            self._log(
                f"[rest] 节律休息已启用：连续训练 {self.args.work_sec/3600:.2f}h "
                f"→ 休息 {self.args.rest_sec/60:.0f}min"
                + (f" · 已有累计 {self.s.active_accum/3600:.2f}h / 已休 "
                   f"{self.s.rest_count} 次" if self.s.active_accum > 0 or self.s.rest_count else "")
            )

    # ---------------------------------------------------------------- 持久化
    def _load_state(self) -> None:
        """读取累计状态。任何失败都静默降级（不拦训练）。"""
        p = self.args.state_path
        if not p:
            return
        try:
            f = Path(p)
            if f.exists():
                d = json.loads(f.read_text(encoding="utf-8"))
                self.s.active_accum = float(d.get("active_accum", 0.0))
                self.s.rest_count = int(d.get("rest_count", 0))
                self.s.rest_total_sec = float(d.get("rest_total_sec", 0.0))
                ts = d.get("last_rest_at")
                self.s.last_rest_at = float(ts) if ts else None
        except Exception:                                          # noqa: BLE001
            pass

    def _save_state(self) -> None:
        p = self.args.state_path
        if not p:
            return
        try:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            Path(p).write_text(json.dumps({
                "active_accum": round(self.s.active_accum, 1),
                "rest_count": self.s.rest_count,
                "rest_total_sec": round(self.s.rest_total_sec, 1),
                "last_rest_at": self.s.last_rest_at,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:                                          # noqa: BLE001
            pass

    # ---------------------------------------------------------------- helpers
    def _emit(self, ev: dict) -> None:
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:                                      # noqa: BLE001
                pass

    @property
    def resting(self) -> bool:
        return self.s.resting

    def snapshot(self) -> dict:
        """供大屏推送：是否在休息 / 累计训练时长 / 已休次数。"""
        return {
            "rest_enabled": self.args.enable,
            "resting": self.s.resting,
            "rest_count": self.s.rest_count,
            "rest_total_sec": round(self.s.rest_total_sec, 1),
            "rest_active_sec": round(self.s.active_accum, 1),
            "rest_work_sec": self.args.work_sec,
            "rest_sec": self.args.rest_sec,
        }

    def _tick(self, now: float) -> None:
        """累积真实训练时间（单次增量设上限，防长阻塞被误算）。"""
        if self._last_tick is None:
            self._last_tick = now
            return
        dt = now - self._last_tick
        self._last_tick = now
        if dt < 0:
            return
        self.s.active_accum += min(dt, self.args.max_tick_sec)

    # ------------------------------------------------------------------ 核心
    def guard(self, now: Optional[float] = None) -> Optional[dict]:
        """训练步内的轻量检查（纯算术，无 subprocess）。

        返回 None 表示"还不到点"；返回 dict 且 rest=True 表示**该休息了**，
        调用方应立即停下并调用 wait_rest()。
        """
        if not self.args.enable:
            return None
        now = now if now is not None else time.time()
        self._tick(now)
        if self.s.active_accum < self.args.work_sec:
            return None
        # 到点：立刻清零累计，避免同一步被重复触发
        due_h = self.s.active_accum / 3600.0
        self.s.active_accum = 0.0
        self.s.rest_count += 1
        self.s.resting = True
        self._log(f"[rest] 🟡 已连续训练 {due_h:.2f}h（阈值 "
                  f"{self.args.work_sec/3600:.2f}h）→ 休息 "
                  f"{self.args.rest_sec/60:.0f}min 后自动继续")
        self._emit({"type": "rest_begin", "trained_sec": round(due_h * 3600, 1),
                    "rest_sec": self.args.rest_sec,
                    "rest_count": self.s.rest_count})
        self._save_state()
        return {"rest": True, **self.snapshot()}

    def wait_rest(self, sleep_fn: Optional[Callable[[float], None]] = None) -> float:
        """休息 rest_sec 秒后自动恢复。返回实际休息秒数。

        与 thermal.wait_cool() 的区别：那个是"等到温度达标"（条件轮询），
        这个是"等够固定时长"（定时）。故此处按秒睡，分段以便打日志。
        """
        if not self.s.resting:
            return 0.0
        sleeper = sleep_fn or time.sleep
        total = max(0.0, self.args.rest_sec)
        seg = max(1.0, self.args.log_every_sec)
        waited = 0.0
        while waited < total:
            chunk = min(seg, total - waited)
            sleeper(chunk)
            waited += chunk
            if waited < total:
                self._log(f"[rest] 休息中 {waited/60:.1f}/{total/60:.0f}min "
                          f"（训练已暂停，GPU 空闲）")
        self.s.resting = False
        self.s.rest_total_sec += waited
        self.s.last_rest_at = time.time()
        # ⚠ 重置计时基准：否则下次 guard() 会把这段休息算进训练时长
        self._last_tick = None
        self._log(f"[rest] 🟢 休息结束（{waited/60:.0f}min），恢复训练 · "
                  f"累计已休 {self.s.rest_count} 次 / "
                  f"{self.s.rest_total_sec/3600:.2f}h")
        self._emit({"type": "rest_end", "rested_sec": round(waited, 1),
                    "rest_count": self.s.rest_count})
        self._save_state()
        return waited

    def close(self) -> None:
        """收尾：写一次汇总事件与落盘。"""
        try:
            self._save_state()
            if self.s.rest_count:
                self._log(f"[rest] 会话统计：已休 {self.s.rest_count} 次 · "
                          f"累计 {self.s.rest_total_sec/3600:.2f}h")
                self._emit({"type": "rest_summary", **self.snapshot()})
        except Exception:                                          # noqa: BLE001
            pass