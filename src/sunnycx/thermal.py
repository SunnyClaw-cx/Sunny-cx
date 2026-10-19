# -*- coding: utf-8 -*-
"""GPU 温度熔断（thermal guard）—— 自研栈 sunny-cx

目标
----
长时间训练时，6GB 卡（GTX 1060 / sm_61）若散热不足会持续升温；
高温下 CUDA 可能降频、报错，甚至损伤硬件。
本模块提供「超温 → 暂停训练 → 降温 → 自动恢复」的闭环保护。

设计原则（与项目既有工程约守一致）
----------------------------------
1. **零依赖训练核心**：只读温度、只请示"现在能不能继续"，不碰模型/优化器；
   真正决定暂停/恢复的是 train loop 里的调用方。
2. **不阻塞主线程的状态机**：
   · 每个 optimizer step 调用 `guard()`，内部有 1 秒级节流（nvidia-smi 调用约 30~60ms，
     每步都查会明显拖慢 13s/step 的吞吐）；
   · 超温持续 `hold_sec` 秒才真正熔断 —— 避免瞬时尖峰误触发；
   · 熔断后进入"冷却等待"，由 `wait_cool()` 轮询，降到 `resume_c` 以下才放行。
3. **双阈值滞回（hysteresis）**：暂停 82℃ / 恢复 74℃。
   若用单一阈值，温度会在阈值附近反复穿越 → 训练"抽风式"启停（最坏情况是
   每一步都暂停一次）。滞回带 ≥5℃ 是这类保护的基本要求。
4. **失败安全**：拿不到温度（无 nvidia-smi / 非 CUDA / 驱动异常）时**不拦截训练**，
   只记录一次警告，绝不因为"测不到温度"而让训练停摆。
5. **全程可观测**：每次熔断/恢复都写日志 + 通过 on_event 回调下发（供大屏显示），
   并把累计暂停时长与次数记入 state，便于事后判断散热是否真的够用。

温度读取
--------
用 `nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,clocks_throttle_reasons.active`
取真实读数（本机 GTX 1060 已实测可用：572.16 驱动）。
比 NVML Python 绑定更省事，也不给 6GB 卡的进程再挂一个常驻依赖。
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class ThermalArgs:
    """熔断参数。默认值按 GTX 1060 的常见安全区间设定。"""
    enable: bool = True
    pause_c: float = 82.0          # 达到该温度并持续 hold_sec → 暂停
    resume_c: float = 74.0         # 降到该温度以下 → 恢复（必须 < pause_c，形成滞回）
    hold_sec: float = 20.0         # 超温需持续这么久才熔断（抵御瞬时尖峰）
    poll_sec: float = 5.0          # 熔断后每多久查一次温度
    max_wait_min: float = 60.0     # 单次冷却最长等待（超时则告警并继续，避免永久卡死）
    check_every_sec: float = 3.0   # 正常训练时的温度检查节流间隔
    quiet: bool = False


@dataclass
class _ThermalState:
    over_since: Optional[float] = None      # 连续超温起始时刻
    paused: bool = False
    pause_count: int = 0
    pause_total_sec: float = 0.0
    last_temp: Optional[float] = None
    last_check: float = 0.0
    peak_temp: float = 0.0
    warned_once: bool = False


def read_gpu_status() -> Optional[dict]:
    """读取 GPU 温度/利用率/功耗/降频原因。失败返回 None（不抛异常）。

    ⚠ `clocks_throttle_reasons.active` 的位定义（NVIDIA 官方，实测踩过坑）：
        0x01 GpuIdle            0x02 ApplicationsClocksSetting
        0x04 SwPowerCap   ←     0x08 HwSlowdown
        0x10 SyncBoost          0x20 SwThermalSlowdown  ← 热
        0x40 HwThermalSlowdown   ← 热
        0x80 HwPowerBrakeSlowdown   0x100 DisplayClockSetting

    **只有 0x20 / 0x40 才是热降频。**
    第一版误把 0x4 当成热降频（那是 SwPowerCap，即"功耗上限"）——
    GTX 1060 是 78W 笔记本卡，**空闲 43℃ 时 0x4 也一直置位**，
    结果会让训练在完全正常的温度下反复熔断、根本起不来。
    经验：任何"bit mask 判定"都必须先用真实读数验证，不能凭位序猜。
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,utilization.gpu,power.draw,clocks_throttle_reasons.active",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
            encoding="utf-8", errors="replace",
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        temp = float(parts[0])
        util = float(parts[1]) if len(parts) > 1 and parts[1] not in ("", "[N/A]") else None
        power = float(parts[2]) if len(parts) > 2 and parts[2] not in ("", "[N/A]") else None
        throttle = parts[3] if len(parts) > 3 else ""
        th = None
        try:
            th = int(throttle, 16) if throttle.lower().startswith("0x") else int(throttle)
        except Exception:
            th = None
        # 仅 0x20(SW 热降频) / 0x40(HW 热降频) 才算热保护
        THERMAL_MASK = 0x20 | 0x40
        return {"temp": temp, "util": util, "power": power,
                "throttle_mask": th,
                "thermal_throttle": bool(th is not None and (th & THERMAL_MASK))}
    except Exception:
        return None


class ThermalGuard:
    """温度熔断状态机。

    用法（训练循环内）：::

        guard = ThermalGuard(ThermalArgs(), on_event=cb)
        ...
        while True:
            st = guard.guard()          # 每个 optimizer step 调一次（内部自动节流）
            if st and st.get("paused"):
                guard.wait_cool()       # 阻塞轮询，直到降温 / 超时
            ...继续训练...
    """

    def __init__(self, args: Optional[ThermalArgs] = None,
                 on_event: Optional[Callable[[dict], None]] = None,
                 log: Optional[Callable[[str], None]] = None):
        self.args = args or ThermalArgs()
        self.on_event = on_event
        self._log = log or (lambda s: print(s, flush=True))
        self.s = _ThermalState()
        # 首次探测：确认温度可读。读不到就整体停用，绝不拦训练。
        if self.args.enable:
            probe = read_gpu_status()
            if probe is None:
                self.args.enable = False
                self._log("[thermal] ⚠ 读取不到 GPU 温度（无 nvidia-smi / 非 CUDA），"
                          "温度熔断已自动停用；训练不受影响。")
            else:
                self.s.last_temp = probe["temp"]
                self.s.peak_temp = probe["temp"]
                self._log(f"[thermal] 温度熔断已启用：暂停 {self.args.pause_c:.0f}℃ "
                          f"(持续 {self.args.hold_sec:.0f}s) / 恢复 {self.args.resume_c:.0f}℃ "
                          f"· 当前 {probe['temp']:.0f}℃")

    # ---------------------------------------------------------------- helpers
    def _emit(self, ev: dict):
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:
                pass

    @property
    def paused(self) -> bool:
        return self.s.paused

    def snapshot(self) -> dict:
        """供大屏推送：温度 / 是否熔断 / 累计暂停。"""
        return {
            "gpu_temp": self.s.last_temp,
            "gpu_peak_temp": round(self.s.peak_temp, 1) if self.s.peak_temp else None,
            "thermal_paused": self.s.paused,
            "thermal_pause_count": self.s.pause_count,
            "thermal_pause_sec": round(self.s.pause_total_sec, 1),
            "thermal_pause_c": self.args.pause_c,
            "thermal_resume_c": self.args.resume_c,
            "thermal_enabled": self.args.enable,
        }

    # ------------------------------------------------------------------ 核心
    def guard(self) -> Optional[dict]:
        """训练步内的轻量检查（自动节流）。

        返回 None 表示"无需处理"；返回 dict 且 paused=True 表示**已熔断**，
        调用方应立即停下训练并调用 wait_cool()。
        """
        if not self.args.enable:
            return None
        now = time.time()
        if now - self.s.last_check < self.args.check_every_sec:
            return None
        self.s.last_check = now

        st = read_gpu_status()
        if st is None:
            return None
        self.s.last_temp = st["temp"]
        if st["temp"] > self.s.peak_temp:
            self.s.peak_temp = st["temp"]

        hot = st["temp"] >= self.args.pause_c

        # 热降频（HW thermal throttle）优先于温度阈值：驱动已经明确在保护硬件
        if st.get("thermal_throttle"):
            hot = True

        if not hot:
            self.s.over_since = None
            return None

        if self.s.over_since is None:
            self.s.over_since = now
            self._log(f"[thermal] 温度 {st['temp']:.0f}℃ ≥ {self.args.pause_c:.0f}℃，"
                      f"观察 {self.args.hold_sec:.0f}s 后决定是否熔断"
                      + ("（驱动已热降频）" if st.get("thermal_throttle") else ""))
            return None

        if now - self.s.over_since >= self.args.hold_sec:
            self.s.paused = True
            self.s.pause_count += 1
            self._log(f"[thermal] 🔴 熔断触发：{st['temp']:.0f}℃ 已持续 "
                      f"{now - self.s.over_since:.0f}s（阈值 {self.args.pause_c:.0f}℃）"
                      f"→ 暂停训练，等待降温至 {self.args.resume_c:.0f}℃")
            self._emit({"type": "thermal_pause", "temp": st["temp"],
                        "pause_c": self.args.pause_c, "resume_c": self.args.resume_c,
                        "hold_sec": round(now - self.s.over_since, 1),
                        "power": st.get("power"), "util": st.get("util")})
            return {"paused": True, **self.snapshot()}
        return None

    def wait_cool(self, sleep_fn: Optional[Callable[[float], None]] = None) -> float:
        """熔断后的冷却等待。返回实际暂停秒数。

        每 poll_sec 查一次；降到 resume_c 以下即放行。
        超过 max_wait_min 分钟仍未降温 → 告警并**继续等待**（不放弃保护），
        但把状态标为 timeout 以便人工介入（例如手动清灰/降 batch）。
        """
        if not self.s.paused:
            return 0.0
        t_start = time.time()
        sleeper = sleep_fn or time.sleep
        log_every = max(1, int(30 / max(self.args.poll_sec, 1)))
        i = 0
        while True:
            sleeper(self.args.poll_sec)
            i += 1
            st = read_gpu_status()
            waited = time.time() - t_start
            if st is not None:
                self.s.last_temp = st["temp"]
                cool = st["temp"] <= self.args.resume_c
                if i % log_every == 0 or cool:
                    self._log(f"[thermal] 冷却中 {waited/60:.1f}min · "
                              f"当前 {st['temp']:.0f}℃ → 目标 ≤{self.args.resume_c:.0f}℃")
                if cool:
                    self.s.paused = False
                    self.s.over_since = None
                    self.s.pause_total_sec += waited
                    self._log(f"[thermal] 🟢 已降温至 {st['temp']:.0f}℃，"
                              f"恢复训练（本次暂停 {waited/60:.1f}min）")
                    self._emit({"type": "thermal_resume", "temp": st["temp"],
                                "paused_sec": round(waited, 1)})
                    return waited
            if waited > self.args.max_wait_min * 60:
                self._log(f"[thermal] ⚠ 冷却等待已超 {self.args.max_wait_min:.0f}min "
                          f"仍未降到 {self.args.resume_c:.0f}℃，继续等待（保护不放弃）；"
                          f"建议检查散热/机箱风道。")
                self._emit({"type": "thermal_timeout",
                            "waited_sec": round(waited, 1),
                            "temp": (st or {}).get("temp")})
                t_start = time.time()   # 重置计时，避免每轮都刷告警

    def close(self):
        """收尾：把累计统计写入一次事件，便于大屏/日志汇总。"""
        if self.s.pause_count or self.s.peak_temp:
            self._log(f"[thermal] 会话统计：峰值 {self.s.peak_temp:.0f}℃ · "
                      f"熔断 {self.s.pause_count} 次 · "
                      f"累计暂停 {self.s.pause_total_sec/60:.1f}min")
            self._emit({"type": "thermal_summary", **self.snapshot()})