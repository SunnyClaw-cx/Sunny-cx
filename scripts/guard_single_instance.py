# -*- coding: utf-8 -*-
"""单实例守卫：检查是否已有 sunny-cx 训练进程在运行。

为什么独立成脚本：在 .bat 里内联 PowerShell + 多层引号转义会被 cmd.exe
吞掉反斜杠，导致条件永远为假、守卫静默失效（实测过）。用 Python 调用
wmic/psutil 风格的进程枚举则稳定可靠。

判定依据：python.exe 且命令行含 scripts/train.py 且工作集 > 500MB。
之所以加内存阈值——venv 的 python.exe 只是个转发 shim（约 11MB），
真正的训练进程才持有 1.6GB+ 显存驻留，两者命令行相同。

退出码：0 = 无训练在跑（可启动）；2 = 已在运行（应拒绝启动）
"""
import subprocess
import sys

THRESHOLD = 500 * 1024 * 1024


def count_trainers() -> int:
    """用 PowerShell CIM 枚举 python 进程，返回真实训练进程数。"""
    ps = (
        "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*train.py*' -and "
        f"$_.WorkingSetSize -gt {THRESHOLD} }}).Count"
    )
    for exe in ("powershell", "pwsh"):
        try:
            r = subprocess.run(
                [exe, "-NoProfile", "-NonInteractive", "-Command", ps],
                capture_output=True, text=True, timeout=30, errors="replace",
            )
            out = (r.stdout or "").strip()
            if out.isdigit():
                return int(out)
        except (OSError, subprocess.SubprocessError):
            continue
    return -1        # 探测失败：不阻止启动，仅提示


def main() -> int:
    n = count_trainers()
    if n < 0:
        print("[guard] 无法探测进程状态，放行启动（请自行确认没有重复实例）")
        return 0
    if n == 0:
        print("[guard] 未发现运行中的训练进程，可以启动")
        return 0
    print(f"[guard] 已有 {n} 个训练进程在运行，拒绝启动第二个实例")
    print("[guard] 两个实例会争抢同一块 6GB 显存并双双 OOM。")
    print("[guard] 如需重启：先在任务管理器结束那个 python.exe，再运行本启动器。")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())