# -*- coding: utf-8 -*-
"""生成 SFT 阶段的两个启动脚本（GBK + CRLF）。

为什么要用本脚本写 .bat：
Edit/Write 工具落盘是 UTF-8 + LF，cmd.exe 按 GBK 解析会乱码甚至语法错。
历史上已踩过一次（run_pretrain.bat）。故统一由本脚本按 gbk 编码 + CRLF 写出。

SFT 阶段的三个关键差异（全部来自实测，不是猜的）：
  1. 长度分桶 + 动态 pad：15.73 s/step → 4.01 s/step，单轮 12.31 天 → 3.14 天（3.92x）
     实测口径 scripts/bench_sft_bucket.py（batch2 × accum8，覆盖整块 32 step）
  2. 不能打包（会串掉多轮对话结构），所以只能用分桶这条路
  3. 验证集仍用固定 pad：各次 val_loss 必须同口径，否则早停判据失真
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SCRIPTS = Path(__file__).resolve().parent

RUN = r"""@echo off
REM ============================================================
REM  sunny-cx 0.5B  SFT  (v2 corpus)  --  length-bucketed
REM  HW: GTX 1060 6GB / sm_61 / Win11
REM
REM  WHY BUCKETING (measured on this card, not theory):
REM    SFT samples average 92.2 tokens (P50 77 / P99 249).
REM    Fixed pad to 512  -> only 18.3% of computed positions are real tokens
REM                      -> 15.73 s/step x 67,596 step = 12.31 days
REM    Bucket + dynamic pad -> 96.0% real
REM                      ->  4.01 s/step x 67,596 step =  3.14 days  (3.92x)
REM    The loss mask is IDENTICAL (pad = label -100 / mask 0); only the
REM    padding length changes, so this is pure waste removal.
REM
REM  WHY NOT PACKING: packing concatenates documents and would destroy the
REM    multi-turn user/assistant structure. Bucketing keeps each sample intact.
REM
REM  VAL stays on fixed pad on purpose: val_loss must be measured the same
REM    way every time, otherwise the early-stop signal drifts (v1 lost 1000
REM    steps to exactly this class of bug).
REM ============================================================
setlocal
cd /d "%~dp0.."

set "PY="
if exist "%~dp0python_path.txt" for /f "usebackq delims=" %%P in ("%~dp0python_path.txt") do set "PY=%%P"
if not defined PY if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"
if not defined PY set "PY=python"

set LOG=logs\sft_0.5b_v2.log
set CKPT=checkpoints\sunnycx_0.5b_sft_v2_latest.pt

REM ---- init weights: prefer the v2 pretrain best, fall back to warm5600 ----
set INIT=checkpoints\sunnycx_0.5b_pretrain_v2_best.pth
if not exist "%INIT%" set INIT=checkpoints\sunnycx_0.5b_pretrain_v2_latest.pt
if not exist "%INIT%" set INIT=checkpoints\sunnycx_0.5b_pretrain_best.pth
if not exist "%INIT%" set INIT=checkpoints\sunnycx_0.5b_pretrain_warm5600.pth
if not exist "%INIT%" (
  echo [error] no pretrain weights found - run pretrain first
  exit /b 1
)

if exist "%CKPT%" (
  set MODE=resume
  set INITOPT=--from_resume
) else (
  set MODE=fresh
  set INITOPT=--from_weight "%INIT%"
)

echo ============================================================
echo  [mode] %MODE%
if "%MODE%"=="resume" echo  [info] found %CKPT%
if "%MODE%"=="fresh"  echo  [info] init weights: %INIT%
echo  [plan] 67,596 steps / 1 epoch / ~3.2 days bucket / ~4.0 days incl. rest+val
echo  [log ] %LOG%
echo ============================================================
echo.

"%PY%" -u scripts\train.py ^
  --stage sft ^
  --config 0.5b ^
  --data_path "..\tcm-llm\data\processed_v2\tcm_classics_qa.train.jsonl" ^
  --val_data_path "..\tcm-llm\data\processed_v2\tcm_classics_qa.val.jsonl" ^
  %INITOPT% ^
  --save_weight sunnycx_0.5b_sft_v2 ^
  --batch_size 2 ^
  --max_seq_len 512 ^
  --accumulation_steps 8 ^
  --learning_rate 1e-4 ^
  --epochs 1 ^
  --val_interval 2000 ^
  --val_samples 200 ^
  --early_stop_patience 3 ^
  --save_interval 500 ^
  --log_interval 20 ^
  --warmup_steps 200 ^
  --sft_bucket_chunk 512 ^
  --gradient_checkpointing ^
  --num_workers 0 ^
  --thermal_pause_c 82 ^
  --thermal_resume_c 74 ^
  --thermal_hold_sec 20 ^
  --rest_work_hours 2 ^
  --rest_minutes 30 ^
  > "%LOG%" 2>&1

echo.
echo [done] exit=%ERRORLEVEL%  log: %LOG%
endlocal
"""

START = r"""@echo off
REM ============================================================
REM  sunny-cx 0.5B SFT (v2) - standalone launcher
REM
REM  Same reason as the pretrain launcher: a training process started
REM  from an agent session is killed when that session ends. This window
REM  is independent of any session, so double-click it and leave it.
REM
REM  Resume: after the first run, closing this window and double-clicking
REM  again continues from the last checkpoint (--from_resume).
REM ============================================================
setlocal
cd /d "%~dp0.."

set "PY="
if exist "%~dp0python_path.txt" for /f "usebackq delims=" %%P in ("%~dp0python_path.txt") do set "PY=%%P"
if not defined PY if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"
if not defined PY set "PY=python"

"%PY%" scripts\guard_single_instance.py
if errorlevel 2 (
  echo.
  echo [guard] Startup aborted - see the message above.
  echo.
  pause
  exit /b 2
)

echo ============================================================
echo  sunny-cx 0.5B SFT v2  -  1 epoch, length-bucketed
echo  data : tcm-llm\data\processed_v2\tcm_classics_qa.*.jsonl
echo  init : v2 pretrain best (falls back to warm5600)
echo  ETA  : ~3.2 days bucket / ~4.0 days incl. rest and validation
echo  stop : Ctrl+C is safe; checkpoints every 500 steps
echo ============================================================
echo.

call scripts\run_sft_v2.bat

echo.
echo [done] SFT process exited ^(see log for details^).
echo [done] Press any key to close this window.
pause >nul
endlocal
"""


def write(name: str, text: str) -> None:
    p = SCRIPTS / name
    # cmd.exe 按 GBK 解析；换行必须 CRLF
    p.write_bytes(text.replace("\n", "\r\n").encode("gbk"))
    raw = p.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n"), "CRLF 不一致"
    print(f"[ok] {name}  {len(raw)} bytes  GBK+CRLF")


if __name__ == "__main__":
    write("run_sft_v2.bat", RUN)
    write("start_sft_v2.bat", START)
