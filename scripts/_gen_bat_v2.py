# -*- coding: utf-8 -*-
"""生成 v2 完整预训练的两个启动脚本（GBK + CRLF）。

为什么要用本脚本写 .bat：
Edit/Write 工具落盘是 UTF-8 + LF，cmd.exe 按 GBK 解析会乱码甚至语法错。
历史上已踩过一次（run_pretrain.bat）。故统一由本脚本按 gbk 编码 + CRLF 写出。
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SCRIPTS = Path(__file__).resolve().parent

RUN = r"""@echo off
REM ============================================================
REM  sunny-cx 0.5B pretrain  --  FULL EPOCH on rebuilt corpus (v2)
REM  HW: GTX 1060 6GB / sm_61 / Win11
REM
REM  Five differences from run_pretrain.bat (all measured, not guesses):
REM    1. DATA   -> data\processed_v2
REM       The paragraph splitter used to throw away every sub-40-char
REM       sentence produced by splitting a long paragraph (681,372
REM       fragments). Fixed by greedy accumulation: +13.0% corpus.
REM    2. INIT   -> warm start from the v1 step-5600 weights
REM       NOT --from_resume. Resume inherits step=5600 and would SKIP
REM       the first 5600 steps of the NEW corpus. --from_weight moves
REM       only parameters; optimizer / step / early-stop reset to 0,
REM       so the rebuilt corpus is covered end to end (100%).
REM       (v1 latest@5600 beats v1 best@4600 on a 1200-sample set:
REM        ppl 31.21 vs 34.02, so its weights are the better start.)
REM    3. EPOCH  -> 1 full pass = 13,212 optimizer steps
REM    4. VAL    -> 400 samples every 400 steps
REM       The old 64 samples were drawn from only 2 books (59% weight)
REM       and mis-fired early stop at step 4600.
REM    5. LR     -> 2e-4 (was 3e-4): warm start needs a gentler restart
REM       so the existing convergence is not destroyed.
REM
REM  Measured plan:  13,212 step x 16.5 s = 60.6 h
REM                  + val 32 x 425 s      =  3.8 h
REM                  + rest 30min/2h       -> 80.6 h  (~3.4 days)
REM  VRAM peak 3.17 / 6.00 GB.  Thermal trip + duty-cycle rest ON.
REM ============================================================
setlocal
cd /d "%~dp0.."

set "PY="
if exist "%~dp0python_path.txt" for /f "usebackq delims=" %%P in ("%~dp0python_path.txt") do set "PY=%%P"
if not defined PY if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"
if not defined PY set "PY=python"

set LOG=logs\pretrain_0.5b_v2.log
set CKPT=checkpoints\sunnycx_0.5b_pretrain_v2_latest.pt
set WARM=checkpoints\sunnycx_0.5b_pretrain_warm5600.pth

REM ---- 首启 vs 中断续训：自动判别 ----
if exist "%CKPT%" (
  set MODE=resume
  set INITOPT=--from_resume
) else (
  set MODE=warmstart
  set INITOPT=--from_weight "%WARM%"
)

echo ============================================================
echo  [mode] %MODE%
if "%MODE%"=="resume" echo  [info] found %CKPT%
if "%MODE%"=="warmstart" echo  [info] no v2 checkpoint yet - init from v1 step-5600
echo  [plan] 13,212 steps / 1 full epoch / ETA ~80 h incl. rest
echo  [log ] %LOG%
echo ============================================================
echo.

"%PY%" -u scripts\train.py ^
  --stage pretrain ^
  --config 0.5b ^
  --data_path "..\tcm-llm\data\processed_v2\tcm_classics_pretrain.train.jsonl" ^
  --val_data_path "..\tcm-llm\data\processed_v2\tcm_classics_pretrain.val.jsonl" ^
  %INITOPT% ^
  --save_weight sunnycx_0.5b_pretrain_v2 ^
  --batch_size 2 ^
  --max_seq_len 512 ^
  --accumulation_steps 8 ^
  --learning_rate 2e-4 ^
  --epochs 1 ^
  --val_interval 400 ^
  --val_samples 400 ^
  --early_stop_patience 20 ^
  --save_interval 100 ^
  --log_interval 20 ^
  --warmup_steps 200 ^
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
REM  sunny-cx 0.5B pretrain (v2, full epoch) - standalone launcher
REM
REM  WHY: training launched from an agent session dies when that
REM  session ends (verified twice on 2026-10-16: stopped at step 100,
REM  and once at step 3). This window is independent of any session.
REM
REM  USAGE: double-click. First run warm-starts from v1 step-5600;
REM  after that it auto-resumes from the v2 checkpoint, so you can
REM  close/stop/reboot and simply double-click again.
REM
REM  SINGLE-INSTANCE GUARD: two runs would fight over the single 6GB
REM  GPU and both would OOM. Delegated to guard_single_instance.py
REM  because an inline PowerShell one-liner loses its backslashes
REM  through cmd.exe quoting and silently becomes a no-op.
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
echo  sunny-cx 0.5B pretrain v2  -  FULL EPOCH (13,212 steps)
echo  data  : tcm-llm\data\processed_v2  (split bug fixed, +13%)
echo  init  : v1 step-5600 weights, then auto-resume on restart
echo  ETA   : ~80 h incl. 30min rest every 2h of training
echo  stop  : Ctrl+C is safe; checkpoints every 100 steps (~28 min)
echo ============================================================
echo.

call scripts\run_pretrain_v2.bat

echo.
echo [done] Training process exited ^(see log for details^).
echo [done] Press any key to close this window.
pause >nul
endlocal
"""

for name, text in [("run_pretrain_v2.bat", RUN), ("start_training_v2.bat", START)]:
    p = SCRIPTS / name
    data = text.replace("\n", "\r\n").encode("gbk")
    p.write_bytes(data)
    crlf = data.count(b"\r\n")
    lf = data.count(b"\n")
    print(f"[write] {name}  {len(data)} bytes  CRLF={crlf}  LF={lf}  "
          f"单一换行符类型={crlf == lf}")
