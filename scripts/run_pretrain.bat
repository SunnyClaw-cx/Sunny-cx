@echo off
REM ============================================================
REM  sunny-cx 0.5B pretrain (self-built stack, no MiniMind deps)
REM  HW: GTX 1060 6GB / sm_61 / Win11
REM  measured peak VRAM 3.19 GiB; packing speeds up ~5x
REM
REM  cloud dashboard push ON by default (structured callback):
REM    - per-step real loss / val_loss / grad_norm / weight drift
REM    - 32-layer real activation probe -> every dot on the dial
REM    - GPU temperature live on dashboard (incl. trip state)
REM    - offline: add --no_push to disable
REM
REM  GPU thermal trip ON by default (protect long runs):
REM    - >=82C for 20s -> pause training; <=74C -> auto resume
REM    - measured here (GTX 1060 / 78W cap): 25s at 100% util
REM      only reaches 58C, plenty of headroom
REM    - disable: --no_thermal ;  tune: --thermal_pause_c / --thermal_resume_c
REM
REM  Duty-cycle REST ON by default (proactive, complements thermal):
REM    - after 2h of continuous training -> rest 30min -> auto resume
REM    - rest time is EXCLUDED from elapsed/ETA accounting
REM    - cumulative state persisted, survives restart (cannot be bypassed)
REM    - disable: --no_rest ;  tune: --rest_work_hours / --rest_minutes
REM ============================================================
setlocal
cd /d "%~dp0.."

set "PY="
if exist "%~dp0python_path.txt" for /f "usebackq delims=" %%P in ("%~dp0python_path.txt") do set "PY=%%P"
if not defined PY if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"
if not defined PY set "PY=python"

set STAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set STAMP=%STAMP: =0%
set LOG=logs\pretrain_0.5b_push.log

echo [start] log -> %LOG%
"%PY%" -u scripts\train.py ^
  --stage pretrain ^
  --config 0.5b ^
  --batch_size 2 ^
  --max_seq_len 512 ^
  --accumulation_steps 8 ^
  --learning_rate 3e-4 ^
  --epochs 2 ^
  --val_interval 200 ^
  --val_samples 400 ^
  --early_stop_patience 10 ^
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
  --save_weight sunnycx_0.5b_pretrain ^
  --from_resume ^
  > "%LOG%" 2>&1

echo [done] exit=%ERRORLEVEL%  log: %LOG%
endlocal
