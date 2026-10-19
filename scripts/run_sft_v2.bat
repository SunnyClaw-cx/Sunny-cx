@echo off
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
