@echo off
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
