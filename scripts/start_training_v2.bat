@echo off
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
