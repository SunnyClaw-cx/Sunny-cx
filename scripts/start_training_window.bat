@echo off
REM ============================================================
REM  sunny-cx 0.5B pretrain - standalone window launcher
REM
REM  WHY THIS EXISTS:
REM    Training launched from an agent/automation session dies when
REM    that session ends (verified twice on 2026-10-16: the run
REM    stopped at step 100, and a later attempt at step 3). This
REM    launcher gives training its own console window, fully
REM    independent of any agent session.
REM
REM  USAGE:
REM    Double-click this file. It resumes from the latest checkpoint
REM    automatically (run_pretrain.bat already carries --from_resume).
REM
REM  SINGLE-INSTANCE GUARD:
REM    Two runs would fight over the single 6GB GPU and both would
REM    OOM. The check is delegated to guard_single_instance.py, because
REM    an inline PowerShell one-liner in a .bat loses its backslashes
REM    through cmd.exe quoting and silently makes the guard a no-op.
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
echo  sunny-cx 0.5B pretrain - resuming from latest checkpoint
echo  log: logs\pretrain_0.5b_push.log
echo ============================================================
echo.

call scripts\run_pretrain.bat

echo.
echo [done] Training process exited ^(see log for details^).
echo [done] Press any key to close this window.
pause >nul
endlocal