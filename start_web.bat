@echo off
REM sunny-cx 本地服务 —— 白话问诊 + 训练看板
REM 依赖：pip install -r requirements.txt
chcp 65001 >nul
cd /d "%~dp0"

set "PY="
if exist "%~dp0scripts\python_path.txt" for /f "usebackq delims=" %%P in ("%~dp0scripts\python_path.txt") do set "PY=%%P"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY set "PY=python"

echo.
echo   sunny-cx
echo   白话问诊：http://127.0.0.1:8000
echo   训练看板：http://127.0.0.1:8000/live
echo   停止：Ctrl+C
echo.
"%PY%" -m uvicorn server.app:app --host 127.0.0.1 --port 8000
pause
