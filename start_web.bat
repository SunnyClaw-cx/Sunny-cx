@echo off
REM sunny-cx 白话问诊服务 —— 本地启动
REM 依赖：pip install -r requirements.txt
chcp 65001 >nul
cd /d "%~dp0"

set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

echo.
echo   sunny-cx 白话问诊
echo   地址：http://127.0.0.1:8000
echo   停止：Ctrl+C
echo.
"%PY%" -m uvicorn server.app:app --host 127.0.0.1 --port 8000
pause
