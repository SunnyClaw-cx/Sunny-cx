#!/usr/bin/env bash
# sunny-cx 本地服务 —— 白话问诊 + 训练看板（macOS / Linux）
set -e
cd "$(dirname "$0")"
PY=python3
[ -x ".venv/bin/python" ] && PY=.venv/bin/python
echo "白话问诊  ->  http://127.0.0.1:8000"
echo "训练看板  ->  http://127.0.0.1:8000/live"
exec "$PY" -m uvicorn server.app:app --host 127.0.0.1 --port 8000
