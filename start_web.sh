#!/usr/bin/env bash
# sunny-cx 白话问诊服务 —— 本地启动（macOS / Linux）
set -e
cd "$(dirname "$0")"
PY=python3
[ -x ".venv/bin/python" ] && PY=.venv/bin/python
echo "sunny-cx 白话问诊  ->  http://127.0.0.1:8000"
exec "$PY" -m uvicorn server.app:app --host 127.0.0.1 --port 8000
