# -*- coding: utf-8 -*-
"""sunny-cx 白话问诊 —— HTTP API 服务

设计原则：**服务端只做转发，不新增任何医学判断**。
所有证型 / 方剂 / 药味都来自 `data/consult/` 下的两张离线表，
`ConsultEngine` 负责受控判定，`check_answer()` 负责越界闸门。
本层不调用任何云端服务，不写库，无状态，可直接水平复制。

启动：
    uvicorn server.app:app --reload --port 8000
或：
    python -m server.app
"""
from __future__ import annotations

import re
import sys
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:                     # 让 `pip install -e .` 之外也能跑
    sys.path.insert(0, str(SRC))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sunnycx import consult as core
from server import train_state

WEB_DIR = ROOT / "web"
API_VERSION = "0.1.0"

@asynccontextmanager
async def lifespan(application: FastAPI):
    """启动即加载词表，避免首个请求承担解析开销。"""
    eng = engine()
    application.state.started_at = time.time()
    application.state.n_syndromes = len(eng.syndromes)
    application.state.n_symptoms = len(eng.lex["symptoms"])
    application.state.n_aliases = len(eng._alias)
    yield


app = FastAPI(
    title="sunny-cx 白话问诊 API",
    version=API_VERSION,
    description="受控判定引擎：白话主诉 → 规范症状 → 病位/病性证据 → 35 证型打分 → 查表出方 → 越界闸门。",
    lifespan=lifespan,
)


class ConsultIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="白话主诉")
    top_k: int = Field(3, ge=1, le=10, description="返回候选数")


class CheckIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000, description="待校验文本")


# ---------------------------------------------------------------- 引擎单例
@lru_cache(maxsize=1)
def engine() -> core.ConsultEngine:
    return core.ConsultEngine.load()


# ---------------------------------------------------------------- API
@app.get("/api/health")
def health() -> dict:
    eng = engine()
    return {
        "status": "ok",
        "version": API_VERSION,
        "engine": {
            "syndromes": len(eng.syndromes),
            "symptoms": len(eng.lex["symptoms"]),
            "aliases": len(eng._alias),
        },
        "uptime_sec": round(time.time() - getattr(app.state, "started_at", time.time()), 1),
    }


@app.get("/api/meta")
def meta() -> dict:
    """前端用来渲染阈值说明与统计条。"""
    eng = engine()
    return {
        "version": API_VERSION,
        "stats": {
            "syndromes": len(eng.syndromes),
            "symptoms": len(eng.lex["symptoms"]),
            "aliases": len(eng._alias),
        },
        "thresholds": {
            "ratio_ok": core.RATIO_OK,
            "ratio_weak": core.RATIO_WEAK,
            "tie_eps": core.TIE_EPS,
            "token_cap": core.TOKEN_CAP,
            "w_pos": core.W_POS,
            "w_nat": core.W_NAT,
        },
        "decisions": {
            "ok": "证据充分，给辨证方向与参考方剂",
            "differential": "两个证型证据接近，只给鉴别不给单选",
            "weak": "覆盖不足，只给方向并列出缺失要素",
            "refuse": "信息不足或内部校验未通过，拒绝辨证",
        },
        "disclaimer": core.DISCLAIM,
    }


@app.get("/api/syndromes")
def syndromes() -> dict:
    """35 证型全表（不含剂量）。"""
    eng = engine()
    return {
        "count": len(eng.syndromes),
        "items": [
            {
                "syndrome": s["syndrome"],
                "base": s["base"],
                "zhi_fa": s["zhi_fa"],
                "formula": s["formula"],
                "source": s["source"],
                "composition": s["composition"],
                "alt": s.get("alt") or [],
                "n": s.get("n"),
            }
            for s in eng.syndromes
        ],
    }


@app.get("/api/symptoms")
def symptoms() -> dict:
    """受控词表：规范症状 + 别名（前端做示例与提示）。"""
    eng = engine()
    items = [
        {
            "term": e["term"],
            "alias": list(e.get("alias") or []),
            "axes": e["axes"],
        }
        for e in eng.lex["symptoms"]
    ]
    return {"count": len(items), "items": items}


@app.post("/api/consult")
def consult(body: ConsultIn) -> JSONResponse:
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="主诉不能为空")
    res = engine().consult(text, topk=body.top_k)
    return JSONResponse(res)


@app.post("/api/check")
def check(body: CheckIn) -> dict:
    """对任意文本做越界校验，返回是否放行与命中的违规项。"""
    return core.check_answer(body.text, engine())


# ---------------------------------------------------------------- 训练看板
# 全部数据来自本机 logs/ 与 checkpoints/，不查询、不推送任何云端服务。
RUN_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


@app.get("/api/train/runs")
def train_runs() -> dict:
    runs = train_state._runs()
    latest = train_state.latest_run()
    return {
        "count": len(runs),
        "latest": latest,
        "items": [
            {
                "run": r,
                "is_latest": r == latest,
                "state": train_state.parse_run(r)["state"],
            }
            for r in runs
        ],
    }


@app.get("/api/train/status")
def train_status(run: str | None = None) -> JSONResponse:
    """解析指定 run 的日志与检查点。不传 run 时取最近更新的那个。"""
    name = run or train_state.latest_run()
    if not name:
        raise HTTPException(status_code=404, detail="本机没有任何训练日志")
    if not RUN_RE.match(name):
        raise HTTPException(status_code=422, detail="run 名只允许字母、数字、下划线、点、短横")
    data = train_state.parse_run(name)
    if not data["log_exists"]:
        raise HTTPException(status_code=404, detail=f"未找到 {name} 的日志")
    return JSONResponse(data)


# ---------------------------------------------------------------- 静态前端
@app.get("/live", include_in_schema=False)
def live_page() -> FileResponse:
    """训练看板：读本机日志的静态页，与问诊页同目录。"""
    page = WEB_DIR / "live.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="web/live.html 缺失")
    return FileResponse(page)


if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
else:                                            # 纯 API 部署时的兜底
    @app.get("/")
    def _no_web() -> dict:
        return {"detail": "web/ 目录缺失，仅提供 API"}


def main() -> None:
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
