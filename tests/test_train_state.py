# -*- coding: utf-8 -*-
"""训练看板解析测试（stdlib unittest）

用临时目录造一份日志，验证：
  1) 进度 / loss / val / 休息行都能解析
  2) 日志名与权重名前缀不一致时能正确配对
  3) 三种状态（training / resting / stopped）判定正确
  4) API 层拒绝路径穿越、未知 run 返回 404

运行：
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient          # noqa: E402

from server import train_state                     # noqa: E402
from server.app import app                         # noqa: E402

LOG = """\
[env] sunny-cx  0.5b
Epoch:[1/1](100/1000), loss: 3.2000, lr: 0.00020000, gnorm: 0.900, elapsed: 20.0min, peak: 3.10/6.00GB
Epoch:[1/1](200/1000), loss: 2.5000, lr: 0.00010000, gnorm: 0.700, elapsed: 40.0min, peak: 3.10/6.00GB
[VAL] step 200/1000  train_loss: 2.5000  val_loss: 2.8000  gap: -0.3000  best: 2.8000
[early-stop] 新最优 val=2.8000，已保存 best 权重
[rest] 已连续训练 2.00h（阈值 2.00h）→ 休息 30min 后自动继续
[rest] 休息中 5.0/30min （训练已暂停，GPU 空闲）
[rest] 休息结束（30min），恢复训练 · 累计已休 1 次 / 0.50h
Epoch:[1/1](220/1000), loss: 2.4000, lr: 0.00009900, gnorm: 0.800, elapsed: 44.0min, peak: 3.11/6.00GB
"""


class TrainStateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.logs = base / "logs"
        self.ckpt = base / "checkpoints"
        self.logs.mkdir()
        self.ckpt.mkdir()

        # 故意让日志名与权重名前缀不一致，复现真实工程里的命名
        self.log_path = self.logs / "toy_v2.log"
        self.log_path.write_text(LOG, encoding="utf-8")
        (self.ckpt / "sunnycx_toy_v2_latest.json").write_text(
            json.dumps({"step": 220, "best_val": 2.8, "best_step": 200,
                        "saved_at": "2026-10-19 10:00:00"}),
            encoding="utf-8")
        (self.ckpt / "sunnycx_toy_v2_reststate.json").write_text(
            json.dumps({"rest_count": 3, "rest_total_sec": 5400.0}),
            encoding="utf-8")

        self._saved = (train_state.LOGS, train_state.CKPT)
        train_state.LOGS, train_state.CKPT = self.logs, self.ckpt

    def tearDown(self):
        train_state.LOGS, train_state.CKPT = self._saved
        self.tmp.cleanup()

    def _age_log(self, seconds: int) -> None:
        t = time.time() - seconds
        os.utime(self.log_path, (t, t))

    # ---------------- 解析 ----------------
    def test_parses_progress_and_curve(self):
        d = train_state.parse_run("sunnycx_toy_v2")
        self.assertEqual(d["log_file"], "toy_v2.log")      # 名称模糊配对
        self.assertEqual(d["ckpt_file"], "sunnycx_toy_v2_latest.json")
        self.assertEqual(d["step"], 220)
        self.assertEqual(d["total_steps"], 1000)
        self.assertAlmostEqual(d["progress"], 0.22)
        self.assertAlmostEqual(d["loss"], 2.4)
        self.assertAlmostEqual(d["peak_gb"], 3.11)
        self.assertEqual(len(d["curve"]), 3)
        self.assertEqual(len(d["vals"]), 1)
        self.assertAlmostEqual(d["vals"][0]["val"], 2.8)

    def test_eta_uses_wall_clock_elapsed(self):
        d = train_state.parse_run("sunnycx_toy_v2")
        # elapsed 44min / 220 步 = 12 s/步；剩余 780 步 = 9360s = 2.6h
        self.assertAlmostEqual(d["sec_per_step"], 12.0, places=1)
        self.assertAlmostEqual(d["eta_hours"], 2.6, places=1)

    def test_checkpoint_and_rest_state(self):
        d = train_state.parse_run("sunnycx_toy_v2")
        self.assertAlmostEqual(d["best_val"], 2.8)
        self.assertEqual(d["best_step"], 200)
        self.assertEqual(d["rest_count"], 3)
        self.assertAlmostEqual(d["rest_total_hours"], 1.5, places=2)

    # ---------------- 状态 ----------------
    def test_state_training_when_log_fresh(self):
        self._age_log(5)
        self.assertEqual(train_state.parse_run("sunnycx_toy_v2")["state"], "training")

    def test_state_stopped_when_log_old(self):
        self._age_log(3600)
        self.assertEqual(train_state.parse_run("sunnycx_toy_v2")["state"], "stopped")

    def test_state_resting_when_last_event_is_rest(self):
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write("[rest] 休息中 10.0/30min （训练已暂停，GPU 空闲）\n")
        d = train_state.parse_run("sunnycx_toy_v2")
        self.assertEqual(d["state"], "resting")
        self.assertEqual(d["rest_timer"], [10.0, 30.0])

    # ---------------- API ----------------
    def test_api_serves_status(self):
        with TestClient(app) as c:
            r = c.get("/api/train/status", params={"run": "sunnycx_toy_v2"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["step"], 220)

    def test_api_rejects_path_traversal(self):
        with TestClient(app) as c:
            r = c.get("/api/train/status", params={"run": "../secret"})
        self.assertEqual(r.status_code, 422)

    def test_api_unknown_run_404(self):
        with TestClient(app) as c:
            r = c.get("/api/train/status", params={"run": "no_such_run"})
        self.assertEqual(r.status_code, 404)

    def test_api_lists_runs(self):
        with TestClient(app) as c:
            j = c.get("/api/train/runs").json()
        names = {it["run"] for it in j["items"]}
        self.assertIn("toy_v2", names)          # 日志侧
        self.assertIn("sunnycx_toy_v2", names)  # 检查点侧


if __name__ == "__main__":
    unittest.main(verbosity=2)
