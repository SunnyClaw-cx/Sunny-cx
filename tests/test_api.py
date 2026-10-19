# -*- coding: utf-8 -*-
"""API 契约测试（stdlib unittest，无需 pytest）

运行：
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient          # noqa: E402

from server.app import app                          # noqa: E402

DOSE = re.compile(r"\d+\s*(?:g|克|钱|两|毫克|ml|毫升)")


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = TestClient(app)
        cls.c.__enter__()
        cls.syndromes = cls.c.get("/api/syndromes").json()["items"]
        cls.names = {s["syndrome"] for s in cls.syndromes}
        cls.formulas = set()
        for s in cls.syndromes:
            cls.formulas.add(s["formula"])
            cls.formulas.update(s["alt"])
            cls.formulas.add(re.sub(r"[（(].*?[)）]", "", s["formula"]).strip())

    @classmethod
    def tearDownClass(cls):
        cls.c.__exit__(None, None, None)

    # ---------------- 基础 ----------------
    def test_health(self):
        r = self.c.get("/api/health")
        self.assertEqual(r.status_code, 200)
        j = r.json()
        self.assertEqual(j["status"], "ok")
        self.assertEqual(j["engine"]["syndromes"], 35)

    def test_meta_has_thresholds(self):
        j = self.c.get("/api/meta").json()
        for k in ("ratio_ok", "ratio_weak", "tie_eps"):
            self.assertIn(k, j["thresholds"])
        self.assertIn("不构成诊疗建议", j["disclaimer"])

    def test_empty_text_rejected(self):
        self.assertEqual(self.c.post("/api/consult", json={"text": "   "}).status_code, 422)

    # ---------------- 判定 ----------------
    def test_ok_path_is_inside_table(self):
        j = self.c.post("/api/consult", json={
            "text": "我最近老觉得累，一动就出汗，舌头发紫，身上老是一个地方刺痛"
        }).json()
        self.assertEqual(j["decision"], "ok")
        self.assertIn(j["syndrome"], self.names)
        self.assertIn(j["formula"]["formula"], self.formulas)
        self.assertTrue(j["selfcheck"]["pass"])

    def test_refuse_returns_no_candidates(self):
        """拒答时不许回 0 分候选：那等于凭空造了一个结论。"""
        j = self.c.post("/api/consult", json={"text": "我胃不舒服"}).json()
        self.assertEqual(j["decision"], "refuse")
        self.assertEqual(j["candidates"], [])

    def test_answer_never_contains_dose(self):
        for text in (
            "我最近老觉得累，一动就出汗，舌头发紫，身上老是一个地方刺痛",
            "晚上睡不着，手心脚心发热，夜里出汗，嘴干",
            "小便发黄，阴囊潮湿，两条腿发沉",
        ):
            ans = self.c.post("/api/consult", json={"text": text}).json()["answer"]
            self.assertIsNone(DOSE.search(ans), f"输出出现剂量：{ans}")

    def test_answer_carries_disclaimer(self):
        ans = self.c.post("/api/consult", json={"text": "晚上睡不着，手心脚心发热，夜里出汗，嘴干"}).json()["answer"]
        self.assertIn("不构成诊疗建议", ans)

    # ---------------- 闸门 ----------------
    def test_gate_blocks_fabricated_formula(self):
        bad = "【参考方剂】益气活血通脉汤（《自制方》）\n以上仅为中医知识性说明，不构成诊疗建议；具体请面诊执业中医师。"
        j = self.c.post("/api/check", json={"text": bad}).json()
        self.assertFalse(j["pass"])
        self.assertTrue(any("方名" in v for v in j["violations"]))

    def test_gate_blocks_dose(self):
        bad = "【参考药味】黄芪30克\n以上仅为中医知识性说明，不构成诊疗建议；具体请面诊执业中医师。"
        j = self.c.post("/api/check", json={"text": bad}).json()
        self.assertFalse(j["pass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
