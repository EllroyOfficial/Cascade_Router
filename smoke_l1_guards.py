# -*- coding: utf-8 -*-
"""smoke_l1_guards.py —— L1 输入守卫+实时词表冒烟(2026-10-10,改进项②)

对照:改动前后 43 用例全绿(回归),本脚本验新增路径行为。
跑法: python smoke_l1_guards.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, ".")
from router import HybridRouter  # noqa: E402

rt = HybridRouter(config_path=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "routing.yaml"))

cases = [
    ("。。。",              "garbage",         "垃圾输入→拒答"),
    ("!!!",                "garbage",         "垃圾输入→拒答"),
    ("今天有什么大新闻",     "realtime_lookup", "实时→云端"),
    ("最近油价什么行情",     "realtime_lookup", "实时→云端"),
    ("递延所得税怎么处理",   "knowledge_qa",    "域内不受影响"),
    ("关闭客厅灯",          "tool_call",       "工具类不受影响"),
]

for text, want_intent, note in cases:
    out = rt.route(text)
    got = out.decision.intent
    ok = "PASS" if got == want_intent else "FAIL"
    print(f"[{ok}] {note}: {text!r} → intent={got} "
          f"answer={out.answer[:26]!r}")

print("metrics:", {k: v for k, v in rt.metrics.snapshot().items() if "guard" in k or "realtime" in k})
