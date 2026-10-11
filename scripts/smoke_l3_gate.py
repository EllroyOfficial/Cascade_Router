# -*- coding: utf-8 -*-
"""smoke_l3_gate.py —— L3 质量门行为冒烟(2026-10-10 收尾,桩件无GPU)"""
from __future__ import annotations
# _hr_bootstrap_v1: 本脚本在 scripts\ 下，需要把仓库根加进 sys.path 才能 import 核心模块
import os as _hr_os, sys as _hr_sys
_hr_sys.path.insert(0, _hr_os.path.dirname(_hr_os.path.dirname(_hr_os.path.abspath(__file__))))

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router import HybridRouter  # noqa: E402
from ragas_local import relevancy, faithfulness  # noqa: E402
from interfaces import RoutingDecision  # noqa: E402


class StubEmbedder:  # 含"所得税"→[1,0],否则[0,1]
    def embed(self, text):
        return [1.0, 0.0] if "所得税" in text else [0.0, 1.0]


class StubL3:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def ask(self, _):
        self.calls += 1
        return self.replies.pop(0) if self.replies else ""


rt = HybridRouter(config_path=os.path.join(
    os.path.dirname(os.path.abspath(__file__), ) if False else
    os.path.dirname(os.path.abspath(__file__)), "routing.yaml"))
Q = "递延所得税怎么处理？"
GOOD, BAD = "递延所得税按暂时性差异确认。", "今天天气真不错呀。"

# ① relevancy 本身:相关高、无关低
assert relevancy(Q, GOOD, StubEmbedder()) > 0.9, " relevancy 相关对应为高"
assert relevancy(Q, BAD, StubEmbedder()) < 0.1, " relevancy 无关对应为低"

# ② 门关:透传+原话术
rt.l3, rt.l3_gate, rt.embedder = StubL3([BAD]), None, StubEmbedder()
d = RoutingDecision(intent="chat")
ans = rt._ask_l3_gated(Q, d)
assert ans == BAD and d.reasons[-1] == "L3 provider 答复成功", " 门关应透传"

# ③ 门开:首答不过→重试→次答过
rt.l3, rt.l3_gate, rt.embedder = StubL3([BAD, GOOD]), {"relevancy_min": 0.5, "max_retries": 1}, StubEmbedder()
d = RoutingDecision(intent="chat")
ans = rt._ask_l3_gated(Q, d)
assert ans == GOOD and rt.l3.calls == 2, " 应重试一次后收货"
assert rt.metrics.snapshot()["l3_gate_retry"] == 1, " 重试计数"

# ④ 门开:两答都不过→None+拒绝计数
rt.l3, rt.embedder = StubL3([BAD, BAD]), StubEmbedder()
d = RoutingDecision(intent="chat")
ans = rt._ask_l3_gated(Q, d)
assert ans is None and rt.l3.calls == 2, " 重试耗尽应返回 None"
assert rt.metrics.snapshot()["l3_gate_rejected"] == 1, " 拒绝计数"

# ⑤ faithfulness 冒烟:含材料关键词的句子覆盖
cov = faithfulness("递延所得税确认暂时性差异。今天天气不错。", "递延所得税确认暂时性差异。", StubEmbedder())
assert 0.4 <= cov <= 0.6, f" faithfulness 应约半句覆盖,得到{cov}"

print("L3 质量门冒烟: 5/5 PASS")
