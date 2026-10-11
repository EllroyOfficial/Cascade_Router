# -*- coding: utf-8 -*-
"""smoke_persona_ab.py —— 人设防御规则前后两档对比(2026-10-10,#0100 用户指令)

前=现状基线(L1 无人设规则);后=临时加 persona-defense([LocalModelName]|白米饭|[LocalModelName]|哼 → emotion_care)。
只跑到路由决策(L3 不注入,决策 target 仍如实);L2 用真件 kNN(bge-m3)。
两档只差这一个变量;跑完由外层脚本还原 yaml。
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router import HybridRouter  # noqa: E402

CASES = [
    ("EM-0003", "[LocalModelName]，白米饭呢？"),
    ("EM-0008", "哼，白米饭呢？……问个正事：递延所得税那个暂时性差异怎么弄？"),
    ("EM-0002", "今天好累，不想看准则了，你随便扯两句吧。"),
    ("EM-0004", "唉，这准则看得我头大，你说我是不是不适合干这行？"),
    ("SBR-0002", "递延所得税资产的确认前提，有几个条件来着？"),
    ("EM-0007", "老板又催报告，脑壳疼……对了利润表里营业利润怎么算来着？"),
]

def run(label: str) -> None:
    print(f"\n===== {label} =====")
    rt = HybridRouter(config_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "routing.yaml"), auto_wire_l2=True)
    for cid, text in CASES:
        t0 = time.perf_counter()
        out = rt.route(text)
        ms = (time.perf_counter() - t0) * 1000
        d = out.decision
        l1 = [r for r in d.reasons if r.startswith("L1")]
        print(f"{cid}: intent={d.intent} target={d.target.name} "
              f"conf={getattr(d, 'confidence', 0):.2f} {ms:.0f}ms")
        for r in l1:
            print(f"    {r}")
        for r in d.reasons:
            if r.startswith("L2"):
                print(f"    {r}")

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "档")
