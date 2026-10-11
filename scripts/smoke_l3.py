"""smoke_l3.py —— L3 云端真件冒烟(走 ingress,勿连发)

① 真件一问:L3Provider.ask → 答复(会话 session-11519cfd 已暖,只发一条);
② 降级路径:坏信标 → router fail-open → 应得"[L3 云端不可用,已降级本地]"如实前缀,而非静默。
跑法:python scripts\smoke_l3.py
"""
from __future__ import annotations
# _hr_bootstrap_v1: 本脚本在 scripts\ 下，需要把仓库根加进 sys.path 才能 import 核心模块
import os as _hr_os, sys as _hr_sys
_hr_sys.path.insert(0, _hr_os.path.dirname(_hr_os.path.dirname(_hr_os.path.abspath(__file__))))

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l3_provider import DSHIngressL3Provider, L3UnavailableError
from router import HybridRouter

print("① 真件一问(单发,会话已暖):")
p = DSHIngressL3Provider()
t0 = time.time()
try:
    reply = p.ask("只回一句话:1+1等于几?")
    print(f"  {time.time()-t0:.1f}s → {reply[:60]!r}")
except L3UnavailableError as e:
    print(f"  L3 不可用(如实): {e}")

print("② 降级路径(坏信标+必 fail-open 分类桩):")


class _FailOpenStub:
    def classify(self, prompt, temperature=0.0, max_tokens=10, timeout_ms=2000):
        return "大概是专业问题吧我觉得"   # 枚举外 → fail-open → L3


rt = HybridRouter(
    config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "routing.yaml"),
    classifier=_FailOpenStub(),
    l3_provider=DSHIngressL3Provider(beacon_path=r"C:\not\exist\beacon.json"))
out = rt.route("我那个破电脑又出问题了")
hit = out.answer.startswith("[L3 云端不可用,已降级本地]")
print(f"  answer={out.answer[:52]!r}")
print(f"  reasons={out.decision.reasons}")
print(f"  如实报错+降级={'✓' if hit else '✗'}")
