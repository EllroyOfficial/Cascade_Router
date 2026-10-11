"""smoke_l3_strip.py —— L3 尾缀剥离自测(2026-10-09,派活②交付件)

① 纯字符串用例:锚定行尾精确剥「—— 本回合结束」,正文破折号不受影响;
② 真件单发一条(会话已暖,勿连发),断言答复不再带尾缀。
跑法:python smoke_l3_strip.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l3_provider import strip_session_delimiter, DSHIngressL3Provider, L3UnavailableError

CASES = [
    ("递延所得税按暂时性差异确认。—— 本回合结束", "递延所得税按暂时性差异确认。"),
    ("递延所得税按暂时性差异确认。 ——  本回合结束\n", "递延所得税按暂时性差异确认。"),
    ("借:递延所得税资产 —— 贷:应交税费。", None),      # 正文破折号:不许动
    ("本回合结束了。", None),                            # 非完整分隔符:不许动
    ("", None),
]
ok = True
for src, want in CASES:
    got = strip_session_delimiter(src)
    expect = want if want is not None else src
    mark = "✓" if got == expect else "✗"
    ok = ok and (got == expect)
    print(f"  {mark} {src!r} → {got!r}")

print("② 真件单发(单发一次):")
try:
    reply = DSHIngressL3Provider().ask("只回一句话:1+1等于几?")
    has_suffix = "本回合结束" in reply
    ok = ok and not has_suffix
    print(f"  答复={reply!r}")
    print(f"  尾缀已剥={'✓' if not has_suffix else '✗'}")
except L3UnavailableError as e:
    print(f"  L3 不可用(如实,本用例不判过): {e}")

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
