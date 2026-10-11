# -*- coding: utf-8 -*-
"""selftest_numcalc.py —— 数字口径验收(#0187/#0188,交接文档①的六行验收表)

链路=AnswerGuard.check 的 number_consistency 判据(可推导经 anchors.safe_calc 家族,
单位归一+绝对容差 0.01);语义判据用桩件放行(本表只验数字口径)。
跑法: python scripts\selftest_numcalc.py   —— 全 PASS 即验收过;任何 FAIL 打印明细。
"""
from __future__ import annotations
# _hr_bootstrap_v1: 本脚本在 scripts\ 下，需要把仓库根加进 sys.path 才能 import 核心模块
import os as _hr_os, sys as _hr_sys
_hr_sys.path.insert(0, _hr_os.path.dirname(_hr_os.path.dirname(_hr_os.path.abspath(__file__))))

import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from answer_guard import AnswerGuard  # noqa: E402
from metrics import Metrics  # noqa: E402


class _SemStub:
    """语义桩:恒 0.9(放行),本表只验数字口径;真实语义另由 embed_local 管。"""

    def similarity(self, a: str, b: str) -> float:
        return 0.9


CONFIG = {"answer_guard": {"criteria": {"semantic_sim": {"min_sim": 0.60},
                                        "number_consistency": True,
                                        "length_ratio_max": 3},
                           "number_tol": 0.01,
                           "derivable_ops": ["add", "sub", "mul", "div", "pct"],
                           "on_fail": ["cite_original", "refuse_no_material"]}}
GUARD = AnswerGuard(CONFIG, _SemStub(), Metrics())

R_TAX = "销售额 100 万元，适用税率 13%。"
R_RATE = "适用税率 0.35。"
R_SMALL = "小规模纳税人增值税征收率减按 1% 征收，月销售额 10 万元以下免征。"

CASES = [
    # (说明, query, rag_raw, answer, 期望: pass/动作/含标注?)
    ("验收1 13.00 万元写法", "100万×13% 是多少？", R_TAX, "13.00 万元",
     True, "", False),
    ("验收1 130000.00 元写法", "100万×13% 是多少？", R_TAX, "130000.00 元",
     True, "", False),
    ("验收2 1/3 差0.0033 放行+标注", "1/3 约等于多少？", "把总额按三份平均分，每份占三分之一。", "0.33",
     True, "舍入差0.01", True),
    ("验收3 该0.35答0.36 放行+标注", "税率是多少？", R_RATE, "0.36",
     True, "舍入差0.01", True),
    ("验收4 该0.35答0.40 Fail→直出原文", "税率是多少？", R_RATE, "0.40",
     False, "number_consistency", False),
    ("验收5 单位混写 0.85万元=8500元", "预收租金是多少？", "预收租金 0.85 万元。", "8500 元",
     True, "", False),
    ("验收6 编造数字仍拦", "征收率是多少？", R_SMALL, "减按 0.3% 征收，月销售额 15 万元以下免征。",
     False, "number_consistency", False),
    ("回归 无数字差异不受影响", "递延所得税怎么处理？", "暂时性差异应确认递延所得税资产。", "先找暂时性差异，再确认递延所得税资产，哼。",
     True, "", False),
]

ok = True
for note, q, rag, ans, want_pass, want_note, want_tag in CASES:
    r = GUARD.check(q, rag, ans, number_ref_query=True)
    hit = (r.passed == want_pass) and (want_note in (r.failed_criterion + r.note)) \
        and (want_tag == ("舍入差0.01" in r.note))
    ok &= hit
    mark = "PASS" if hit else "FAIL"
    print(f"[{mark}] {note}: passed={r.passed} action={r.action} "
          f"criterion={r.failed_criterion!r} note={r.note!r}")
print("\n结论:", "全部 PASS —— 计算现在由 Python 出、模型只表述" if ok else "存在 FAIL,见上")
print("能力边界: 只做四则+百分比+单位归一(元/千元/万元);摊余成本/利息计提/多步带单位换算 ⇒ 算不了,转人工或直出原文")
sys.exit(0 if ok else 1)
