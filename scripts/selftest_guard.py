# -*- coding: utf-8 -*-
"""guard 三规则（否定/方向/数字）的复验集 —— 可一键复跑：python scripts\selftest_guard.py

用例设计：每条都给"期望 deny/放行"，跑完直接给 PASS/FAIL，不靠肉眼。
（2026-10-09 用户批准的三防线方案里的"第一道：正则截杀"。）
"""
# _hr_bootstrap_v1: 本脚本在 scripts\ 下，需要把仓库根加进 sys.path 才能 import 核心模块
import os as _hr_os, sys as _hr_sys
_hr_sys.path.insert(0, _hr_os.path.dirname(_hr_os.path.dirname(_hr_os.path.abspath(__file__))))
# ── 路径去标识（不再硬编码用户名；可用环境变量 HR_DATA_ROOT 覆盖数据根）
import os as _hr_os
_HR_HOME = _hr_os.path.expanduser("~")
_HR_DESKTOP = _hr_os.environ.get("HR_DATA_ROOT") or _hr_os.path.join(_HR_HOME, "Desktop")
_HR_DIR = _hr_os.path.dirname(_hr_os.path.abspath(__file__))
import sys

sys.path.insert(0, _HR_DIR)
import anchors as A  # noqa: E402

MAT = "非正常损失对应的进项税额不得从销项税额中抵扣，小规模纳税人征收率减按 1% 征收。"
MAT2 = "可抵扣暂时性差异应确认递延所得税资产。"

CASES = [
    # (标签, 材料, 回答, 期望 deny?)
    ("正-真件改写(曾假阳性)", MAT, "按现行规定，非正常损失的进项税额不得抵扣；小规模纳税人征收率减按 1%。", False),
    ("正-口语全项", MAT, "按现行规定：非正常损失对应的进项税额不得从销项税额中抵扣，小规模纳税人征收率减按 1% 征收。", False),
    ("正-合法省略", MAT, "征收率减按 1%。", False),
    ("反-否定丢了", MAT, "非正常损失可以抵扣进项税额。", True),
    ("反-改数字", MAT, "小规模纳税人征收率减按 3%。", True),
    ("反-资产说成负债", MAT2, "该可抵扣暂时性差异应确认递延所得税负债。", True),
    ("正-照说资产", MAT2, "该可抵扣暂时性差异应确认递延所得税资产。", False),
]


def main() -> None:
    ok = 0
    for tag, mat, ans, expect_deny in CASES:
        g = A.guard_check(mat, ans)
        good = g["deny"] == expect_deny
        ok += good
        print(f"{'PASS' if good else 'FAIL'}  {tag:22s} deny={str(g['deny']):5s} fired={g['fired']}  {g['summary']}")
        if g["deny"] and not expect_deny:
            for _k, v in g["violations"].items():
                for item in v.get("violations") or []:
                    print("        误杀证据:", item.get("reason"))
    print(f"\n{ok}/{len(CASES)} 符合预期")
    assert ok == len(CASES), "有不符合预期的用例"
    print("OK")


if __name__ == "__main__":
    main()
