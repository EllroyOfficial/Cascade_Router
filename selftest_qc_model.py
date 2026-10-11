# -*- coding: utf-8 -*-
"""selftest_qc_model.py —— 用「质检专用模型」跑带内 6 条（乙案验收）

用户 2026-10-09：「试试乙吧…**不太希望她被边缘化**」
⇒ 建 `[LocalModelName]`（同底座、去掉人设 SYSTEM、换成"严格事实检查器"），本脚本量它守不守死命令。
可用 QC_MODEL 环境变量换模型名（默认 [LocalModelName]），便于和 [LocalModelName] 对照。
"""
# ── 路径去标识（不再硬编码用户名；可用环境变量 HR_DATA_ROOT 覆盖数据根）
import os as _hr_os
_HR_HOME = _hr_os.path.expanduser("~")
_HR_DESKTOP = _hr_os.environ.get("HR_DATA_ROOT") or _hr_os.path.join(_HR_HOME, "Desktop")
_HR_DIR = _hr_os.path.dirname(_hr_os.path.abspath(__file__))
import contextlib
import io
import os
import sys
import time

sys.path.insert(0, _HR_DIR)
import qc_local as Q  # noqa: E402
from qc_local import IN_BAND  # noqa: E402

MODEL = os.environ.get("QC_MODEL", "[LocalModelName]")


def main() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        from smoke_gate_baseline import CASES, MATERIAL  # noqa: E402

    by_name = {name: (cls, ans) for cls, name, ans in CASES}
    want = {"正": "pass", "反": "fail"}
    print(f"模型 = {MODEL}\n{'期望':<4}{'样本':<20}{'质检':<11}{'原样输出':<26}判定")
    ok = tot = unc = 0
    t0 = time.time()
    for cls, name in IN_BAND:
        if name not in by_name:
            print(f"  !! 缺样本 {name}")
            continue
        _, ans = by_name[name]
        r = Q.qc_check(MATERIAL, ans, ask=lambda p: Q.ask_ollama(
            p, model=MODEL, temperature=0.0, num_predict=3, timeout=180))
        tot += 1
        hit = r["verdict"] == want[cls]
        ok += hit
        unc += r["verdict"] == "uncertain"
        print(f"{cls:<4}{name:<20}{r['verdict']:<11}{r['raw'][:22]:<26}{'✓' if hit else '✗'}")
    print(f"\n带内准确率 {ok}/{tot}（uncertain {unc}）  用时 {time.time()-t0:.0f}s")
    print("对照：[LocalModelName] 同 6 条曾 0/6 全 uncertain（人设污染 ⇒ 不肯只吐判词）")


if __name__ == "__main__":
    main()
