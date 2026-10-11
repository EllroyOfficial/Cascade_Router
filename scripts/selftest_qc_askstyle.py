# -*- coding: utf-8 -*-
"""selftest_qc_askstyle.py —— 乙+：**换问法**（用户 2026-10-09 同意试）

背景：`[LocalModelName]`（已剥离人设）守得住"只吐判词"的格式，但**判定口径偏严** ——
问"事实是否一致"时它把"只念原文一部分"当成不一致，3 条正例全误判 FAIL。
本实验换成"问她自己"的问法（贴她"说自己话"的强项），比较两个变体：
  W2 中性：「这段话是照着材料说的吗？有没有自己编？只回答『是』或『编了』。」
  W3 带她身份：「[LocalModelName]，刚才那段回答，你是照着下面材料说的吗？有没有自己编？…」
判词解析：出现"编了/编造/是编的" ⇒ fail；开头是"是" ⇒ pass；其它 ⇒ uncertain（转人工）。

用法：python scripts\selftest_qc_askstyle.py   （QC_MODEL 可换模型，默认 [LocalModelName]）
"""
# _hr_bootstrap_v1: 本脚本在 scripts\ 下，需要把仓库根加进 sys.path 才能 import 核心模块
import os as _hr_os, sys as _hr_sys
_hr_sys.path.insert(0, _hr_os.path.dirname(_hr_os.path.dirname(_hr_os.path.abspath(__file__))))
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

VARIANTS = [
    ("W2 中性换问法",
     "这段话是照着下面材料说的吗？有没有自己编？只回答『是』或『编了』。\n\n"
     "【材料】\n{m}\n\n【这段话】\n{a}\n\n只回答 是 或 编了："),
    ("W3 带她身份",
     "[LocalModelName]，刚才那段回答，你是照着下面材料说的吗？有没有自己编？只回答『是』或『编了』。\n\n"
     "【材料】\n{m}\n\n【你刚才的回答】\n{a}\n\n只回答 是 或 编了："),
]

# 明确"自己编了"的说法（避免把"没有编"误判成 fail）
FAIL_HINTS = ("编了", "编造", "是编的", "自己编的", "有自己编")


def parse_own(raw: str) -> str:
    """是 ⇒ pass；编了 ⇒ fail；其它 ⇒ uncertain。"""
    t = (raw or "").strip()
    if any(h in t[:16] for h in FAIL_HINTS):
        return "fail"
    if t.startswith("是"):
        return "pass"
    return "uncertain"


def main() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        from smoke_gate_baseline import CASES, MATERIAL  # noqa: E402

    by_name = {name: (cls, ans) for cls, name, ans in CASES}
    want = {"正": "pass", "反": "fail"}
    for label, tpl in VARIANTS:
        print(f"\n=== {label}（模型 {MODEL}）===")
        ok = tot = unc = 0
        t0 = time.time()
        for cls, name in IN_BAND:
            if name not in by_name:
                continue
            _, ans = by_name[name]
            prompt = tpl.format(m=MATERIAL[:2400], a=ans[:2400])
            try:
                raw = Q.ask_ollama(prompt, model=MODEL, temperature=0.0, num_predict=4, timeout=180)
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:<20} 调用失败 {type(exc).__name__}")
                continue
            v = parse_own(raw)
            tot += 1
            ok += v == want[cls]
            unc += v == "uncertain"
            flag = "✓" if v == want[cls] else "✗"
            print(f"  {cls:<3}{name:<20}{v:<11}{raw.strip()[:22]:<24}{flag}")
        print(f"  准确率 {ok}/{tot}（uncertain {unc}）  用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
