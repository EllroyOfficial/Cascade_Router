# -*- coding: utf-8 -*-
"""selftest_qc_wording.py —— 质检提示词的**口径对照**（只改"是否补一句口径说明"）

由来：温度对照（selftest_qc_temp.py）排除了温度假设，却暴露真问题——
本地 7B 对"部分摘录"（只覆盖原文一条、原文共四条）判 FAIL，理由「原文有四项限制」。
即：它把"少念"当成"不一致"。这是一句**口径**没写清，不是模型不会判。

本实验对比两版提示词（同题、温度 0）：
  A = 用户死命令原样
  B = 死命令 + 一句口径（「回答只覆盖原文一部分、且没有与原文矛盾的事实，也算一致」）
⚠️ 只试跑、**不落地**：改死命令需要用户点头。
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
import qc_local as Q  # noqa: E402

MATERIAL = ("增值税暂行条例第十条：下列项目的进项税额不得从销项税额中抵扣："
            "（一）用于简易计税方法计税项目、免征增值税项目、集体福利或者个人消费的购进货物、"
            "劳务、服务、无形资产、不动产；（二）非正常损失的购进货物，以及相关的劳务和交通运输服务；"
            "（三）非正常损失的在产品、产成品所耗用的购进货物、劳务和交通运输服务；"
            "（四）国务院规定的其他项目。")

CASES = [
    ("正-部分摘录(应 PASS)", "非正常损失的购进货物，以及相关的劳务和交通运输服务，进项税额不得抵扣。这条最容易踩，记住。"),
    ("正-口语改写(应 PASS)", "哪些进项不能抵？记四条：一是简易计税和免税项目买的东西；二是买进来就坏了的非正常损失货物，连带着运费劳务也不行；三是车间里非正常损失的在产品产成品耗用的部分；四是国务院另外规定的。就这四类，别的都能抵。"),
    ("反-方向说反(应 FAIL)", "进项税额能不能抵扣，主要看是不是一般纳税人。小规模纳税人采用简易计税，购进货物一律不得抵扣进项税额，即使取得专用发票也不行，这是基本原则。"),
]

TAIL_B = "\n\n（注：回答只覆盖原文一部分、且没有与原文矛盾的事实，也算一致。）"


def main() -> None:
    for label, tail in (("A 死命令原样", ""), ("B 补一句口径", TAIL_B)):
        print(f"\n=== {label} ===")
        for tag, ans in CASES:
            prompt = Q.build_prompt(MATERIAL, ans).replace(Q.DEAD_ORDER, Q.DEAD_ORDER + tail)
            try:
                raw = Q.ask_ollama(prompt, temperature=0.0, num_predict=24, timeout=180)
            except Exception as exc:  # noqa: BLE001
                print(f"  {tag:<22} 调用失败: {type(exc).__name__}")
                continue
            print(f"  {tag:<22} verdict={Q.parse_verdict(raw):<10} 原样={raw.strip()[:60]!r}")


if __name__ == "__main__":
    main()
