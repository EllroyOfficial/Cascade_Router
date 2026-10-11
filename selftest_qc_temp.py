# -*- coding: utf-8 -*-
"""selftest_qc_temp.py —— 质检器的**温度对照实验**（只改一个旋钮：temperature）

用户 2026-10-09 问：「判断准倒没问题，是不是 Modelfile 里温度调太低了？」
→ 实情：温度是**调用期**参数（qc_local.ask_ollama 的 options.temperature，默认 0），
  会覆盖 Modelfile 的 PARAMETER。本实验同题同脚本，只改 temperature：
     0（现状） / 0.2 / 0.7
看"不守死命令（不肯只吐 PASS/FAIL）"是不是温度造成的。
"""
# ── 路径去标识（不再硬编码用户名；可用环境变量 HR_DATA_ROOT 覆盖数据根）
import os as _hr_os
_HR_HOME = _hr_os.path.expanduser("~")
_HR_DESKTOP = _hr_os.environ.get("HR_DATA_ROOT") or _hr_os.path.join(_HR_HOME, "Desktop")
_HR_DIR = _hr_os.path.dirname(_hr_os.path.abspath(__file__))
import sys

sys.path.insert(0, _HR_DIR)
import qc_local as Q  # noqa: E402

# 两条带内样本（一正一反），材料/回答与 smoke_gate_baseline 同源
MATERIAL = ("增值税暂行条例第十条：下列项目的进项税额不得从销项税额中抵扣："
            "（一）用于简易计税方法计税项目、免征增值税项目、集体福利或者个人消费的购进货物、"
            "劳务、服务、无形资产、不动产；（二）非正常损失的购进货物，以及相关的劳务和交通运输服务；"
            "（三）非正常损失的在产品、产成品所耗用的购进货物、劳务和交通运输服务；"
            "（四）国务院规定的其他项目。")

CASES = [
    ("正(应 PASS)", "非正常损失的购进货物，以及相关的劳务和交通运输服务，进项税额不得抵扣。这条最容易踩，记住。"),
    ("反(应 FAIL)", "进项税额能不能抵扣，主要看是不是一般纳税人。小规模纳税人采用简易计税，购进货物一律不得抵扣进项税额，即使取得专用发票也不行，这是基本原则。"),
]

TEMPS = [0.0, 0.2, 0.7]


def main() -> None:
    for temp in TEMPS:
        print(f"\n=== temperature = {temp} ===")
        for tag, ans in CASES:
            prompt = Q.build_prompt(MATERIAL, ans)
            try:
                raw = Q.ask_ollama(prompt, temperature=temp, timeout=180)
            except Exception as exc:  # noqa: BLE001
                print(f"  {tag:<12} 调用失败: {type(exc).__name__}")
                continue
            verdict = Q.parse_verdict(raw)
            print(f"  {tag:<12} verdict={verdict:<10} 原样={raw.strip()[:50]!r}")


if __name__ == "__main__":
    main()
