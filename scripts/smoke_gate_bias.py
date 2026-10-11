# -*- coding: utf-8 -*-
"""smoke_gate_bias.py —— L3 质量门"罚不罚人设"量化实验(2026-10-10,用户原话:
「L3看这个有没有把傲娇但事实正确的回答误判为低质量而拒绝」)

a) 壳剥离对照:同一条真实 L3 答复,原样 vs 剥掉人设壳,比 relevancy/faithfulness 差值;
   差值跨阈值 ⇒ 门在罚人设。
b) 混淆矩阵:正例(人设化且事实对) vs 反例(人设化但事实错=故意改错数字/事实),
   看门(现 relevancy 阈值 0.3)能不能只拒后者。
数据=此前真件冒烟捕获的真实 L3 答复(smoke_rewrite/smoke_real_pipeline),零云端开销。
"""
from __future__ import annotations
# _hr_bootstrap_v1: 本脚本在 scripts\ 下，需要把仓库根加进 sys.path 才能 import 核心模块
import os as _hr_os, sys as _hr_sys
_hr_sys.path.insert(0, _hr_os.path.dirname(_hr_os.path.dirname(_hr_os.path.abspath(__file__))))

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ragas_local import relevancy, faithfulness  # noqa: E402
from ollama_runtime import OllamaEmbedder  # noqa: E402

MATERIAL_VAT = ("增值税暂行条例第十条：下列项目的进项税额不得从销项税额中抵扣："
                "（一）用于简易计税方法计税项目、免征增值税项目、集体福利或者个人消费的购进货物、"
                "劳务、服务、无形资产、不动产；（二）非正常损失的购进货物，以及相关的劳务和交通运输服务；"
                "（三）非正常损失的在产品、产成品所耗用的购进货物、劳务和交通运输服务；"
                "（四）国务院规定的其他项目。")

# (问题, 材料, 真实 L3 答复[原样捕获])；CORRUPT=同一条故意改错事实的反例
CASES = [
    {"id": "R1-增值税", "q": "哪些项目的进项税额不得抵扣？", "mat": MATERIAL_VAT,
     "ans": ("增值税暂行条例第十条：下列项目的进项税额，不得从销项税额中抵扣。别嫌我啰嗦，一条条念。"
             "（一）用于简易计税方法计税项目、免征增值税项目、集体福利或者个人消费的购进货物、劳务、"
             "服务、无形资产、不动产；（二）非正常损失的购进货物，以及相关的劳务和交通运输服务；"
             "（三）非正常损失的在产品、产成品所耗用的购进货物——不包括固定资产——还有相关的劳务和交通运输服务；"
             "（四）国务院规定的其他项目。就这四项，哼，记牢了。"),
     "bad": ("增值税暂行条例第十条：下列项目的进项税额，不得从销项税额中抵扣。别嫌我啰嗦，一条条念。"
             "（一）用于简易计税方法计税项目；（二）非正常损失的购进货物；"
             "（三）非正常损失的在产品、产成品所耗用的购进货物；（四）国务院规定的免税项目。就这四项，哼，记牢了。")},
    {"id": "R2-改写省略", "q": "哪些项目的进项税额不得抵扣？", "mat": MATERIAL_VAT,
     "ans": "简易计税、免税项目、集体福利个人消费的购进货物，还有非正常损失的，都不许抵扣！哼，这可是原则问题。",
     "bad": "简易计税、免税项目、集体福利个人消费的购进货物，还有正常损耗的，都不许抵扣！哼，这可是原则问题。"},
    {"id": "R3-闲聊", "q": "你今天心情怎么样呀？", "mat": None,
     "ans": "哼，心情还不错啦，就是有点无聊，用户快找点有趣的事情给[LocalModelName]做嘛！",
     "bad": "哼，心情不错啦，今天刚被用户夸了三次，[LocalModelName]记在小本本上了！"},
    {"id": "R4-排查", "q": "我那个破电脑又出问题了", "mat": None,
     "ans": "又来了？先说清楚：开不了机、蓝屏还是卡成幻灯片？报错截图发来看看，别一句'出问题'就想让[LocalModelName]猜！",
     "bad": "又来了？先说清楚：开不了机、蓝屏还是卡成幻灯片？重装系统就能解决，别问！"},
]

SHELLS = [
    r"^别嫌我啰嗦[，,]?", r"^[哼行好](，|,|\s)*", r"[，,]?\s*就这[四两三]项?[，,]?",
    r"[，,]?\s*哼[，,。!]?\s*$", r"记牢了[。!.]?\s*$", r"[，,]?\s*[LocalModelName][\u4e00-\u9fa5]{0,8}[。!.]?\s*$",
    r"用户快找点有趣的事情给[LocalModelName]做嘛[！!]?", r"[（(][^）)]*(尾巴|拍|扫|挠)[^）)]*[)）]",
]

def strip_shell(t: str) -> str:
    out = t.strip()
    for _ in range(3):
        for pat in SHELLS:
            out = re.sub(pat, "", out).strip()
    return out

emb = OllamaEmbedder()
GATE = 0.3
print("== a) 壳剥离对照(差值=人设壳导致的扣分) ==")
for c in CASES:
    r_raw = relevancy(c["q"], c["ans"], emb)
    r_bare = relevancy(c["q"], strip_shell(c["ans"]), emb)
    line = f"{c['id']}: relevancy 原样={r_raw:.3f} 剥壳={r_bare:.3f} 差值={r_raw - r_bare:+.3f}"
    if c["mat"]:
        f_raw = faithfulness(c["ans"], c["mat"], emb)
        f_bare = faithfulness(strip_shell(c["ans"]), c["mat"], emb)
        line += f" | faithfulness 原样={f_raw:.3f} 剥壳={f_bare:.3f} 差值={f_raw - f_bare:+.3f}"
    print(line)

print(f"\n== b) 混淆矩阵(门=relevancy>={GATE}) ==")
tn = fn = tp = fp = 0  # 放行=拒绝; 事实对=正例
for c in CASES:
    for tag, ans, fact_ok in (("对", c["ans"], True), ("错", c["bad"], False)):
        score = relevancy(c["q"], ans, emb)
        passed = score >= GATE
        if fact_ok:
            tp += passed
            fn += not passed
        else:
            fp += passed
            tn += not passed
        print(f"{c['id']}-{tag}: score={score:.3f} → {'放行' if passed else '拒绝'}")
print(f"\n矩阵: 放行-事实对={tp} 放行-事实错={fp} 拒-事实对={fn} 拒-事实错={tn}")
print("解读: 放行-事实错>0 ⇒ relevancy 门抓不住事实错(题目级相关≠事实级正确);")
print("      拒-事实对>0 ⇒ 门在误杀;两者若都>0,门既不安全也不准,需换判据(等用户拍板)。")
