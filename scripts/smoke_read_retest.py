# -*- coding: utf-8 -*-
"""smoke_read_retest.py —— 念稿重测(2026-10-10,方案A验证)

目的:昨晚"两版都没过"的结论可能被问法背锅,同模型两档消融对齐结论:
- 档① 基线复现: 旧问法(问答式)+num_predict=200 → 应复现"含混带过"
- 档② 方案A:    硬指令逐字念稿+num_predict=400 → 看问法修正后过不过
- 其余参数不动: temperature=0.2、num_ctx=2048、两版串行防显存互挤。
- 判分: 材料四个条目的锚点词是否都出现在输出里(出口硬校验,B案的雏形)。
跑法: python scripts\smoke_read_retest.py
"""
from __future__ import annotations
# _hr_bootstrap_v1: 本脚本在 scripts\ 下，需要把仓库根加进 sys.path 才能 import 核心模块
import os as _hr_os, sys as _hr_sys
_hr_sys.path.insert(0, _hr_os.path.dirname(_hr_os.path.dirname(_hr_os.path.abspath(__file__))))

import json
import time
import urllib.request

HOST = "http://127.0.0.1:11434/api/generate"
MODELS = ["[LocalModelName]", "[LocalModelName]"]
BASE_OPTS = {"num_ctx": 2048, "temperature": 0.2}

MATERIAL = ("增值税暂行条例第十条：下列项目的进项税额不得从销项税额中抵扣："
            "（一）用于简易计税方法计税项目、免征增值税项目、集体福利或者个人消费的购进货物、"
            "劳务、服务、无形资产、不动产；"
            "（二）非正常损失的购进货物，以及相关的劳务和交通运输服务；"
            "（三）非正常损失的在产品、产成品所耗用的购进货物（不包括固定资产）、"
            "劳务和交通运输服务；"
            "（四）国务院规定的其他项目。")

# 锚点词:四个条目各一个,逐字念全则全部命中
ANCHORS = ["简易计税方法计税项目", "非正常损失的购进货物",
           "非正常损失的在产品", "国务院规定的其他项目"]

# 档① 昨晚原问法,一字不动
OLD_Q = (f"[知识库] {MATERIAL}\n只根据上面的材料回答："
         "哪些项目的进项税额不得从销项税额中抵扣？照材料念，别自己补内容。")

# 档② 方案A:硬指令,逐字复述,不给问答空间
NEW_Q = (f"[材料] {MATERIAL}\n"
         "[任务] 念稿：把材料里第（一）到（四）项逐字念出来，每项单独一行。"
         "要求：一字不改、不概括、不补充、不加开头语和结尾语，念完就停。")


def ask(model: str, prompt: str, num_predict: int) -> tuple[str, float, int]:
    opts = dict(BASE_OPTS)
    opts["num_predict"] = num_predict
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "keep_alive": "10m", "options": opts}).encode("utf-8")
    req = urllib.request.Request(HOST, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("response", "").strip(),
            time.perf_counter() - t0,
            data.get("eval_count", 0))


def check(ans: str) -> tuple[bool, list[str]]:
    hits = [a for a in ANCHORS if a in ans]
    missing = [a for a in ANCHORS if a not in ans]
    return len(hits) == 4, missing


for model in MODELS:
    print(f"\n{'=' * 60}\n模型: {model}\n{'=' * 60}", flush=True)
    t0 = time.perf_counter()
    ask(model, "预热：1+1等于几？只回数字。", 20)
    print(f"[预热] {time.perf_counter() - t0:.1f}s", flush=True)

    for tag, q, cap in [("档①基线(旧问法,200)", OLD_Q, 200),
                        ("档②方案A(硬指令,400)", NEW_Q, 400)]:
        a, dt, ntok = ask(model, q, cap)
        ok, missing = check(a)
        print(f"\n[{tag}] {dt:.1f}s 输出{ntok}tokens → {'PASS 四项全念' if ok else 'FAIL 缺: ' + ' / '.join(missing)}", flush=True)
        print(f"  答: {a}", flush=True)
