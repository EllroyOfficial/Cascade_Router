# -*- coding: utf-8 -*-
"""smoke_rewrite.py —— 两段式念稿双模式冒烟(2026-10-10,设计稿v0验证)

改写模式: L3 云端把检索材料改成[LocalModelName]腔口语稿(事实锚点逐字保留) → 出口锚点校验
忠读模式: 本地 [LocalModelName] 接龙式逐字续写(不过 LLM 改写) → 同一套锚点校验
材料沿用 smoke_read_retest 的条例第十条;阈值全部"待标定",本脚本只验通不通。
跑法: python smoke_rewrite.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

sys.path.insert(0, ".")
from l3_provider import DSHIngressL3Provider, strip_session_delimiter  # noqa: E402

MATERIAL = ("增值税暂行条例第十条：下列项目的进项税额不得从销项税额中抵扣："
            "（一）用于简易计税方法计税项目、免征增值税项目、集体福利或者个人消费的购进货物、"
            "劳务、服务、无形资产、不动产；"
            "（二）非正常损失的购进货物，以及相关的劳务和交通运输服务；"
            "（三）非正常损失的在产品、产成品所耗用的购进货物（不包括固定资产）、"
            "劳务和交通运输服务；"
            "（四）国务院规定的其他项目。")

# 出口硬校验锚点:四项各一 + 条款号
ANCHORS = ["第十条", "简易计税方法计税项目", "非正常损失的购进货物",
           "非正常损失的在产品", "国务院规定的其他项目"]

REWRITE_PROMPT = (
    "你是念稿写手，把下面的检索材料改写成适合朗读的口语稿，风格傲娇、直、有活人感。\n"
    "铁律：\n"
    "①材料中的数字、条款编号、专有名词、限定语必须逐字保留，一个不许改写；\n"
    "②材料里没有的信息一个字不许加；\n"
    "③只输出念稿正文：无标题、无序号符号、无舞台说明；\n"
    "④长句拆短句，书面语换口语；\n"
    "⑤总长约为材料的一点二倍以内。\n"
    f"材料：{MATERIAL}"
)

CHAIN_PROMPT = (
    f"{MATERIAL}\n"
    "继续从「（一）」开始把上面四项逐字接写出来，一字不改、不概括，写完第（四）项就停。"
)


def local_ask(prompt: str) -> str:
    body = json.dumps({"model": "[LocalModelName]", "prompt": prompt,
                       "stream": False, "keep_alive": "5m",
                       "options": {"num_ctx": 2048, "temperature": 0.2,
                                   "num_predict": 400}}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "").strip()


def check(text: str) -> tuple[bool, str]:
    missing = [a for a in ANCHORS if a not in text]
    return (not missing), ("全命中" if not missing else "缺: " + " / ".join(missing))


print("== 改写模式(L3 云端) ==", flush=True)
t0 = time.perf_counter()
try:
    draft = strip_session_delimiter(DSHIngressL3Provider().ask(REWRITE_PROMPT))
    ok, why = check(draft)
    print(f"[{time.perf_counter() - t0:.1f}s] {'PASS' if ok else 'FAIL'} {why}", flush=True)
    print(f"  稿: {draft}", flush=True)
except Exception as e:
    print(f"[{time.perf_counter() - t0:.1f}s] L3 异常: {e}", flush=True)

print("\n== 忠读模式(本地接龙) ==", flush=True)
t0 = time.perf_counter()
cont = local_ask(CHAIN_PROMPT)
ok, why = check(cont)
print(f"[{time.perf_counter() - t0:.1f}s] {'PASS' if ok else 'FAIL'} {why}", flush=True)
print(f"  稿: {cont}", flush=True)
