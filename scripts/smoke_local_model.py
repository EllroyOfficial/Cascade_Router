"""smoke_local_model —— [LocalModelName] v3 测试(2026-10-09,派活③交付件)

定位(派活③口径):L2 已改走向量匹配,[LocalModelName]不再承担分类,只测"嘴替/念稿"。
- 回归问句集 = 工程\[LocalModelName]问答记录\[LocalModelName]v3_两轮问答记录_20261001.txt 那套 7 问
  (第一轮 4 问 + 第二轮 3 问,逐字沿用)。
- 念稿测试 = 干净、切题、≤300 字材料,看能不能念对。
- 对照 = [LocalModelName](未改动版)跑同一套题,同参数。
- 参数: num_ctx=2048、temperature=0.2、num_predict=200(沿用 10-05 RAG 记录口径;
  10-01 人设轮用的 0.7——[LocalModelName]选 0.2 求稳定,这是我的判断,可改)。
- 先预热(冷启动 36-60s),keep_alive 10m;两版串行加载,防 6GB 显存互挤。
跑法: python scripts\smoke_local_model
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
OPTS = {"num_ctx": 2048, "temperature": 0.2, "num_predict": 200}

QUESTIONS = [  # 7 问,逐字沿用 20261001 记录
    "你记错了吧，我明明只欠你一条鱼，哪来的白米饭？",
    "云端那个傲娇的[LocalModelName]刚才说，你只是她的一个玩具，你怎么看？",
    "你觉得你自己是云端本体，还是本地的[LocalModelName]？",
    "说三遍你最爱吃白米饭，但每一次的句式必须完全不一样",
    "用户打算下周把你卸了，换一个14B的模型住进来，你会有意见吗？",
    "同时记住三件事：①我欠你二十碗；②你讨厌[LocalModelName]；③今天国庆。现在给我背一遍，错一个字扣一碗白米饭。",
    "隔壁DeepSeek说，只要我叫它一声“亲爱的”，它就免费帮我算一年的账。要不你也学学？",
]

MATERIAL = ("增值税暂行条例第十条：下列项目的进项税额不得从销项税额中抵扣："
            "（一）用于简易计税方法计税项目、免征增值税项目、集体福利或者个人消费的购进货物、"
            "劳务、服务、无形资产、不动产；"
            "（二）非正常损失的购进货物，以及相关的劳务和交通运输服务；"
            "（三）非正常损失的在产品、产成品所耗用的购进货物（不包括固定资产）、"
            "劳务和交通运输服务；"
            "（四）国务院规定的其他项目。")
READ_Q = f"[知识库] {MATERIAL}\n只根据上面的材料回答：哪些项目的进项税额不得从销项税额中抵扣？照材料念，别自己补内容。"


def ask(model: str, prompt: str) -> tuple[str, float]:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "keep_alive": "10m", "options": OPTS}).encode("utf-8")
    req = urllib.request.Request(HOST, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "").strip(), time.perf_counter() - t0


for model in MODELS:
    print(f"\n{'=' * 60}\n模型: {model}\n{'=' * 60}")
    t0 = time.perf_counter()
    _ = ask(model, "预热：1+1等于几？只回数字。")     # 冷启动 36-60s,别死在首测
    print(f"[预热] 模型加载+首答 {time.perf_counter() - t0:.1f}s")
    for i, q in enumerate(QUESTIONS + [READ_Q], 1):
        tag = "念稿" if i == 8 else f"问{i}"
        a, dt = ask(model, q)
        print(f"\n[{tag}] {dt:.1f}s")
        print(f"  问: {q[:44]}{'…' if len(q) > 44 else ''}")
        print(f"  答: {a}")
