# -*- coding: utf-8 -*-
"""smoke_real_pipeline.py —— 全真件流水线冒烟(2026-10-10,用户指示"注入真实生成端")

真实件: L2=bge-m3 kNN3(auto_wire_l2) / 本地生成=Ollama [LocalModelName] /
云端=L3 DSHIngressL3Provider。检索件未注入(骨架占位,rag 必 no_hit,如实呈现)。
问题换一批,覆盖六类: 域内/实时/chat/情感/垃圾/工具。
跑法: python smoke_real_pipeline.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router import HybridRouter  # noqa: E402
from l3_provider import DSHIngressL3Provider  # noqa: E402

HOST = "http://127.0.0.1:11434/api/generate"
MODEL = "[LocalModelName]"
OPTS = {"num_ctx": 2048, "temperature": 0.2, "num_predict": 300}


class OllamaLocalExecutor:
    """最小本地生成端:/api/generate 单轮,无历史(冒烟用;生产化另派活)。"""

    name = "ollama-[LocalModelName]"

    def execute(self, req, rag_ctx):
        prompt = req.text if not rag_ctx else f"{rag_ctx}\n\n{req.text}"
        body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                           "keep_alive": "10m", "options": OPTS}).encode("utf-8")
        r = urllib.request.Request(HOST, data=body,
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8")).get("response", "").strip()

    def is_available(self):
        return True


rt = HybridRouter(
    config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "routing.yaml"),
    l3_provider=DSHIngressL3Provider(),
    auto_wire_l2=True,
    executors={"local": OllamaLocalExecutor()},
)

QUESTIONS = [
    "递延所得税怎么处理？",               # 域内 knowledge_qa(无检索件→本地直答)
    "今天有什么大新闻？",                 # realtime → L3 云端真件
    "你今天心情怎么样呀？",               # chat → 本地人设
    "我那个破电脑又出问题了，好烦。",       # 情感类看 L2 判什么
    "。。。",                             # 垃圾 → 守卫拒答
    "关闭客厅灯",                         # tool → 待确认流程
]

for q in QUESTIONS:
    t0 = time.perf_counter()
    out = rt.route(q)
    dt = time.perf_counter() - t0
    d = out.decision
    print(f"\n问: {q}", flush=True)
    print(f"  intent={d.intent} target={d.target.value} [{dt:.1f}s]", flush=True)
    print(f"  链路: {d.reasons[0] if d.reasons else '-'}", flush=True)
    print(f"  答: {out.answer[:120]}", flush=True)
