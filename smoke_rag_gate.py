"""smoke_rag_gate.py —— RAG 门接真实 ollama/bge-m3 冒烟

链路:真件 Embedder(bge-m3 余弦) + 本地小语料(取自已入库的增值税条例 wiki 卡片)
→ RagGate.evaluate(改写钩子可插)→ Injector(红线7 截断/位置) → 打印判定细节。
验证点:真件嵌入能算出分、阈值/margin 判定按 yaml 阈值走、注入块带 [知识库] 且提问在最后、
丢弃字数回报。检索后端是临时列表检索器(quaesto_kb 是 MCP,冒烟不直连)。
跑法:python smoke_rag_gate.py
"""
from __future__ import annotations
# ── 路径去标识（不再硬编码用户名；可用环境变量 HR_DATA_ROOT 覆盖数据根）
import os as _hr_os
_HR_HOME = _hr_os.path.expanduser("~")
_HR_DESKTOP = _hr_os.environ.get("HR_DATA_ROOT") or _hr_os.path.join(_HR_HOME, "Desktop")
_HR_DIR = _hr_os.path.dirname(_hr_os.path.abspath(__file__))

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_loader import load_config
from metrics import Metrics
from ollama_runtime import OllamaEmbedder
from rag_gate import RagGate
from injection import Injector
from interfaces import RetrievedChunk, RequestProfile

TIDDLERS = _hr_os.path.join(_HR_HOME, r"tiddlywiki", r"main", r"tiddlers")

# 小语料:8 张增值税条例卡片(逐字原文,来源已入库卡片)+2 条噪声
CORPUS_KEYS = ["第十条", "第二十三条", "第二十五条", "第二十六条", "第二十七条",
               "第二十八条", "第三十条", "第三十三条"]


class ListRetriever:
    """临时检索器:全量候选给后端(排序不可信,由 gate 本地复算重排)。"""

    name = "local_cards_smoke"

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 5) -> list:
        return [RetrievedChunk(text=c, source=self.name, backend_rank=i)
                for i, c in enumerate(self.chunks)]


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(root, "routing.yaml"))
    metrics = Metrics()

    chunks = []
    for key in CORPUS_KEYS:
        p = os.path.join(TIDDLERS, f"《增值税法实施条例》{key}.md")
        text = open(p, encoding="utf-8").read()
        body = text.split("## 原文（逐字）", 1)[1].split("## 锚点", 1)[0].strip()
        chunks.append(body)
    chunks += ["今天天气不错,适合出门散步。", "本鲸今天要吃三碗白米饭才够。"]

    from embed_local import Embedder   # 统一嵌入入口
    embedder = Embedder()
    gate = RagGate(config, [ListRetriever(chunks)], embedder, metrics)
    injector = Injector(config, metrics)

    cases = [
        ("纳税人取得的下列进项税额不得从销项税额中抵扣", "finance_tax"),   # 域内,应命中第十条
        ("小规模纳税人的标准是怎么定的", "finance_tax"),                    # 域内
        ("今天穿什么衣服出门比较好", ""),                                   # 域外,应低置信不注入
    ]
    for query, domain in cases:
        req = RequestProfile(text=query, domain=domain)
        rag, top1 = gate.evaluate(req)
        line = (f"q={query[:22]!r} domain={domain or '-':<12} "
                f"conf={rag.confidence:.3f} margin={rag.margin:.3f} "
                f"reason={rag.reason or 'hit'} injected={rag.injected}")
        if rag.injected and top1 is not None:
            composed, dropped = injector.build_user_turn(top1.text, query)
            rag.rag_ctx = "[知识库] " + composed[:composed.rfind(query)].strip().removeprefix("[知识库] ").strip()
            rag.dropped_chars = dropped
            tail_ok = composed.rstrip().endswith(query)
            line += (f" 注入={len(rag.rag_ctx)}字 丢弃={dropped} 提问在末尾={tail_ok} "
                     f"命中片段={top1.text[:18]!r}")
        print(line)
    print("[计数器]", {k: v for k, v in metrics.snapshot().items() if v})
    print("[冒烟口径] 后端为本地列表检索器;quaesto_kb(MCP)与 query_rewrite 真件另接线,本冒烟只验嵌入/判定/注入链路。")


if __name__ == "__main__":
    main()
