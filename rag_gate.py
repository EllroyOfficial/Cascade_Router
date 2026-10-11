"""rag_gate.py —— RAG 检索门(骨架)

流程:改写前置(#0016)→ 检索 → 本地复算重排(红线8:后端顺序不可信)
→ 阈值 + margin 双判据 → 归因(#0019)。

规约:
- 阈值全部从 config["rag_gate"] 读,零硬编码(红线5)。
- 归因枚举只用 interfaces.LOW_CONF_REASONS。
- "dirty_note"(命中但源笔记质量差)本骨架不判定,留 TODO。
"""
from __future__ import annotations

from interfaces import LOW_CONF_REASONS, RagGateResult, RequestProfile, RetrievedChunk


class RagGate:
    """检索门:仅 trigger_intents 触发;置信度一律本地 embedder 复算。"""

    def __init__(self, config, retrievers: list, embedder, metrics, rewriter=None) -> None:
        """retrievers:已按 enabled 过滤的 interfaces.Retriever 列表;
        rewriter:可注入改写器(接口 rewrite(text)->str),None=不改写。"""
        gate = config["rag_gate"]
        self._trigger_intents = list(gate["trigger_intents"])
        self._max_retries = int(gate["query_rewrite"]["max_retries"])
        thresholds = gate["confidence"]["thresholds"]
        self._threshold_in_domain = float(thresholds["in_domain"])
        self._threshold_out_of_domain = float(thresholds["out_of_domain"])
        self._margin_min = float(gate["confidence"]["margin_min"])

        self._retrievers = list(retrievers)
        self._embedder = embedder
        self._metrics = metrics
        self._rewriter = rewriter

    # ── 触发判定 ────────────────────────────────────────────
    def should_trigger(self, intent: str) -> bool:
        """仅 config rag_gate.trigger_intents 内的意图触发检索。"""
        return intent in self._trigger_intents

    # ── 主评估 ──────────────────────────────────────────────
    def evaluate(self, req: RequestProfile) -> tuple[RagGateResult, object]:
        """返回 (RagGateResult, top1_chunk_or_None)。

        调用方应先用 should_trigger(intent) 判定后再调 evaluate;
        故本方法内 result.needed 恒置 True。
        """
        result = RagGateResult(needed=True, rewritten_query=req.text)

        # 1) 原话检索:所有 retriever 合并候选(backend_rank/score 随块记录)
        candidates = self._retrieve_all(req.text)
        query = req.text

        if not candidates:
            # 0 命中且有改写器且还有重试额度 → 改写后重试一次(#0016/#0025-3)
            if self._rewriter is not None and self._max_retries > 0:
                result.rewritten_query = self._rewriter.rewrite(req.text)
                candidates = self._retrieve_all(result.rewritten_query)
                query = result.rewritten_query
                if candidates:
                    result.reason = "rewrite_fixable"
                    self._metrics.inc("rag_rewrite_fixable")
                else:
                    result.reason = "no_hit_after_rewrite"
                    self._metrics.inc("rag_no_hit_after_rewrite")
                    return result, None
            else:
                # 首轮即 0 且无改写器(或无额度)→ 未改写档 0 命中
                result.reason = "no_hit"
                self._metrics.inc("rag_no_hit")
                return result, None

        # 2) 重排(红线8:trust_backend_rank=False)——本地复算相似度
        scored = [(chunk, float(self._embedder.similarity(query, chunk.text)))
                  for chunk in candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        top1_chunk, top1_sim = scored[0]
        top2_sim = scored[1][1] if len(scored) > 1 else 0.0

        result.confidence = top1_sim
        result.margin = top1_sim - top2_sim

        # 3) 判定:margin 不足 → 不采信 top1,按低置信处理(#0012-2)
        threshold = (self._threshold_in_domain if req.domain == "finance_tax"
                     else self._threshold_out_of_domain)
        if result.margin < self._margin_min or result.confidence < threshold:
            result.reason = "retrieval_miss"
            result.injected = False
            result.rag_ctx = None
            self._metrics.inc("rag_retrieval_miss")
            return result, None

        # 4) 达标 → 注入 top1 文本
        # TODO: dirty_note(命中但源笔记质量差)本骨架不判定,待语料质量信号接入。
        result.injected = True
        result.rag_ctx = top1_chunk.text
        return result, top1_chunk

    # ── 内部:合并全部 retriever 候选 ────────────────────────
    def _retrieve_all(self, query: str) -> list:
        """所有后端各检索一次,合并候选;后端顺序不可信,只收候选。

        后端可能返回裸字符串(骨架 mock 常见),统一包成 RetrievedChunk。
        """
        merged: list = []
        for retriever in self._retrievers:
            for item in retriever.retrieve(query):
                if isinstance(item, RetrievedChunk):
                    merged.append(item)
                elif isinstance(item, str):
                    merged.append(RetrievedChunk(text=item, source=getattr(retriever, "name", "")))
                else:
                    merged.append(item)
        return merged


assert set(LOW_CONF_REASONS) >= {"no_hit", "no_hit_after_rewrite",
                                 "rewrite_fixable", "retrieval_miss"}


# ── 自测:mock 两个 retriever + mock embedder ──────────────────
if __name__ == "__main__":
    from interfaces import RetrievedChunk
    from metrics import Metrics

    config = {
        "rag_gate": {
            "trigger_intents": ["knowledge_qa"],
            "query_rewrite": {"enabled": True, "strategy": "term_extract",
                              "max_retries": 1},
            "confidence": {
                "trust_backend_rank": False,
                "method": "top1_cosine",
                "margin_judge": "enabled",
                "margin_min": 0.05,
                "thresholds": {"in_domain": 0.58, "out_of_domain": 0.73},
            },
        },
    }

    KB = "递延所得税资产以未来期间可取得的应纳税所得额为限计量。"

    class RetrieverA:
        """仅当查询含术语"递延所得税"才命中(模拟口语原话 0 命中)。"""
        def retrieve(self, query, top_k=5):
            if "递延所得税" not in query:
                return []
            return [RetrievedChunk(text=KB, source="quaesto_kb",
                                   backend_rank=1, backend_score=0.9)]

    class RetrieverB:
        def retrieve(self, query, top_k=5):
            return [RetrievedChunk(text="无关笔记占位文本", source="local_wiki",
                                   backend_rank=1, backend_score=0.1)]

    class EmptyRetriever:
        def retrieve(self, query, top_k=5):
            return []

    class MockEmbedder:
        """与 KB 词面重合越高分越高;重合<2 字记 0.2。"""
        def similarity(self, a, b):
            overlap = sum(1 for ch in set(a) if ch in b)
            return min(0.95, 0.2 + 0.1 * overlap) if overlap >= 2 else 0.2

    class Rewriter:
        def rewrite(self, text):
            return "递延所得税 应纳税所得额 计量"

    m = Metrics()
    gate = RagGate(config, [RetrieverA(), RetrieverB()], MockEmbedder(), m)

    # ① 域内命中,margin 足 → 注入
    req = RequestProfile(text="递延所得税资产怎么计量", domain="finance_tax")
    r, top1 = gate.evaluate(req)
    assert r.injected and r.rag_ctx == KB and r.reason == "" and top1 is not None
    print("① 域内达标注入: ok, conf=%.2f margin=%.2f" % (r.confidence, r.margin))

    # ② 原话 0 命中 → 改写后命中 → rewrite_fixable
    gate2 = RagGate(config, [EmptyRetriever(), RetrieverA()],
                    MockEmbedder(), m, rewriter=Rewriter())
    r2, _ = gate2.evaluate(RequestProfile(text="所得税那个东西咋算", domain="finance_tax"))
    assert r2.reason == "rewrite_fixable" and r2.injected
    assert r2.rewritten_query == "递延所得税 应纳税所得额 计量"
    print("② rewrite_fixable: ok, 改写=%s" % r2.rewritten_query)

    # ③ 改写后仍 0 命中 → no_hit_after_rewrite
    class RewriteNoop:
        def rewrite(self, text):
            return text + " 术语化"

    gate3 = RagGate(config, [EmptyRetriever()], MockEmbedder(), m,
                    rewriter=RewriteNoop())
    r3, top3 = gate3.evaluate(RequestProfile(text="库里根本没有的东西"))
    assert r3.reason == "no_hit_after_rewrite" and not r3.injected and top3 is None
    print("③ no_hit_after_rewrite: ok")

    # ④ 无改写器首轮 0 → no_hit
    gate4 = RagGate(config, [EmptyRetriever()], MockEmbedder(), m)
    r4, _ = gate4.evaluate(RequestProfile(text="库里根本没有的东西"))
    assert r4.reason == "no_hit" and not r4.injected
    print("④ no_hit: ok")

    # ⑤ 有候选但低于域外阈值 → retrieval_miss
    r5, _ = gate.evaluate(RequestProfile(text="跟笔记毫无关系的陌生问题", domain=""))
    assert r5.reason == "retrieval_miss" and not r5.injected
    print("⑤ retrieval_miss(阈值): ok, conf=%.2f" % r5.confidence)

    # ⑥ margin 不足(两候选同分)→ retrieval_miss
    class TwinRetriever:
        def retrieve(self, query, top_k=5):
            return [RetrievedChunk(text=KB, source="a", backend_rank=1),
                    RetrievedChunk(text=KB + "补充", source="b", backend_rank=2)]

    gate6 = RagGate(config, [TwinRetriever()], MockEmbedder(), m)
    r6, _ = gate6.evaluate(RequestProfile(text="递延所得税资产怎么计量",
                                          domain="finance_tax"))
    assert r6.reason == "retrieval_miss" and r6.margin < 0.05
    print("⑥ retrieval_miss(margin): ok, margin=%.2f" % r6.margin)

    # ⑦ 触发判定
    assert gate.should_trigger("knowledge_qa") and not gate.should_trigger("chat")
    print("⑦ should_trigger: ok")

    snap = m.snapshot()
    for k in ("rag_rewrite_fixable", "rag_no_hit", "rag_no_hit_after_rewrite",
              "rag_retrieval_miss"):
        assert snap[k] >= 1, k
    print("计数器: %s" % {k: v for k, v in snap.items() if v})
    print("ALL PASS")
