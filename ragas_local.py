# -*- coding: utf-8 -*-
"""ragas_local.py —— 本地 RAGAS 评估件(2026-10-10,L3 收尾,借 RouteLM evaluation.py)

纯嵌入实现、无 LLM 评审、零外部依赖:嵌入件注入(OllamaEmbedder/bge-m3 走向量,
骨架件的 .similarity() 退化文本比值——与 semantic_cache 同一双接口套路)。
两个消费者:①router 的 L3 出口质量门(relevancy);②题库标定/v9 验收的客观分
(faithfulness/answer_relevancy 连续版,即念稿锚点校验的一般化)。

┌─ 实测基线口径(2026-10-10 #0101/#0102 双方独立复核,待用户拍板,未接线) ──
│ cos(材料,答案) 实测:正例 0.778–0.975 / 反例 0.596–0.809,区间重叠;
│   最标准的忠读逐字仅 0.778(材料长答案短稀释),"方向说反"反例反而 0.809。
│   ⇒ 建议分界(宁漏不误):cos≥0.70 放行(拦主题漂移);
│     0.70–0.85 且 anchors.extra_in_answer>0 ⇒ 转人工复核(不自动拒);
│     同域胡编纯 cos 拦不住,靠锚点+人工。
│ faithfulness 句级覆盖:同域文本实测全 1.000(会计句对会计材料句句过 0.45),
│   本域失效勿单独当判据;relevancy(问题,答案)是跑题检测器抓不住事实错
│   (正反例 8/8 全放行,smoke_gate_bias.py)。判据粒度要匹配要抓的错。
└──────────────────────────────────────────────────────
"""
from __future__ import annotations

import re


def _embed(embedder, text: str) -> list[float] | None:
    if hasattr(embedder, "embed"):
        try:
            return list(embedder.embed(text))
        except Exception:
            return None
    return None


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(y * y for y in b) ** 0.5 or 1.0
    return dot / (na * nb)


def relevancy(question: str, answer: str, embedder) -> float:
    """答案与问题的相关度 0-1(RouteLM answer_relevancy 同口径)。"""
    if not question.strip() or not answer.strip():
        return 0.0
    if hasattr(embedder, "similarity") and not hasattr(embedder, "embed"):
        return max(0.0, embedder.similarity(question, answer))
    va, vb = _embed(embedder, question), _embed(embedder, answer)
    if va is None or vb is None:
        return 0.0
    return max(0.0, _cos(va, vb))


def select_baseline(answer: str, material: str, embedder, top: int = 2) -> tuple[float, str]:
    """第三道防线(#0104 用户拍板):比对基准从整篇材料换成"最相关的 top-k 句"。

    治"材料长答案短稀释"(忠读逐字整段 cos 只有 0.778 的根因)。检索层若已暴露
    命中句,应优先把命中项传进 material 参数或直接替换本函数切句结果。
    返回 (cos(答案, top-k 句拼接), 基准文本)。
    """
    if not answer.strip() or not material.strip():
        return 0.0, ""
    sents = [s.strip() for s in re.split(r"(?<=[。！？!?；;])|\n", material)
             if len(s.strip()) >= 6]
    av = _embed(embedder, answer)
    if not sents or av is None:
        return 0.0, ""
    scored = []
    for s in sents:
        sv = _embed(embedder, s)
        if sv is not None:
            scored.append((_cos(av, sv), s))
    if not scored:
        return 0.0, ""
    scored.sort(reverse=True)
    baseline = "。".join(s for _, s in scored[:top])
    bv = _embed(embedder, baseline)
    return (_cos(av, bv), baseline) if bv is not None else (0.0, "")


def faithfulness(answer: str, material: str, embedder, threshold: float = 0.45) -> float:
    """答案逐句对材料的覆盖度 0-1(RouteLM faithfulness 同口径;材料单文本简化版)。

    shortcut: 材料按整段单向量比对,不如 RouteLM 逐块精确;标定阶段升级为分块。
    ⚠️ 实测(#0101):同域文本句句过 0.45 ⇒ 全 1.000,本域勿单独当判据。
    """
    if not answer.strip() or not material.strip():
        return 0.0
    mv = _embed(embedder, material)
    if mv is None:
        return 0.0
    sentences = [s.strip() for s in re.split(r"[。！？!?\n]", answer) if len(s.strip()) >= 6]
    if not sentences:
        return 0.0
    covered = 0
    for s in sentences:
        sv = _embed(embedder, s)
        if sv is not None and _cos(sv, mv) >= threshold:
            covered += 1
    return covered / len(sentences)
