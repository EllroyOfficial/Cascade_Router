# -*- coding: utf-8 -*-
"""semantic_cache.py —— 语义缓存(2026-10-10,改进项①,借 vLLM-SR 思路)

位置:route() 内 knowledge_qa 分支——RAG 门之前查缓存,guard 通过后写缓存。
命中:查询向量与缓存条目余弦 ≥ similarity_threshold(待标定) → 直接复用答案。
收益:省云端 token + 绕开 L3 改写时延(实测改写单发 86s)。
嵌入:注入件有 .embed() 走向量余弦(OllamaEmbedder/bge-m3);只有 .similarity()
的骨架件退化为文本相似度比对(测试/骨架可跑)。缓存读写失败不影响路由正确性
(降级为无缓存,静默跳过——缓存属加速层,不是正确性层)。
"""
from __future__ import annotations

import json
import math
import os


class SemanticCache:
    """文件制语义缓存:查询向量 + 答案,JSON 落盘,LRU 式截断。"""

    def __init__(self, config: dict, embedder, base_dir: str = ""):
        sc = config.get("semantic_cache", {})
        self.enabled = bool(sc.get("enabled", False))
        self.threshold = float(sc.get("similarity_threshold", 0.92))  # 待标定
        self.max_entries = int(sc.get("max_entries", 200))
        path = sc.get("path", "semantic_cache.json")
        self.path = path if os.path.isabs(path) else os.path.join(base_dir, path)
        self.embedder = embedder
        self._entries: list[dict] = []

    # ── 向量:有 .embed() 用真向量,否则退化为文本相似度 ────
    def _vec(self, text: str) -> list[float] | None:
        if hasattr(self.embedder, "embed"):
            try:
                return list(self.embedder.embed(text))
            except Exception:
                return None
        return None

    def _score(self, query: str, qv: list[float] | None, entry: dict) -> float:
        if qv is not None and entry.get("vec"):
            ev = entry["vec"]
            dot = sum(a * b for a, b in zip(qv, ev))
            nq = math.sqrt(sum(a * a for a in qv)) or 1.0
            ne = math.sqrt(sum(b * b for b in ev)) or 1.0
            return dot / (nq * ne)
        if hasattr(self.embedder, "similarity"):
            return self.embedder.similarity(query, entry["q"])
        return 0.0

    # ── 落盘 ─────────────────────────────────────────────
    def _load(self) -> None:
        if self._entries:
            return
        try:
            if os.path.exists(self.path):
                with open(self.path, encoding="utf-8") as f:
                    self._entries = json.load(f)
        except Exception:
            self._entries = []          # 缓存坏 = 无缓存,不影响正确性

    def _save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False)
        except Exception:
            pass

    # ── 主接口 ───────────────────────────────────────────
    def lookup(self, query: str) -> str | None:
        self._load()
        if not self._entries:
            return None
        qv = self._vec(query)
        best, best_score = None, -1.0
        for e in self._entries:
            if qv is not None and e.get("vec") and len(e["vec"]) != len(qv):
                continue        # 维度不符=别的模型时代写入的条目 ⇒ 跳过(宁漏不误)
            s = self._score(query, qv, e)
            if s > best_score:
                best, best_score = e, s
        if best is not None and best_score >= self.threshold:
            return best["a"]
        return None

    def store(self, query: str, answer: str) -> None:
        self._load()
        self._entries.append({"q": query, "a": answer, "vec": self._vec(query)})
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]
        self._save()
