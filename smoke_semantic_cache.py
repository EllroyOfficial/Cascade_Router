# -*- coding: utf-8 -*-
"""smoke_semantic_cache.py —— 语义缓存真件冒烟(2026-10-10,改进项①)

验证:store→近似查询命中→无关查询不命中;嵌入走 OllamaEmbedder/bge-m3。
跑法: python smoke_semantic_cache.py
"""
from __future__ import annotations

import os
import tempfile

from semantic_cache import SemanticCache
from embed_local import Embedder

CACHE_FILE = os.path.join(tempfile.gettempdir(), "smoke_semcache_test.json")
if os.path.exists(CACHE_FILE):
    os.remove(CACHE_FILE)

config = {"semantic_cache": {"enabled": True, "similarity_threshold": 0.75,
                             "max_entries": 50, "path": CACHE_FILE}}
emb = Embedder()
cache = SemanticCache(config, emb)

Q1 = "增值税进项税额哪些项目不得抵扣？"
A1 = "十条念给你听——简易计税、免税项目、集体福利、非正常损失这几类都不能抵，哼，记牢了。"
Q2 = "进项税里哪些项目是不让抵扣的？"          # 换说法,同义 → 应命中
Q3 = "递延所得税资产怎么确认？"                # 无关 → 不应命中

cache.store(Q1, A1)
hit = cache.lookup(Q2)
miss = cache.lookup(Q3)
print(f"同义查询命中: {'PASS' if hit == A1 else 'FAIL'} → {(hit or '')[:30]}…")
print(f"无关查询放行: {'PASS' if miss is None else 'FAIL'} → {miss!r}")

# 落盘重载后仍命中(进程重启场景)
cache2 = SemanticCache(config, emb)
print(f"落盘重载命中: {'PASS' if cache2.lookup(Q2) == A1 else 'FAIL'}")
