"""ollama_runtime.py —— Ollama 真件:SmallClassifier(L2) 与 Embedder(bge-m3)

冒烟/联调用真实实现,替代 router.py 的骨架缺省假件。
- 分类:POST /api/generate,temperature=0、num_predict≤10、format=json schema 约束
  (DESIGN.md:输出卡死枚举);timeout 由调用方传,本侧用 requests timeout 硬保证。
- 嵌入:POST /api/embeddings(bge-m3),余弦相似度。
零硬编码枚举:分类 prompt 的枚举词由调用方(L2Classifier)拼入,本件只管跑模型。
"""
from __future__ import annotations

import json
import math
import urllib.request

DEFAULT_HOST = "http://127.0.0.1:11434"


class OllamaClassifier:
    """interfaces.SmallClassifier 的 Ollama 实现(真件,旧 LLM 零样本路,弱,已被
    l2_vector 向量匹配取代为备选;保留供 method=llm 场景与回归对照)。"""

    # prompt 前缀收在 LLM 件内部(2026-10-09 起 L2Classifier 传原句);
    # 只作为 prompt 常量,枚举判据仍在 L2Classifier(红线1)。
    _PROMPT_PREFIX = ("分类这句话的意图,只输出以下标签之一:"
                      "chat/professional/tool/emotion。句子:")

    def __init__(self, model: str = "[LocalModelName]", host: str = DEFAULT_HOST) -> None:
        self.model = model
        self.host = host.rstrip("/")

    def classify(self, prompt: str, temperature: float = 0.0,
                 max_tokens: int = 10, timeout_ms: int = 2000) -> str:
        body = json.dumps({
            "model": self.model,
            "prompt": self._PROMPT_PREFIX + prompt,
            "stream": False,
            "format": {"type": "object",
                       "properties": {"category": {"type": "string",
                                                    "enum": ["chat", "professional", "tool", "emotion"]}},
                       "required": ["category"]},
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }).encode("utf-8")
        req = urllib.request.Request(self.host + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_ms / 1000.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        raw = data.get("response", "")
        # num_predict=10 会把 format=json 的输出截断(如 '{ "category": "chat'),
        # 容错提取 category 值;枚举校验不在这里——严格校验在 L2Classifier(红线1)。
        try:
            return str(json.loads(raw)["category"])
        except (KeyError, ValueError):
            import re
            m = re.search(r'"category"\s*:\s*"(\w+)', raw)
            return m.group(1) if m else raw


class OllamaEmbedder:
    is_semantic = True   # 语义嵌入件标志:cos 类判据(跑题/兜底)只在真语义件上生效
    """interfaces.Embedder 的 bge-m3 实现(真件),余弦相似度。"""

    def __init__(self, model: str = "bge-m3", host: str = DEFAULT_HOST) -> None:
        self.model = model
        self.host = host.rstrip("/")

    def _embed(self, text: str) -> list[float]:
        body = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(self.host + "/api/embeddings", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))["embedding"]

    def embed(self, text: str) -> list[float]:
        """公开取向量(l2_vector 用);similarity 语义不变。"""
        return self._embed(text)

    def similarity(self, a: str, b: str) -> float:
        va, vb = self._embed(a), self._embed(b)
        dot = sum(x * y for x, y in zip(va, vb))
        na = math.sqrt(sum(x * x for x in va)) or 1.0
        nb = math.sqrt(sum(x * x for x in vb)) or 1.0
        return dot / (na * nb)
