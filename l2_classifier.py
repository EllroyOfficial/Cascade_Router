"""l2_classifier.py —— L2 轻量语义分类层(本地小模型,失败即 fail-open,红线1)

规约:
- 分类器本体(SmallClassifier / 向量件 l2_vector)由外部注入,模块内不发网络请求。
  ⚠️ classify 收到的是**原句**(2026-10-09 起):向量匹配按原句嵌入,prompt 前缀
  会污染向量;prompt 由 LLM 件(ollama_runtime.OllamaClassifier)自己拼。
- 枚举/映射/temperature/max_tokens/timeout 全部读 config(routing.yaml
  intent_classifier 节),代码零硬编码枚举值(红线5)。
- 输出必须"恰好等于"枚举词;枚举外/低置信/超时/异常一律 ok=False 直升 L3,
  禁止重试、禁止人脑纠偏硬猜(红线1)。
"""
from __future__ import annotations

import time

from interfaces import L2Result


class L2Classifier:
    """L2 层:调注入的小模型做语义分类,枚举校验后映射为路由意图。"""

    def __init__(self, config: dict, classifier, metrics):
        ic = config["intent_classifier"]
        self._enum = tuple(ic["enum"])
        self._intent_map: dict[str, str] = dict(ic["intent_map"])
        self._temperature = ic["temperature"]
        self._max_output_tokens = ic["max_output_tokens"]
        self._timeout_ms = ic["timeout_ms"]
        self._classifier = classifier
        self._metrics = metrics

    def classify(self, text: str) -> L2Result:
        """分类一段话;任何失败(枚举外/超时/异常)都 fail-open 上 L3,不重试。"""
        start = time.perf_counter()
        raw = ""
        try:
            raw = self._classifier.classify(
                text,                      # 原句直传;LLM 件的 prompt 在其内部拼
                temperature=self._temperature,
                max_tokens=self._max_output_tokens,
                timeout_ms=self._timeout_ms,
            )
        except Exception:
            return self._fail_open(raw, start)

        category = raw.strip() if isinstance(raw, str) else ""
        if category not in self._enum:
            return self._fail_open(raw, start)

        mapped = self._intent_map[category]
        self._metrics.inc("L2_hit")
        if mapped == "emotion_care":
            self._metrics.inc("emotion_route_count")
        # 置信度口径由注入件声明:vote_ratio=票数比 / cosine=原型余弦,
        # ⚠️ 都不是校准概率,下游不得当概率用。
        return L2Result(
            ok=True,
            category=category,
            mapped_intent=mapped,
            raw_output=getattr(self._classifier, "last_detail", raw),
            elapsed_ms=self._elapsed_ms(start),
            confidence=getattr(self._classifier, "last_confidence", 0.0),
            confidence_kind=getattr(self._classifier, "confidence_kind", ""),
        )

    # ── 内部 ────────────────────────────────────────────────
    def _fail_open(self, raw: str, start: float) -> L2Result:
        """红线1:枚举外/超时/异常 → ok=False,不重试,直升 L3。"""
        self._metrics.inc("L2_fail_open")
        self._metrics.inc("L3_hit")
        return L2Result(
            ok=False,
            category="",
            mapped_intent="",
            raw_output=raw,
            elapsed_ms=self._elapsed_ms(start),
        )

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)
