"""l1_rules.py —— 第一层硬规则层(正则词表,配置驱动,命中即停)

规约(DESIGN.md):
- L1 只做匹配,不做仲裁;冲突即降级 L2(红线3)。
- 词表全部来自 config(routing.yaml hard_rules),代码零硬编码关键词(红线5)。
- 规则顺序 = 配置顺序;返回全部命中,由 unique_intents 数量决定是否升级。
"""
from __future__ import annotations

import re

import interfaces
from interfaces import HardRuleHit, L1Result


class L1Layer:
    """第一层硬规则:正则词表匹配,0 命中或 intent 冲突 → escalated(降 L2)。"""

    def __init__(self, config: dict, metrics):
        self._metrics = metrics
        self._rules: list[dict] = []
        self.reload(config)

    def reload(self, config: dict) -> None:
        """替换已编译规则(配合 routing.yaml 热加载);顺序 = 配置顺序。"""
        compiled: list[dict] = []
        for rule in config.get("hard_rules", []):
            compiled.append({
                "id": rule.get("id", ""),
                "regex": re.compile(rule["pattern"]),
                "intent": rule.get("intent", ""),
                "domain": rule.get("domain", ""),
                "confirm": bool(rule.get("confirm", False)),
                "args_schema": rule.get("args_schema"),
            })
        self._rules = compiled

    def match(self, text: str) -> L1Result:
        """返回全部命中;每命中一条计一次 L1_hit。冲突/0命中不仲裁,只升级。"""
        hits: list[HardRuleHit] = []
        for rule in self._rules:
            if rule["regex"].search(text):
                hits.append(HardRuleHit(
                    rule_id=rule["id"],
                    intent=rule["intent"],
                    domain=rule["domain"],
                    confirm=rule["confirm"],
                    args_schema=rule["args_schema"],
                ))
                self._metrics.inc("L1_hit")

        unique_intents: list[str] = []
        for hit in hits:
            if hit.intent not in unique_intents:
                unique_intents.append(hit.intent)

        escalated = len(unique_intents) != 1  # 0 命中或 ≥2 种 intent → 降 L2
        return L1Result(hits=hits, unique_intents=unique_intents, escalated=escalated)
