"""privacy.py —— 隐私判定(纸⑤ D 混合:A 显式标记 + B 词表 双入口)

职责:在任何路由决策前,对用户原话做隐私敏感判定,产出 RequestProfile 画像
与归因字符串(attribution)。判定规则全部来自 routing.yaml 的
privacy_detection 段;B 入口词表复用 hard_rules 中 id=privacy-keywords 的
pattern,本模块零硬编码词表(红线5)。

计数口径(T9.6):两个计数器 privacy_explicit_mark_hit /
privacy_keyword_hit 按"最终归因"只记一个、不双计——即 A、B 同时命中时,
按 verdict_priority 取最优先入口作为归因,且只为该归因的计数器 +1,
另一个入口虽然命中也不计数。verdict_priority 同时决定归因字符串;
sensitive 标志本身由各入口命中独立成立(显式标记与词表命中均判敏感)。

剥除口径(T9.4):A 入口命中且 strip_before_route=true 时,把前缀从查询词
中剥掉(剥后 lstrip),RequestProfile.text 存剥除后的文本;
RequestProfile.raw_text 恒保留用户原话,保证下游(如审计/日志)可见原文,
而检索收到的 query 不含前缀。
"""
from __future__ import annotations

import re

from interfaces import RequestProfile

# 归因 → 计数器 名单映射(计数器名见 routing.yaml metrics.counters)
_ATTR_COUNTER = {
    "explicit_mark": "privacy_explicit_mark_hit",
    "keyword_rules": "privacy_keyword_hit",
}


class PrivacyDetector:
    """隐私判定 A+B 双入口(纸⑤D)。

    config: 整份 routing.yaml 解析后的 dict;本类只读
        config["privacy_detection"](explicit_mark.prefixes /
        explicit_mark.strip_before_route / keyword_rules /
        verdict_priority / default_sensitive)
        与 config["hard_rules"](取 id=privacy-keywords 的 pattern,
        不复制词表)。
    metrics: 计数器对象,需提供 inc(name: str)。
    """

    def __init__(self, config: dict, metrics):
        pd = config["privacy_detection"]

        # A 入口:显式标记前缀
        em = pd.get("explicit_mark", {}) or {}
        self._prefixes: tuple[str, ...] = tuple(em.get("prefixes", ()) or ())
        self._strip_before_route: bool = bool(em.get("strip_before_route", False))

        # B 入口:keyword_rules=from_hard_rules → 复用 hard_rules 里
        # id=privacy-keywords 的 pattern,本模块不复制词表
        self._keyword_re = None
        if pd.get("keyword_rules") == "from_hard_rules":
            pattern = self._privacy_keyword_pattern(config.get("hard_rules", []))
            if pattern:
                self._keyword_re = re.compile(pattern)

        # verdict_priority:归因优先级,按 yaml 顺序展开为元组
        self._priority: tuple[str, ...] = tuple(pd.get("verdict_priority", ()) or ())

        # 默认判定(未标记且无词表命中)
        self._default_sensitive: bool = bool(pd.get("default_sensitive", False))

        self._metrics = metrics

    @staticmethod
    def _privacy_keyword_pattern(hard_rules: list) -> str:
        """从 hard_rules 取 id=privacy-keywords 的 pattern(只引用,不复制)。"""
        for rule in hard_rules or []:
            if isinstance(rule, dict) and rule.get("id") == "privacy-keywords":
                return rule.get("pattern", "") or ""
        return ""

    def detect(self, raw_text: str) -> tuple[RequestProfile, str]:
        """判定 raw_text 的隐私敏感性,返回 (RequestProfile, 归因)。

        归因 ∈ {"explicit_mark", "keyword_rules", "default"},由
        verdict_priority 决定:A、B 同时命中时取优先级最高者,且计数器
        只按最终归因记一次(T9.6,不双计)。
        """
        explicit_hit = False
        text = raw_text
        if self._prefixes:
            for prefix in self._prefixes:
                if raw_text.startswith(prefix):
                    explicit_hit = True
                    if self._strip_before_route:
                        text = raw_text[len(prefix):].lstrip()
                    break  # 命中即停,按 yaml 中 prefixes 首个匹配者剥除

        keyword_hit = bool(self._keyword_re is not None
                           and self._keyword_re.search(raw_text))

        # 归因:按 verdict_priority 取首个命中入口;均未命中 → default
        attribution = "default"
        for candidate in self._priority:
            if candidate == "explicit_mark" and explicit_hit:
                attribution = "explicit_mark"
                break
            if candidate == "keyword_rules" and keyword_hit:
                attribution = "keyword_rules"
                break
            if candidate == "default":
                attribution = "default"
                break

        # sensitive:各入口命中独立成立;default 入口取 default_sensitive
        if attribution == "explicit_mark":
            sensitive = True
        elif attribution == "keyword_rules":
            sensitive = True
        else:
            sensitive = self._default_sensitive

        # 计数器按归因只记一个(T9.6)
        counter = _ATTR_COUNTER.get(attribution)
        if counter is not None and self._metrics is not None:
            self._metrics.inc(counter)

        profile = RequestProfile(
            text=text,                       # T9.4:剥前缀后的查询词
            raw_text=raw_text,               # 原话恒保留
            privacy_sensitive=sensitive,
            privacy_source=attribution,
        )
        return profile, attribution
