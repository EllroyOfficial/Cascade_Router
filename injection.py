"""injection.py —— RAG 注入出口硬约束(红线7)

职责:
- 把 rag_ctx 按句子边界截断到 inject_budget_chars 以内,拼进用户轮:
  知识库块紧挨提问之前,提问必须是消息最后内容(Ollama 上下文溢出保尾)。
- 出口无条件回报截断丢弃字数(metrics: injected_dropped_chars,即使 0 也调)。
- position 只支持 user_turn_before_query,配置写错立刻抛 ValueError(早爆炸)。

规约:
- 只 import 标准库 + interfaces;阈值一律从 config(routing.yaml)读,零硬编码。
- 本模块不发任何网络请求,metrics 通过注入对象回报。
"""
from __future__ import annotations

# 句子边界字符:按这些符号切句(含换行)。数据即词表,集中在常量区。
_SENTENCE_DELIMS = frozenset("。;;!!??\n\r")

_POSITION_USER_TURN_BEFORE_QUERY = "user_turn_before_query"


class Injector:
    """RAG 注入出口:截断 + 拼装用户轮 + 回报丢弃量。"""

    def __init__(self, config: dict, metrics) -> None:
        inj = config["rag_gate"]["injection"]
        self.budget_chars: int = inj["inject_budget_chars"]
        self.position: str = inj["position"]
        self.report_dropped_chars: bool = inj["report_dropped_chars"]
        self.metrics = metrics

        # 配置写错早爆炸:position 只支持这一种
        if self.position != _POSITION_USER_TURN_BEFORE_QUERY:
            raise ValueError(
                f"injection.position 只支持 {_POSITION_USER_TURN_BEFORE_QUERY},"
                f"当前为 {self.position!r}"
            )
        if not isinstance(self.budget_chars, int) or self.budget_chars <= 0:
            raise ValueError(
                f"inject_budget_chars 必须为正整数,当前为 {self.budget_chars!r}"
            )

    # ── 对外主入口 ────────────────────────────────────────────
    def build_user_turn(self, rag_ctx: str | None, query: str) -> tuple[str, int]:
        """返回 (拼接后的用户轮消息, 截断丢弃字数)。

        - rag_ctx 为 None:直接返回 (query, 0),counter 仍以增量 0 调用。
        - rag_ctx 非空:按句子边界截到 ≤ budget_chars,拼
          「[知识库] {材料}\n\n{query}」;提问保证在最后。
        """
        if rag_ctx is None:
            self._report(0)
            return query, 0

        kept = self._truncate_at_sentence_boundary(rag_ctx, self.budget_chars)
        dropped = len(rag_ctx) - len(kept)
        turn = f"[知识库] {kept}\n\n{query}"
        self._report(dropped)
        return turn, dropped

    # ── 内部实现 ──────────────────────────────────────────────
    def _report(self, dropped: int) -> None:
        # 红线7:出口无条件调用并回报,即使 dropped == 0 也调。
        self.metrics.inc("injected_dropped_chars", dropped)

    @staticmethod
    def _truncate_at_sentence_boundary(text: str, budget: int) -> str:
        """按句子边界(。;;!!??换行)截到 ≤ budget 字符。

        规则:整句放得下就收;首句单独超预算时硬切到 budget(不空手返回)。
        """
        if len(text) <= budget:
            return text
        sentences = _split_sentences(text)
        kept: list[str] = []
        used = 0
        for s in sentences:
            if used + len(s) <= budget:
                kept.append(s)
                used += len(s)
            else:
                break
        if not kept:
            # 首句就超预算:硬切保底,保证注入不空转
            return text[:budget]
        return "".join(kept)


def _split_sentences(text: str) -> list[str]:
    """把文本按句界字符切成句段,句界字符保留在句尾。"""
    parts: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in _SENTENCE_DELIMS:
            parts.append("".join(buf))
            buf = []
    if buf:
        parts.append("".join(buf))
    return parts
