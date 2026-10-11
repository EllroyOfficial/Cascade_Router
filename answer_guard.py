"""answer_guard.py —— 答案兜底校验 v2(红线8:生成之后、返回之前)

依据:TESTS.md T5.4–T5.8 + routing.yaml answer_guard(v0.2 语义判据版)
- v0.1 字面判据曾误伤人设化口语回答(#0024,重合 0.119 被丢),v0.2 以语义判据为主,
  字面重合降为辅助信号(本模块留 TODO,见 check 内说明)。
- 编数字类幻觉语义相似度可高达 0.848(#0026-4),语义判据抓不住,故数字一致
  必须独立成判据(T5.4)。
- 本模块只判定不降级:action 交给调用方执行(cite_original / refuse_no_material)。
- 阈值/开关全部来自 config(routing.yaml answer_guard),代码零硬编码(红线5)。
只 import 标准库 + interfaces。
"""
from __future__ import annotations

import re

from interfaces import GuardResult


# 数字/百分数/金额序列:匹配 15、15%、1,000、1,000.50、3.5% 等。
# 千分位逗号在归一化时剥除,再按"数值字面(含可选 % 后缀)"精确比对。
_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\d+(?:\.\d+)?%?")


def _normalize_number(token: str) -> str:
    """归一化数字 token:去千分位逗号,保留 % 后缀与小数点。"""
    return token.replace(",", "")


class AnswerGuard:
    """答案兜底校验器:对生成答案做判据链检查,产出 GuardResult。"""

    def __init__(self, config: dict, embedder, metrics) -> None:
        self.config = config
        self.embedder = embedder        # interfaces.Embedder 协议注入
        self.metrics = metrics
        criteria = (config.get("answer_guard") or {}).get("criteria") or {}
        self.criteria = criteria
        # 判据参数(全部可配,零硬编码阈值)
        self.min_sim: float = float(criteria.get("semantic_sim", {}).get("min_sim", 0.65))
        self.length_ratio_max: float = float(criteria.get("length_ratio_max", 3))
        self.number_consistency: bool = bool(criteria.get("number_consistency", True))
        self.number_tol: float = float((config.get("answer_guard") or {}).get(
            "number_tol", 0.01))     # #0187 用户口径:统一单位后绝对 0.01 容差
        self.derivable_ops = tuple((config.get("answer_guard") or {}).get(
            "derivable_ops", ["add", "sub", "mul", "div", "pct"]))
        self.literal_overlap_mode: str = str(criteria.get("literal_overlap", "auxiliary"))
        # 字面重合放行下限:yaml 的 literal_overlap 只有 auxiliary 无数值(零硬编码
        # 红线5 下不能造配置键),此常量为模块级骨架默认,待标定后迁入配置。
        self.LITERAL_RESCUE_MIN: float = 0.5
        # on_fail 降级顺序(config["answer_guard"]["on_fail"],如 cite_original → refuse_no_material)
        self.on_fail: list[str] = list(
            (config.get("answer_guard") or {}).get("on_fail")
            or ["cite_original", "refuse_no_material"]
        )

    # ── 数字集合抽取 ─────────────────────────────────────────
    def _extract_numbers(self, text: str) -> set[str]:
        """抽出原料/答案中的全部数字 token,归一化后返回集合。

        覆盖:整数、小数、千分位金额(1,000)、百分数(15%)。
        % 后缀参与比对:原料是 15% 而答案写成 15 同样视为不一致(量纲不同)。
        """
        return {_normalize_number(m.group(0)) for m in _NUM_RE.finditer(text or "")}

    # ── 主判据链 ─────────────────────────────────────────────
    def check(self, query: str, rag_raw: str | None, answer: str,
              number_ref_query: bool = False, material_for_derive: str | None = None) -> GuardResult:
        """按判据顺序校验;全过 → pass;任一不过 → 记失败判据与降级动作。

        rag_raw 为 None/空时,依赖原料的判据(数字一致/长度比/语义相似)跳过。
        """
        ans = answer or ""

        # 1) 空答案(T5.6):直接拦
        if not ans.strip():
            return self._reject("empty", rag_raw)

        has_material = bool(rag_raw and rag_raw.strip())

        # 2) 数字一致(T5.4,独立判据):答案中的每个数字必须能在原料数字集合中找到
        #    无材料分支(#0184 (b) 用户拍):top1=None(检索未命中)时,答案数字必须
        #    ⊆ 问题数字(走既有归一化)——防"无注入自由发挥编数"(#0181 样本5);
        #    问题本身无数字 ⇒ 任何答案数字都算编(计算类误杀已知,用户知情)。
        #    #0187/#0188 三选一判据:不在原料的数字,可由 问题/材料 推出(Python 真算,
        #    单位归一后绝对容差 0.01)即放行;差 ≤0.01 打观察标签[舍入差0.01],不 Fail;
        #    差 >0.01 ⇒ Fail ⇒ 修正动作走既有 cite_original(原料直出,正确值在原文里)。
        if self.number_consistency:
            if has_material:
                raw_nums = self._extract_numbers(rag_raw)
            elif number_ref_query:
                raw_nums = self._extract_numbers(query)
            else:
                raw_nums = None
            if raw_nums is not None:
                from anchors import derivable_numbers, num_matches_derived
                derived = derivable_numbers(query, rag_raw or "", self.derivable_ops)
                note_parts: list[str] = []
                for num in self._extract_numbers(ans):
                    if num in raw_nums:
                        continue
                    try:
                        v = float(num)
                    except ValueError:
                        return self._reject("number_consistency", rag_raw)
                    if num_matches_derived(v, derived, tol_abs=self.number_tol):
                        nearest = min(abs(v - d) for d in derived)
                        if nearest > 1e-9:      # 精确相等不算舍入;只有"差但 ≤0.01"才标
                            note_parts.append(num)
                        continue                # 可推导即放行(#0187 红线15:算对的必须放)
                    return self._reject("number_consistency", rag_raw)
                if note_parts:
                    self.metrics.inc("number_rounding_tolerated")
                    return GuardResult(passed=True, failed_criterion="", action="pass",
                                       note="[舍入差0.01]:" + ",".join(note_parts))

        # 3) 长度比:答案 > 原料字数×length_ratio_max → 防扩写
        if has_material and len(ans) > len(rag_raw) * self.length_ratio_max:
            return self._reject("length_ratio", rag_raw)

        # 4) 语义相似(T5.7 回归:干净原料 + 人设化口语回答 sim≈0.81 必须放行)
        #    比对方向=「答案 对 原料」:注入件的相似度脚本多以答案文本为锚
        #    (a_rules 命中答案中的子串),方向反了会漏掉全部锚点。
        if has_material:
            sim = self.embedder.similarity(ans, rag_raw)
            if sim < self.min_sim:
                # 4b) 字面重合辅助判据:语义低但字面高度重合 = 忠实改写/直抄变体,
                #     放行(辅助判据只用于"抓完全离题",不用于加严)。语义低+字面
                #     重合也≈0(如答非所问)才落 semantic_sim 拦截(T5.5)。
                if self.literal_overlap_mode == "auxiliary":
                    import difflib
                    overlap = difflib.SequenceMatcher(None, ans, rag_raw).ratio()
                    if overlap >= self.LITERAL_RESCUE_MIN:
                        return GuardResult(passed=True, failed_criterion="", action="pass")
                return self._reject("semantic_sim", rag_raw)

        return GuardResult(passed=True, failed_criterion="", action="pass")

    # ── 拦截处置(只判定,不执行降级) ─────────────────────────
    def _reject(self, failed_criterion: str, rag_raw: str | None) -> GuardResult:
        """按 on_fail 顺序确定 action:cite_original 需原料非空,否则 refuse_no_material。"""
        action = "refuse_no_material"
        has_material = bool(rag_raw and rag_raw.strip())
        for step in self.on_fail:
            if step == "cite_original" and has_material:
                action = "cite_original"
                break
            if step == "refuse_no_material":
                action = "refuse_no_material"
                break
        self.metrics.inc("answer_guard_rejected")
        return GuardResult(passed=False, failed_criterion=failed_criterion, action=action)
