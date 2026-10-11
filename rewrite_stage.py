# -*- coding: utf-8 -*-
"""rewrite_stage.py —— 念稿两段式改写段(2026-10-10,设计稿 v0)

位置:RAG 门注入之后、生成出口之前(仅 knowledge_qa 且已注入原料时接管)。
两种模式(routing.yaml rewrite: 节配置,红线5):
- rewrite    改写模式: L3 云端把检索原文改成[LocalModelName]腔口语稿,事实锚点逐字保留;
- faith_read 忠读模式: 不走 LLM 改写,材料直接交本地模型接龙逐字续写。
模式选择: 用户原话命中 faith_read_patterns(L1 式词表) → 忠读;否则默认改写。
出口硬校验: 锚点(数字串+中文括号序号)覆盖率 < min_anchor_coverage →
    rewrite 先降 faith_read 重试,仍不足 → 直出材料原文(cite_original 精神,红线8)。
全部阈值挂"待标定";L3 不可用由调用方(router)按既有降级链处理。
"""
from __future__ import annotations

import re

from anchors import guard_check
from l3_provider import strip_session_delimiter
from qc_local import qc_check
from ragas_local import select_baseline


class RewriteStage:
    """念稿改写段:模式选择 + 改写/接龙 + 三道防线出口校验(#0104 用户拍板)。

    第一道=guard_check 正则截杀(否定/方向/数字;先 review 统计假阳性再切 deny);
    第二道=0.75–0.85 定点微型质检(本地死命令 PASS/FAIL,严格解析,异常转人工);
    第三道=比对基准换"材料最相关 top-2 句"(治材料长答案短稀释)。
    分档:≥0.85 直放;<0.70 拒;0.75–0.85 过质检;0.70–0.75 与质检异常转人工。
    全部阈值挂"待标定/待用户拍";embedder=None 时语义分档跳过(骨架测试可跑)。
    """

    def __init__(self, config: dict, l3, local_ask, embedder=None):
        rc = config.get("rewrite", {})
        self.enabled = bool(rc.get("enabled", False))
        self.faith_patterns = [re.compile(p) for p in rc.get(
            "faith_read_patterns", ["念原文", "逐字念", "照材料念", "原文照念"])]
        self.target_ratio = float(rc.get("target_ratio", 1.2))          # 待标定
        self.min_coverage = float(rc.get("min_anchor_coverage", 0.8))   # 待标定
        self.max_retries = int(rc.get("max_retries", 1))
        self.cos_pass = float(rc.get("cos_pass", 0.85))       # 待用户拍
        self.cos_floor = float(rc.get("cos_floor", 0.70))     # 待用户拍
        self.qc_low = float(rc.get("qc_band_low", 0.75))      # 待用户拍
        self.qc_high = float(rc.get("qc_band_high", 0.85))    # 待用户拍
        self.guard_action = str(rc.get("guard_action", "review"))  # 先 review 再切 deny
        # 契约头(#0167 B):生成端要求首行输出标记,exit_gate 验头+剥头;标记可配
        self.contract_marker = str((config.get("exit_gate") or {}).get("contract_marker", ""))
        self.last_raw_output = ""         # 原始输出(未剥头),供 Fail 留档两栏
        self.last_contract_ok = None      # 契约头是否出现(None=未走生成端)
        self.l3 = l3
        self.local_ask = local_ask      # callable(prompt) -> str(本地生成端)
        self.embedder = embedder        # None=骨架模式,语义分档跳过

    # ── 模式选择 ──────────────────────────────────────────
    def select_mode(self, text: str) -> str:
        return "faith_read" if any(p.search(text) for p in self.faith_patterns) else "rewrite"

    # ── 锚点抽取与覆盖判据 ────────────────────────────────
    @staticmethod
    def _anchors(material: str) -> list[str]:
        hits = set(re.findall(r"\d+(?:\.\d+)?%?", material))
        hits |= set(re.findall(r"（[一二三四五六七八九十]+）", material))
        return [x for x in hits if len(x) <= 12]

    def _coverage(self, anchors: list[str], out: str) -> float:
        if not anchors:
            return 1.0
        return sum(1 for a in anchors if a in out) / len(anchors)

    # ── 清洗(借 voiceapp1 骨架,中文口径) ─────────────────
    @staticmethod
    def _pre_clean(material: str) -> str:
        keep = []
        for line in material.splitlines():
            s = line.strip()
            if not s:
                continue
            if re.search(r"版权所有|all rights reserved|第\s*\d+\s*页|https?://", s, re.I):
                continue
            keep.append(s)
        return "\n".join(keep)

    @staticmethod
    def _post_clean(text: str) -> str:
        t = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
        t = re.sub(r"#{1,4}\s*", "", t)
        t = re.sub(r"https?://\S+", "", t)
        return re.sub(r"\n{3,}", "\n\n", t).strip()

    # ── 提示词(设计稿 v0 草案,可调;契约头 #0167 B) ────────
    @staticmethod
    def _rewrite_prompt(material: str, query: str, marker: str = "") -> str:
        head = (f"输出第一行必须是契约头 {marker}，第二行起才是念稿正文。\n" if marker else "")
        return (head +
                "你是念稿写手，把检索材料改写成适合朗读的口语稿，风格傲娇、直、有活人感。\n"
                "铁律：①材料中的数字、条款编号、专有名词、限定语必须逐字保留，一个不许改写；\n"
                "②材料里没有的信息一个字不许加，拿不准就直说材料里没写；\n"
                "③只输出念稿正文：无标题、无序号符号、无舞台说明；\n"
                "④长句拆短句，书面语换口语；⑤扣住用户的问题：" + query + "\n"
                "⑥涉及计算的金额：以千元/万元为单位、保留两位小数（如 13.00 万元）。\n"
                "材料：\n" + material)

    def _strip_contract(self, raw: str) -> str:
        """验契约头+剥头(#0167 B):有头剥头;缺头原样返回(调用方记 contract_missing)。"""
        marker = self.contract_marker
        if not marker:
            return raw
        stripped = raw.lstrip()
        if stripped.startswith(marker):
            return stripped[len(marker):].lstrip()
        return raw

    @staticmethod
    def _chain_prompt(material: str, marker: str = "") -> str:
        head = (f"输出第一行必须是契约头 {marker}，然后逐字接写。\n" if marker else "")
        return (head + material + "\n继续把上面材料中列举的各条目逐字接写出来，"
                "一字不改、不概括、不加开头语和结尾语，念完就停。")

    # ── 主入口 ────────────────────────────────────────────
    def run(self, material: str, query: str) -> tuple[str, str, str]:
        """返回 (answer, mode, reason)。material=注入前的检索原文。"""
        mat = self._pre_clean(material)
        if not mat:
            return "", "rewrite", "材料清洗后为空 → 交还原链路"
        anchors = self._anchors(mat)
        mode = self.select_mode(query)
        last = ""
        notes: list[str] = []
        for attempt in range(self.max_retries + 1):
            if mode == "rewrite":
                raw = self.l3.ask(self._rewrite_prompt(mat, query, self.contract_marker))
            else:
                raw = self.local_ask(self._chain_prompt(mat, self.contract_marker))
            self.last_raw_output = raw                      # Fail 留档"原始"栏(#0167 B)
            content = self._strip_contract(raw)             # 剥契约头,剥头后才给前端
            self.last_contract_ok = (content != raw) or not self.contract_marker
            if mode == "rewrite":
                out = self._post_clean(strip_session_delimiter(content))
            else:
                out = self._post_clean(content)
            notes = []
            failed = False
            cov = self._coverage(anchors, out)
            if cov < self.min_coverage:
                notes.append(f"锚点覆盖{cov:.0%}<{self.min_coverage:.0%}")
                failed = True
            else:
                notes.append(f"锚点覆盖{cov:.0%}")
            # ── 第三道:换基准语义分档 ──
            if self.embedder is not None:
                cos_v, base = select_baseline(out, mat, self.embedder)
                if cos_v >= self.cos_pass:
                    notes.append(f"材料级cos={cos_v:.0%}≥{self.cos_pass:.0%} 直放")
                elif cos_v < self.cos_floor:
                    notes.append(f"材料级cos={cos_v:.0%}<{self.cos_floor:.0%}")
                    failed = True
                elif cos_v >= self.qc_low:
                    # ── 第二道:定点微型质检(#0107 实测本地 7B 不守死命令 ⇒ 走云端 l3) ──
                    qc = qc_check(base, out, ask=self.l3.ask)
                    if qc["verdict"] == "pass":
                        notes.append(f"cos={cos_v:.0%} 微型质检PASS")
                    elif qc["verdict"] == "fail":
                        notes.append(f"cos={cos_v:.0%} 微型质检FAIL")
                        failed = True
                    else:
                        notes.append(f"cos={cos_v:.0%} {qc['reason']}")
                else:
                    notes.append(f"cos={cos_v:.0%} 人工带({self.cos_floor:.0%}–{self.qc_low:.0%})→转人工")
            # ── 第一道:正则截杀 ──
            guard = guard_check(mat, out, action=self.guard_action)
            if guard["deny"]:
                tag = "" if self.guard_action == "deny" else "(review只标记)"
                notes.append(f"正则截杀{tag}:{guard['summary']}")
                if self.guard_action == "deny":
                    failed = True
            if not failed:
                return out, mode, "；".join(notes) + " PASS"
            last = out
            if attempt < self.max_retries and mode == "rewrite":
                mode = "faith_read"    # 改写丢事实 → 降忠读重试
                continue
        # 硬约束兜底:改写/接龙都保不住事实 → 直出材料原文(绝不返回自编)
        return mat, mode, "；".join(notes) + " → 直出材料原文"
