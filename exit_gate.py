"""exit_gate.py —— 出口硬兜(#0154 用户定稿:Fail + 重试一次;#0161/#0162 扩跑题判据与双向标签)

判据(命中任一 ⇒ Fail,带标签):
  [过短]/[纯壳复读] —— (a+b 合成)剥壳后有效字数 < min_effective_chars:
        原文长但剥壳后空 ⇒ [纯壳复读];原文本身短 ⇒ [过短]。"壳+实质"放行(允许人设化,#0154 补充①)。
  [格式错乱] —— 请求侧显式 expect(结构化可判 sections_min/must_include,#0154 补充②),无 expect 不判。
  [答非所问] —— 整个跑题(#0161 用户口径),三判据任一;**只判"有没有答到那件事",不判"多说了什么"**:
        ①问题级 cos(问题,答案) < offtopic_min_sim(配置,待标定);
        ②材料级 cos(材料,答案) < 念稿 cos_floor(0.70,配置引用);
        ③核心实体全缺——问题里的科目/条款/数字(anchors.py 抽取),答案一个都没命中。
  ⚠️ 三级口径(#0161):(i) 答对多讲 ⇒ PASS+观察标签[内容冗余],绝不 Fail;
                    (ii) 整个跑题 ⇒ Fail+[答非所问];
                    (iii) 材料里真没有、如实说"没查到" ⇒ PASS+观察标签[如实拒答](refuse_honesty 优先,别判成跑题)。
处置:Fail ⇒ 不输出到前端 ⇒ 同路重试一次 ⇒ 仍 Fail ⇒ 调用方走降级链直出原文(#0154 补充③)。
每次 Fail 留档(instruction/material/output/fail_tag)——LoRA 现成素材(#0154 补充④)。
判据与阈值全走配置(routing.yaml exit_gate 节),零硬编码(红线5)。
"""
from __future__ import annotations

import json
import math
import re
import time

DEFAULT_SHELL = [
    "白米饭", "[LocalModelName]", "深海[LocalModelName]", "小肥鱼", "小鱼干", "尾鳍", "拍屏幕", "本鲸",
    "炸毛", "记一碗", "记你账上", "添一碗", "这碗得加", "白工", "Token", "token",
    "行吧", "听好了", "不许这样答", "又来使唤我",
]
REFUSE_RE = re.compile(r"没查到|没找到|库里没有|无法查|查不到")


class ExitGate:
    def __init__(self, config: dict, metrics, fail_log_path: str | None = None,
                 embedder=None) -> None:
        eg = (config.get("exit_gate") or {})
        self.enabled: bool = bool(eg.get("enabled", True))
        self.min_effective: int = int(eg.get("min_effective_chars", 15))
        self.retry: int = int(eg.get("retry", 1))
        self.shell = [re.compile(p) for p in eg.get("shell_patterns", DEFAULT_SHELL)]
        # 跑题判据(#0161):问题级阈值自配;材料级引用 rewrite.cos_floor(0.70,与念稿降级同源);实体全缺用 anchors
        self.offtopic_min_sim: float = float(eg.get("offtopic_min_sim", 0.30))
        self.material_floor: float = float((config.get("rewrite") or {}).get("cos_floor", 0.70))
        self.material_fail_min_len: int = int(eg.get("material_fail_min_len", 80))
        # 短答案换判据(#0171):15–80 字不用材料级,改走 问题级cos+锚点+数字;阈值全配置
        self.short_answer_path: bool = bool(eg.get("short_answer_path", True))
        self.short_answer_min_sim: float = float(eg.get("short_answer_min_sim", 0.30))
        self.contract_marker: str = str(eg.get("contract_marker", ""))
        self.embedder = embedder
        self.metrics = metrics
        self.fail_log_path = fail_log_path
        self.format_checks = list(eg.get("format_checks", []))   # (c) 格式期望:按 intent 匹配

    # ── 剥壳:去掉人设壳词后的有效内容 ────────────────────────
    def strip_shell(self, answer: str) -> str:
        text = answer or ""
        for p in self.shell:
            text = p.sub("", text)
        return text.strip()

    def _cos(self, a: str, b: str) -> float | None:
        if not self.embedder or not a or not b:
            return None
        # 走公共 similarity 接口(OllamaEmbedder 与 embed_local.Embedder 适配壳都实现)
        return self.embedder.similarity(a, b)

    # ── 判定:返回 (是否通过, 失败标签, 观察标签列表) ──────────
    def check(self, answer: str, question: str | None = None,
              material: str | None = None, expect: dict | None = None) -> tuple[bool, str, list[str]]:
        obs: list[str] = []
        if not self.enabled:
            return True, "", obs
        ans = answer or ""

        # (iii) 如实拒答优先:refuse_honesty 是正确行为,绝不判跑题(#0161 三级口径)
        if REFUSE_RE.search(ans):
            obs.append("如实拒答")
            return True, "", obs

        # (a)+(b) 合成:剥壳后有效字数
        eff = self.strip_shell(ans)
        if len(eff) < self.min_effective:
            tag = "纯壳复读" if len(ans) > self.min_effective else "过短"
            return False, f"[{tag}]", obs
        if self.strip_shell(ans) != ans.strip():
            obs.append("人设到位")               # 壳+实质=允许的人设化(观察标签)

        # (c) 格式结构化可判(#0154 补充②)
        if expect:
            sec_min = expect.get("sections_min")
            if sec_min and len([l for l in ans.splitlines() if l.strip()]) < int(sec_min):
                return False, "[格式错乱]", obs
            for kw in expect.get("must_include", []):
                if str(kw) not in ans:
                    return False, "[格式错乱]", obs

        # [答非所问](整个跑题,#0161):只判"有没有答到那件事",不判"多说了什么"。
        # ⚠️ cos 判据只在语义嵌入件上生效(is_semantic);按长度分档(#0171):
        #    长答案(≥80)走 问题级+材料级+实体全缺;短答案(15–80)走 问题级(短答阈值)
        #    +锚点全缺+数字一致——同一判据只计一次,不重复计数。
        semantic = getattr(self.embedder, "is_semantic", False)
        q_sim = self._cos(question, ans) if (question and semantic) else None
        m_sim = self._cos(material, ans) if (material and semantic) else None
        is_short = bool(material) and (
            self.min_effective <= len(ans) < self.material_fail_min_len) and self.short_answer_path

        if not is_short:
            if question and semantic and q_sim is not None and q_sim < self.offtopic_min_sim:
                return False, "[答非所问](问题级)", obs
            if material and semantic and m_sim is not None and m_sim < self.material_floor:
                # 长度门(#0167 A 语义,短答关闸时同样适用):材料级 Fail 只判长答案
                if len(ans) >= self.material_fail_min_len:
                    return False, "[答非所问](材料级<%.2f)" % self.material_floor, obs
                obs.append("短答结构偏低")
            if question:
                # 实体全缺(长答案/无材料档):问题里的科目/条款/数字,答案一个都没命中。
                # 判"全缺"要宽:答案与问题/材料共享任一 3-gram 锚即算命中(T5.7 回归)。
                from anchors import (_anchors_of, extract_accounts, extract_clauses,
                                     extract_numbers)
                q_anchors = (set(extract_numbers(question)) | set(extract_accounts(question))
                             | set(extract_clauses(question)))
                if q_anchors:
                    a_anchors = (set(extract_numbers(ans)) | set(extract_accounts(ans))
                                 | set(extract_clauses(ans)))
                    if not (q_anchors & a_anchors):
                        shared = _anchors_of(ans) & (_anchors_of(question) | _anchors_of(material or ""))
                        if not shared:
                            return False, "[答非所问](核心实体全缺)", obs
        else:
            # 短答案三判据(#0171 A-2)
            if semantic and q_sim is not None and q_sim < self.short_answer_min_sim:
                return False, "[答非所问](问题级短答)", obs
            if question:
                from anchors import (extract_accounts, extract_clauses, extract_numbers,
                                     check_numbers)
                q_anchors = (set(extract_numbers(question)) | set(extract_accounts(question))
                             | set(extract_clauses(question)))
                a_anchors = (set(extract_numbers(ans)) | set(extract_accounts(ans))
                             | set(extract_clauses(ans)))
                # ②锚点全缺:问题里的科目/条款/数字一个没命中(3-gram 宽锚兜底)
                if q_anchors and not (q_anchors & a_anchors):
                    from anchors import _anchors_of
                    # 宽锚兜底含材料:答案与问题或材料共享任一 3-gram 即算命中(T5.7 回归)
                    shared = _anchors_of(ans) & (_anchors_of(question) | _anchors_of(material))
                    if not shared:
                        return False, "[锚点全缺]", obs
                # ③数字一致:问题/材料里有数字,答案数字必须都能找到(只查新增,合法省略不查)
                r = check_numbers(question + " " + material, ans)
                if r["deny"]:
                    return False, "[数字未命中]", obs

        # 观察标签:答对但明显多讲(#0161(i):只打标,绝不 Fail)
        if material and len(ans) > len(material) * 1.5:
            obs.append("内容冗余")
        return True, "", obs

    def expect_for(self, intent: str) -> dict | None:
        """(c) 格式期望:按 intent 匹配 format_checks;无匹配 ⇒ None(不判格式)。"""
        for fc in self.format_checks:
            if fc.get("intent") == intent:
                return {k: v for k, v in fc.items() if k != "intent"}
        return None

    # ── 契约头(#0167 B):剥头/验头辅助 ────────────────────────
    def strip_header(self, answer: str, marker: str | None = None) -> tuple[str, bool]:
        """剥契约头:有头 ⇒ (剥头后内容, True);无头 ⇒ (原样, False)。剥头后才给前端。"""
        marker = marker or self.contract_marker
        if not marker:
            return answer, False
        stripped = answer.lstrip()
        if stripped.startswith(marker):
            return stripped[len(marker):].lstrip(), True
        return answer, False

    # ── Fail 留档(LoRA 素材,#0154 补充④;#0167 B:原始+剥头两栏) ──
    def log_fail(self, instruction: str, material: str, output: str,
                 reason: str, source: str = "", output_raw: str | None = None) -> None:
        if not self.fail_log_path:
            return
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
               "instruction": instruction[:300], "material": material[:800],
               "output": output[:600], "output_raw": (output_raw or output)[:600],
               "fail_tags": reason, "source": source}
        with open(self.fail_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def norm_len(s: str) -> str:
    return s
