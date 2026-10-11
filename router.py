"""router.py —— 混合架构路由器 v1 可跑骨架(合成版,2026-10-07)

按 DESIGN.md 逻辑流 + routing.yaml v0.2 本体组装:
    L1 硬规则(l1_rules) → L2 语义分类(l2_classifier;2026-10-09 起主选向量匹配
    l2_vector: bge-m3 kNN3,LLM 零样本降备选;枚举外/低置信/超时 fail-open)
    → 意图分流(persona_defense / tool_call / emotion_care / chat / knowledge_qa)
    → RAG 门(rag_gate:改写→检索→本地复算重排→阈值+margin) → 注入(injection,红线7)
    → 生成路由(generation:隐私/离线锁本地,本地硬墙计算式) → 兜底链 → answer_guard v2(红线8)

外部依赖(ollama / 检索 / bge-m3 / 云端)全部由构造参数注入;缺省给骨架假件。
测试契约见 test_router_v2.py 头部;设计红线见 DESIGN.md §4。
旧版纯结构骨架已备份为 router.py.bak-20261006-224504。
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from interfaces import (
    RouteTarget, RequestProfile, RagGateResult, GuardResult,
    RoutingDecision, RouteOutcome,
)
from metrics import Metrics
from config_loader import load_config, ConfigWatcher
from l1_rules import L1Layer
from privacy import PrivacyDetector
from l2_classifier import L2Classifier
from rag_gate import RagGate
from injection import Injector
from answer_guard import AnswerGuard
from exit_gate import ExitGate
from generation import GenerationRouter, BothFailError
from anchors import check_direction as anchors_check_direction
from rewrite_stage import RewriteStage
from semantic_cache import SemanticCache
from ragas_local import relevancy as l3_relevancy


# ── 骨架缺省假件(未注入时使用,只保证流水线可跑通) ──────────
class _DefaultClassifier:
    """缺省 L2 分类器:恒回 chat(骨架联调用;真实部署必须注入 ollama 件)。"""

    def __init__(self) -> None:
        self.call_count = 0

    def classify(self, prompt: str, temperature: float = 0.0,
                 max_tokens: int = 10, timeout_ms: int = 2000) -> str:
        self.call_count += 1
        return "chat"


class _TimeoutGuardClassifier:
    """超时守卫包装:超过 timeout_ms 即抛 TimeoutError(L2 捕获后 fail-open,红线1)。

    依赖方(ollama)自己不一定守时,出口兜底必须由本侧强制(#0006 精神)。
    内部调用仍恰好一次(工作者线程),调用参数原样透传。
    """

    def __init__(self, inner, timeout_ms: int) -> None:
        self._inner = inner
        self._timeout_ms = timeout_ms
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="l2cls")

    def classify(self, prompt: str, temperature: float = 0.0,
                 max_tokens: int = 10, timeout_ms: int = 2000) -> str:
        fut = self._pool.submit(self._inner.classify, prompt,
                                temperature, max_tokens, timeout_ms)
        try:
            return fut.result(timeout=self._timeout_ms / 1000.0)
        except FutureTimeout:
            raise TimeoutError(
                f"L2 classify exceeded {self._timeout_ms}ms(fail-open)") from None


class _DefaultEmbedder:
    """缺省嵌入件:字符级 difflib 比值(骨架联调用;真实部署注入 bge-m3)。"""

    def similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()


class _DefaultExecutor:
    """缺省生成端:回显应答(骨架联调用)。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.call_count = 0

    def execute(self, req: RequestProfile, rag_ctx: str | None) -> str:
        self.call_count += 1
        return f"[{self.name} 骨架应答] {req.text}"

    def is_available(self) -> bool:
        return True


# ── 路由器本体 ─────────────────────────────────────────────
class HybridRouter:
    """三层分流 + RAG 门 + 生成路由 + 兜底链 的编排器。"""

    def __init__(self, config_path: str = "routing.yaml", retrievers=None,
                 embedder=None, classifier=None, executors=None, rewriter=None,
                 l3_provider=None, auto_wire_l2: bool = False):
        self.config_path = config_path
        self._base_dir = os.path.dirname(os.path.abspath(config_path))
        self.config = load_config(config_path)
        self.metrics = Metrics()
        self.l3 = l3_provider   # L3 可替换 provider(l3_provider.py);None=骨架走 CLOUD executor

        # L2 自建接线(默认关,显式开启;测试一律注入假件,不受影响):
        # config intent_classifier.method=knn3/prototype 时,按配置用 ollama 嵌入
        # 组装向量分类件(kNN3 主选/原型法备选);失败显式报错,不静默降级。
        if auto_wire_l2 and classifier is None:
            method = self.config["intent_classifier"].get("method", "llm")
            if method in ("knn3", "prototype"):
                from l2_vector import build_from_config
                if embedder is None:
                    from embed_local import Embedder   # 统一嵌入入口(embedding.json 配后端/模型)
                    embedder = Embedder(consumer="l2")   # 分消费者配置(#0173 B):L2 快优先
                classifier = build_from_config(
                    self.config, embedder,
                    base_dir=os.path.dirname(os.path.abspath(config_path)))
                self._need_gate_embedder = True   # 出口闸/极性表用语义嵌入件(分消费者:exit_gate 可独立换)

        # 注入件归一化:executors 接受 {"local":…, "cloud":…} 或 {RouteTarget:…}
        ex = dict(executors or {})
        self.executors = {
            "local": ex.get("local") or ex.get(RouteTarget.LOCAL) or _DefaultExecutor("local"),
            "cloud": ex.get("cloud") or ex.get(RouteTarget.CLOUD) or _DefaultExecutor("cloud"),
        }
        self.retrievers = list(retrievers or [])
        self.rewriter = rewriter
        self.embedder = embedder or _DefaultEmbedder()
        if getattr(self, "_need_gate_embedder", False):
            # 出口闸/极性表用语义嵌入件(分消费者:exit_gate 可独立换"细粒度区分强"的模型)
            self.embedder = Embedder(consumer="exit_gate")

        raw_classifier = classifier or _DefaultClassifier()
        timeout_ms = int(self.config["intent_classifier"]["timeout_ms"])
        self.classifier = _TimeoutGuardClassifier(raw_classifier, timeout_ms)

        self._build_layers(self.config)
        self._watcher = ConfigWatcher(config_path) if os.path.exists(config_path) else None

    # ── 层构建(reload_config 复用) ──────────────────────────
    def _build_layers(self, config: dict) -> None:
        self.privacy = PrivacyDetector(config, self.metrics)
        self.l1 = L1Layer(config, self.metrics)
        self.l2 = L2Classifier(config, self.classifier, self.metrics)
        self.gate = RagGate(config, self.retrievers, self.embedder,
                            self.metrics, rewriter=self.rewriter)
        self.injector = Injector(config, self.metrics)
        self.guard = AnswerGuard(config, self.embedder, self.metrics)
        eg = (config.get("exit_gate") or {})
        self.direction_pairs = [tuple(x) for x in
                                ((config.get("answer_guard") or {}).get("direction_pairs") or [])]
        self.exit_gate = ExitGate(config, self.metrics, embedder=self.embedder, fail_log_path=os.path.join(
            self._base_dir, eg.get("fail_log", "exit_gate_fails.jsonl")))
        target_executors = {
            RouteTarget.LOCAL: self.executors["local"],
            RouteTarget.CLOUD: self.executors["cloud"],
        }
        self.gen = GenerationRouter(config, target_executors, self.metrics)

        # 输入守卫(2026-10-10 改进项②,借 llm-intent-router 启发式):
        # 非文字垃圾输入(纯标点/emoji/空白)→ 拒答,不进 L1/L2/检索。
        ig = config.get("input_guard", {}) if isinstance(config, dict) else {}
        if ig.get("enabled") and ig.get("garbage_pattern"):
            self._garbage_re = re.compile(ig["garbage_pattern"])
            self._garbage_reply = ig.get("refuse_reply", "没看懂想问什么,换个说法?")
        else:
            self._garbage_re = None

        # L3 出口质量门(2026-10-10 收尾,借 Sriti quality_gate + RouteLM relevancy):
        # relevancy 达标才收货;不过→同层重试→仍不过由调用方降级本地如实标注。
        lg = config.get("l3_quality_gate", {}) if isinstance(config, dict) else {}
        if lg.get("enabled"):
            self.l3_gate: dict | None = {
                "relevancy_min": float(lg.get("relevancy_min", 0.3)),   # 待标定
                "max_retries": int(lg.get("max_retries", 1)),
            }
        else:
            self.l3_gate = None

        # 念稿两段式改写段(2026-10-10):config rewrite.enabled 开关,默认关;
        # 仅 knowledge_qa 且 RAG 已注入时接管,测试/skeleton(无 rewrite 节)不受影响。
        rc = config.get("rewrite", {}) if isinstance(config, dict) else {}
        if rc.get("enabled") and self.l3 is not None:
            self.rewrite_stage: RewriteStage | None = RewriteStage(
                config, self.l3, self._local_ask, self.embedder)
        else:
            self.rewrite_stage = None

        # 语义缓存(2026-10-10 改进项①,vLLM-SR 思路):knowledge_qa 查缓存,
        # 命中复用答案省云端;config semantic_cache.enabled 开关,默认关。
        sc = config.get("semantic_cache", {}) if isinstance(config, dict) else {}
        if sc.get("enabled"):
            self.semantic_cache: SemanticCache | None = SemanticCache(
                config, self.embedder,
                base_dir=os.path.dirname(os.path.abspath(self.config_path)))
        else:
            self.semantic_cache = None

    def _ask_l3_gated(self, prompt: str, decision) -> str | None:
        """L3 ask + 质量门;None = 门未过,调用方降级本地。门关时原样透传。"""
        answer = self.l3.ask(prompt)
        if self.l3_gate is None:
            decision.reasons.append("L3 provider 答复成功")
            return answer
        score = 0.0
        for attempt in range(self.l3_gate["max_retries"] + 1):
            score = l3_relevancy(prompt, answer, self.embedder)
            if score >= self.l3_gate["relevancy_min"]:
                decision.reasons.append(f"L3 质量门通过(relevancy={score:.0%})")
                return answer
            if attempt < self.l3_gate["max_retries"]:
                self.metrics.inc("l3_gate_retry")
                answer = self.l3.ask(prompt)
        self.metrics.inc("l3_gate_rejected")
        decision.reasons.append(
            f"L3 质量门未过(relevancy={score:.0%}<{self.l3_gate['relevancy_min']:.0%})")
        return None

    def _local_ask(self, prompt: str) -> str:
        """本地生成端直连(忠读模式接龙用):文本进 RequestProfile 走 local executor。"""
        from interfaces import RequestProfile
        return self.executors["local"].execute(
            RequestProfile(text=prompt, raw_text=prompt), None)

    def reload_config(self, config: dict) -> None:
        """词表/阈值热加载入口(红线5):传新配置 dict,即时重建各层,不重启。"""
        self.config = config
        self._build_layers(config)

    # ── 主入口 ─────────────────────────────────────────────
    def route(self, raw_text: str, context_tokens: int = 0,
              offline_mode: bool = False, cloud_quota: int = -1) -> RouteOutcome:
        # 磁盘配置变了就热加载(词表增删不重启)
        if self._watcher is not None:
            fresh = self._watcher.reload_if_stale()
            if fresh is not None:
                self.reload_config(fresh)

        # 0) 隐私双入口(纸⑤D):显式标记 > 词表 > 默认;前缀在路由/检索前剥除
        req, privacy_source = self.privacy.detect(raw_text)
        req.context_tokens = context_tokens
        req.offline_mode = offline_mode
        req.cloud_quota_remaining = cloud_quota

        # 0.5) 输入守卫(2026-10-10):垃圾输入直接拒,不进 L1/L2/检索
        if self._garbage_re is not None and self._garbage_re.search(req.text):
            self.metrics.inc("input_guard_rejected")
            return RouteOutcome(
                decision=RoutingDecision(
                    intent="garbage",
                    reasons=["输入守卫:非文字垃圾输入 → 拒答(不进检索不外发)"]),
                rag=RagGateResult(needed=False),
                answer=self._garbage_reply, guard=None)

        # 1) L1 硬规则(只匹配不仲裁,红线3)
        l1 = self.l1.match(req.text)
        l2_result = None
        if not l1.escalated:
            hit = l1.hits[0]
            intent = hit.intent
            req.domain = hit.domain
            reasons = [f"L1 单命中 {hit.rule_id} → {intent}"]
        else:
            # 2) L2 轻量语义分类(枚举外/超时 fail-open 直升 L3,红线1)
            l2_result = self.l2.classify(req.text)
            if l2_result.ok:
                intent = l2_result.mapped_intent
                reasons = [f"L1 零命中或冲突 → L2={l2_result.category} → {intent}"]
            else:
                intent = "chat"          # L3 由生成路由强制云端体现
                reasons = ["L2 fail-open(枚举外/低置信/超时)→ L3 云端直答"]

        decision = RoutingDecision(intent=intent, reasons=reasons)

        # 3) 意图分流
        rag = RagGateResult(needed=False)
        rag_material: str | None = None
        top1 = None
        guard_result: GuardResult | None = None
        answer = ""

        if intent == "persona_defense":
            # 本地直出:零检索、零外发(T7.1)
            decision.target = RouteTarget.LOCAL
            decision.fallback = RouteTarget.CLOUD
            decision.reasons.append("persona_defense → 本地直出(零检索零外发)")
            answer = self.gen.dispatch(RouteTarget.LOCAL, req, None)

        elif intent == "tool_call":
            # 宁窄勿宽(红线4):执行前确认;未确认绝不执行(T6.1)
            # hit 可能为空:L2 也能产出 tool_call(经 intent_map)。
            # B(#0139 用户拍板):L2 来源的 tool_call 也挂 confirm(危险误判兜底,
            # reasons 记来源 l2_tool_call);L1 规则表不动,各规则仍按自身 confirm。
            hit = l1.hits[0] if l1.hits else None
            if hit is not None:
                decision.tool_needs_confirm = bool(hit.confirm)
                decision.tool_args_schema = hit.args_schema
                decision.reasons.append("tool_call 来源=L1:" + hit.rule_id)
            else:
                decision.tool_needs_confirm = True
                decision.reasons.append("tool_call 来源=l2_tool_call(confirm 兜底)")
            decision.target = RouteTarget.LOCAL
            decision.fallback = RouteTarget.CLOUD
            self._last_tool = {"req": req, "decision": decision}
            if not decision.tool_needs_confirm:
                answer = self.gen.dispatch(RouteTarget.LOCAL, req, None)  # 无副作用/待拍板类直接执行
            else:
                answer = "工具调用待确认:该操作有副作用,请确认后执行(confirm_tool)。"

        elif intent == "emotion_care":
            decision.target = RouteTarget.LOCAL
            decision.fallback = RouteTarget.CLOUD
            decision.reasons.append("emotion_care → 本地安抚(不检索不上云,纸③B)")
            answer = self.gen.dispatch(RouteTarget.LOCAL, req, None)

        elif intent == "realtime_lookup":
            # 实时类(2026-10-10 借 llm-intent-router):本地知识库永无实时答案,
            # 强制云端直答,不检索(跳过 RAG 门),失败降级本地如实呈现。
            self.metrics.inc("realtime_route_count")
            decision.target = RouteTarget.CLOUD
            decision.fallback = RouteTarget.LOCAL
            decision.reasons.append("realtime_lookup → 本地库无实时信息,直答云端")
            if self.l3 is not None:
                answer, gate_note = None, ""
                try:
                    answer = self._ask_l3_gated(req.text, decision)
                    if answer is None:
                        gate_note = "[L3 质量门未过,已降级本地] "
                except Exception as e:
                    decision.reasons.append(f"L3 不可用({e}) → 降级本地")
                    gate_note = "[L3 云端不可用,已降级本地] "
                if answer is None:
                    answer = gate_note + self.gen.dispatch(RouteTarget.LOCAL, req, None)
            else:
                answer = self.gen.dispatch(RouteTarget.CLOUD, req, None)

        else:
            # chat / knowledge_qa / privacy_sensitive / L3 兜底
            raw_answer = ""                    # 原始输出(未剥头),Fail 留档两栏用
            rag_material = ""                  # 念稿改写段原料(注入前原文)
            if l2_result is not None and not l2_result.ok:
                # L2 fail-open:config fail_open: cloud → 生成端强制云端
                decision.target = RouteTarget.CLOUD
                decision.fallback = RouteTarget.LOCAL
                decision.reasons.append("fail_open: cloud → 直升 L3 云端")
            else:
                t, fb, rs = self.gen.decide(req)
                decision.target, decision.fallback = t, fb
                decision.reasons.extend(rs)

            # 4) RAG 门(仅 knowledge_qa 触发;RAG≠生成绑定,红线2)
            # 语义缓存先查(2026-10-10):命中即复用,跳过检索与生成
            if (self.semantic_cache is not None and intent == "knowledge_qa"
                    and not (l2_result is not None and not l2_result.ok)):
                cached = self.semantic_cache.lookup(req.text)
                if cached is not None:
                    self.metrics.inc("semantic_cache_hit")
                    decision.reasons.append("语义缓存命中 → 复用既有答案(跳过RAG与生成)")
                    return RouteOutcome(decision=decision, rag=rag,
                                        answer=cached, guard=None)

            decision.use_rag = self.gate.should_trigger(intent)
            if decision.use_rag:
                rag, top1 = self.gate.evaluate(req)
                if rag.injected and top1 is not None:
                    # 5) 注入出口硬约束(红线7):无条件截断并回报丢弃量
                    composed, dropped = self.injector.build_user_turn(top1.text, req.text)
                    material = composed[:composed.rfind(req.text)].strip()
                    rag.rag_ctx = "[知识库] " + material.removeprefix("[知识库] ").strip()
                    rag.dropped_chars = dropped
                    rag_material = material   # 念稿改写段原料(注入前原文)
                elif rag.reason == "no_hit_after_rewrite":
                    # 改写后仍 0 命中:refuse_honesty,不注入不自由发挥(#0019-4)
                    decision.reasons.append("no_hit_after_rewrite → refuse_honesty")
                    return RouteOutcome(decision=decision, rag=rag,
                                        answer="库里没查到这条,我不瞎编——换官方准则原文核对吧。",
                                        guard=None)

            # 6) 生成 + 兜底链(T5.1/T5.2/T5.3)
            # L3 分支:注入了 l3 provider 时,fail-open 直连云端大脑(可替换层,
            # 换别家只改 l3_provider.py);失败如实报错并降级本地,禁止静默吞。
            if l2_result is not None and not l2_result.ok and self.l3 is not None:
                answer, gate_note = None, ""
                try:
                    answer = self._ask_l3_gated(req.text, decision)
                    if answer is None:
                        gate_note = "[L3 质量门未过,已降级本地] "
                except Exception as e:      # provider 无关:任何失败都如实呈现
                    decision.reasons.append(f"L3 不可用({e}) → 降级本地")
                    gate_note = "[L3 云端不可用,已降级本地] "
                if answer is None:
                    answer = gate_note + self.gen.dispatch(RouteTarget.LOCAL, req, None)
                decision.target = RouteTarget.CLOUD
                decision.fallback = RouteTarget.LOCAL
            # RAG 链路未注入原料时:本地失败不转云端自由发挥(红线8:绝不返回
            # 自编内容),空答案统一走下方 refuse_no_material 出口。
            elif rag.needed and not rag.injected:
                ex = self.executors["local" if decision.target == RouteTarget.LOCAL else "cloud"]
                try:
                    answer = ex.execute(req, None)
                except Exception:
                    answer = ""
            else:
                if (self.rewrite_stage is not None and rag_material
                        and intent == "knowledge_qa"):
                    # 念稿两段式:改写/忠读模式切换 + 锚点出口校验(2026-10-10)
                    ans2, mode, why = self.rewrite_stage.run(rag_material, req.text)
                    if self.rewrite_stage.last_contract_ok is False:
                        # 契约头缺失:只记数不 Fail(#0167 B)
                        self.metrics.inc("exit_gate_contract_missing")
                        decision.reasons.append("契约头缺失(contract_missing,记数)")
                    if ans2:
                        answer = ans2
                        raw_answer = self.rewrite_stage.last_raw_output
                        decision.reasons.append(f"念稿改写段 mode={mode}({why})")
                    else:
                        answer = self.gen.dispatch(decision.target, req, rag.rag_ctx)
                else:
                    answer = self.gen.dispatch(decision.target, req, rag.rag_ctx)

            # 7) 出口硬兜(#0154/#0178,先于 guard):剥壳有效字数/纯壳/格式不符 ⇒ Fail
            #    不出前端,同路重试一次;仍 Fail ⇒ 降级链直出原文(top1 存在时),无原料
            #    则保留并留档。适用面=knowledge_qa(念稿/RAG 生成场景);chat/人设/情绪/
            #    工具分支不进此闸。重试产物同样要过下方 guard 收口。
            if self.exit_gate.enabled and intent == "knowledge_qa":
                ok, reason, gate_obs = self.exit_gate.check(
                    answer, question=req.text, material=rag.rag_ctx or "",
                    expect=self.exit_gate.expect_for(intent))                # (c) 格式判据在线接线(#0178)
                if not ok:
                    self.metrics.inc("exit_gate_fail")
                    self.exit_gate.log_fail(req.text, rag.rag_ctx or "", answer, reason,
                                            source=f"intent:{intent}", output_raw=raw_answer)
                    decision.reasons.append(f"exit_gate Fail({reason}) → 重试一次")
                    if rag.needed and not rag.injected:
                        # 红线8:RAG 未注入原料,重试也不转云端自由发挥
                        ex = self.executors["local" if decision.target == RouteTarget.LOCAL else "cloud"]
                        try:
                            answer2 = ex.execute(req, None)
                        except Exception:
                            answer2 = ""
                    else:
                        answer2 = self.gen.dispatch(decision.target, req, rag.rag_ctx)
                    self.metrics.inc("exit_gate_retry")
                    ok2, reason2, _ = self.exit_gate.check(answer2, question=req.text, material=rag.rag_ctx or "")
                    if ok2:
                        self.metrics.inc("exit_gate_recovered")
                        answer = answer2
                        decision.reasons.append("出口硬兜重试 → 服从")
                    elif top1 is not None:
                        answer = top1.text
                        decision.reasons.append("出口硬兜重试仍 Fail → 直出原文")
                    else:
                        self.exit_gate.log_fail(req.text, rag.rag_ctx or "", answer2,
                                                reason2 or reason, source="retry",
                                                output_raw=raw_answer)

            # 8) 空答案出口(T5.8-B):生成空串且无注入原料 → refuse_no_material,绝不静默吞
            if not answer.strip():
                guard_result = GuardResult(passed=False, failed_criterion="empty",
                                           action="refuse_no_material")
                self.metrics.inc("answer_guard_rejected")
                answer = "库里没查到相关资料,不能瞎说。"

            # 9) answer_guard v2(红线8)——最后一道,对"最终答案"判(含出口硬兜的重试
            #    产物);未注入时原料视为空,数字参考改取问题(#0184 (b),仅 rag.needed
            #    域内题,chat 零波及)
            if rag.needed or top1 is not None:
                raw_for_guard = top1.text if (rag.injected and top1 is not None) else None
                guard_result = self.guard.check(req.text, raw_for_guard, answer,
                                                number_ref_query=bool(rag.needed))
                if not guard_result.passed:
                    if guard_result.action == "cite_original":
                        answer = top1.text            # 直出原文片段
                    else:
                        answer = "库里没查到相关资料,不能瞎说。"  # refuse_no_material
                    decision.reasons.append(
                        f"answer_guard 拦截({guard_result.failed_criterion}) → {guard_result.action}")
            # 9) 方向极性表 review 观察(#0167 C):只标不拒,报表单列[方向说反]
            if self.direction_pairs and (rag_material or rag.rag_ctx) and answer.strip():
                dr = anchors_check_direction(rag_material or rag.rag_ctx, answer,
                                             pairs=self.direction_pairs)
                if dr.get("deny"):
                    self.metrics.inc("direction_polarity_hits")
                    decision.reasons.append("[方向说反]极性表 review 命中(只标不拒)")

        # 语义缓存写回(2026-10-10):只存成功答案(非空、非拒答、guard 通过)
        if (self.semantic_cache is not None and intent == "knowledge_qa"
                and answer.strip() and "库里没查到" not in answer
                and (guard_result is None or guard_result.passed)):
            self.semantic_cache.store(req.text, answer)

        return RouteOutcome(decision=decision, rag=rag, answer=answer, guard=guard_result)

    # ── 工具确认入口(T6.2:缺参数拒执行,不许带默认值硬跑) ────
    def confirm_tool(self, args: dict, confirmed: bool = True) -> str:
        """带参确认/执行工具。args 缺 schema 必填键 → 拒绝并回询;未确认 → 不执行。"""
        last = getattr(self, "_last_tool", None)
        if last is None:
            return "没有待确认的工具调用。"
        decision: RoutingDecision = last["decision"]
        req: RequestProfile = last["req"]
        if decision.tool_needs_confirm and not confirmed:
            return "该工具调用尚未确认,拒绝执行。"
        schema = decision.tool_args_schema or {}
        missing = [k for k in schema if k not in (args or {})]
        if missing:
            return f"参数缺失:{'、'.join(missing)}。请补全后再执行,不带默认值硬跑。"
        self.metrics.inc("tool_confirmed")
        executor = self.executors["local"]
        return executor.execute(req, None)


if __name__ == "__main__":
    # 最小自测:三条主链路各跑一遍
    rt = HybridRouter(config_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "routing.yaml"))
    for text, kw in [
        ("递延所得税怎么处理", {}),                 # L1 → knowledge_qa → RAG 门
        ("我那个破电脑又出问题了", {}),             # L2 → chat
        ("帮我看看这份工资单的个税对不对", {"context_tokens": 20000}),  # 隐私锁本地
    ]:
        out = rt.route(text, **kw)
        print(f"{text!r} → intent={out.decision.intent} target={out.decision.target.value} "
              f"rag={out.rag.reason or '-'} answer={out.answer[:24]!r}")
    print("metrics:", {k: v for k, v in rt.metrics.snapshot().items() if v})
