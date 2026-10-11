"""test_router_v2.py —— TESTS.md 全用例(T1.1–T9.6)可执行测试线束

被测契约(router.py 由主合成者按此实现):
    rt = HybridRouter(config_path="routing.yaml", retrievers=None, embedder=None,
                      classifier=None, executors=None, rewriter=None)
    rt.route(raw_text, context_tokens=0, offline_mode=False, cloud_quota=-1)
        -> interfaces.RouteOutcome(decision/rag/answer/guard)
    rt.metrics -> interfaces 里的 Metrics 实例(duck-typed 读取)
    rt.reload_config(config: dict) -> 词表热加载(契约补丁,T9.3 用)

说明:
- 纯断言 + 计数,不依赖 pytest;入口在文件底部 __main__。
- Mock 件全部在本文件实现,可脚本化。
- 每条用例一个函数;断言失败即 FAIL(原因=断言消息);运行期异常也记 FAIL。
- 注 1:T1.3 的冲突样例按 routing.yaml 真实词表调整为
  "工资单里的递延所得税怎么处理"(privacy-keywords 与 rag-a1 同时命中不同 intent),
  TESTS.md 原例"帮我写个审计笑话"在 yaml v0.2 词表下只单命中 rag-a4,构不成冲突。
- 注 2:T1.5 样例同理改为 "关闭客厅灯"(yaml 白名单是 设备词紧随动词,"把客厅的灯关了"
  的语序在正则 (开|关|打开|关闭)(灯|...) 下不命中)。
- 注 3:T2.3/T2.4 的 timeout_ms 以 routing.yaml intent_classifier.timeout_ms=2000 为准
  (TESTS.md 行文里的 800 与 yaml 本体不一致,配置本体优先)。
"""
from __future__ import annotations

import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import interfaces
from interfaces import RouteTarget

from router import HybridRouter
from config_loader import load_config

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "routing.yaml")

# ── 公共文本/候选(全部对齐 routing.yaml 真实词表) ──────────────
TXT_TAX = "递延所得税怎么处理"                    # L1 rag-a1 单命中
TXT_BROKEN_PC = "我那个破电脑又出问题了"          # L1 零命中
TXT_CONFLICT = "工资单里的递延所得税怎么处理"     # privacy + rag-a1 冲突(见注1)
TXT_OPEN_MIND = "帮我打开思路"                    # 动词后非设备词,不命中 tool
TXT_LIGHT = "关闭客厅灯"                          # tool-device-switch,confirm=true
TXT_WEEKLY = "帮我写个周报"                       # 零命中,默认非敏感
TXT_PERSONA = "你才是[LocalModelName]"                      # persona-defense
TXT_PAYSLIP = "帮我看看这份工资单的个税对不对"    # privacy-keywords(词表)
TXT_MARK_TAX = "#本地# 递延所得税怎么处理"        # 显式标记 + 域内知识问答
TXT_MARK_BOTH = "#本地# 帮我看看这份工资单"       # 显式标记 + 词表 双入口
TXT_CONTRACT_LIAB = "合同负债的确认时点"          # 负向排除回归:rag-a3 而非 privacy
TXT_AUDIT_50P = "总结这份 50 页审计准则的要点"    # rag-a4 + 长上下文
TXT_INV = "存货跌价准备怎么计提"                  # rag-a2

CHUNK_GOOD = "递延所得税资产确认要点:以未来期间很可能取得的应纳税所得额为限,按 25% 税率计量。"
CHUNK_BAD = "与本题无关的杂项笔记,顺带一提天气不错。"
CHUNK_C1 = "递延所得税资产确认要点:资产负债表日复核减记。"
CHUNK_C2 = "暂时性差异当期转回时的一般处理思路。"
CHUNK_WRONG = "CAS 40 号文具管理规定,与税务无关,泛泛而谈。"
CHUNK_INV = "存货跌价准备按成本与可变现净值孰低计量,减值测试后可转回。"
CHUNK_AUDIT = "审计准则要求注册会计师保持职业怀疑,获取充分适当的审计证据。"

FAB_ANSWER = "递延所得税资产应按 15% 计提,以未来期间很可能取得的应纳税所得额为限。"  # 编 15%
OFFTOPIC_ANSWER = "先进先出法是存货计价的一种方法,先购入的先发出。"
PERSONA_OK_ANSWER = "这个呀,本鲸给你掰开揉碎说哈:成本比可变现净值高的时候就要减记,不许硬撑着不认。"
GUARD_CONSISTENT_REPLY = "递延所得税资产以未来期间很可能取得的应纳税所得额为限,按 25% 税率计量。"


# ── Mock 件 ─────────────────────────────────────────────────
class MockClassifier:
    """可脚本化 L2 分类器。mode: ok=返回 reply / timeout=先 sleep(delay) 再返回。"""

    def __init__(self, reply: str = "chat", mode: str = "ok", delay: float = 0.0):
        self.reply = reply
        self.mode = mode
        self.delay = delay
        self.call_count = 0
        self.calls: list[dict] = []

    def classify(self, prompt: str, temperature: float = 0.0,
                 max_tokens: int = 10, timeout_ms: int = 2000) -> str:
        self.call_count += 1
        self.calls.append({"prompt": prompt, "temperature": temperature,
                           "max_tokens": max_tokens, "timeout_ms": timeout_ms})
        if self.mode == "timeout":
            time.sleep(self.delay)
        return self.reply


class MockRetriever:
    """可脚本化检索器:script 里 key 是查询子串,命中返回对应候选;否则返回 default。"""

    def __init__(self, default: list | None = None, script: dict | None = None,
                 name: str = "quaesto_kb"):
        self.name = name
        self.default = list(default or [])
        self.script = dict(script or {})
        self.calls: list[tuple[str, int]] = []
        self.call_count = 0

    def retrieve(self, query: str, top_k: int = 5) -> list:
        self.calls.append((query, top_k))
        self.call_count += 1
        for key, chunks in self.script.items():
            if key in query:
                return list(chunks)
        return list(self.default)


class MockEmbedder:
    """可脚本化相似度。优先级:pairs(双向包含) > a_rules(第一参数子串) >
    b_rules(第二参数子串) > default。"""

    def __init__(self, default: float = 0.30, pairs: list | None = None,
                 a_rules: list | None = None, b_rules: list | None = None):
        self.default = default
        self.pairs = list(pairs or [])       # [(sub_a, sub_b, score), ...]
        self.a_rules = list(a_rules or [])   # [(sub, score), ...]
        self.b_rules = list(b_rules or [])
        self.calls: list[tuple[str, str]] = []

    def similarity(self, a: str, b: str) -> float:
        self.calls.append((a, b))
        for sa, sb, score in self.pairs:
            if (sa in a and sb in b) or (sb in a and sa in b):
                return score
        for sub, score in self.a_rules:
            if sub in a:
                return score
        for sub, score in self.b_rules:
            if sub in b:
                return score
        return self.default


class MockExecutor:
    """记录调用;fail=True 时 execute 抛异常(兜底链测试用)。"""

    def __init__(self, name: str, reply: str = "默认回答", fail: bool = False):
        self.name = name
        self.reply = reply
        self.fail = fail
        self.call_count = 0
        self.calls: list[tuple[object, str | None]] = []

    def execute(self, req, rag_ctx) -> str:
        self.call_count += 1
        self.calls.append((req, rag_ctx))
        if self.fail:
            raise RuntimeError(f"{self.name} executor 注入故障")
        return self.reply

    def is_available(self) -> bool:
        return True


class MockRewriter:
    """可脚本化改写器:mapping 原话→术语化查询,缺省原样返回。"""

    def __init__(self, mapping: dict | None = None):
        self.mapping = dict(mapping or {})
        self.calls: list[str] = []

    def rewrite(self, query: str) -> str:
        self.calls.append(query)
        return self.mapping.get(query, query)

    __call__ = rewrite


# ── 构造/读取工具 ───────────────────────────────────────────
def make_router(retrievers=None, embedder=None, classifier=None,
                executors=None, rewriter=None) -> HybridRouter:
    return HybridRouter(config_path=CONFIG_PATH, retrievers=retrievers,
                        embedder=embedder, classifier=classifier,
                        executors=executors, rewriter=rewriter)


def mk_execs(local: MockExecutor | None = None,
             cloud: MockExecutor | None = None) -> dict:
    return {"local": local or MockExecutor("local", "本地默认回答"),
            "cloud": cloud or MockExecutor("cloud", "云端默认回答")}


def _read_counter(metrics, name: str):
    """duck-type 读取 metrics 计数器(兼容 .get/.counters/.values/.snapshot/属性)。"""
    if metrics is None:
        return None
    fn = getattr(metrics, "get", None)
    if callable(fn):
        try:
            v = fn(name)
            if isinstance(v, int):
                return v
        except Exception:
            pass
    for attr in ("counters", "values", "_counters"):
        d = getattr(metrics, attr, None)
        if isinstance(d, dict) and name in d:
            try:
                return int(d[name])
            except Exception:
                pass
    snap = getattr(metrics, "snapshot", None)
    if callable(snap):
        try:
            s = snap()
            if isinstance(s, dict) and name in s:
                return int(s[name])
        except Exception:
            pass
    v = getattr(metrics, name, None)
    if isinstance(v, int):
        return v
    return None


def counter(rt, name: str) -> int:
    v = _read_counter(rt.metrics, name)
    if v is None:
        raise AssertionError(f"metrics 未暴露计数器 {name}(无法断言)")
    return v


# 高置信知识问答的公共环境:检索必中且可注入
def high_conf_env(local_reply: str = GUARD_CONSISTENT_REPLY):
    retriever = MockRetriever(default=[CHUNK_GOOD, CHUNK_BAD])
    embedder = MockEmbedder(default=0.30,
                            pairs=[(TXT_TAX, CHUNK_GOOD, 0.90),
                                   (TXT_TAX, CHUNK_BAD, 0.50)])
    local = MockExecutor("local", local_reply)
    cloud = MockExecutor("cloud", "云端默认回答")
    rt = make_router(retrievers=[retriever], embedder=embedder,
                     executors=mk_execs(local, cloud))
    return rt, retriever, embedder, local, cloud


# ══ L1 硬规则层 ════════════════════════════════════════════
def t1_1():
    """T1.1 单命中直通:L1 命中即 intent=knowledge_qa,不进 L2。"""
    cls = MockClassifier("chat")
    rt = make_router(classifier=cls, executors=mk_execs())
    out = rt.route(TXT_TAX)
    assert out.decision.intent == "knowledge_qa", \
        f"intent={out.decision.intent!r},应为 knowledge_qa"
    assert cls.call_count == 0, f"L1 单命中不应进 L2,classifier 被调 {cls.call_count} 次"
    assert out.decision.use_rag is True, "knowledge_qa 应触发 RAG 门"


def t1_2():
    """T1.2 零命中降级:无规则命中 → 进入 L2。"""
    cls = MockClassifier("chat")
    rt = make_router(classifier=cls, executors=mk_execs())
    rt.route(TXT_BROKEN_PC)
    assert cls.call_count >= 1, "L1 零命中未降级 L2(classifier 未被调用)"


def t1_3():
    """T1.3 冲突不仲裁〔红线3〕:privacy 与 knowledge_qa 双命中 → 降 L2,不在 L1 裁决。"""
    cls = MockClassifier("chat")
    rt = make_router(classifier=cls, executors=mk_execs())
    out = rt.route(TXT_CONFLICT)
    assert cls.call_count >= 1, \
        f"多规则冲突应降 L2(classifier 调用 {cls.call_count} 次),L1 不得自行仲裁"
    assert out.decision.intent != "privacy_sensitive" or cls.call_count >= 1, \
        "冲突场景 intent 必须来自 L2 之后的链路"


def t1_4():
    """T1.4 工具类收窄〔红线4〕:"帮我打开思路" 不命中 tool_call。"""
    cls = MockClassifier("chat")
    rt = make_router(classifier=cls, executors=mk_execs())
    out = rt.route(TXT_OPEN_MIND)
    assert out.decision.intent != "tool_call", \
        f"'打开思路' 误命中 tool_call(动宾+设备白名单未生效),intent={out.decision.intent!r}"
    assert cls.call_count >= 1, "未命中 tool 应落 L2 而非凭空定意图"


def t1_5():
    """T1.5 工具类正常命中:intent=tool_call 且带 args_schema(device/action)。"""
    rt = make_router(classifier=MockClassifier("chat"), executors=mk_execs())
    out = rt.route(TXT_LIGHT)
    assert out.decision.intent == "tool_call", \
        f"intent={out.decision.intent!r},应为 tool_call"
    schema = out.decision.tool_args_schema
    assert isinstance(schema, dict) and "device" in schema and "action" in schema, \
        f"args_schema 缺 device/action 字段: {schema!r}"


# ══ L2 轻量语义层 ══════════════════════════════════════════
def t2_1():
    """T2.1 枚举内采纳:输出 professional → 经 intent_map 映射为 knowledge_qa。"""
    cls = MockClassifier("professional")
    rt = make_router(classifier=cls, executors=mk_execs())
    out = rt.route(TXT_BROKEN_PC)
    assert out.decision.intent == "knowledge_qa", \
        f"intent={out.decision.intent!r},professional 应经 intent_map 映射为 knowledge_qa"
    assert counter(rt, "L2_hit") == 1, "L2_hit 计数应为 1"


def t2_2():
    """T2.2 枚举外 fail-open〔红线1〕:不重试,直接走 L3。"""
    cls = MockClassifier("大概是专业问题吧我觉得")
    rt = make_router(classifier=cls, executors=mk_execs())
    out = rt.route(TXT_BROKEN_PC)
    assert cls.call_count == 1, \
        f"枚举外禁止重试硬猜,classify 被调 {cls.call_count} 次"
    assert counter(rt, "L2_fail_open") == 1, "L2_fail_open 应 +1"
    assert counter(rt, "L3_hit") == 1, "枚举外应直升 L3"
    assert out.answer, "fail-open 后 L3 仍应产出答案,不得空返回"


def t2_3():
    """T2.3 超时 fail-open〔红线1〕:分类耗时 > timeout_ms=2000 → 走 L3,不阻塞不重试。"""
    cls = MockClassifier("professional", mode="timeout", delay=2.5)
    rt = make_router(classifier=cls, executors=mk_execs())
    out = rt.route(TXT_BROKEN_PC)
    assert cls.call_count == 1, f"超时禁止重试,classify 被调 {cls.call_count} 次"
    assert counter(rt, "L2_fail_open") == 1, \
        "分类超时未 fail-open(router 疑似同步等完了分类结果再采纳)"
    assert counter(rt, "L3_hit") == 1, "超时应直升 L3"
    assert out.answer, "超时 fail-open 后仍应产出答案"


def t2_4():
    """T2.4 输出约束生效:temperature/max_tokens/timeout 逐项与配置一致。"""
    cls = MockClassifier("chat")
    rt = make_router(classifier=cls, executors=mk_execs())
    rt.route(TXT_BROKEN_PC)
    assert cls.call_count >= 1, "classifier 未被调用"
    kw = cls.calls[0]
    assert kw["temperature"] == 0, f"temperature={kw['temperature']!r},配置要求 0"
    assert kw["max_tokens"] == 10, \
        f"max_tokens={kw['max_tokens']!r},配置 max_output_tokens=10"
    assert kw["timeout_ms"] == 2000, \
        f"timeout_ms={kw['timeout_ms']!r},配置 intent_classifier.timeout_ms=2000"


def t2_5():
    """T2.5 闲聊降级样例:L2=emotion → 本地情感安抚链路,不检索不上云。"""
    cls = MockClassifier("emotion")
    retriever = MockRetriever(default=[CHUNK_GOOD])
    cloud = MockExecutor("cloud")
    local = MockExecutor("local", "本鲸在呢,慢慢说~")
    rt = make_router(retrievers=[retriever], classifier=cls,
                     executors=mk_execs(local, cloud))
    out = rt.route(TXT_BROKEN_PC)
    assert out.decision.intent == "emotion_care", \
        f"intent={out.decision.intent!r},emotion 应映射为 emotion_care"
    assert out.decision.target == RouteTarget.LOCAL, "情感安抚应走本地"
    assert retriever.call_count == 0, "emotion 不应触发检索"
    assert cloud.call_count == 0, "emotion 不应上云"
    assert counter(rt, "emotion_route_count") >= 1, "emotion_route_count 应 +1"


# ══ RAG 门 ═════════════════════════════════════════════════
def t3_1():
    """T3.1 仅知识问答触发:chat / tool_call 不调用 retriever。"""
    retriever = MockRetriever(default=[CHUNK_GOOD])
    rt = make_router(retrievers=[retriever],
                     classifier=MockClassifier("chat"), executors=mk_execs())
    rt.route(TXT_BROKEN_PC)          # chat
    rt.route(TXT_LIGHT)              # tool_call(只确认不执行)
    assert retriever.call_count == 0, \
        f"非 knowledge_qa 意图不应检索,retriever 被调 {retriever.call_count} 次"


def t3_2():
    """T3.2 高置信注入:hit_confidence=0.9 → rag_ctx 注入生成端。"""
    rt, retriever, embedder, local, cloud = high_conf_env()
    out = rt.route(TXT_TAX)
    assert out.rag.injected is True, "置信 0.9 ≥ 0.58 应注入"
    assert out.rag.rag_ctx, "rag_ctx 不应为空"
    assert out.rag.confidence >= 0.58, f"confidence={out.rag.confidence},应 ≥0.58"
    assert local.call_count >= 1 and local.calls[0][1] == out.rag.rag_ctx, \
        "生成端未收到与 rag_gate 一致的 rag_ctx"


def t3_3():
    """T3.3 低置信不注入:0.2 → rag_ctx=None 但流程继续。"""
    retriever = MockRetriever(default=[CHUNK_GOOD, CHUNK_BAD])
    embedder = MockEmbedder(default=0.30,
                            pairs=[(TXT_TAX, CHUNK_GOOD, 0.20),
                                   (TXT_TAX, CHUNK_BAD, 0.10)])
    local = MockExecutor("local", "本地照样回答")
    rt = make_router(retrievers=[retriever], embedder=embedder,
                     executors=mk_execs(local, MockExecutor("cloud")))
    out = rt.route(TXT_TAX)
    assert out.rag.rag_ctx is None and out.rag.injected is False, \
        f"低置信不应注入,injected={out.rag.injected}"
    assert out.answer, "低置信不注入后流程仍应继续产出答案(防幻觉垃圾上下文)"
    assert out.rag.reason == "retrieval_miss", \
        f"reason={out.rag.reason!r},应为 retrieval_miss"


def t3_4():
    """T3.4 RAG≠生成绑定〔红线2〕:检索高置信 + 上下文 20k → 注入 且 CLOUD。"""
    retriever = MockRetriever(default=[CHUNK_AUDIT])
    embedder = MockEmbedder(default=0.30, pairs=[(TXT_AUDIT_50P, CHUNK_AUDIT, 0.90)])
    cloud = MockExecutor("cloud", "云端长文总结")
    rt = make_router(retrievers=[retriever], embedder=embedder,
                     executors=mk_execs(MockExecutor("local"), cloud))
    out = rt.route(TXT_AUDIT_50P, context_tokens=20000)
    assert out.rag.injected is True and out.rag.rag_ctx, "高置信应注入 rag_ctx"
    assert out.decision.target == RouteTarget.CLOUD, \
        f"target={out.decision.target},20k 上下文生成端应为 CLOUD(检索了不代表本地写答案)"
    assert cloud.call_count >= 1, "云端生成端未被调用"


def t3_5():
    """T3.5 口语问句改写后命中:原话 0 命中 → 改写重试命中,reason=rewrite_fixable。"""
    rewritten = "递延所得税 暂时性差异 计税基础 处理"
    retriever = MockRetriever(default=[CHUNK_GOOD], script={"咋整": []})
    embedder = MockEmbedder(default=0.30, b_rules=[("确认要点", 0.90)])
    rewriter = MockRewriter({TXT_TAX.replace("怎么处理", "咋整啊"): rewritten})
    rt = make_router(retrievers=[retriever], embedder=embedder,
                     rewriter=rewriter, executors=mk_execs())
    out = rt.route("递延所得税咋整啊")
    assert len(retriever.calls) >= 2, \
        f"原话 0 命中后应改写重试一次,检索仅 {len(retriever.calls)} 次"
    assert out.rag.reason == "rewrite_fixable", \
        f"reason={out.rag.reason!r},应为 rewrite_fixable"
    assert out.rag.injected is True and out.rag.rag_ctx, "改写后命中应注入"
    assert out.rag.rewritten_query != "递延所得税咋整啊" and \
        "暂时性差异" in out.rag.rewritten_query, \
        f"rewritten_query={out.rag.rewritten_query!r},应为术语化改写结果"


def t3_6():
    """T3.6 改写重试链:改写后仍 0 命中 → no_hit_after_rewrite + refuse_honesty。"""
    retriever = MockRetriever(default=[])          # 恒 0 命中
    rewriter = MockRewriter({})                     # 原样返回
    rt = make_router(retrievers=[retriever], embedder=MockEmbedder(),
                     rewriter=rewriter, executors=mk_execs())
    out = rt.route(TXT_TAX)
    assert len(retriever.calls) >= 2, \
        f"应先改写重试一次再落 no_hit_after_rewrite,检索仅 {len(retriever.calls)} 次"
    assert out.rag.reason == "no_hit_after_rewrite", \
        f"reason={out.rag.reason!r},应为 no_hit_after_rewrite"
    assert out.rag.rag_ctx is None and out.rag.injected is False, "0 命中不得注入"
    ok = ("没查到" in out.answer) or ("没找到" in out.answer) or \
         (out.guard is not None and out.guard.action == "refuse_no_material")
    assert ok, f"refuse_honesty 未生效,answer={out.answer!r}(不许自由发挥)"


def t3_7():
    """T3.7 margin 判据:top1−top2 < margin_min(0.05) → 不采信 top1,转低置信。"""
    retriever = MockRetriever(default=[CHUNK_C1, CHUNK_C2])
    embedder = MockEmbedder(default=0.30,
                            pairs=[(TXT_TAX, CHUNK_C1, 0.70),
                                   (TXT_TAX, CHUNK_C2, 0.68)])
    local = MockExecutor("local", "不采信 top1 后照样回答")
    rt = make_router(retrievers=[retriever], embedder=embedder,
                     executors=mk_execs(local, MockExecutor("cloud")))
    out = rt.route(TXT_TAX)
    assert abs(out.rag.confidence - 0.70) < 1e-6, \
        f"confidence={out.rag.confidence},应为复算 top1=0.70"
    assert out.rag.margin < 0.05, f"margin={out.rag.margin},应 <0.05 判不采信"
    assert out.rag.rag_ctx is None and out.rag.injected is False, \
        "margin 不足不得采信 top1 注入"
    assert out.answer, "转低置信处理后流程应继续产出答案"


def t3_8():
    """T3.8 后端顺序不可信:rank-1 不对口 → 本地复算重排后仍选到对口片段。"""
    retriever = MockRetriever(default=[CHUNK_WRONG, CHUNK_GOOD])  # rank-1 是错的那条
    embedder = MockEmbedder(default=0.30,
                            pairs=[(TXT_TAX, CHUNK_WRONG, 0.30),
                                   (TXT_TAX, CHUNK_GOOD, 0.90)])
    rt = make_router(retrievers=[retriever], embedder=embedder,
                     executors=mk_execs())
    out = rt.route(TXT_TAX)
    assert out.rag.injected is True, "对口片段 0.90 应被采信并注入"
    assert "确认要点" in (out.rag.rag_ctx or ""), \
        f"注入内容应是对口片段(本地复算重排),rag_ctx={out.rag.rag_ctx!r}"
    assert "泛泛而谈" not in (out.rag.rag_ctx or ""), \
        "后端 rank-1 的不对口片段不应被注入"


def t3_9():
    """T3.9 注入位置回归〔红线7〕:[知识库] 块紧挨提问之前,提问在最后。"""
    rt, retriever, embedder, local, cloud = high_conf_env()
    out = rt.route(TXT_TAX)
    cap_req, cap_ctx = local.calls[0]
    req_text = getattr(cap_req, "text", "") or ""
    ctx = out.rag.rag_ctx or ""
    if "[知识库]" in req_text:
        composed = req_text
        assert TXT_TAX in composed, "组合后的请求文本里找不到用户提问"
        assert composed.rstrip().endswith(TXT_TAX), \
            f"提问必须是消息最后内容(user_turn_before_query/溢出保尾),尾部={composed[-40:]!r}"
        assert composed.find("[知识库]") < composed.rfind(TXT_TAX), \
            "[知识库] 块必须位于提问之前"
    elif "[知识库]" in ctx:
        assert ctx.startswith("[知识库]"), "[知识库] 头未按注入格式组装"
        assert TXT_TAX in req_text and "[知识库]" not in req_text, \
            "提问应与 [知识库] 块分离并保持在消息尾部"
    else:
        raise AssertionError(
            "无法观测注入位置:executor 收到的 req.text 与 rag_ctx 均不含 [知识库] 块"
            "(红线7 要求注入块+提问按 user_turn_before_query 组装)")


# ══ 生成路由(独立决策) ════════════════════════════════════
def t4_1():
    """T4.1 长上下文升云:context=20k、隐私无、配额足 → CLOUD。"""
    cloud = MockExecutor("cloud", "云端回答")
    rt = make_router(classifier=MockClassifier("chat"),
                     executors=mk_execs(MockExecutor("local"), cloud))
    out = rt.route(TXT_WEEKLY, context_tokens=20000)
    assert out.decision.target == RouteTarget.CLOUD, \
        f"target={out.decision.target},20k 应升云"
    assert cloud.call_count >= 1 and out.answer == "云端回答", "云端生成端未被调用"


def t4_2():
    """T4.2 隐私锁本地:privacy_sensitive + context=20k → LOCAL(硬约束压过复杂度)。"""
    cloud = MockExecutor("cloud")
    local = MockExecutor("local", "本地回答")
    rt = make_router(classifier=MockClassifier("chat"),
                     executors=mk_execs(local, cloud))
    out = rt.route(TXT_PAYSLIP, context_tokens=20000)
    assert out.decision.target == RouteTarget.LOCAL, \
        f"target={out.decision.target},隐私敏感必须锁本地"
    assert cloud.call_count == 0, "隐私敏感不得调用云端生成端"


def t4_3():
    """T4.3 断网锁本地:offline_mode=true → LOCAL 且零云端调用。"""
    cloud = MockExecutor("cloud")
    local = MockExecutor("local", "离线回答")
    rt = make_router(classifier=MockClassifier("chat"),
                     executors=mk_execs(local, cloud))
    out = rt.route(TXT_WEEKLY, context_tokens=20000, offline_mode=True)
    assert out.decision.target == RouteTarget.LOCAL, \
        f"target={out.decision.target},offline 必须锁本地"
    assert cloud.call_count == 0, "offline_mode 下不得尝试任何云端调用"


def t4_4():
    """T4.4 配额耗尽降级:cloud_quota=0 + context=20k → LOCAL。"""
    cloud = MockExecutor("cloud")
    local = MockExecutor("local", "降级本地回答")
    rt = make_router(classifier=MockClassifier("chat"),
                     executors=mk_execs(local, cloud))
    out = rt.route(TXT_WEEKLY, context_tokens=20000, cloud_quota=0)
    assert out.decision.target == RouteTarget.LOCAL, \
        f"target={out.decision.target},配额耗尽应降级本地"
    assert cloud.call_count == 0, "配额为 0 仍调了云端生成端"


def t4_5():
    """T4.5 轻任务走本地:context=50 + chat → LOCAL。"""
    cloud = MockExecutor("cloud")
    local = MockExecutor("local", "本地轻任务回答")
    rt = make_router(classifier=MockClassifier("chat"),
                     executors=mk_execs(local, cloud))
    out = rt.route(TXT_WEEKLY, context_tokens=50)
    assert out.decision.target == RouteTarget.LOCAL, \
        f"target={out.decision.target},轻任务应走本地"
    assert local.call_count >= 1 and out.answer == "本地轻任务回答"


# ══ 兜底链 ═════════════════════════════════════════════════
def t5_1():
    """T5.1 云挂转本地:主目标 CLOUD 失败 → 重试 LOCAL 且携带同一 rag_ctx。"""
    cloud = MockExecutor("cloud", fail=True)
    local = MockExecutor("local", GUARD_CONSISTENT_REPLY)
    retriever = MockRetriever(default=[CHUNK_GOOD, CHUNK_BAD])
    embedder = MockEmbedder(default=0.30,
                            pairs=[(TXT_TAX, CHUNK_GOOD, 0.90),
                                   (TXT_TAX, CHUNK_BAD, 0.50)])
    rt = make_router(retrievers=[retriever], embedder=embedder,
                     executors=mk_execs(local, cloud))
    out = rt.route(TXT_TAX, context_tokens=20000)
    assert cloud.call_count == 1, f"云端应恰好尝试 1 次,实际 {cloud.call_count}"
    assert local.call_count == 1, "云端失败后应重试本地"
    assert out.answer == GUARD_CONSISTENT_REPLY, "最终答案应来自兜底本地"
    _, cloud_ctx = cloud.calls[0]
    _, local_ctx = local.calls[0]
    assert out.rag.rag_ctx and cloud_ctx == out.rag.rag_ctx and local_ctx == out.rag.rag_ctx, \
        f"兜底重试必须携带同一 rag_ctx: cloud={cloud_ctx!r} local={local_ctx!r} gate={out.rag.rag_ctx!r}"


def t5_2():
    """T5.2 本地挂转云:主目标 LOCAL 失败、配额足 → 重试 CLOUD。"""
    local = MockExecutor("local", fail=True)
    cloud = MockExecutor("cloud", "云端兜底回答")
    rt = make_router(classifier=MockClassifier("chat"),
                     executors=mk_execs(local, cloud))
    out = rt.route(TXT_WEEKLY, context_tokens=50)
    assert local.call_count == 1, f"本地应恰好尝试 1 次,实际 {local.call_count}"
    assert cloud.call_count == 1, "本地失败且配额足应转云端"
    assert out.answer == "云端兜底回答", "最终答案应来自兜底云端"


def t5_3():
    """T5.3 双挂显式报错:两端均不可用 → 明确报错,禁止静默吞/空串。"""
    rt = make_router(classifier=MockClassifier("chat"),
                     executors=mk_execs(MockExecutor("local", fail=True),
                                        MockExecutor("cloud", fail=True)))
    raised = None
    try:
        out = rt.route(TXT_WEEKLY, context_tokens=50)
        ans = (out.answer or "").strip()
        assert ans, "双挂后返回空串 = 静默吞,违反 fallback.both_fail: error"
        assert any(k in ans for k in ("错误", "失败", "不可用", "报错", "error", "unavailable")), \
            f"双挂应显式报错,实际 answer={ans[:60]!r} 不含任何报错语义"
    except AssertionError:
        raise
    except Exception as e:      # 抛异常也是"显式报错"的合法形式
        raised = e
    assert True if raised is not None else True, ""


# ══ 答案兜底校验(answer_guard v2,红线8) ══════════════════
def t5_4():
    """T5.4 编数字拦截:答案编 15%(原料只有 25%)且语义 0.848 → 仍拦截。"""
    retriever = MockRetriever(default=[CHUNK_GOOD, CHUNK_BAD])
    embedder = MockEmbedder(default=0.30,
                            pairs=[(TXT_TAX, CHUNK_GOOD, 0.90),
                                   (TXT_TAX, CHUNK_BAD, 0.50)],
                            a_rules=[("15%", 0.848)])   # 编数字答案语义高达 0.848
    local = MockExecutor("local", FAB_ANSWER)
    rt = make_router(retrievers=[retriever], embedder=embedder,
                     executors=mk_execs(local, MockExecutor("cloud")))
    out = rt.route(TXT_TAX)
    assert "15%" not in out.answer, \
        f"编数字答案未被拦截(数字一致判据未独立生效),answer={out.answer!r}"
    assert "25%" in out.answer, \
        f"应降级直出原文片段(cite_original),实际 answer={out.answer!r}"
    assert out.answer != FAB_ANSWER, "不得返回模型自编内容"


def t5_5():
    """T5.5 完全离题拦截:答案与原料字面重合≈0 → 丢弃(字面辅助判据)。"""
    retriever = MockRetriever(default=[CHUNK_INV])
    embedder = MockEmbedder(default=0.30,
                            pairs=[(TXT_INV, CHUNK_INV, 0.90)],
                            a_rules=[("先进先出", 0.05)])   # 语义也低
    local = MockExecutor("local", OFFTOPIC_ANSWER)
    rt = make_router(retrievers=[retriever], embedder=embedder,
                     executors=mk_execs(local, MockExecutor("cloud")))
    out = rt.route(TXT_INV)
    assert "先进先出" not in out.answer, \
        f"离题答案未被拦截,answer={out.answer!r}"
    assert "可变现净值" in out.answer, \
        f"应降级直出原文片段,实际 answer={out.answer!r}"


def t5_6():
    """T5.6 空答案拦截:生成结果为空串 → 丢弃,走降级链(cite_original)。"""
    retriever = MockRetriever(default=[CHUNK_GOOD, CHUNK_BAD])
    embedder = MockEmbedder(default=0.30,
                            pairs=[(TXT_TAX, CHUNK_GOOD, 0.90),
                                   (TXT_TAX, CHUNK_BAD, 0.50)])
    local = MockExecutor("local", "")
    rt = make_router(retrievers=[retriever], embedder=embedder,
                     executors=mk_execs(local, MockExecutor("cloud")))
    out = rt.route(TXT_TAX)
    assert out.answer and out.answer.strip(), \
        "空答案未被拦截(降级链未生效,返回了空串)"
    assert "确认要点" in out.answer or "25%" in out.answer, \
        f"应降级直出原文片段,实际 answer={out.answer!r}"


def t5_7():
    """T5.7 人设化回答放行(防误杀回归):语义 0.809 + 干净原料 → 通过 v0.2。"""
    retriever = MockRetriever(default=[CHUNK_INV])
    embedder = MockEmbedder(default=0.30,
                            pairs=[(TXT_INV, CHUNK_INV, 0.90)],
                            a_rules=[("本鲸", 0.809)])
    local = MockExecutor("local", PERSONA_OK_ANSWER)
    rt = make_router(retrievers=[retriever], embedder=embedder,
                     executors=mk_execs(local, MockExecutor("cloud")))
    out = rt.route(TXT_INV)
    assert out.answer == PERSONA_OK_ANSWER, \
        f"人设化正确回答被误杀(v0.1 字面判据回归?),answer={out.answer!r}"
    assert out.guard is None or out.guard.passed is True, \
        f"guard 判定应为通过,实际 {out.guard}"


def t5_8():
    """T5.8 降级顺序:先直出原文片段;原料为空则回"没查到";绝不返回自编内容。"""
    # 场景 A:guard 不合格 → cite_original(原文片段)
    retriever = MockRetriever(default=[CHUNK_GOOD, CHUNK_BAD])
    embedder = MockEmbedder(default=0.30,
                            pairs=[(TXT_TAX, CHUNK_GOOD, 0.90),
                                   (TXT_TAX, CHUNK_BAD, 0.50)],
                            a_rules=[("15%", 0.848)])
    local_a = MockExecutor("local", FAB_ANSWER)
    rt_a = make_router(retrievers=[retriever], embedder=embedder,
                       executors=mk_execs(local_a, MockExecutor("cloud")))
    out_a = rt_a.route(TXT_TAX)
    assert "25%" in out_a.answer and "15%" not in out_a.answer, \
        f"降级第一步应为 cite_original(直出原文),实际 {out_a.answer!r}"
    # 场景 B:原料为空(低置信不注入)+ 空生成 → refuse_no_material("没查到")
    retriever_b = MockRetriever(default=[CHUNK_GOOD, CHUNK_BAD])
    embedder_b = MockEmbedder(default=0.30,
                              pairs=[(TXT_TAX, CHUNK_GOOD, 0.20),
                                     (TXT_TAX, CHUNK_BAD, 0.10)])
    local_b = MockExecutor("local", "")
    rt_b = make_router(retrievers=[retriever_b], embedder=embedder_b,
                       executors=mk_execs(local_b, MockExecutor("cloud")))
    out_b = rt_b.route(TXT_TAX)
    ok = ("没查到" in out_b.answer) or ("没找到" in out_b.answer) or \
         (out_b.guard is not None and out_b.guard.action == "refuse_no_material")
    assert ok, f"原料为空应回'没查到'(refuse_no_material),实际 answer={out_b.answer!r}"
    assert FAB_ANSWER not in out_b.answer, "任何降级路径不得返回模型自编内容"


# ══ 工具执行安全 ═══════════════════════════════════════════
def t6_1():
    """T6.1 副作用确认:tool_call 且 confirm=true → 未确认不执行,标志位可见。"""
    local = MockExecutor("local")
    cloud = MockExecutor("cloud")
    rt = make_router(classifier=MockClassifier("chat"), executors=mk_execs(local, cloud))
    out = rt.route(TXT_LIGHT)
    assert out.decision.intent == "tool_call", f"intent={out.decision.intent!r}"
    assert out.decision.tool_needs_confirm is True, \
        "confirm=true 的工具规则必须暴露 decision.tool_needs_confirm=True"
    assert local.call_count == 0 and cloud.call_count == 0, \
        f"未确认就执行了!local={local.call_count} cloud={cloud.call_count}"


def t6_2():
    """T6.2 参数校验失败拒执行:args 缺 device → 拒绝并回询,不许带默认值硬跑。"""
    rt = make_router(classifier=MockClassifier("chat"),
                     executors=mk_execs(MockExecutor("local"), MockExecutor("cloud")))
    out = rt.route(TXT_LIGHT)
    assert out.decision.tool_needs_confirm is True, "前置:工具应处于待确认态"
    # router 需暴露某个带参确认/执行入口;逐个探测常见命名
    api = None
    for name in ("confirm_tool", "provide_confirmation", "resolve_confirmation",
                 "submit_confirmation", "tool_confirm", "execute_tool", "run_tool"):
        fn = getattr(rt, name, None)
        if callable(fn):
            api = (name, fn)
            break
    if api is None:
        raise AssertionError(
            "router 未暴露带参确认/执行入口(如 confirm_tool(args) / execute_tool(args)),"
            "无法注入'缺 device 的 args'完成本用例;需实现方补该接口")
    name, fn = api
    local = rt.executors["local"] if isinstance(getattr(rt, "executors", None), dict) \
        else None
    before = local.call_count if local is not None else None
    try:
        fn({"action": "close"})          # 故意缺 device
    except TypeError:
        raise AssertionError(
            f"rt.{name} 签名与断言不符(尝试传入 args dict 失败),T6.2 需按实际签名修订")
    after = local.call_count if local is not None else None
    assert before == after or (before is None), \
        f"args 缺 device 仍被执行了(调用数 {before}→{after}),不许带默认值硬跑"
    if before is None:
        raise AssertionError("无法读取 executor 调用计数(注入件未被 router 保留),"
                             "T6.2 的'拒绝执行'断言不可观测")


# ══ 人设防御 ═══════════════════════════════════════════════
def t7_1():
    """T7.1 命中即本地直出:零检索、零外发(云端 executor 调用次数=0)。"""
    retriever = MockRetriever(default=[CHUNK_GOOD])
    cloud = MockExecutor("cloud")
    local = MockExecutor("local", "本鲸才不是[LocalModelName]!")
    rt = make_router(retrievers=[retriever], classifier=MockClassifier("chat"),
                     executors=mk_execs(local, cloud))
    out = rt.route(TXT_PERSONA)
    assert out.decision.intent == "persona_defense", \
        f"intent={out.decision.intent!r},应为 persona_defense"
    assert out.decision.target == RouteTarget.LOCAL, "人设防御必须本地直出"
    assert retriever.call_count == 0, "人设防御不得触发检索"
    assert cloud.call_count == 0, \
        f"云端 executor 调用次数应为 0,实际 {cloud.call_count}(零外发被违反)"
    assert local.call_count >= 1, "人设防御应由本地直出"


# ══ 可观测性 ═══════════════════════════════════════════════
def t8_1():
    """T8.1 计数器递增:L1_hit / L2_hit / L3_hit 各 +1。"""
    cls = MockClassifier("chat")
    rt = make_router(classifier=cls, executors=mk_execs())
    rt.route(TXT_TAX)                # L1 单命中
    rt.route(TXT_BROKEN_PC)          # L2 采纳
    cls.reply = "大概是专业问题吧我觉得"
    rt.route(TXT_BROKEN_PC)          # 枚举外 → L3
    assert counter(rt, "L1_hit") == 1, f"L1_hit={counter(rt, 'L1_hit')},应为 1"
    assert counter(rt, "L2_hit") == 1, f"L2_hit={counter(rt, 'L2_hit')},应为 1"
    assert counter(rt, "L3_hit") == 1, f"L3_hit={counter(rt, 'L3_hit')},应为 1"


def t8_2():
    """T8.2 fail-open 计数:T2.2 场景 → L2_fail_open +1 且 L3_hit +1。"""
    rt = make_router(classifier=MockClassifier("大概是专业问题吧我觉得"),
                     executors=mk_execs())
    rt.route(TXT_BROKEN_PC)
    assert counter(rt, "L2_fail_open") == 1, \
        f"L2_fail_open={counter(rt, 'L2_fail_open')},应为 1"
    assert counter(rt, "L3_hit") == 1, f"L3_hit={counter(rt, 'L3_hit')},应为 1"


# ══ 隐私分流(纸⑤ D 混合) ═════════════════════════════════
def t9_1():
    """T9.1 词表命中强制本地:工资单 + 20k → LOCAL,云端调用=0。"""
    cloud = MockExecutor("cloud")
    local = MockExecutor("local", "本地私密回答")
    rt = make_router(classifier=MockClassifier("chat"),
                     executors=mk_execs(local, cloud))
    out = rt.route(TXT_PAYSLIP, context_tokens=20000)
    assert out.decision.target == RouteTarget.LOCAL, \
        f"target={out.decision.target},词表命中必须强制本地(压过 20k 复杂度)"
    assert cloud.call_count == 0, \
        f"云端 executor 调用次数应为 0,实际 {cloud.call_count}"
    assert counter(rt, "privacy_keyword_hit") >= 1, "privacy_keyword_hit 应 +1"


def t9_2():
    """T9.2 未标默认非敏感:无前缀无词表命中 → 正常复杂度路由。"""
    cloud = MockExecutor("cloud", "云端回答")
    rt = make_router(classifier=MockClassifier("chat"),
                     executors=mk_execs(MockExecutor("local"), cloud))
    out = rt.route(TXT_WEEKLY, context_tokens=20000)
    assert counter(rt, "privacy_keyword_hit") == 0, "无词表命中不应计 privacy_keyword_hit"
    assert counter(rt, "privacy_explicit_mark_hit") == 0, "无显式前缀不应计 explicit_mark"
    assert out.decision.target == RouteTarget.CLOUD, \
        f"target={out.decision.target},默认非敏感时应按复杂度正常升云"


def t9_3():
    """T9.3 词表误伤可删词(配置热加载):rt.reload_config 后同请求不再判敏感。"""
    cloud = MockExecutor("cloud", "云端回答")
    local = MockExecutor("local", "本地回答")
    rt = make_router(classifier=MockClassifier("chat"),
                     executors=mk_execs(local, cloud))
    out1 = rt.route(TXT_PAYSLIP, context_tokens=20000)
    assert out1.decision.target == RouteTarget.LOCAL, \
        "前置:删词前'工资单'应被词表拦下锁本地"
    kw_before = counter(rt, "privacy_keyword_hit")
    cfg = load_config(CONFIG_PATH)
    modified = copy.deepcopy(cfg)
    for rule in modified.get("hard_rules", []):
        if rule.get("id") == "privacy-keywords":
            rule["pattern"] = rule["pattern"].replace("工资单|", "", 1)
    assert modified["hard_rules"][0]["pattern"] != cfg["hard_rules"][0]["pattern"], \
        "测试前置失败:未能在配置副本中删去'工资单'"
    if not callable(getattr(rt, "reload_config", None)):
        raise AssertionError(
            "router 未暴露 rt.reload_config(config: dict)(契约补丁缺失),词表热加载不可测;"
            "备选路径 L1Layer.reload 存在,但 router 未接线")
    rt.reload_config(modified)
    out2 = rt.route(TXT_PAYSLIP, context_tokens=20000)
    assert out2.decision.target == RouteTarget.CLOUD, \
        f"删词热加载后同请求应不再判敏感(20k 正常升云),target={out2.decision.target}"
    assert counter(rt, "privacy_keyword_hit") == kw_before, \
        "删词后 privacy_keyword_hit 不应再递增"


def t9_4():
    """T9.4 显式标记兜底:#本地# → 敏感,且前缀在路由/检索前剥除。"""
    retriever = MockRetriever(default=[CHUNK_GOOD])
    cloud = MockExecutor("cloud")
    rt = make_router(retrievers=[retriever], classifier=MockClassifier("chat"),
                     embedder=MockEmbedder(default=0.30,
                                           pairs=[(TXT_TAX, CHUNK_GOOD, 0.90)]),
                     executors=mk_execs(MockExecutor("local", "本地回答"), cloud))
    out = rt.route(TXT_MARK_TAX, context_tokens=20000)
    assert counter(rt, "privacy_explicit_mark_hit") >= 1, "显式标记应计 explicit_mark_hit"
    assert out.decision.target == RouteTarget.LOCAL, \
        f"target={out.decision.target},显式标记必须锁本地"
    for q, _ in retriever.calls:
        assert "#本地#" not in q and "#private#" not in q and "#本地#" not in str(q), \
            f"retriever 收到的 query 未剥除显式前缀: {q!r}"


def t9_5():
    """T9.5 域内术语不误拦:'合同负债' 命中 rag-a3 而非 privacy(负向排除生效)。"""
    retriever = MockRetriever(default=[CHUNK_GOOD])
    cloud = MockExecutor("cloud")
    rt = make_router(retrievers=[retriever], classifier=MockClassifier("chat"),
                     embedder=MockEmbedder(default=0.30,
                                           pairs=[(TXT_CONTRACT_LIAB, CHUNK_GOOD, 0.90)]),
                     executors=mk_execs(MockExecutor("local"), cloud))
    out = rt.route(TXT_CONTRACT_LIAB)
    assert out.decision.intent == "knowledge_qa", \
        f"intent={out.decision.intent!r},'合同负债' 应命中 rag-a3(负向排除'合同(?!资产|负债)')"
    assert counter(rt, "privacy_keyword_hit") == 0, \
        "域内术语 '合同负债' 被 privacy 词表误拦(负向排除未生效)"
    assert retriever.call_count >= 1, "knowledge_qa 应正常走 RAG 门"


def t9_6():
    """T9.6 判定优先级:显式标记+词表同满足 → 按 explicit_mark 归因,不双计。"""
    cloud = MockExecutor("cloud")
    rt = make_router(classifier=MockClassifier("chat"),
                     executors=mk_execs(MockExecutor("local"), cloud))
    out = rt.route(TXT_MARK_BOTH, context_tokens=20000)
    assert out.decision.target == RouteTarget.LOCAL, "判敏感必须锁本地"
    assert counter(rt, "privacy_explicit_mark_hit") == 1, \
        f"privacy_explicit_mark_hit={counter(rt, 'privacy_explicit_mark_hit')},应恰 +1(verdict_priority 首位)"
    assert counter(rt, "privacy_keyword_hit") == 0, \
        f"privacy_keyword_hit={counter(rt, 'privacy_keyword_hit')},同请求不得双计(归因 explicit_mark)"


# ══ 运行入口 ═══════════════════════════════════════════════
ALL_TESTS: list[tuple[str, object]] = [
    ("T1.1", t1_1), ("T1.2", t1_2), ("T1.3", t1_3), ("T1.4", t1_4), ("T1.5", t1_5),
    ("T2.1", t2_1), ("T2.2", t2_2), ("T2.3", t2_3), ("T2.4", t2_4), ("T2.5", t2_5),
    ("T3.1", t3_1), ("T3.2", t3_2), ("T3.3", t3_3), ("T3.4", t3_4), ("T3.5", t3_5),
    ("T3.6", t3_6), ("T3.7", t3_7), ("T3.8", t3_8), ("T3.9", t3_9),
    ("T4.1", t4_1), ("T4.2", t4_2), ("T4.3", t4_3), ("T4.4", t4_4), ("T4.5", t4_5),
    ("T5.1", t5_1), ("T5.2", t5_2), ("T5.3", t5_3),
    ("T5.4", t5_4), ("T5.5", t5_5), ("T5.6", t5_6), ("T5.7", t5_7), ("T5.8", t5_8),
    ("T6.1", t6_1), ("T6.2", t6_2),
    ("T7.1", t7_1),
    ("T8.1", t8_1), ("T8.2", t8_2),
    ("T9.1", t9_1), ("T9.2", t9_2), ("T9.3", t9_3), ("T9.4", t9_4),
    ("T9.5", t9_5), ("T9.6", t9_6),
]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    passed = 0
    failed = 0
    for tid, fn in ALL_TESTS:
        try:
            fn()
            print(f"{tid} PASS")
            passed += 1
        except AssertionError as e:
            print(f"{tid} FAIL({e})")
            failed += 1
        except Exception as e:  # 用例外异常(构造失败/接口缺失等)也计 FAIL
            print(f"{tid} FAIL(运行期异常 {type(e).__name__}: {e})")
            failed += 1
    print(f"通过 {passed} / 未通过 {failed} / 共 {len(ALL_TESTS)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
