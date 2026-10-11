"""interfaces.py —— 全模块共享的数据类与协议(接口定死,各模块不得另行定义重复结构)

规约:
- 所有模块只从本文件 import 数据类/协议/枚举;禁止互相 import 对方内部类。
- 阈值/词表一律从 config(routing.yaml)读,代码零硬编码关键词(红线5)。
- 骨架阶段:外部 IO(ollama / MCP 检索 / bge-m3 / 云端 API)全部走 Protocol 注入,
  模块内不许直接发网络请求;真实实现以后换注入对象即可。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


# ── 枚举与字面量 ─────────────────────────────────────────────
class RouteTarget(Enum):
    LOCAL = "local"
    CLOUD = "cloud"


# L1/L2 产出的意图标签(L1 词表可产出;L2 枚举经 intent_map 映射到前四种)
INTENT_CHAT = "chat"
INTENT_KNOWLEDGE_QA = "knowledge_qa"
INTENT_TOOL_CALL = "tool_call"
INTENT_PERSONA_DEFENSE = "persona_defense"
INTENT_PRIVACY_SENSITIVE = "privacy_sensitive"
INTENT_EMOTION_CARE = "emotion_care"

# L2 分类枚举(routing.yaml intent_classifier.enum)
L2_ENUM = ("chat", "professional", "tool", "emotion")

# low_confidence_reason 归因枚举(routing.yaml rag_gate)
LOW_CONF_REASONS = ("no_hit", "no_hit_after_rewrite", "rewrite_fixable",
                    "retrieval_miss", "dirty_note")


# ── 请求画像(路由输入) ───────────────────────────────────────
@dataclass
class RequestProfile:
    text: str                          # 已剥除显式隐私前缀后的查询词
    raw_text: str = ""                 # 用户原话(未剥前缀)
    context_tokens: int = 0
    privacy_sensitive: bool = False
    privacy_source: str = ""           # explicit_mark / keyword_rules / default / ""
    offline_mode: bool = False
    cloud_quota_remaining: int = -1    # -1 表示不限
    domain: str = ""                   # L1 命中携带的 domain(如 finance_tax)


# ── 各层产出 ────────────────────────────────────────────────
@dataclass
class HardRuleHit:
    rule_id: str
    intent: str
    domain: str = ""
    confirm: bool = False
    args_schema: dict | None = None


@dataclass
class L1Result:
    hits: list[HardRuleHit] = field(default_factory=list)
    unique_intents: list[str] = field(default_factory=list)  # 去重后的 intent
    escalated: bool = False           # True = 0 命中或冲突 → 降 L2(红线3)


@dataclass
class L2Result:
    ok: bool = False                   # False = 枚举外/低置信/超时/异常 → fail-open(红线1)
    category: str = ""                 # ok 时为 L2_ENUM 之一
    mapped_intent: str = ""            # 经 intent_map 映射后的路由意图
    raw_output: str = ""
    elapsed_ms: int = 0
    confidence: float = 0.0            # ⚠️ 口径见 confidence_kind,不是校准概率
    confidence_kind: str = ""          # vote_ratio(kNN票数比) / cosine(原型余弦) / ""


@dataclass
class RetrievedChunk:
    text: str
    source: str = ""                   # 来源 retriever 名
    backend_rank: int = 0
    backend_score: float = 0.0


@dataclass
class RagGateResult:
    needed: bool = False               # intent 是否触发检索
    injected: bool = False
    rag_ctx: str | None = None         # 截断后的注入文本(None=不注入)
    confidence: float = 0.0            # 本地复算 top1 余弦
    margin: float = 0.0                # top1 - top2
    reason: str = ""                   # LOW_CONF_REASONS 之一;""
    rewritten_query: str = ""          # 术语化改写结果(未改写则=原话)
    dropped_chars: int = 0             # enforce_cap 截断丢弃字数(红线7)


@dataclass
class GuardResult:
    passed: bool = True
    failed_criterion: str = ""         # semantic_sim / number_consistency / length_ratio / literal_overlap / empty
    action: str = ""                   # cite_original / refuse_no_material / pass
    note: str = ""                     # 观察标签(#0187:如「舍入差0.01」——放行但进报表,不当 Fail)


@dataclass
class RoutingDecision:
    intent: str = ""
    target: RouteTarget = RouteTarget.LOCAL
    fallback: RouteTarget = RouteTarget.CLOUD
    use_rag: bool = False
    reasons: list[str] = field(default_factory=list)   # 决策依据,测试断言用
    tool_needs_confirm: bool = False
    tool_args_schema: dict | None = None


@dataclass
class RouteOutcome:
    decision: RoutingDecision
    rag: RagGateResult
    answer: str
    guard: GuardResult | None = None


# ── 外部组件协议(骨架只注入假件,真实实现后换对象) ─────────────
class Retriever(Protocol):
    """检索后端(如 quaesto_kb)。只负责给候选,顺序不可信(红线8)。"""
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]: ...


class Embedder(Protocol):
    """本地 bge-m3。所有置信度以它复算为准(红线8)。"""
    def similarity(self, a: str, b: str) -> float: ...


class SmallClassifier(Protocol):
    """本地小模型(ollama)。L2 用,temperature=0, ≤10 token, 超时由实现方遵守。"""
    def classify(self, prompt: str, temperature: float = 0.0,
                 max_tokens: int = 10, timeout_ms: int = 2000) -> str: ...


class Executor(Protocol):
    """生成端。LOCAL 与 CLOUD 各一份。"""
    def execute(self, req: RequestProfile, rag_ctx: str | None) -> str: ...
    def is_available(self) -> bool: ...


# ── 计数器名单(routing.yaml metrics.counters,模块不得私加) ───
METRIC_COUNTERS = (
    "L1_hit", "L2_hit", "L2_fail_open", "L3_hit", "rag_no_hit",
    "rag_no_hit_after_rewrite", "rag_rewrite_fixable", "rag_retrieval_miss",
    "tool_confirmed", "injected_dropped_chars", "answer_guard_rejected",
    "emotion_route_count", "emotion_followup_count",
    "privacy_keyword_hit", "privacy_explicit_mark_hit",
    "semantic_cache_hit",          # 2026-10-10 改进项①:语义缓存命中
    "input_guard_rejected",        # 2026-10-10 改进项②:输入守卫拒答
    "realtime_route_count",        # 2026-10-10 改进项②:实时类路由云端
    "l3_gate_retry",               # 2026-10-10 L3 收尾:质量门同层重试
    "l3_gate_rejected",            # 2026-10-10 L3 收尾:质量门拒绝降级
    "exit_gate_fail",              # 2026-10-11 出口硬兜(#0154):Fail 触发
    "exit_gate_retry",             # 2026-10-11 出口硬兜:重试一次
    "exit_gate_recovered",         # 2026-10-11 出口硬兜:重试后服从
    "exit_gate_contract_missing",  # 2026-10-11 契约头缺失(#0167 B,只记数不 Fail)
    "direction_polarity_hits",     # 2026-10-11 方向极性表 review 命中(#0167 C,只标不拒)
    "number_rounding_tolerated",   # 2026-10-11 数字口径(#0187):舍入差≤0.01 放行+标注计数
)
