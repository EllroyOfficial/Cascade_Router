"""路由测试用例 —— 与骨架同步落地

约定:骨架阶段所有用例 xfail(决策未实现)。
规则表每填一条,对应用例 xpass,即完成验收。
"""
# _hr_bootstrap_v1: 本脚本在 scripts\ 下，需要把仓库根加进 sys.path 才能 import 核心模块
import os as _hr_os, sys as _hr_sys
_hr_sys.path.insert(0, _hr_os.path.dirname(_hr_os.path.dirname(_hr_os.path.abspath(__file__))))

import pytest

from router import (
    HybridRouter,
    RagGateResult,
    RequestProfile,
    RouteTarget,
    Retriever,
    Executor,
)


# ── 测试替身(不是实现,只服务于测试) ──────────────────────
class FakeExecutor:
    def __init__(self, available: bool = True):
        self._available = available
        self.calls: list[str] = []

    def execute(self, prompt: str, rag_context: str | None) -> str:
        self.calls.append(prompt)
        return f"reply[{prompt}]"

    def is_available(self) -> bool:
        return self._available


class FakeRetriever(Retriever):
    def __init__(self, hit_confidence: float = 0.0):
        self.hit_confidence = hit_confidence

    def retrieve(self, query: str, top_k: int = 5) -> RagGateResult:
        return RagGateResult(needed=True, hit_confidence=self.hit_confidence)


def make_router(local_ok=True, cloud_ok=True, rag_hit=0.0) -> HybridRouter:
    return HybridRouter(
        local=FakeExecutor(local_ok),
        cloud=FakeExecutor(cloud_ok),
        retriever=FakeRetriever(rag_hit),
    )


# ── 1. 预检层:硬约束优先 ──────────────────────────────────
def test_offline_forces_local():
    """断网模式必须锁本地,即使任务再重、配额再多。"""
    r = make_router()
    d = r.decide(RequestProfile(
        prompt="总结这份 50 页文档", context_tokens=50_000, offline_mode=True))
    assert d.target is RouteTarget.LOCAL_OLLAMA


def test_privacy_forces_local():
    """隐私敏感请求锁本地,不许外发云端。"""
    r = make_router()
    d = r.decide(RequestProfile(
        prompt="处理我的工资单", privacy_sensitive=True))
    assert d.target is RouteTarget.LOCAL_OLLAMA
    assert not d.use_rag or d.fallback is RouteTarget.LOCAL_OLLAMA


# ── 2. RAG 检索门 ──────────────────────────────────────────
def test_knowledge_qa_triggers_rag():
    """知识问答类任务必须走检索门。"""
    r = make_router(rag_hit=0.9)
    d = r.decide(RequestProfile(prompt="XX 协议第几条说了什么", task_type="knowledge_qa"))
    assert d.use_rag is True


def test_chat_skips_rag():
    """闲聊/代码任务不触发检索,省一次向量库开销。"""
    r = make_router()
    for t in ("chat", "code"):
        d = r.decide(RequestProfile(prompt="hi", task_type=t))
        assert d.use_rag is False


def test_low_rag_confidence_no_injection():
    """检索命中置信度低于门槛 → 检索了也不注入,防止幻觉垃圾上下文。"""
    r = make_router(rag_hit=0.2)
    d = r.decide(RequestProfile(prompt="某个冷门问题", task_type="knowledge_qa"))
    assert d.use_rag is True          # 门过了,检索发生
    # TODO 决策层应带 injection=False 或等价语义,骨架阶段先断言不炸
    assert d.target in (RouteTarget.LOCAL_OLLAMA, RouteTarget.CLOUD_API)


# ── 3. 复杂度层 ────────────────────────────────────────────
def test_long_context_routes_cloud():
    """上下文超阈值 → 云端。"""
    r = make_router()
    d = r.decide(RequestProfile(prompt="长文分析", context_tokens=20_000))
    assert d.target is RouteTarget.CLOUD_API


def test_light_task_routes_local():
    """短上下文轻任务 → 本地,省钱省延迟。"""
    r = make_router()
    d = r.decide(RequestProfile(prompt="帮我把这句改通顺", context_tokens=50))
    assert d.target is RouteTarget.LOCAL_OLLAMA


# ── 4. 配额层 ──────────────────────────────────────────────
def test_quota_exhausted_degrades_local():
    """云端配额耗尽 → 强制降级本地。"""
    r = make_router()
    d = r.decide(RequestProfile(
        prompt="长文分析", context_tokens=20_000, cloud_quota_remaining=0))
    assert d.target is RouteTarget.LOCAL_OLLAMA


# ── 5. 兜底链 ──────────────────────────────────────────────
def test_dispatch_falls_back_on_failure():
    """主目标执行失败 → 自动切 fallback,并携带 RAG 上下文(若注入过)。"""
    r = make_router(cloud_ok=False)
    profile = RequestProfile(prompt="重推理任务", context_tokens=20_000)
    assert r.decide(profile).fallback is RouteTarget.LOCAL_OLLAMA
    # dispatch 的完整重试链路待实现后补断言


# ── 6. 决策可解释性 ────────────────────────────────────────
def test_decision_carries_reasons():
    """每次决策必须带 reasons,路由出问题要能审计。"""
    r = make_router(rag_hit=0.9)
    d = r.decide(RequestProfile(prompt="知识问题", task_type="knowledge_qa"))
    assert len(d.reasons) >= 1
