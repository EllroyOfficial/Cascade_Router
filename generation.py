"""generation.py —— 生成路由(独立于意图路由,红线2)+ 兜底链

依据 TESTS.md T4.1–T4.5 / T5.1–T5.3,routing.yaml generation_routing / fallback。
阈值全部从 config 读取,零硬编码(红线5)。
"""
from __future__ import annotations

from interfaces import Executor, RequestProfile, RouteTarget


class BothFailError(Exception):
    """兜底链双挂:显式报错,禁止静默吞或返回空串(T5.3)。"""


class GenerationRouter:
    def __init__(self, config, executors: dict, metrics):
        gen = config["generation_routing"]
        wall = gen["local_context_wall"]
        self._num_ctx = wall["num_ctx"]
        self._system_tokens = wall["system_tokens"]
        self._inject_tokens = wall["inject_tokens"]
        self._cloud_context_threshold = gen["cloud_context_threshold"]
        self._quota_policy = gen["quota_exhausted_policy"]
        fb = config["fallback"]
        self._fallback_map = {
            RouteTarget.LOCAL: RouteTarget[fb["local_fail"].upper()],
            RouteTarget.CLOUD: RouteTarget[fb["cloud_fail"].upper()],
        }
        self._both_fail_policy = fb["both_fail"]
        self._executors = executors
        self._metrics = metrics

    # ── 路由决策(T4.x) ─────────────────────────────────────
    def decide(self, req: RequestProfile) -> tuple[RouteTarget, RouteTarget, list[str]]:
        reasons: list[str] = []

        # 1) 硬约束:隐私敏感 / 断网 → 锁本地,压过一切(T4.2/T4.3)
        if req.privacy_sensitive:
            reasons.append("privacy_sensitive=true → force_local(硬约束压过复杂度,T4.2)")
            return RouteTarget.LOCAL, self._fallback_map[RouteTarget.LOCAL], reasons
        if req.offline_mode:
            reasons.append("offline_mode=true → force_local,不尝试任何云端调用(T4.3)")
            return RouteTarget.LOCAL, self._fallback_map[RouteTarget.LOCAL], reasons

        # 2) 本地硬墙:context_tokens > num_ctx − system_tokens − inject_tokens → 升云(T4.1)
        wall_limit = self._num_ctx - self._system_tokens - self._inject_tokens
        if req.context_tokens > wall_limit:
            target = RouteTarget.CLOUD
            reasons.append(
                f"context_tokens={req.context_tokens} > 硬墙 {wall_limit}"
                f"(num_ctx {self._num_ctx} - system {self._system_tokens}"
                f" - inject {self._inject_tokens}) → CLOUD(T4.1)"
            )
            # 3) 云端配额耗尽 → 按 policy 降本地(T4.4)
            if req.cloud_quota_remaining == 0:
                reasons.append(
                    f"cloud_quota_remaining=0 且 quota_exhausted_policy="
                    f"{self._quota_policy} → 降级 LOCAL(T4.4)"
                )
                target = RouteTarget.LOCAL
            return target, self._fallback_map[target], reasons

        # 4) 其余走本地(T4.5)
        reasons.append(
            f"context_tokens={req.context_tokens} ≤ 硬墙 {wall_limit} → LOCAL(T4.5)"
        )
        return RouteTarget.LOCAL, self._fallback_map[RouteTarget.LOCAL], reasons

    # ── 执行 + 兜底链(T5.1–T5.3) ───────────────────────────
    def dispatch(self, target, req, rag_ctx, max_hops: int = 2) -> str:
        if target is RouteTarget.CLOUD and req.privacy_sensitive:
            raise AssertionError(
                "防御性断言:privacy_sensitive 请求不得派发到 CLOUD"
            )
        current = target
        last_exc: Exception | None = None
        for _ in range(max_hops):
            executor = self._executors.get(current)
            if executor is not None and executor.is_available():
                try:
                    answer = executor.execute(req, rag_ctx)
                    if not isinstance(answer, str) or answer == "":
                        raise ValueError("executor 返回空结果(视同失败进入兜底)")
                    return answer
                except Exception as exc:  # 执行异常 → 转 fallback 重试
                    last_exc = exc
            else:
                last_exc = RuntimeError(f"executor[{current}] 不可用")
            current = self._fallback_map[current]  # 携带同一 req/rag_ctx 下一跳
        raise BothFailError(
            f"兜底链双挂(fallback.both_fail={self._both_fail_policy}):"
            f"主目标 {target.name} 与 fallback {self._fallback_map[target].name} "
            f"均执行失败;last_error={last_exc!r}"
        )
