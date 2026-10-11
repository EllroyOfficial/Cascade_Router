"""metrics.py —— 计数器注册表(线程安全,只做内存计数,零网络 IO)

计数器名单唯一来源:interfaces.METRIC_COUNTERS(routing.yaml metrics.counters),
模块不得私加计数器。
"""
from __future__ import annotations

import threading

from interfaces import METRIC_COUNTERS


class Metrics:
    """路由各环节计数器的注册表;线程安全(内部 threading.Lock)。"""

    def __init__(self) -> None:
        """按 METRIC_COUNTERS 初始化全部计数器为 0,并创建锁。"""
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {name: 0 for name in METRIC_COUNTERS}

    def inc(self, name: str, n: int = 1) -> None:
        """递增计数器;未知名字抛 ValueError(不许私加计数器)。"""
        with self._lock:
            if name not in self._counters:
                raise ValueError(f"unknown metric counter: {name!r}")
            self._counters[name] += n

    def snapshot(self) -> dict[str, int]:
        """返回全部计数器的副本。"""
        with self._lock:
            return dict(self._counters)

    def reset(self) -> None:
        """全部计数器归零。"""
        with self._lock:
            for name in self._counters:
                self._counters[name] = 0
