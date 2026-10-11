"""config_loader.py —— routing.yaml 加载与热加载(词表热加载落地,红线5)

规约:
- 只 import 标准库 + yaml + interfaces;零硬编码关键词/阈值,全部从文件读。
- yaml 中 args_schema 的值(如 "enum[open, close]")是字符串,原样保留不解析。
"""
from __future__ import annotations

import os

try:
    import yaml
except ImportError as e:  # 不自动安装,只提示
    raise ImportError(
        "缺少依赖 PyYAML,请执行: pip install pyyaml"
    ) from e

# 必含节(缺任一抛 ConfigError)
_REQUIRED_SECTIONS = (
    "hard_rules",
    "privacy_detection",
    "intent_classifier",
    "rag_gate",
    "answer_guard",
    "generation_routing",
    "fallback",
    "metrics",
)


class ConfigError(Exception):
    """配置文件缺失/非法/缺节时抛出。"""


def load_config(path: str = "routing.yaml") -> dict:
    """读取并校验 routing.yaml,返回整份 dict。

    - yaml.safe_load 全文读取;文件不存在或非映射抛 ConfigError。
    - 缺必含节抛 ConfigError(节名见 _REQUIRED_SECTIONS)。
    - args_schema 中的 "enum[...]" 等值为字符串,原样保留,不做任何解析。
    """
    if not os.path.isfile(path):
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML 解析失败: {path}: {e}") from e

    if not isinstance(data, dict):
        raise ConfigError(f"配置根必须是映射(dict),实际: {type(data).__name__}")

    missing = [s for s in _REQUIRED_SECTIONS if s not in data]
    if missing:
        raise ConfigError(f"配置缺必含节: {', '.join(missing)}")
    return data


class ConfigWatcher:
    """基于 mtime 的热加载:词表增删只改 routing.yaml,不重启(红线5)。"""

    def __init__(self, path: str):
        self.path = path
        self._cache: dict | None = None
        self._mtime: float | None = self._current_mtime()

    def _current_mtime(self) -> float | None:
        try:
            return os.path.getmtime(self.path)
        except OSError:
            return None

    def is_stale(self) -> bool:
        """文件 mtime 相对上次加载已变化(或文件曾消失/新出现)返回 True。"""
        return self._current_mtime() != self._mtime

    def reload_if_stale(self) -> dict | None:
        """变了就重读并返回新 dict;没变返回 None。"""
        if not self.is_stale():
            return None
        self._cache = load_config(self.path)
        self._mtime = self._current_mtime()
        return self._cache


if __name__ == "__main__":
    cfg = load_config("routing.yaml")
    print("routing.yaml 加载成功,顶层节:")
    for k in cfg:
        print(" -", k)
