"""smoke_l2_mapping.py —— L2 枚举映射真件冒烟

做两件事:
1) 真件([LocalModelName] 模型)对意图分类样本 TSV 抽样跑 L2 分类,对比人工标签,报一致率;
2) 验证 routing.yaml 的 intent_map 全覆盖四枚举、无缺键(枚举映射接线检查)。
跑法:python smoke_l2_mapping.py [--n 24]
"""
from __future__ import annotations
# ── 路径去标识（不再硬编码用户名；可用环境变量 HR_DATA_ROOT 覆盖数据根）
import os as _hr_os
_HR_HOME = _hr_os.path.expanduser("~")
_HR_DESKTOP = _hr_os.environ.get("HR_DATA_ROOT") or _hr_os.path.join(_HR_HOME, "Desktop")
_HR_DIR = _hr_os.path.dirname(_hr_os.path.abspath(__file__))

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_loader import load_config
from l2_classifier import L2Classifier
from metrics import Metrics
from ollama_runtime import OllamaClassifier

TSV = _hr_os.path.join(_HR_DESKTOP, r"send", r"意图分类样本_结果.tsv")


def main(n: int = 24) -> None:
    config = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "routing.yaml"))
    ic = config["intent_classifier"]

    # ② 枚举映射接线检查:enum 与 intent_map 键必须一一对应
    enum, imap = list(ic["enum"]), dict(ic["intent_map"])
    missing = [e for e in enum if e not in imap]
    print(f"[映射接线] enum={enum}")
    print(f"[映射接线] intent_map={imap} 缺键:{missing or '无'}")
    assert not missing, "intent_map 缺键,枚举映射不闭合"

    # ① 真件抽样对齐(TSV 人工标签为基准;分层取样保证小类命中)
    rows = [l.rstrip("\n").split("\t") for l in open(TSV, encoding="utf-8").read().splitlines()[1:] if l.strip()]
    by_label = {}
    for sid, sent, tag in rows:
        by_label.setdefault(tag, []).append(sent)
    rng = random.Random(20261008)
    sample = []
    for tag, sents in by_label.items():
        sample += [(s, tag) for s in rng.sample(sents, min(max(2, n // 8), len(sents)))]
    clf = L2Classifier(config, OllamaClassifier(), Metrics())
    # 预热:模型冷启动实测 36–60s,先用长超时跑一枪把模型拽进显存/内存,
    # 之后稳态(实测 ~27 t/s)才可能在 2000ms 配置超时内返回。
    OllamaClassifier().classify("预热:只输出 chat", temperature=0, max_tokens=10, timeout_ms=120000)
    ok = fail = 0
    for sent, gold in sample:
        r = clf.classify(sent)
        hit = r.ok and r.category == gold
        ok += hit
        fail += not hit
        print(f"  {'✓' if hit else '✗'} gold={gold:<13} got={r.category or 'fail-open':<13} {sent[:32]}")
    print(f"[真件冒烟] 一致 {ok} / 不一致 {fail} / 共 {len(sample)}(小样本冒烟口径,非正式标定)")


if __name__ == "__main__":
    main(int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 24)
