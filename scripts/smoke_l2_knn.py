"""smoke_l2_knn.py —— L2 向量匹配自测 + 留出集评测(2026-10-09,派活①交付件)

口径:
- 训练集 = TSV 的 8 成(seed=42 确定性切分);留出 2 成只在这里评测报指标,
  绝不回灌训练/调参。阈值读 routing.yaml,脚本不许私调(待用户正式标定)。
- 用法:  .venv/Scripts/python.exe scripts\smoke_l2_knn.py          # 全量:构建+留出集评测
          .venv/Scripts/python.exe scripts\smoke_l2_knn.py --demo   # 只演示三条样本分类
"""
from __future__ import annotations
# _hr_bootstrap_v1: 本脚本在 scripts\ 下，需要把仓库根加进 sys.path 才能 import 核心模块
import os as _hr_os, sys as _hr_sys
_hr_sys.path.insert(0, _hr_os.path.dirname(_hr_os.path.dirname(_hr_os.path.abspath(__file__))))

import os
import sys
import time

from config_loader import load_config
from ollama_runtime import OllamaEmbedder
from l2_vector import load_tsv, split_train_holdout, build_from_config

BASE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    config = load_config(os.path.join(BASE, "routing.yaml"))
    ic = config["intent_classifier"]
    enum = tuple(ic["enum"])
    knn_cfg = ic["knn"]

    tsv = knn_cfg["train_tsv"]
    if not os.path.isabs(tsv):
        tsv = os.path.normpath(os.path.join(BASE, tsv))
    rows = load_tsv(tsv)
    train, holdout = split_train_holdout(rows, float(knn_cfg["holdout_ratio"]),
                                         int(knn_cfg["seed"]))
    print(f"[split] 总 {len(rows)} 条 → 训练 {len(train)} / 留出 {len(holdout)} "
          f"(seed={knn_cfg['seed']},留出不回灌)")

    from embed_local import Embedder   # 统一嵌入入口(随 embedding.json 走,不再写死模型)
    embedder = Embedder()
    t0 = time.perf_counter()
    _ = embedder.embed("预热一枪,排除冷启动")      # bge-m3 冷启动别死在首测
    print(f"[warmup] embedder 首次嵌入 {time.perf_counter() - t0:.1f}s")

    cache = os.path.normpath(os.path.join(BASE, knn_cfg.get("vector_cache", ""))) \
        if knn_cfg.get("vector_cache") else None
    t0 = time.perf_counter()
    knn = build_from_config(config, embedder, BASE)
    print(f"[build] method={ic.get('method')} 训练向量就绪 "
          f"({time.perf_counter() - t0:.1f}s,含缓存命中)"
          + (f" cache={os.path.basename(cache)}" if cache else ""))

    if "--demo" in sys.argv:
        for text in ["递延所得税怎么处理", "帮我关一下客厅灯", "今天好累啊不想动了"]:
            t0 = time.perf_counter()
            label = knn.classify(text)
            dt = (time.perf_counter() - t0) * 1000
            print(f"[demo] {text!r} → {label or '(低置信→fail-open)'} "
                  f"{knn.confidence_kind}={knn.last_confidence:.2f} {dt:.0f}ms | {knn.last_detail}")
        return 0

    # ── 留出集评测(只报指标,不回灌) ──
    correct, per = 0, {}
    lat, low_conf = 0.0, 0
    for _, text, gold in holdout:
        t0 = time.perf_counter()
        pred = knn.classify(text)
        lat += (time.perf_counter() - t0) * 1000
        if not pred:
            low_conf += 1
        per.setdefault(gold, {"n": 0, "ok": 0})
        per[gold]["n"] += 1
        if pred == gold:
            correct += 1
            per[gold]["ok"] += 1
    n = len(holdout)
    print(f"[eval] 留出集 {n} 条:acc={correct}/{n}={correct / n:.1%} "
          f"低置信fail-open={low_conf} 平均耗时={lat / n:.0f}ms")
    for lab, s in sorted(per.items()):
        print(f"[eval]   {lab:<13} {s['ok']}/{s['n']}={s['ok'] / s['n']:.1%}")
    print(f"[conf] 口径={knn.confidence_kind}(⚠️ 票数比,不是概率) "
          f"阈值 min_vote_ratio={knn_cfg['min_vote_ratio']}(待标定)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
