"""eval_baseline_local_model —— 第四档对照:[LocalModelName]零样本(现基线)同口径评估

与腿A/腿B同 seed 同切分(复用 load_split);走 ollama format=json 分类(同 L2Classifier 语义)。
跑法:.venv\Scripts\python.exe scripts\eval_baseline_local_model
"""
from __future__ import annotations
# _hr_bootstrap_v1: 本脚本在 scripts\ 下，需要把仓库根加进 sys.path 才能 import 核心模块
import os as _hr_os, sys as _hr_sys
_hr_sys.path.insert(0, _hr_os.path.dirname(_hr_os.path.dirname(_hr_os.path.abspath(__file__))))

import json
import os
import time
from collections import Counter

import urllib.request

from eval_prototypes import LABELS, load_split

ENUM = ["chat", "professional", "emotion", "tool"]


def classify(text: str, timeout: float = 30.0) -> str:
    body = json.dumps({
        "model": "[LocalModelName]",
        "prompt": "分类这句话的意图,只输出以下标签之一:chat/professional/tool/emotion。句子:" + text,
        "stream": False,
        "format": {"type": "object",
                   "properties": {"category": {"type": "string", "enum": ENUM}},
                   "required": ["category"]},
        "options": {"temperature": 0, "num_predict": 10},
    }).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    import re
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = json.loads(resp.read().decode("utf-8")).get("response", "")
    try:
        return str(json.loads(raw)["category"])
    except (KeyError, ValueError):
        m = re.search(r'"category"\s*:\s*"(\w+)', raw)
        return m.group(1) if m else ""     # 枚举外=fail-open,计入错误


def main() -> None:
    train, test = load_split()
    print(f"[切分] 训练 {len(train)} / 留出 {len(test)}(与腿A/腿B同 seed 同切分)")
    classify("预热")                      # 冷启动一枪
    lat, rows = [], []
    correct = 0
    cm = {g: {p: 0 for p in LABELS} for g in LABELS}
    fo = 0
    for i, (sent, gold) in enumerate(test):
        t1 = time.time()
        pred = classify(sent)
        dt = (time.time() - t1) * 1000
        if i < 2:
            continue
        lat.append(dt)
        if pred not in LABELS:
            pred, fo = "chat", fo + 1     # fail-open 直升 L3,作为分类器算错
        cm[gold][pred] += 1
        correct += pred == gold
        rows.append((sent, gold, pred))
    n = len(test) - 2
    print(f"\n== [LocalModelName] 零样本(现基线) ==")
    print(f"  acc={correct/n:.3f}  fail-open {fo}/{n}")
    for g in LABELS:
        print(f"    {g:<13} → " + " ".join(f"{cm[g][p]:>3}" for p in LABELS))
    print(f"  延迟: mean={sum(lat)/len(lat):.0f}ms max={max(lat):.0f}ms")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "L2_CLS_EVAL_BASELINE.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": "[LocalModelName] zero-shot", "accuracy": correct / n,
                   "fail_open": fo, "confusion": cm,
                   "latency_ms": {"mean": sum(lat) / len(lat), "max": max(lat)},
                   "rows": rows}, f, ensure_ascii=False, indent=1)
    print(f"[落盘] {out}")


if __name__ == "__main__":
    main()
