"""eval_laya.py —— 腿B:Laya 决策引擎零样本 4 分类评估(与腿A同切分同口径)

复用 eval_prototypes.load_split(同 seed 同 TSV,留出集一致);choice 问题带中文
criteria(零样本靠它);报 准确率+混淆矩阵+置信度+本机单次延迟。
首次运行会经 hf-mirror 下载 multilingual 检查点(约 614MB)。
跑法:HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 .venv\Scripts\python.exe scripts\eval_laya.py
"""
from __future__ import annotations
# _hr_bootstrap_v1: 本脚本在 scripts\ 下，需要把仓库根加进 sys.path 才能 import 核心模块
import os as _hr_os, sys as _hr_sys
_hr_sys.path.insert(0, _hr_os.path.dirname(_hr_os.path.dirname(_hr_os.path.abspath(__file__))))

import json
import os
import time
from collections import Counter

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import laya
from eval_prototypes import LABELS, load_split

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "L2_CLS_EVAL.json")

CRITERIA = {
    "chat": "日常闲聊、寒暄、人设问答、生活常识闲聊、泛技术/架构闲谈;没有明确的财会专业点,也不需要执行外部动作",
    "professional": "财会/税法/审计/法规/会计准则相关的专业知识问答,含具体准则概念、科目处理、报表税务问题",
    "emotion": "情绪倾诉、抱怨、吐槽、求安慰、带情绪的调侃或攻击(如『我好累』『你好蠢』)",
    "tool": "需要执行外部动作或查询实时信息的指令(开关设备、音量、定时、查天气/时间/限行/提醒)",
}


def main() -> None:
    train, test = load_split()
    print(f"[切分] 训练 {len(train)} / 留出 {len(test)}(与腿A同 seed 同切分)")
    print(f"[留出分布] {dict(Counter(t for _, t in test))}")

    t0 = time.time()
    agent = laya.load("convaiinnovations/laya", subfolder="multilingual")
    print(f"[加载] multilingual 检查点 {time.time()-t0:.0f}s")

    question = {"category": {
        "type": "choice",
        "instructions": "判断这句中文用户输入的意图类别,只选一个最贴切的标签。",
        "criteria": CRITERIA,
    }}

    # 预热 2 条 + 全留出集判定
    lat = []
    rows = []
    correct = 0
    cm = {g: {p: 0 for p in LABELS} for g in LABELS}
    confs = []
    for i, (sent, gold) in enumerate(test):
        t1 = time.time()
        res = laya.decide(agent, sent, questions=question, return_details=True)
        dt = (time.time() - t1) * 1000
        pred = str(res.values.get("category", "")) if hasattr(res, "values") else str(res)
        conf = None
        if hasattr(res, "confidence"):
            c = res.confidence
            conf = float(c.get("category")) if isinstance(c, dict) and "category" in c else (float(c) if isinstance(c, (int, float)) else None)
        if i < 2:
            continue                     # 前两条算预热,不计延迟与指标
        lat.append(dt)
        if pred not in LABELS:
            pred = "chat"                # 枚举外兜底(与线上 fail-open 同向,计入错误)
        cm[gold][pred] += 1
        confs.append(conf if conf is not None else 0.0)
        correct += pred == gold
        rows.append((sent, gold, pred, conf))

    n = len(test) - 2
    print(f"\n== laya/convaiinnovations multilingual(零样本) ==")
    print(f"  acc={correct/n:.3f} 置信均值={sum(confs)/len(confs):.3f}")
    for g in LABELS:
        print(f"    {g:<13} → " + " ".join(f"{cm[g][p]:>3}" for p in LABELS))
    print(f"  延迟: mean={sum(lat)/len(lat):.0f}ms max={max(lat):.0f}ms(预算 2000ms;CPU 0.7–5s 风险实测)")
    for s, g, p, c in rows[:10]:
        flag = "✓" if g == p else "✗"
        print(f"  {flag} gold={g:<13} pred={p:<13} conf={c if c is not None else '-'} {s[:24]}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "L2_CLS_EVAL_LAYA.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": "convaiinnovations/laya multilingual (zero-shot)",
                   "accuracy": correct / n, "confusion": cm,
                   "latency_ms": {"mean": sum(lat) / len(lat), "max": max(lat)},
                   "rows": rows}, f, ensure_ascii=False, indent=1)
    print(f"[落盘] {out}")


if __name__ == "__main__":
    main()
