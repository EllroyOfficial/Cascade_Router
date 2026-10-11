"""eval_prototypes.py —— 腿A:向量最近原型/kNN 分类评估(bge-small-zh vs bge-m3 对照)

口径(照 #0082):①分层 8:2 切分,留出集不参与原型构建与调参(种子固定可复跑);
②报 准确率+四类混淆矩阵+置信度分布+本机单次延迟;③同题同料同脚本,只换嵌入模型;
④负例全部来自真实语料,零手造(#0012)。
跑法:.venv\Scripts\python.exe scripts\eval_prototypes.py [--knn 3] [--sample-cap 0]
"""
from __future__ import annotations
# _hr_bootstrap_v1: 本脚本在 scripts\ 下，需要把仓库根加进 sys.path 才能 import 核心模块
import os as _hr_os, sys as _hr_sys
_hr_sys.path.insert(0, _hr_os.path.dirname(_hr_os.path.dirname(_hr_os.path.abspath(__file__))))
# ── 路径去标识（不再硬编码用户名；可用环境变量 HR_DATA_ROOT 覆盖数据根）
import os as _hr_os
_HR_HOME = _hr_os.path.expanduser("~")
_HR_DESKTOP = _hr_os.environ.get("HR_DATA_ROOT") or _hr_os.path.join(_HR_HOME, "Desktop")
_HR_DIR = _hr_os.path.dirname(_hr_os.path.abspath(__file__))

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict

import urllib.request

import numpy as np

TSV = _hr_os.path.join(_HR_DESKTOP, r"send", r"意图分类样本_结果.tsv")
SEED = 20261008
LABELS = ["chat", "professional", "emotion", "tool"]


def ollama_embed(model: str, text: str, timeout: float = 30.0) -> list[float]:
    body = json.dumps({"model": model, "prompt": text}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:11434/api/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))["embedding"]


def load_split(test_ratio: float = 0.2):
    rows = [l.rstrip("\n").split("\t") for l in open(TSV, encoding="utf-8").read().splitlines()[1:] if l.strip()]
    by_label = defaultdict(list)
    for sid, sent, tag in rows:
        by_label[tag].append(sent)
    rng = random.Random(SEED)
    train, test = [], []
    for tag in sorted(by_label):
        sents = sorted(by_label[tag])
        rng.shuffle(sents)
        k = max(1, round(len(sents) * test_ratio))
        test += [(s, tag) for s in sents[:k]]
        train += [(s, tag) for s in sents[k:]]
    return train, test


def cos(a, b) -> float:
    na, nb = np.linalg.norm(a) or 1.0, np.linalg.norm(b) or 1.0
    return float(np.dot(a, b) / (na * nb))


def evaluate(model: str, train, test, knn: int):
    # 嵌入(带进度)
    t0 = time.time()
    tr_vec = [(np.array(ollama_embed(model, s), dtype=np.float32), tag) for s, tag in train]
    te_vec = [(np.array(ollama_embed(model, s), dtype=np.float32), tag) for s, tag in test]
    embed_secs = time.time() - t0

    # 类别原型(仅训练集)
    protos = {}
    for tag in LABELS:
        vecs = [v for v, t in tr_vec if t == tag]
        m = np.mean(vecs, axis=0)
        protos[tag] = m / (np.linalg.norm(m) or 1.0)

    # 单次延迟:预发热身后测 12 次
    for s, _ in test[:3]:
        ollama_embed(model, s)
    lat = []
    for s, _ in test[3:15]:
        t1 = time.time()
        ollama_embed(model, s)
        lat.append((time.time() - t1) * 1000)

    results = {}
    for mode in ("proto", f"knn{knn}"):
        cm = {g: {p: 0 for p in LABELS} for g in LABELS}
        confs = []
        correct = 0
        for v, gold in te_vec:
            if mode == "proto":
                sims = {t: cos(v, p) for t, p in protos.items()}
                pred = max(sims, key=sims.get)
                conf = sims[pred]
            else:
                sims = sorted(((cos(v, tv), t) for tv, t in tr_vec), reverse=True)[:knn]
                votes = Counter(t for _, t in sims)
                pred = votes.most_common(1)[0][0]
                conf = votes.most_common(1)[0][1] / knn
            cm[gold][pred] += 1
            confs.append(conf)
            correct += pred == gold
        results[mode] = {
            "accuracy": correct / len(te_vec),
            "confusion": cm,
            "conf_mean": float(np.mean(confs)),
            "conf_min": float(np.min(confs)),
        }
    per_class = {t: sum(1 for _, g in te_vec if g == t) for t in LABELS}
    return {"model": model, "n_train": len(train), "n_test": len(test),
            "per_class_test": per_class, "embed_total_s": round(embed_secs, 1),
            "latency_ms": {"mean": round(float(np.mean(lat)), 1),
                           "p95": round(float(np.percentile(lat, 95)), 1),
                           "max": round(float(np.max(lat)), 1)},
            "results": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--knn", type=int, default=3)
    ap.add_argument("--models", default="qllama/bge-small-zh-v1.5,bge-m3")
    args = ap.parse_args()

    train, test = load_split()
    print(f"[切分] 训练 {len(train)} / 留出 {len(test)}(分层 8:2,seed={SEED})")
    print(f"[留出分布] {dict(Counter(t for _, t in test))}")
    report = []
    for model in args.models.split(","):
        try:
            r = evaluate(model, train, test, args.knn)
        except Exception as e:
            print(f"[跳过] {model}: {e}")
            continue
        report.append(r)
        print(f"\n== {model} ==")
        for mode, res in r["results"].items():
            print(f"  [{mode}] acc={res['accuracy']:.3f} 置信均值={res['conf_mean']:.3f} "
                  f"最低={res['conf_min']:.3f}")
            for g in LABELS:
                row = " ".join(f"{res['confusion'][g][p]:>3}" for p in LABELS)
                print(f"    {g:<13} → {row}")
        print(f"  延迟: mean={r['latency_ms']['mean']}ms p95={r['latency_ms']['p95']}ms "
              f"max={r['latency_ms']['max']}ms(预算 2000ms)")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "L2_CLS_EVAL.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "split": "8:2 stratified holdout", "arms": report},
                  f, ensure_ascii=False, indent=1)
    print(f"\n[落盘] {out}")


if __name__ == "__main__":
    main()
