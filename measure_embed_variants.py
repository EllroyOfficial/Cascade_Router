# -*- coding: utf-8 -*-
"""measure_embed_variants.py —— 嵌入换模型三档 × 四组量测(2026-10-10,#0115 分工)

三档(纯环境变量驱动+importlib.reload,不改任何文件):
  V1 = bge-m3 / ollama(现状基线)  V2 = bge-small / ollama  V3 = bge-small / inproc
四组:
  ①L2 留出集(出手率/正确率 + vote_ratio P5/P50/P95,临时缓存不污染生产)
  ②RAG 门(域内/域外 top1-cos 分布,语料=增值税条例 54 卡)
  ③语义缓存(同义 vs 不同义 cos 分布)
  ④answer_guard —— 直接引用 #0115 已量的念稿组(同样本同指标),不重跑
纪律:只测量不改阈值;不调云端。
"""
from __future__ import annotations
# ── 路径去标识（不再硬编码用户名；可用环境变量 HR_DATA_ROOT 覆盖数据根）
import os as _hr_os
_HR_HOME = _hr_os.path.expanduser("~")
_HR_DESKTOP = _hr_os.environ.get("HR_DATA_ROOT") or _hr_os.path.join(_HR_HOME, "Desktop")
_HR_DIR = _hr_os.path.dirname(_hr_os.path.abspath(__file__))

import copy
import importlib
import json
import os
import sys
import tempfile
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

VARIANTS = [
    ("V1 m3/ollama",   {"EMBED_BACKEND": "ollama", "EMBED_OLLAMA_MODEL": "bge-m3:latest"}),
    ("V2 small/ollama", {"EMBED_BACKEND": "ollama", "EMBED_OLLAMA_MODEL": "qllama/bge-small-zh-v1.5:latest"}),
    ("V3 small/inproc", {"EMBED_BACKEND": "inproc", "EMBED_INPROC_MODEL": "BAAI/bge-small-zh-v1.5"}),
]

def pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, int(round(p / 100 * (len(sorted_vals) - 1)))))
    return sorted_vals[i]

# ── 样本 ─────────────────────────────────────────────────
SAME_PAIRS = [  # 语义缓存组:同义对
    ("递延所得税资产的确认前提，有几个条件来着？", "确认递延所得税资产要满足哪些前提条件？"),
    ("公允价值模式那个投资性房地产，是不是不计提减值来着？", "以公允价值计量的投资性房地产不用计提减值准备吧？"),
    ("把今天跑的那批问答记录存到 archive 文件夹里。", "把今天生成的问答记录归档到 archive 目录。"),
    ("我那个语义缓存阈值设成0.7，是不是太严了？", "语义缓存的相似度阈值调到0.7会不会太严格？"),
    ("帮我把这份报告导出成 PDF。", "把这份报告转成 PDF 导出来。"),
]
DIFF_PAIRS = [  # 不同义对:跨题
    (SAME_PAIRS[i][0], SAME_PAIRS[(i + 2) % 5][1]) for i in range(5)
] + [(SAME_PAIRS[i][1], SAME_PAIRS[(i + 3) % 5][0]) for i in range(3)]

def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

TESTSET = load_jsonl(os.path.join(_hr_os.path.join(_HR_DESKTOP, r"send"), "测试集_合并_150.jsonl"))
SBR_Q = [r["question"] for r in TESTSET if r["domain"] == "SBR准则"
         and any(k in r["question"] for k in ("增值税", "征收率", "加计扣除", "视同销售"))][:4]
GEN_Q = [r["question"] for r in TESTSET if r["domain"] == "通用"][:8]

# ── 各组计算(嵌入件注入) ─────────────────────────────────

def group_l2(embedder):
    from config_loader import load_config
    from l2_vector import load_tsv, split_train_holdout, build_from_config
    config = load_config(os.path.join(BASE, "routing.yaml"))
    knn_cfg = config["intent_classifier"]["knn"]
    tsv = knn_cfg["train_tsv"]
    if not os.path.isabs(tsv):
        tsv = os.path.normpath(os.path.join(BASE, tsv))
    rows = load_tsv(tsv)
    train, holdout = split_train_holdout(rows, float(knn_cfg["holdout_ratio"]), int(knn_cfg["seed"]))
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name  # 临时缓存,不碰生产
    cfg2 = copy.deepcopy(config)
    cfg2["intent_classifier"]["knn"]["vector_cache"] = os.path.relpath(tmp, BASE)
    knn = build_from_config(cfg2, embedder, BASE)
    out, low = [], 0
    for _, text, gold in holdout:
        pred = knn.classify(text)
        conf = knn.last_confidence
        if not pred:
            low += 1
            conf = 0.0
        out.append((pred == gold, conf))
    os.unlink(tmp)
    confs = sorted(c for _, c in out if c > 0)
    fired = [c for ok, c in out if ok and c > 0]
    res = {"n": len(out), "low_conf": low,
           "acc_fired": f"{len(fired)}/{len(confs)}" if confs else "-",
           "vote_p5": pct(confs, 5), "vote_p50": pct(confs, 50), "vote_p95": pct(confs, 95)}
    # 阈值扫描:0.67/0.80/1.0 下的出手率与出手正确率
    for th in (0.67, 0.80, 1.01):
        fired_n = sum(1 for _, c in out if c >= th)
        ok_n = sum(1 for ok, c in out if c >= th and ok)
        res[f"th{th}"] = f"出手{fired_n} 正确{ok_n}" + (f"={ok_n / fired_n:.0%}" if fired_n else "")
    return res

def group_rag(embedder):
    tiddlers = _hr_os.path.join(_HR_HOME, r"tiddlywiki", r"main", r"tiddlers")
    corpus = [open(os.path.join(tiddlers, fn), encoding="utf-8").read()
              for fn in os.listdir(tiddlers)
              if fn.startswith("《增值税法实施条例》") and fn.endswith(".md")]
    cv = embedder.embed_batch(corpus)
    import math
    def top1(qv):
        sims = []
        for v in cv:
            dot = sum(x * y for x, y in zip(qv, v))
            na = math.sqrt(sum(x * x for x in qv)) or 1.0
            nb = math.sqrt(sum(x * x for x in v)) or 1.0
            sims.append(dot / (na * nb))
        return max(sims)
    pos = [top1(embedder.embed(q)) for q in (SBR_Q + [
        "进项税额不得抵扣的情形有哪些", "小规模纳税人的标准怎么定",
        "纳税人兼营免税减税项目怎么核算", "增值税纳税义务发生时间怎么确认"])]
    neg = [top1(embedder.embed(q)) for q in (GEN_Q + [
        "今天中午吃什么好", "帮我写一首关于秋天的诗", "最近哪部电影好看", "股票明天会涨吗"])]
    return {"pos_min": min(pos), "pos_mean": sum(pos) / len(pos),
            "neg_max": max(neg), "neg_mean": sum(neg) / len(neg),
            "pos_all": [round(x, 3) for x in sorted(pos)], "neg_all": [round(x, 3) for x in sorted(neg)]}

def group_cache(embedder):
    same = []
    for a, b in SAME_PAIRS:
        va, vb = embedder.embed_batch([a, b])
        same.append(sum(x * y for x, y in zip(va, vb)))   # 归一化向量,点积即余弦
    diff = []
    for a, b in DIFF_PAIRS:
        va, vb = embedder.embed_batch([a, b])
        diff.append(sum(x * y for x, y in zip(va, vb)))
    return {"same_min": min(same), "same_mean": sum(same) / len(same),
            "diff_max": max(diff), "diff_mean": sum(diff) / len(diff),
            "same_all": [round(x, 3) for x in sorted(same)], "diff_all": [round(x, 3) for x in sorted(diff)]}

# ── 主流程:三档循环 ──────────────────────────────────────
results = {}
for label, env in VARIANTS:
    for k, v in env.items():
        os.environ[k] = v
    import embed_local as em
    em = importlib.reload(em)
    t0 = time.perf_counter()
    _ = em.embed_one("预热一枪")
    warm = time.perf_counter() - t0
    embedder = em.Embedder()
    print(f"\n===== {label} (dim={em.dim()}, warm {warm:.1f}s) =====", flush=True)
    t0 = time.perf_counter()
    results[label] = {
        "L2": group_l2(embedder),
        "RAG": group_rag(embedder),
        "CACHE": group_cache(embedder),
    }
    print(f"[耗时 {time.perf_counter() - t0:.0f}s]", flush=True)
    for g, d in results[label].items():
        print(f"[{g}] {json.dumps(d, ensure_ascii=False)}", flush=True)

print("\n===== 汇总 =====")
print(json.dumps(results, ensure_ascii=False, indent=1))
