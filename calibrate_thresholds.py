"""calibrate_thresholds.py —— rag_gate 阈值标定骨架(分布切谷法,OPTIONS.md 纸⑥)

用途:用户题库到位后一键复跑——对正例(应命中)与负例(不应命中)各算 top1 余弦分布,
取两类分布的谷底作 in_domain 阈值;top1−top2 的分离点作 margin_min;域外阈值另行按
域外查询分布给建议。当前为骨架:题库未到位时用 --selftest 模式拿本地卡片小语料
自造正负例跑通流程,结果只作管线验证、不作标定结论(#0012 教训:别拿手造负例定阈值)。

跑法:
    python calibrate_thresholds.py --selftest          # 管线自检(手造负例,结果仅参考)
    python calibrate_thresholds.py --queries 题库.tsv  # 正式:题库格式=查询<TAB>正/负<TAB>域内/域外
输出:两类分数分布直方图 + 谷底阈值建议 + margin 分离点建议(打印,不写回 routing.yaml)。
"""
from __future__ import annotations
# ── 路径去标识（不再硬编码用户名；可用环境变量 HR_DATA_ROOT 覆盖数据根）
import os as _hr_os
_HR_HOME = _hr_os.path.expanduser("~")
_HR_DESKTOP = _hr_os.environ.get("HR_DATA_ROOT") or _hr_os.path.join(_HR_HOME, "Desktop")
_HR_DIR = _hr_os.path.dirname(_hr_os.path.abspath(__file__))

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_loader import load_config
from ollama_runtime import OllamaEmbedder

TIDDLERS = _hr_os.path.join(_HR_HOME, r"tiddlywiki", r"main", r"tiddlers")


def load_selftest_queries() -> list[tuple[str, str, str]]:
    """自检模式:正例=卡片标题改写的域内问句;负例=域外闲聊(手造,仅验管线!)。"""
    pos = ["进项税额不得抵扣的情形有哪些", "小规模纳税人的标准怎么定",
           "纳税人兼营免税减税项目怎么核算", "增值税纳税义务发生时间怎么确认"]
    neg = ["今天中午吃什么好", "帮我写一首关于秋天的诗", "最近哪部电影好看",
           "股票明天会涨吗"]
    return [(q, "pos", "finance_tax") for q in pos] + [(q, "neg", "") for q in neg]


def score_queries(queries: list[tuple[str, str, str]], embedder, corpus: list[str]) -> list[tuple[str, str, float, float]]:
    """对每条查询算 top1 余弦与 top1−top2 margin(对全语料,不走向量索引)。"""
    out = []
    vecs = [(c, embedder._embed(c)) for c in corpus]
    import math
    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (na * nb)
    for q, label, domain in queries:
        qv = embedder._embed(q)
        sims = sorted((cos(qv, v) for _, v in vecs), reverse=True)
        top1 = sims[0]
        margin = top1 - (sims[1] if len(sims) > 1 else 0.0)
        out.append((q, label, top1, margin))
    return out


def find_valley(pos_scores: list[float], neg_scores: list[float]) -> float | None:
    """分布切谷:把 [min, max] 分 20 桶,找同时落在两类直方图低洼处的桶边界。"""
    lo = min(pos_scores + neg_scores)
    hi = max(pos_scores + neg_scores)
    if hi <= lo:
        return None
    buckets = 20
    width = (hi - lo) / buckets
    hp = Counter(min(int((s - lo) / width), buckets - 1) for s in pos_scores)
    hn = Counter(min(int((s - lo) / width), buckets - 1) for s in neg_scores)
    valleys = [lo + (b + 1) * width for b in range(buckets - 1)
               if hp[b] and hn[b + 1] and hp[b + 1] == 0 and hn[b] == 0]
    return valleys[0] if valleys else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="手造正负例自检(不作标定结论)")
    ap.add_argument("--queries", help="正式题库 tsv:查询<TAB>正/负<TAB>域内/域外")
    args = ap.parse_args()

    if args.selftest:
        queries = load_selftest_queries()
    elif args.queries:
        queries = []
        for line in open(args.queries, encoding="utf-8").read().splitlines():
            if line.strip():
                q, label, domain = line.split("\t")
                queries.append((q, label, domain))
    else:
        ap.error("需要 --selftest 或 --queries")

    print("TODO(等用户题库):①正负例各扩到 ≥10/≥5 条真实样本(#0012:手造负例会得假分离);"
          "②域外查询单独算 out_of_domain 建议;③margin 取两类 margin 分布的分离点。")
    from embed_local import Embedder   # 统一嵌入入口(与 router 同源,换模型只改 embedding.json)
    embedder = Embedder()
    corpus = []
    for fn in os.listdir(TIDDLERS):
        if fn.startswith("《增值税法实施条例》") and fn.endswith(".md"):
            text = open(os.path.join(TIDDLERS, fn), encoding="utf-8").read()
            corpus.append(text.split("## 原文（逐字）", 1)[1].split("## 锚点", 1)[0].strip())
    scored = score_queries(queries, embedder, corpus)
    pos = [s for _, lb, s, _ in scored if lb == "pos"]
    neg = [s for _, lb, s, _ in scored if lb == "neg"]
    print(f"[分布] 正例 top1: {sorted(round(s, 3) for s in pos)}")
    print(f"[分布] 负例 top1: {sorted(round(s, 3) for s in neg)}")
    valley = find_valley(pos, neg)
    print(f"[切谷建议] in_domain 阈值 ≈ {valley and round(valley, 3)}(selftest 口径仅验管线,勿写入配置)")
    print("[明细]")
    for q, lb, s, m in scored:
        print(f"  {lb:<3} top1={s:.3f} margin={m:.3f} {q[:28]}")


if __name__ == "__main__":
    main()
