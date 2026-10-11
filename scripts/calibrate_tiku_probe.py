"""calibrate_tiku_probe.py —— 题库最小试标定(#0139 用户拍:先 30 条级试三处)

三处:①RAG 门 in_domain(域内=题干问句 vs 域外=测试集闲聊,对本地 wiki 语料算 top1 余弦)
     ②念稿 cos_pass 0.85 / cos_floor 0.70(代理口径:材料=题干+全部选项;正例=正确选项原文
       (忠读);反例=错误选项原文(说反/编数字)——代理是上界,真念稿生成会偏低,只作方向参考)
     ③兜底 answer_guard.min_sim 0.65(正答=正确选项 vs 幻觉答=错误选项,材料同上)
配对:题号 N ↔ "N.【答案】key"+【解析】(参考答案节)。样本 30 条级,分层抽 6 份。
只读原素材;产出落 send\题库_probe\;结论先不落 yaml,等用户拍。
跑法:.venv\\Scripts\\python.exe scripts\calibrate_tiku_probe.py
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

import glob
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ollama_runtime import OllamaEmbedder

PROBE = _hr_os.path.join(_HR_DESKTOP, r"send", r"题库_probe")
NSEED = 30
TIDDLERS = _hr_os.path.join(_HR_HOME, r"tiddlywiki", r"main", r"tiddlers")


def parse_file(path: str) -> list[dict]:
    text = open(path, encoding="utf-8").read()
    body, _, answers = text.partition("参考答案")
    qs = {}
    for m in re.finditer(r"(\d+)\.【(单选题|多选题|判断题)】", body):
        qs[int(m.group(1))] = {"no": int(m.group(1)), "type": m.group(2), "start": m.start()}
    ends = sorted(qs.values(), key=lambda q: q["start"])
    for i, q in enumerate(ends):
        q_end = ends[i + 1]["start"] if i + 1 < len(ends) else len(body)
        q["stem"] = body[q["start"]:q_end]
    out = []
    for q in ends:
        am = re.search(rf"\n\s*{q['no']}\.【答案】\s*([A-Da-d]+|对|错|√|×)", answers)
        pm = re.search(rf"\n\s*{q['no']}\.【解析】(.+?)(?=\n\s*\d+\.【答案】|\Z)", answers, re.S)
        if not am or q["type"] == "判断题":
            continue
        key = am.group(1).upper()
        stem = re.sub(r"\s+", "", q["stem"])
        options = dict(re.findall(r"([A-D])\s*\.\s*([^\nA-D]{6,})", q["stem"]))
        if len(options) < 3 or key not in "ABCD" or len(key) != 1:   # 单选起步,配对最干净
            continue
        out.append({"no": q["no"], "type": q["type"], "stem": stem,
                    "options": options, "key": key,
                    "解析": re.sub(r"\s+", "", pm.group(1))[:200] if pm else ""})
    return out


def cos(a, b) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def main() -> None:
    # 1) 配对题池
    pool = []
    for p in sorted(glob.glob(os.path.join(PROBE, "中级会计实务*", "押题班-客观题（?）.pdf"))):
        fname = os.path.basename(p)
        for q in parse_file(os.path.join(PROBE, fname.replace(".pdf", ".full.txt"))):
            q["src"] = fname
            pool.append(q)
    print(f"[题池] 可配对单选 {len(pool)} 题(6 份)")
    rng = random.Random(20261011)
    sample = rng.sample(pool, min(NSEED, len(pool)))

    emb = OllamaEmbedder("bge-m3")
    t0 = time.time()

    # 2) 三处探针
    # ②/③ 念稿与兜底:材料=题干+全部选项;正答=正确选项;幻觉答=错误选项
    pos, neg = [], []
    for q in sample:
        material = q["stem"] + "".join(q["options"][k] for k in sorted(q["options"]))
        mv = emb._embed(material)
        pos.append(cos(mv, emb._embed(q["options"][q["key"]])))
        wrong = [v for k, v in q["options"].items() if k != q["key"]]
        neg.append(max(cos(mv, emb._embed(w)) for w in wrong))   # 最像的错误选项=最强幻觉
    def stats(xs):
        xs = sorted(xs)
        return {"n": len(xs), "min": round(xs[0], 3), "p25": round(xs[len(xs)//4], 3),
                "中位": round(xs[len(xs)//2], 3), "p75": round(xs[3*len(xs)//4], 3), "max": round(xs[-1], 3)}
    print(f"\n[②念稿/③兜底 共用配对] n={len(sample)} 耗时 {time.time()-t0:.0f}s")
    print(f"  正例 cos(材料,正确选项): {stats(pos)}")
    print(f"  反例 cos(材料,最强错误选项): {stats(neg)}")
    both = sorted(pos + neg)
    gap = [x for x in both if x > max(min(pos), min(neg)) and x < min(max(pos), max(neg))]
    print(f"  阈值线: 0.85(念稿pass)/0.70(念稿floor)/0.65(兜底) — 正例低于 0.65 的 {sum(1 for x in pos if x < 0.65)} 条,"
          f" 反例高于 0.70 的 {sum(1 for x in neg if x > 0.70)} 条、高于 0.85 的 {sum(1 for x in neg if x > 0.85)} 条")

    # ① RAG 门 in_domain:域内=题干 vs 域外=测试集闲聊;语料=本地 wiki 增值税条例
    corpus = []
    for fn in os.listdir(TIDDLERS):
        if fn.startswith("《增值税法实施条例》") and fn.endswith(".md"):
            t = open(os.path.join(TIDDLERS, fn), encoding="utf-8").read()
            corpus.append(t.split("## 原文（逐字）", 1)[1].split("## 锚点", 1)[0].strip())
    cvecs = [(c, emb._embed(c)) for c in corpus]
    import math
    def top1(q):
        qv = emb._embed(q)
        sims = sorted((cos(qv, v) for _, v in cvecs), reverse=True)
        return sims[0]
    in_dom = [top1(q["stem"]) for q in sample]
    # 域外负例:测试集里期望层含 L3 且 fail-open 的 chat/情绪题(通用/情绪闲聊)
    ts_rows = [json.loads(l) for l in open(_hr_os.path.join(_HR_DESKTOP, r"send", r"测试集_合并_150.jsonl"), encoding="utf-8") if l.strip()]
    ooc = [r["question"] for r in ts_rows if "L3" in ",".join(r.get("expect_layers", []))
           and not re.search(r"税|会计|审计|准则|折旧|摊销|报表", r["question"])][:NSEED]
    out_dom = [top1(q) for q in ooc]
    print(f"\n[①RAG门 in_domain] 语料=wiki增值税条例{len(corpus)}条 | 域内 n={len(in_dom)} 域外 n={len(out_dom)}")
    print(f"  域内(题干) top1: {stats(in_dom)}")
    print(f"  域外(闲聊) top1: {stats(out_dom)}")
    print(f"  域内低于 0.58 的 {sum(1 for x in in_dom if x < 0.58)} 条;域外高于 0.58 的 {sum(1 for x in out_dom if x > 0.58)} 条、"
          f"高于 0.73 的 {sum(1 for x in out_dom if x > 0.73)} 条")

    json.dump({"sample": [{"src": q["src"], "no": q["no"], "type": q["type"]} for q in sample],
               "pos": pos, "neg": neg, "in_dom": in_dom, "out_dom": out_dom},
              open(os.path.join(PROBE, "标定试算.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n[落盘] send\\题库_probe\\标定试算.json(先不落 yaml,等用户拍)")


if __name__ == "__main__":
    main()
