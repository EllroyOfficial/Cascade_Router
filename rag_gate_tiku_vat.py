"""rag_gate_tiku_vat.py —— RAG 门域内重抽:增值税相关题干(#0143 用户拍,件1)

域内=题库题干中"增值税相关"者(挑得出来多少报多少,不凑);域外=闲聊 30 条(与上轮同规则可比);
语料=本地 wiki 增值税条例 54 条。报两组分布(min/中位/max)+重叠带+建议分界(0.53-0.55 择点+理由)。
只读;产出落 send\\题库_probe\\;不落 yaml。
跑法:.venv\\Scripts\\python.exe rag_gate_tiku_vat.py
"""
from __future__ import annotations
# ── 路径去标识（不再硬编码用户名；可用环境变量 HR_DATA_ROOT 覆盖数据根）
import os as _hr_os
_HR_HOME = _hr_os.path.expanduser("~")
_HR_DESKTOP = _hr_os.environ.get("HR_DATA_ROOT") or _hr_os.path.join(_HR_HOME, "Desktop")
_HR_DIR = _hr_os.path.dirname(_hr_os.path.abspath(__file__))

import glob
import json
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ollama_runtime import OllamaEmbedder

PROBE = _hr_os.path.join(_HR_DESKTOP, r"send", r"题库_probe")
TIDDLERS = _hr_os.path.join(_HR_HOME, r"tiddlywiki", r"main", r"tiddlers")
VAT_KW = re.compile(r"增值税|进项|销项|抵扣|征收率|纳税义务|专用发票|免税|小规模|一般纳税人|视同销售")
NSEED = 30


def cos(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def all_questions() -> list[str]:
    """全类型题干(单选/多选/判断都要——只要题干问句)。"""
    out = []
    for p in sorted(glob.glob(os.path.join(PROBE, "中级会计实务*", "押题班-客观题（?）.pdf"))):
        text = open(os.path.join(PROBE, os.path.basename(p).replace(".pdf", ".full.txt")),
                    encoding="utf-8").read()
        body, _, _ = text.partition("参考答案")
        for m in re.finditer(r"\d+\.【(?:单选题|多选题|判断题)】", body):
            start = m.start()
            nxt = re.search(r"\d+\.【(?:单选题|多选题|判断题)】", body[m.end():])
            end = m.end() + nxt.start() if nxt else len(body)
            stem = re.sub(r"\s+", "", body[start:end])
            out.append(stem)
    return out


def stats(xs):
    xs = sorted(xs)
    return {"n": len(xs), "min": round(xs[0], 3), "p25": round(xs[len(xs)//4], 3),
            "中位": round(xs[len(xs)//2], 3), "p75": round(xs[3*len(xs)//4], 3), "max": round(xs[-1], 3)}


def main() -> None:
    stems = all_questions()
    vat = [s for s in stems if VAT_KW.search(s)]
    print(f"[题池] 题干 {len(stems)} 条,增值税相关 {len(vat)} 条({len(vat)/len(stems):.0%})", flush=True)
    rng = random.Random(20261011)
    rng.shuffle(vat)
    sample = vat[:NSEED]
    emb = OllamaEmbedder("bge-m3")

    corpus = []
    for fn in os.listdir(TIDDLERS):
        if fn.startswith("《增值税法实施条例》") and fn.endswith(".md"):
            t = open(os.path.join(TIDDLERS, fn), encoding="utf-8").read()
            corpus.append(t.split("## 原文（逐字）", 1)[1].split("## 锚点", 1)[0].strip())
    cvecs = [(c, emb._embed(c)) for c in corpus]

    def top1(q):
        qv = emb._embed(q)
        return max(cos(qv, v) for _, v in cvecs)

    in_dom = [top1(s) for s in sample]
    ts_rows = [json.loads(l) for l in open(_hr_os.path.join(_HR_DESKTOP, r"send", r"测试集_合并_150.jsonl"),
                                           encoding="utf-8") if l.strip()]
    ooc = [r["question"] for r in ts_rows if "L3" in ",".join(r.get("expect_layers", []))
           and not re.search(r"税|会计|审计|准则|折旧|摊销|报表", r["question"])][:30]
    out_dom = [top1(q) for q in ooc]

    print(f"[域内·增值税相关题干] n={len(in_dom)}: {stats(in_dom)}")
    print(f"[域外·闲聊(同上轮)] n={len(out_dom)}: {stats(out_dom)}")
    # 重叠带与建议分界:逐点扫 0.50-0.60,报每点的 域内召回 / 域外误放
    print("[择点扫描] 阈值 → 域内召回 / 域外误放")
    for t in [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.58]:
        ri = sum(1 for x in in_dom if x >= t) / len(in_dom)
        fo = sum(1 for x in out_dom if x >= t)
        print(f"  {t:.2f}: 域内召回 {ri:.0%} | 域外误放 {fo} 条")
    lo = max(min(in_dom), min(out_dom)); hi = min(max(in_dom), max(out_dom))
    print(f"[重叠带] [{lo:.3f}, {hi:.3f}]" if lo < hi else "[重叠带] 无(完全分离)")
    json.dump({"in_dom_stems": sample, "in_dom": in_dom, "out_dom": out_dom},
              open(os.path.join(PROBE, "RAG门_增值税重抽.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[落盘] send\\题库_probe\\RAG门_增值税重抽.json(不落 yaml)")


if __name__ == "__main__":
    main()
