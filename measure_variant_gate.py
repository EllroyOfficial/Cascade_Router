# -*- coding: utf-8 -*-
"""measure_variant_gate.py —— 按**当前 embed_local 配置**量"念稿分档"这批样本的分布。
用途：三档对照（V1 bge-m3/ollama 基线、V2 small/ollama、V3 small/inproc）看阈值该移到哪。
样本直接复用 smoke_gate_baseline 的 MATERIAL/CASES（它自己那份走老路径，导入时静默）。
用法:  set EMBED_BACKEND=inproc  &&  .venv\\Scripts\\python measure_variant_gate.py
"""
import contextlib
import importlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    gate = importlib.import_module("smoke_gate_baseline")

from embed_local import Embedder, info              # noqa: E402
from ragas_local import _cos, _embed, faithfulness, select_baseline  # noqa: E402

emb = Embedder()
mv = _embed(emb, gate.MATERIAL)
print("生效配置: " + json.dumps(info(), ensure_ascii=False))
print("%-4s%-16s%9s%9s%9s" % ("类", "名称", "整段cos", "换基准", "faith"))

pos_w, pos_r, neg_w, neg_r, rows = [], [], [], [], []
for cls, name, ans in gate.CASES:
    av = _embed(emb, ans)
    whole = _cos(mv, av)
    rebased, _base = select_baseline(ans, gate.MATERIAL, emb)
    faith = faithfulness(ans, gate.MATERIAL, emb)
    rows.append((cls, name, whole, rebased, faith))
    if cls == "正":
        pos_w.append(whole); pos_r.append(rebased)
    else:
        neg_w.append(whole); neg_r.append(rebased)
    print("%-4s%-16s%9.3f%9.3f%9.3f" % (cls, name, whole, rebased, faith))

print("-" * 52)
print("正(5) 整段cos  最低 %.3f  均值 %.3f" % (min(pos_w), sum(pos_w) / len(pos_w)))
print("反(5) 整段cos  最高 %.3f  均值 %.3f" % (max(neg_w), sum(neg_w) / len(neg_w)))
print("正(5) 换基准   最低 %.3f  均值 %.3f" % (min(pos_r), sum(pos_r) / len(pos_r)))
print("反(5) 换基准   最高 %.3f  均值 %.3f" % (max(neg_r), sum(neg_r) / len(neg_r)))
for tag, p, n in (("整段", pos_w, neg_w), ("换基准", pos_r, neg_r)):
    lo, hi = min(p), max(n)
    print("%s: 正例最低 %.3f ／ 反例最高 %.3f ⇒ %s（带宽 %.3f）"
          % (tag, lo, hi, "有重叠" if hi > lo else "无重叠", abs(lo - hi)))
