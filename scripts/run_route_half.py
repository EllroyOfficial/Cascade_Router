# -*- coding: utf-8 -*-
"""run_route_half.py —— 150 条真跑·路由半程(2026-10-10,#0124 用户拍"先甲")

真实 L1/L2 路由(small 嵌入+生产阈值,一律不动);凡 target=cloud 只记录不调 L3(零云端);
不生成本地答案(骨架 executor 占位,零 GPU)。产出 observations.jsonl 喂 judge_testset --actual。
观察字段:{"id","l1_rule","l1_hit","l2_label","fail_open","target","confirm","latency_ms"(,"error")}
l1_hit 口径(#0125-A1 修订)=L1 **裁定即记规则名**(如 "rag-a1-income-tax");未裁定记 False。
冲突降 L2 记 False(与判分器的层语义一致;judge_testset 按 "bool 或规则名" 消费)。
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

import json
import os
import re
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from router import HybridRouter  # noqa: E402

TESTSET = os.path.join(_hr_os.path.join(_HR_DESKTOP, r"send"), "测试集_合并_150.jsonl")
OUT = os.path.join(BASE, "observations.jsonl")
RE_L1 = re.compile(r"L1 单命中 (\S+)")
RE_L2 = re.compile(r"L2=(\w+)")

rows = [json.loads(l) for l in open(TESTSET, encoding="utf-8") if l.strip()]
rt = HybridRouter(config_path=os.path.join(BASE, "routing.yaml"), auto_wire_l2=True)

out_f = open(OUT, "w", encoding="utf-8")
errors = 0
for i, r in enumerate(rows, 1):
    t0 = time.perf_counter()
    try:
        out = rt.route(r["question"])
        d = out.decision
        reasons = d.reasons
        m1 = next((RE_L1.search(x) for x in reasons if RE_L1.search(x)), None)
        m2 = next((RE_L2.search(x) for x in reasons if RE_L2.search(x)), None)
        obs = {"id": r["id"],
               "l1_rule": m1.group(1) if m1 else "",
               "l1_hit": m1.group(1) if m1 else False,   # A1(#0125):l1_hit 记规则名(判分器约定 bool 或规则名),误杀可定位
               "l2_label": m2.group(1) if m2 else "",
               "fail_open": any(x.startswith("L2 fail-open") for x in reasons),
               "target": d.target.name.lower(),
               "confirm": bool(d.tool_needs_confirm),
               "latency_ms": round((time.perf_counter() - t0) * 1000)}
    except Exception as e:                      # 崩点也是发现:记下来继续跑
        errors += 1
        obs = {"id": r["id"], "l1_rule": "", "l1_hit": False, "l2_label": "",
               "fail_open": False, "target": "error", "confirm": False,
               "latency_ms": round((time.perf_counter() - t0) * 1000),
               "error": f"{type(e).__name__}: {e}"}
    out_f.write(json.dumps(obs, ensure_ascii=False) + "\n")
    if i % 30 == 0 or i == len(rows):
        print(f"[进度] {i}/{len(rows)} errors={errors}", flush=True)
out_f.close()
print(f"观察 {len(rows)} 条(errors {errors}) → {OUT}")
