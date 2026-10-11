# -*- coding: utf-8 -*-
"""show_mismatches.py —— 打印判分明细里未命中条目(四类清单)"""
# _hr_bootstrap_v1: 本脚本在 scripts\ 下，需要把仓库根加进 sys.path 才能 import 核心模块
import os as _hr_os, sys as _hr_sys
_hr_sys.path.insert(0, _hr_os.path.dirname(_hr_os.path.dirname(_hr_os.path.abspath(__file__))))
# ── 路径去标识（不再硬编码用户名；可用环境变量 HR_DATA_ROOT 覆盖数据根）
import os as _hr_os
_HR_HOME = _hr_os.path.expanduser("~")
_HR_DESKTOP = _hr_os.environ.get("HR_DATA_ROOT") or _hr_os.path.join(_HR_HOME, "Desktop")
_HR_DIR = _hr_os.path.dirname(_hr_os.path.abspath(__file__))
import io
import json

BASE = _HR_DIR
details = [json.loads(l) for l in io.open(BASE + r"\judge_detail.jsonl", encoding="utf-8")]
obs = {json.loads(l)["id"]: json.loads(l)
       for l in io.open(BASE + r"\observations.jsonl", encoding="utf-8")}
for d in details:
    if d["matched"]:
        continue
    o = obs[d["id"]]
    src = o.get("l1_rule") or o.get("l2_label") or o.get("target", "")
    extra = " fail_open" if o.get("fail_open") else ""
    extra += " confirm" if o.get("confirm") else ""
    print(f"{d['id']} [{d['domain']}] {d['category']}")
    print(f"    实际={d['actual_layer']} ({src}{extra})  期望层={','.join(d['expect_layers'])}")
