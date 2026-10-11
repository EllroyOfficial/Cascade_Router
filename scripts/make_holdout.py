# -*- coding: utf-8 -*-
"""make_holdout.py —— 从 v9 通用对话语料切测试留出集(2026-10-11)

规矩(用户已核的方向):训练语料改编测试集 = 考前发考卷,故切 15% 做留出集,
单独存档、永训不入;按主题分层、seed=42 可复现。
跑法: .venv/Scripts/python.exe scripts\make_holdout.py
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
import random
import re
from pathlib import Path

SRC = Path(_hr_os.path.join(_HR_DESKTOP, r"工程", r"公共对话集_General_321_重写_v9.jsonl"))
OUT = SRC.with_name("通用对话_留出测试集48.jsonl")

def bucket(item: dict) -> str:
    """按关键词分主题桶(分层用;这是[LocalModelName]的判断,可改)。"""
    t = item["instruction"] + item["output"]
    if re.search(r"\d", item["instruction"]) or re.search(r"算|几|多少|平均", item["instruction"]):
        return "算术数字"
    if re.search(r"法|诉|庭|合同|税|婚|离|继|罪|警|房|租|劳", t):
        return "法律生活"
    if re.search(r"累|焦虑| emo|难过|压力|失眠|emo|烦|迷茫|孤独|自卑|内耗", t):
        return "情绪安慰"
    if re.search(r"菜|做|吃|穿|洗|修|电脑|手机|网|sleep|睡|健|病|药", t):
        return "生活常识"
    return "闲聊杂谈"

items = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
buckets: dict[str, list] = {}
for it in items:
    buckets.setdefault(bucket(it), []).append(it)

rng = random.Random(42)
picked: list[dict] = []
stats = []
for name, group in sorted(buckets.items()):
    n = max(1, round(len(group) * 0.15))
    rng.shuffle(group)
    picked.extend(group[:n])
    stats.append(f"{name}:{len(group)}条抽{n}")

OUT.write_text(
    "\n".join(json.dumps(p, ensure_ascii=False) for p in picked) + "\n",
    encoding="utf-8")
print("分层:", ", ".join(stats))
print(f"留出集 {len(picked)} 条 → {OUT.name}  (v9 原件未动,训练时须排除这批)")
