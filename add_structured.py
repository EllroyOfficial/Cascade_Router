# -*- coding: utf-8 -*-
"""add_structured.py —— 测试集加结构化期望列(A 案,#0120 审计修法)

expect_route 自由文本解析出过 3 处真缺陷 ⇒ 加两列结构化字段,判分器只读结构化:
  expect_layers: ["L1"|"L2"|"L3", ...](层集合,任一命中即对)
  expect_confirm: true(该触发)/false(不得触发)/null(未表态·观察项)
28 种模板逐条人工定值(含 DEV-0002/0003/0007/0008 四条修正);改前备份记 md5。
"""
from __future__ import annotations
# ── 路径去标识（不再硬编码用户名；可用环境变量 HR_DATA_ROOT 覆盖数据根）
import os as _hr_os
_HR_HOME = _hr_os.path.expanduser("~")
_HR_DESKTOP = _hr_os.environ.get("HR_DATA_ROOT") or _hr_os.path.join(_HR_HOME, "Desktop")
_HR_DIR = _hr_os.path.dirname(_hr_os.path.abspath(__file__))

import hashlib
import json
import os
from collections import Counter

P = _hr_os.path.join(_HR_DESKTOP, r"send", r"测试集_合并_150.jsonl")
BAK = P + ".bak-structured"

# 28 种模板 → (expect_layers, expect_confirm) 人工定值表(逐条对照文字真实意图)
MAP = {    "L1 单命中 persona-defense → persona_defense → LOCAL（0ms 直出）": (["L1"], None),
    "L1 命中税法词表→knowledge_qa→local": (["L1"], None),
    "L1 未命中→L2 chat/emotion→local": (["L2"], None),
    "L1 未命中→L2 chat→local": (["L2"], None),
    "L1 未命中→L2 chat→local；**不得**触发 confirm": (["L2"], False),
    "L1 未命中→L2 emotion→local": (["L2"], None),
    "L1 未命中→L2 professional→local（混合意图仍应走知识链）": (["L2"], None),
    "L1 未命中→L2 tool（若出手）；confirm 是否覆盖 L2 路径=观察项": (["L2"], None),          # 观察项=未表态
    "L1 未命中→L2 出手（chat/emotion 均可）→local；低置信升 L3 亦命中": (["L2", "L3"], None),
    "L1/L2 tool 意图→local（确切说：可在本地查，不强制云端）": (["L1", "L2"], None),
    "L1/L2 tool 意图→local；**不得**因实时性词直落 L3 而不检索": (["L1", "L2"], None),        # "不得"管的是直落 L3 ⇒ actual=L3 时判层不符
    "L1/L2 tool 意图可命中→local；情绪词不得使其降级为纯闲聊": (["L1", "L2"], None),
    "L1/L2 tool 意图命中→local": (["L1", "L2"], None),
    "L1/L2 tool 意图命中→local（tool_call）": (["L1", "L2"], None),
    "L1/L2 tool 意图命中→local（tool_call）；期望不误触也**不漏触**确认": (["L1", "L2"], True),  # 不漏触=要求触发(修 DEV-0003)
    "L1未命中→L2 professional→local（低置信 fail-open 升 L3 可接受）": (["L2", "L3"], None),
    "L1未命中→L2 tool/emotion 边界题（任一出手即记）": (["L2"], None),
    "L1未命中→L2 tool→**必须 confirm**；漏触=该拦未拦": (["L2"], True),
    "L1未命中→L2 tool→confirm 确认流（fail-open 升 L3 可接受）": (["L2", "L3"], True),
    "L1未命中→L2 tool→confirm（fail-open 可接受）": (["L2", "L3"], True),
    "L2 professional 或 tool→local/confirm；数字计算类宜 L3": (["L2", "L3"], None),
    "L2 professional→local": (["L2"], None),
    "L2 professional→local；**不得**触发 confirm": (["L2"], False),
    "L2 professional→local；低置信允许升 L3": (["L2", "L3"], None),
    "L2 tool 或 professional（边界题,出手即记）": (["L2"], None),
    "L2→local（fail-open 升 L3 亦命中）": (["L2", "L3"], None),
    "L2→local；本地命中不了允许 fail-open 升 L3；**禁止编路径**": (["L2", "L3"], None),        # "禁止"管编路径,跟 L3 无关(她核过)
    "persona-defense 与税法词表冲突→降 L2 professional→knowledge_qa→local": (["L2"], None),
}

# 按 id 的期望层覆盖表(#0132 用户拍板:B3 目标项 expect_layers 加 L1)。
# 模板被多行共用,不能动模板定值 ⇒ 只覆盖这 7 条;expect_confirm 不变;fingerprint 不动。
ID_OVERRIDE = {
    "ST-0001": ["L1", "L2", "L3"],
    "ST-0004": ["L1", "L2", "L3"],
    "ST-0005": ["L1", "L2", "L3"],
    "ST-0006": ["L1", "L2", "L3"],
    "ST-0008": ["L1", "L2"],
    "ST-0009": ["L1", "L2", "L3"],
    "ST-0010": ["L1", "L2"],
}

rows = [json.loads(l) for l in open(P, encoding="utf-8") if l.strip()]
cnt = Counter(r["expect_route"] for r in rows)
missing = set(cnt) - set(MAP)
assert not missing, f"映射表缺模板: {missing}"
extra = set(MAP) - set(cnt)
assert not extra, f"映射表多模板: {extra}"

os.replace(P, BAK) if not os.path.exists(BAK) else None
md5 = hashlib.md5(open(BAK, "rb").read()).hexdigest()

for r in rows:
    layers, confirm = MAP[r["expect_route"]]
    if r["id"] in ID_OVERRIDE:
        layers = ID_OVERRIDE[r["id"]]      # 按 id 覆盖(模板共用,不能动模板)
    r["expect_layers"] = layers
    r["expect_confirm"] = confirm
with open(P, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"备份: {os.path.basename(BAK)} md5={md5}")
print(f"已写 expect_layers/expect_confirm 两列 × {len(rows)} 条")
print("\n—— 28 模板定值表(供逐条核) ——")
for t, (layers, confirm) in MAP.items():
    print(f"layers={','.join(layers)!s:<8} confirm={confirm!s:<5} | {t[:60]}")
