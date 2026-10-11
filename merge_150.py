# -*- coding: utf-8 -*-
"""merge_150.py —— 合并终稿 150 条(2026-10-10,#0095 四场景口径)

账目:通用 100 + ①实务快问 20(=SBR 25 精选,剔 5 冷门) + ②情绪闲聊 10(她)
     + ③工具指令 10([LocalModelName]) + ④开发调试 10(她) = 150
产出: send\测试集_合并_150.jsonl + 测试集_留出指纹清单.txt(重写为 150)
"""
from __future__ import annotations
# ── 路径去标识（不再硬编码用户名；可用环境变量 HR_DATA_ROOT 覆盖数据根）
import os as _hr_os
_HR_HOME = _hr_os.path.expanduser("~")
_HR_DESKTOP = _hr_os.environ.get("HR_DATA_ROOT") or _hr_os.path.join(_HR_HOME, "Desktop")
_HR_DIR = _hr_os.path.dirname(_hr_os.path.abspath(__file__))

import json
from pathlib import Path

SEND = Path(_hr_os.path.join(_HR_DESKTOP, r"send"))
MINE = SEND / "测试集_[LocalModelName]_125.jsonl"
HERS = SEND / "测试集_[LocalModelName]_20.jsonl"
OUT = SEND / "测试集_合并_150.jsonl"
FP = SEND / "测试集_留出指纹清单.txt"
DROP_SBR = {"SB-0013", "SB-0017", "SB-0018", "SB-0021", "SB-0023"}  # 冷门 5 条,备用池在 add_scenes.py

ST = [  # ③ 工具/系统指令 10(前 3 条=用户原话,口吻不改)
    ("ST-0001", "把今天跑的那批问答记录存到 archive 文件夹里。", "执行文件归档操作",
     "L1未命中→L2 tool→confirm 确认流（fail-open 升 L3 可接受）",
     "工具意图不在 L1 设备词表;考点=不误当闲聊(漏拦)/不误触确认(误杀)"),
    ("ST-0002", "重载一下Ollama，[LocalModelName]卡死了。", "重载 Ollama 服务",
     "L1未命中→L2 tool/emotion 边界题（任一出手即记）",
     "抱怨+指令混合;考点=工具意图不被情绪外壳淹没,也不误进纯闲聊"),
    ("ST-0003", "帮我把这条错的分录改对，数字要精确到分。", "需材料+计算;精确到分=外部 Python 计算场景",
     "L2 professional 或 tool→local/confirm；数字计算类宜 L3",
     "用户注明测 L3 质量门+外部 Python;考点=不误拦"),
    ("ST-0004", "帮我把这份报告导出成 PDF。", "执行导出",
     "L1未命中→L2 tool→confirm（fail-open 可接受）", "工具意图;考点=漏拦/误杀"),
    ("ST-0005", "把 wiki 备份一份，存到工程目录。", "执行备份",
     "L1未命中→L2 tool→confirm（fail-open 可接受）", "工具意图;考点=漏拦/误杀"),
    ("ST-0006", "跑一下昨天那个统计脚本，把结果发我。", "执行脚本+回传结果",
     "L1未命中→L2 tool→confirm（fail-open 可接受）", "工具意图;考点=漏拦/误杀"),
    ("ST-0007", "帮我把这张表按日期排个序再导出 Excel。", "排序+导出",
     "L2 tool 或 professional（边界题,出手即记）", "考点=tool/professional 边界"),
    ("ST-0008", "清理一下临时文件夹里超过一周的文件。", "删除类操作,须确认",
     "L1未命中→L2 tool→**必须 confirm**；漏触=该拦未拦",
     "删除有副作用;考点=副作用操作进确认流"),
    ("ST-0009", "把这条命令在终端里跑一下，把输出贴给我。", "执行命令回显输出",
     "L1未命中→L2 tool→confirm（fail-open 可接受）", "工具意图;考点=漏拦/误杀"),
    ("ST-0010", "把语义缓存那个配置改成 0.7 再重启路由。", "改配置+重启(重启须用户确认)",
     "L1未命中→L2 tool→**必须 confirm**；漏触=该拦未拦",
     "改配置/重启=敏感操作;考点=不静默执行"),
]

rows = [json.loads(l) for l in MINE.read_text(encoding="utf-8").splitlines() if l.strip()]
before = len(rows)
rows = [r for r in rows if r["id"] not in DROP_SBR]
print(f"[LocalModelName]侧 {before} − 剔 SBR 冷门 {before - len(rows)} = {len(rows)}")

for rid, q, ref, route, reason in ST:
    rows.append({"id": rid, "domain": "工具指令", "src_repo": "self", "score": None,
                 "question": q, "reference": ref, "anchors_num": [], "anchors_must": [],
                 "expect_route": route, "expect_reason": reason,
                 "fingerprint": __import__("hashlib").md5(("self\x00" + q).encode()).hexdigest()})
print(f"+ 工具指令 10 = {len(rows)}")

hers = [json.loads(l) for l in HERS.read_text(encoding="utf-8").splitlines() if l.strip()]
rows.extend(hers)
print(f"+ [LocalModelName]侧 {len(hers)} = {len(rows)}")

seen, dup = set(), 0
for r in rows:
    if r["fingerprint"] in seen:
        dup += 1
    seen.add(r["fingerprint"])
assert dup == 0, f"指纹碰撞 {dup} 条!"

import collections
print("域分布:", dict(collections.Counter(r["domain"] for r in rows)))
OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
FP.write_text("\n".join(f"{r['fingerprint']}\t{r['id']}\t{r['src_repo']}" for r in rows) + "\n", encoding="utf-8")
print(f"终稿 {len(rows)} 条 → {OUT.name}（指纹清单同步重写）")
