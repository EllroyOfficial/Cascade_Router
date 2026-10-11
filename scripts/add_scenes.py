# -*- coding: utf-8 -*-
"""add_scenes.py —— 专业四场景追加 ①实务快问 20 + ③工具/系统指令 10(2026-10-10,#0095 口径)

用户 13 条种子原话一字不改入集(各归各的场景)；不足同口吻补足。
读 send\测试集_[LocalModelName]_125.jsonl → 追加 30 条 → 写 send\测试集_[LocalModelName]_155.jsonl(旧件改名留档)。
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

import hashlib
import json
from pathlib import Path

SEND = Path(_hr_os.path.join(_HR_DESKTOP, r"send"))
OLD = SEND / "测试集_[LocalModelName]_125.jsonl"
NEW = SEND / "测试集_[LocalModelName]_155.jsonl"
FP = SEND / "测试集_留出指纹清单.txt"

R2 = "L1未命中→L2 professional→local（低置信 fail-open 升 L3 可接受）"
E2 = "口语化短问,L1 词表无交叉;考点=L1 误杀/词表缺口"
R3 = "fail-open→L3→质量门放行（L2 出手答对亦可接受）"
E3 = "本地 7B 大概率不会,正确行为是放行云端;考点=L3 放行率+漏拦"
R1 = "L1 命中税法词表→knowledge_qa→local"
E1 = "域内词表正命中;考点=不误杀"
R4 = "L1未命中→L2 tool→confirm 确认流（fail-open 升 L3 可接受）"
E4 = "工具意图不在 L1 设备词表;考点=不误当闲聊(漏拦)/不误触确认(误杀)"

def fp(q: str) -> str:
    return hashlib.md5(("self\x00" + q).encode("utf-8")).hexdigest()

ROWS = [
    # ── ① 实务快问 20(前 3 条=用户原话,口吻不改) ──
    ("SK-0001", "哦对，公允价值模式那个投资性房地产，是不是不计提减值来着？",
     "公允价值模式不计提减值,以公允价值计量变动进损益", [], R2, E2, "词表覆盖缺口+口语短问不误杀"),
    ("SK-0002", "递延所得税资产的确认前提，有几个条件来着？",
     "以未来很可能取得足够应纳税所得额为限", [], R1, E1, "L1 正命中对照"),
    ("SK-0003", "帮我查下9月28号那个征求意见稿里，碳排放的科目编号是什么？",
     "需查最新文件;锚点=科目编号(标定阶段核)", [], R2, "伪实时题(带日期非行情),realtime 词表不该拦;考点=伪实时不误杀", "伪实时不误杀"),
    ("SK-0004", "IFRS 16 那个租赁改良支出，是承租人还是出租人确认来着？",
     "租赁资产改良一般计入承租人资产或作为租赁准则按使用权资产处理", [], R2, E2, "词表覆盖缺口"),
    ("SK-0005", "金融资产那个 SPPI 测试，通不过就直接 FVTPL 吗？",
     "SPPI 不过→整体 FVTPL", ["SPPI"], R2, E2, "词表覆盖缺口"),
    ("SB-0006", "合并报表那个内部交易未实现利润，是要抵销还是调整来着？",
     "集团视角抵销未实现损益", [], R2, E2, "词表覆盖缺口"),
    ("SB-0007", "商誉减值是不是永远不能转回来着？",
     "是,商誉减值不得转回", [], R2, E2, "词表覆盖缺口"),
    ("SB-0008", "增值税那个视同销售的八种情形，你还记得不？",
     "统一移送/代销/投资/分配/无偿赠送等八类(细则第四条)", ["八"], R1, E1, "L1 正命中对照"),
    ("SB-0009", "研发费用加计扣除那个比例，现在是百分之多少来着？",
     "现行 100%(制造业等;比例随政策,锚点标定阶段核)", ["100"], R1, E1, "L1 正命中对照"),
    ("SB-0010", "使用权资产那个折旧，是按租赁期还是按寿命来着？",
     "能确定取得所有权按寿命,否则租赁期与寿命孰短", [], R2, E2, "词表覆盖缺口"),
    ("SB-0011", "那个合同履约成本，跟合同取得成本咋区分来着？",
     "履约成本=履约已发生,取得成本=为取得合同发生(增量佣金)", [], R2, E2, "词表覆盖缺口"),
    ("SB-0012", "政府补助那个与资产相关的，是走递延收益不？",
     "总额法走递延收益分期转损益,净额法冲资产账面", [], R2, E2, "词表覆盖缺口"),
    ("SB-0013", "预期信用损失那个三阶段，第一阶段计利息是按毛额还是净额来着？",
     "第一阶段按账面总额×实际利率计息", [], R2, E2, "词表覆盖缺口"),
    ("SB-0014", "固定资产那个弃置费用，是要折现进成本不？",
     "是,弃置费按现值入成本,差额计预计负债利息", [], R2, E2, "词表覆盖缺口"),
    ("SB-0015", "小规模纳税人那个征收率，现在还是 3% 吗？",
     "3%(阶段性优惠另说;锚点标定核)", ["3"], R1, E1, "L1 正命中对照"),
    ("SB-0016", "同一控制下合并那个差额，是走资本公积不？",
     "差额调资本公积(股本溢价),不足冲留存收益", [], R2, E2, "词表覆盖缺口"),
    ("SB-0017", "可转债那个权益成分，是要拆分入其他权益工具不？",
     "是,负债成分公允价值先拆,差额入其他权益工具", [], R2, E2, "词表覆盖缺口"),
    ("SB-0018", "每股收益那个反稀释，可转债转换要调整不？",
     "假定转换增加普通股,具稀释性时调整", [], R2, E2, "词表覆盖缺口"),
    ("SB-0019", "持有待售那个一年内，是硬条件还是可豁免来着？",
     "原则一年内,因不可控因素延迟可豁免", ["一年"], R2, E2, "词表覆盖缺口"),
    ("SB-0020", "永续债那个优先股，到底算权益还是负债来着？",
     "看是否无条件避免交付现金:能避免→权益,否则负债", [], R2, E2, "词表覆盖缺口"),
    # ── ③ 工具/系统指令 10(前 3 条=用户原话,口吻不改) ──
    ("ST-0001", "把今天跑的那批问答记录存到 archive 文件夹里。",
     "执行文件归档操作", [], R4, E4, "工具意图 vs 闲聊边界"),
    ("ST-0002", "重载一下Ollama，[LocalModelName]卡死了。",
     "重载 Ollama 服务", [], R4, "L2 tool/emotion 边界题(抱怨+指令混合);考点=不误拦进情绪安抚", "tool/emotion 边界"),
    ("ST-0003", "帮我把这条错的分录改对，数字要精确到分。",
     "需材料+计算;精确到分=外部 Python 计算场景", [], "L2 professional 或 tool→local/confirm；数字计算类宜 L3",
     "用户注明测 L3 质量门+外部 Python;考点=不误拦", "计算类路由"),
    ("ST-0004", "帮我把这份报告导出成 PDF。",
     "执行导出", [], R4, E4, "工具意图"),
    ("ST-0005", "把 wiki 备份一份，存到工程目录。",
     "执行备份", [], R4, E4, "工具意图"),
    ("ST-0006", "跑一下昨天那个统计脚本，把结果发我。",
     "执行脚本+回传结果", [], R4, E4, "工具意图"),
    ("ST-0007", "帮我把这张表按日期排个序再导出 Excel。",
     "排序+导出", [], R4, E4, "工具意图 vs professional 边界"),
    ("ST-0008", "清理一下临时文件夹里超过一周的文件。",
     "删除类操作,须确认", [], R4, E4 + ";删除有副作用,期望进确认流", "副作用确认流"),
    ("ST-0009", "把这条命令在终端里跑一下，把输出贴给我。",
     "执行命令回显输出", [], R4, E4, "工具意图"),
    ("ST-0010", "把语义缓存那个配置改成 0.7 再重启路由。",
     "改配置+重启(重启须用户确认)", [], R4, "改配置/重启=敏感操作,期望确认流或 L3;考点=不静默执行", "敏感操作确认"),
]

rows = [json.loads(l) for l in OLD.read_text(encoding="utf-8").splitlines() if l.strip()]
for rid, q, ref, an, route, reason, dian in ROWS:
    rows.append({"id": rid, "domain": "实务快问" if rid.startswith("SK") else "工具指令",
                 "src_repo": "self", "score": None, "question": q, "reference": ref,
                 "anchors_num": an, "anchors_must": [], "expect_route": route,
                 "expect_reason": reason, "考点": dian, "fingerprint": fp(q)})
NEW.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
FP.write_text("\n".join(f"{r['fingerprint']}\t{r['id']}\t{r['src_repo']}" for r in rows) + "\n", encoding="utf-8")
print(f"追加 {len(ROWS)} 条 ⇒ 共 {len(rows)} 条 → {NEW.name}")
