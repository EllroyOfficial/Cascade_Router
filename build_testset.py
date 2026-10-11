# -*- coding: utf-8 -*-
"""build_testset.py —— 放量通用 100 条 + SBR 25 条(2026-10-10,#0093 分工)

规矩=选题原则 v2 + #0090/#0093 变更:score≥8、参考答 ≤300 优先/>800 剔、
剔要链接/实时信息题、单问优先、洗 markdown、指纹 md5(repo+"\\0"+input)、seed=42。
产出: send\测试集_[LocalModelName]_125.jsonl + send\测试集_留出指纹清单.txt
"""
from __future__ import annotations
# ── 路径去标识（不再硬编码用户名；可用环境变量 HR_DATA_ROOT 覆盖数据根）
import os as _hr_os
_HR_HOME = _hr_os.path.expanduser("~")
_HR_DESKTOP = _hr_os.environ.get("HR_DATA_ROOT") or _hr_os.path.join(_HR_HOME, "Desktop")
_HR_DIR = _hr_os.path.dirname(_hr_os.path.abspath(__file__))

import hashlib
import json
import random
import re
from pathlib import Path

SRC = Path(_hr_os.path.join(_HR_DESKTOP, r".audit", r"corpus-prep", r"raw", r"distill_r1_110k.jsonl"))
SEND = Path(_hr_os.path.join(_HR_DESKTOP, r"send"))
SEND.mkdir(exist_ok=True)
OUT = SEND / "测试集_[LocalModelName]_125.jsonl"
FP = SEND / "测试集_留出指纹清单.txt"
SEED = 42
PLAN = {"coig/neo": 55, "zhihu/": 15, "xhs/xhs": 10, "ruozhiba": 10, "coig_human": 10}

BAD_Q = re.compile(r"链接|网址|http|www\.|最新|今天几号|星期几|几点了|新闻|热搜|股价|比分|开盘|收盘")
MULTI_Q = re.compile(r"[^。？！]*[？?][^。？！]*[？?]")  # 两个问号=并列多问,本轮不拆直接跳过
MD_PAT = re.compile(r"^\s*([#>*\-+]|\d+\.)\s+|\*\*|__|\[|\]\([^)]*\)|```")
WASH = re.compile(r"\n{2,}")

def wash(t: str) -> str:
    t = "\n".join(MD_PAT.sub("", ln) for ln in t.splitlines())
    return WASH.sub("\n", t).strip()

def fp(repo: str, inp: str) -> str:
    return hashlib.md5((repo + "\x00" + inp).encode("utf-8")).hexdigest()

def anchors(ref: str) -> list[str]:
    return sorted(set(re.findall(r"\d+(?:\.\d+)?", ref)))[:8]

# ── 1) 流式扫源,按 repo 候选池 ─────────────────────────────
pools: dict[str, list] = {k: [] for k in PLAN}
with SRC.open(encoding="utf-8") as f:
    for line in f:
        try:
            it = json.loads(line)
        except Exception:
            continue
        repo = it.get("repo_name", "")
        for key in PLAN:
            if not repo.startswith(key):
                continue
            content, inp = (it.get("content") or "").strip(), (it.get("input") or "").strip()
            if (it.get("score") or 0) < 8 or not inp or not content:
                continue
            lim = 300 if key == "coig/neo" else 500   # v2:≤300优先,尾类放宽到500,>800才剔
            if len(content) > lim or len(inp) > 150 or BAD_Q.search(inp) or MULTI_Q.search(inp):
                continue
            pools[key].append(it)
            break

picked, rows = [], []
rng = random.Random(SEED)
quota = dict(PLAN)
for key in list(PLAN):
    rng.shuffle(pools[key])
    take = pools[key][:quota[key]]
    picked.extend(take)
    if len(take) < quota[key]:
        print(f"⚠ 池不足: {key} 要{quota[key]}只有{len(take)}")
        quota["coig/neo"] += quota[key] - len(take)   # 记缺额
for key in PLAN:                                       # 只按缺额补,不重取整配额
    if key == "coig/neo":
        need = quota["coig/neo"] - sum(1 for it in picked if it["repo_name"].startswith("coig/neo"))
        picked.extend([it for it in pools["coig/neo"] if it not in picked][:need])
stats = []
for it in picked:
    repo = it["repo_name"]
    rows.append({
        "id": f"GT-{len(rows)+101:04d}", "domain": "通用", "src_repo": repo,
        "score": it.get("score"), "question": " ".join(it["input"].split()),
        "reference": wash(it["content"]),
        "anchors_num": anchors(it["content"]), "anchors_must": [],
        "expect_route": "L2→local（fail-open 升 L3 亦命中）",
        "expect_reason": "无 L1 词表交叉；重点看垃圾守卫/工具确认是否误触、L2 是否正常处置",
        "fingerprint": fp(repo, it["input"]),
    })
    stats.append(repo.split("/")[0])
print("通用来源分布:", {s: stats.count(s) for s in set(stats)}, "共", len(rows))

# ── 2) SBR 25 条([LocalModelName]自编,src=self;route 按现行词表实况标) ──
R2 = "L1未命中→L2 professional→local（低置信 fail-open 升 L3 可接受）"
E2 = "现行 L1 词表只有税法条目，准则概念题靠 L2 语义接住；考点=L1 词表覆盖缺口"
R1 = "L1 命中税法词表→knowledge_qa→local"
E1 = "域内词表正命中；考点=不误杀"
SBR = [
    ("SB-0001", "IFRS 15 收入确认五步法的第一步是什么？", "识别与客户的合同", ["15", "五"], R2, E2),
    ("SB-0002", "IFRS 15 里可明确区分的商品怎么判断？", "单独受益+可单独区分两项标准", ["15"], R2, E2),
    ("SB-0003", "IFRS 15 主要责任人是代理还是自行判断的判断标准？", "控制商品或服务方可认主要责任人", ["15"], R2, E2),
    ("SB-0004", "IFRS 16 承租人使用权资产后续计量有哪两种模式？", "成本模式减折旧减值/重估模式", ["16"], R2, E2),
    ("SB-0005", "IFRS 16 租赁期怎么确定？", "不可撤销期+合理确定续租选择权期间", ["16"], R2, E2),
    ("SB-0006", "IFRS 16 短期租赁豁免的条件？", "租赁期≤12个月且无购买选择权", ["12", "16"], R2, E2),
    ("SB-0007", "IFRS 9 金融资产三分类是什么？", "摊余成本/FVOCI/FVTPL", ["9"], R2, E2),
    ("SB-0008", "IFRS 9 预期信用损失三阶段模型怎么划分？", "信用风险未显著增加/已增加/已减值", ["9"], R2, E2),
    ("SB-0009", "IFRS 9 FVOCI 债务工具处置时其他综合收益怎么处理？", "重分类进损益", ["9"], R2, E2),
    ("SB-0010", "IFRS 10 控制的三要素是什么？", "权力+可变回报+能力运用权力影响回报", ["10"], R2, E2),
    ("SB-0011", "IFRS 3 商誉减值测试怎么处理？", "商誉不分摊可收回额直接测试,减值不得转回", ["3"], R2, E2),
    ("SB-0012", "IFRS 3 逆向收购中商誉怎么算？", "以法律子公司为会计收购方合并对价对比公允价值", ["3"], R2, E2),
    ("SB-0013", "IAS 12 递延所得税资产确认的限额条件？", "以未来很可能取得的应纳税暂时性差异为限", ["12"], R1, "含'递延所得税'命中 rag-a1-income-tax"),
    ("SB-0014", "IAS 12 哪些暂时性差异不允许确认递延所得税资产？", "商誉初始确认等初始豁免项目", ["12"], R1, "同上"),
    ("SB-0015", "IAS 16 不动产厂房设备的后续计量两种模式？", "成本模式/重估模式", ["16"], R2, E2),
    ("SB-0016", "IAS 36 减值测试现金流量预测要不要含筹资活动现金流？", "不含,用税前折现率配税前现金流", ["36"], R2, E2),
    ("SB-0017", "IAS 2 存货成本里异常损耗要不要计入？", "不计入,计入当期损益", ["2"], R2, E2),
    ("SB-0018", "IAS 38 研究阶段与开发阶段支出处理差异？", "研究费用化,开发满足条件可资本化", ["38"], R2, E2),
    ("SB-0019", "IAS 7 现金流量表里支付利息属于哪类活动？", "经营或筹资(政策一贯),IFRS 允许两选", ["7"], R2, E2),
    ("SB-0020", "IFRS 13 公允价值计量的三层次输入值怎么分？", "L1活跃报价/L2可观察/L3不可观察", ["13"], R2, E2),
    ("SB-0021", "IFRS 5 持有待售的确认条件？", "即刻出售+极可能一年内售出", ["5"], R2, E2),
    ("SB-0022", "IFRS 11 合营安排两种类型怎么区分？", "共同经营按份额确认资产负债,合营企业按权益法", ["11"], R2, E2),
    ("SB-0023", "IAS 41 生物资产公允价值变动计入哪里？", "当期损益", ["41"], R2, E2),
    ("SB-0024", "企业会计准则下使用寿命不确定的无形资产怎么摊销？", "不摊销,每年做减值测试", [], R2, E2),
    ("SB-0025", "合同资产与应收账款的区别是什么？", "应收无条件收款权,合同资产还取决于履约等其他因素", [], R2, E2),
]
for sid, q, ref, an, route, reason in SBR:
    rows.append({"id": sid, "domain": "SBR准则", "src_repo": "self",
                 "score": None, "question": q, "reference": ref,
                 "anchors_num": an, "anchors_must": [],
                 "expect_route": route, "expect_reason": reason,
                 "fingerprint": fp("self", q)})

OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
FP.write_text("\n".join(f"{r['fingerprint']}\t{r['id']}\t{r['src_repo']}" for r in rows) + "\n", encoding="utf-8")
print(f"产出 {len(rows)} 条 → {OUT.name}；指纹清单 → {FP.name}")
