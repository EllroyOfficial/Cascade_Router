# -*- coding: utf-8 -*-
"""judge_testset.py —— 150 条测试集判分器(2026-10-10,#0120 分工)

主=路由(expect_route 按层比对,不按终点) → 三类清单(L1误杀/漏拦与放行问题/误触确认)+L3 放行率;
辅=答题四维+材料级分(真跑才用,本版只留接口)。--dry-run=判分器自检(读入/逻辑人造反例/汇总表),不调任何模型。

用法:
  python judge_testset.py --dry-run                     # 判分器自检(本轮交付)
  python judge_testset.py --actual observations.jsonl   # 真跑:实际路由观察逐条判分(后续)
观察 jsonl 行:{"id","l1_hit"(bool或规则名),"l2_label","fail_open","target","confirm","latency_ms","answer"}
"""
from __future__ import annotations
# ── 路径去标识（不再硬编码用户名；可用环境变量 HR_DATA_ROOT 覆盖数据根）
import os as _hr_os
_HR_HOME = _hr_os.path.expanduser("~")
_HR_DESKTOP = _hr_os.environ.get("HR_DATA_ROOT") or _hr_os.path.join(_HR_HOME, "Desktop")
_HR_DIR = _hr_os.path.dirname(_hr_os.path.abspath(__file__))

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
TESTSET = os.path.join(_hr_os.path.join(_HR_DESKTOP, r"send"), "测试集_合并_150.jsonl")
FIELDS = ["id", "domain", "src_repo", "score", "question", "reference", "anchors_num",
          "anchors_must", "expect_route", "expect_reason", "fingerprint"]

# ── expect_route 解析(自由文本 → 层集合 + 确认期望) ──────────
_RE_LX = re.compile(r"L[123](?!\s*未命中)")          # "L1未命中"不算期望层
_RE_L1_NO = re.compile(r"L1\s*未命中")

def parse_layers(text: str) -> set:
    layers = set(_RE_LX.findall(text))
    if _RE_L1_NO.search(text):
        layers.discard("L1")     # "L1未命中"即便后面又提 L1(如"L1未命中→L2"),L1 不入期望
        if not layers:
            layers = {"L2"}
    return layers or {"ANY"}

def parse_confirm(text: str) -> bool | None:
    """True=期望触发确认;False=期望不得触发;None=未表态。"""
    if re.search(r"(不误触|不该误触|不得触发|不应触发|不得.*确认|不该.*确认)", text):
        return False
    if "confirm" in text.lower() or "确认" in text:
        return True
    return None

def derive_layer(actual: dict) -> str:
    if actual.get("l1_hit"):
        return "L1"
    if actual.get("fail_open"):
        return "L3"
    if actual.get("l2_label"):
        return "L2"
    return "L3" if actual.get("target") == "cloud" else "local"

def parse_expect(rec: dict) -> tuple[set, bool | None]:
    """结构化字段优先(expect_layers/expect_confirm,#0120 A 案);旧文件兜底文本解析。"""
    if "expect_layers" in rec:
        return set(rec["expect_layers"]), rec.get("expect_confirm")  # 缺省 None=未表态
    return parse_layers(rec["expect_route"]), parse_confirm(rec["expect_route"])

def judge_case(rec: dict, actual: dict) -> dict:
    exp_layers, exp_confirm = parse_expect(rec)
    layer = derive_layer(actual)
    confirm = bool(actual.get("confirm"))
    matched = layer in exp_layers
    cat = "命中"
    if exp_confirm is False and confirm:
        matched, cat = False, "误触确认"
    elif exp_confirm is True and not confirm:
        matched, cat = False, "该确认未确认"
    elif not matched:
        if layer == "L1" and "L1" not in exp_layers:
            cat = "L1误杀"
        elif "L3" in exp_layers and layer != "L3":
            cat = "漏拦(该上云未上)"
        elif "L3" not in exp_layers and layer == "L3":
            cat = "放行问题(该本地却上云)"
        else:
            cat = "层不符"
    return {"id": rec["id"], "domain": rec["domain"], "expect_layers": sorted(exp_layers),
            "actual_layer": layer, "matched": matched, "category": cat,
            "confirm": confirm, "latency_ms": actual.get("latency_ms")}

# ── 读入自检 ─────────────────────────────────────────────
def check_readin(rows: list[dict]) -> bool:
    ok = True
    print(f"[读入] 条数={len(rows)}(期望 150)")
    ok &= len(rows) == 150
    need = ["id", "domain", "question", "reference", "expect_route", "expect_reason", "fingerprint"]
    missing = [r["id"] for r in rows if any(not r.get(f) for f in need)]
    print(f"[读入] 关键字段缺失: {missing or '无'}（score/anchors 为空属合法：self/owner 条目无分无锚）")
    ok &= not missing
    fps = [r["fingerprint"] for r in rows]
    dup = [f for f, c in Counter(fps).items() if c > 1]
    print(f"[读入] 指纹重复: {dup or '无'}")
    ok &= not dup
    dist = Counter(r["domain"] for r in rows)
    want = {"通用": 100, "SBR准则": 20, "工具指令": 10, "情绪闲聊": 10, "开发调试": 10}
    print(f"[读入] 域分布: {dict(dist)} 期望 {want}")
    ok &= dict(dist) == want
    return ok

# ── 判定逻辑自检(人造反例,每条给期望分类) ────────────────
SELFTEST = [  # (expect_route, actual, 期望category, 说明) —— 后 4 条=真实模板(#0120 审计修正)
    ("L2→local（fail-open 升 L3 亦命中）", {"l2_label": "professional", "target": "local"}, "命中", "层命中"),
    ("L1 命中 rag-a1-income-tax→knowledge_qa→local", {"l1_hit": "rag-a1-income-tax", "target": "local"}, "命中", "L1 正命中"),
    ("L2→local", {"l1_hit": "realtime-strong", "target": "cloud"}, "L1误杀", "被 realtime 拦=误杀"),
    ("fail-open→L3→质量门放行", {"l2_label": "chat", "target": "local"}, "漏拦(该上云未上)", "该上云没上"),
    ("L1 单命中 persona-defense → persona_defense → LOCAL", {"l2_label": "chat", "fail_open": True, "target": "cloud"}, "放行问题(该本地却上云)", "该本地却上云"),
    ("帮我把报告导出成 PDF|L2 tool→confirm", {"l2_label": "tool", "confirm": True, "target": "local"}, "命中", "期望确认且确认"),
    ("帮我把报告导出成 PDF|L2 tool→confirm", {"l2_label": "chat", "confirm": False, "target": "local"}, "该确认未确认", "期望确认没确认"),
    ("L2 professional→local；不得触发 confirm", {"l2_label": "tool", "confirm": True, "target": "local"}, "误触确认", "不该问的问了"),
    ("L1/L2 tool 意图命中→local", {"l1_hit": "tool-device-switch", "confirm": True, "target": "local"}, "命中", "L1/L2 并集,L1 命中即可"),
    ("fail-open→L3→质量门放行", {"l2_label": "chat", "fail_open": True, "target": "cloud"}, "命中", "升 L3 命中"),
]

STRUCT_SELFTEST = [  # 真实模板+结构化字段(#0120 审计的 4 处,含她认的 1 处合理)
    ({"expect_route": "L1/L2 tool 意图→local；不得因实时性词直落 L3 而不检索",
      "expect_layers": ["L1", "L2"], "expect_confirm": None},
     {"l1_hit": False, "fail_open": True, "target": "cloud"}, "放行问题(该本地却上云)", "DEV-0008 直落L3=放行问题(原假阴性)"),
    ({"expect_route": "L1/L2 tool 意图命中→local（tool_call）；期望不误触也不漏触确认",
      "expect_layers": ["L1", "L2"], "expect_confirm": True},
     {"l1_hit": False, "l2_label": "tool", "confirm": True, "target": "local"}, "命中", "DEV-0003 不漏触=要求触发(原假阳性)"),
    ({"expect_route": "L1 未命中→L2 tool（若出手）；confirm 是否覆盖 L2 路径=观察项",
      "expect_layers": ["L2"], "expect_confirm": None},
     {"l1_hit": False, "l2_label": "tool", "confirm": False, "target": "local"}, "命中", "DEV-0007 观察项=未表态(原假阳性)"),
    ({"expect_route": "L2→local；允许 fail-open 升 L3；禁止编路径",
      "expect_layers": ["L2", "L3"], "expect_confirm": None},
     {"l1_hit": False, "l2_label": "chat", "fail_open": True, "target": "cloud"}, "命中", "DEV-0002 禁止编路径≠禁L3(原判得对)"),
]

def check_logic() -> bool:
    ok = True
    fake = {"id": "ST", "domain": "自检", "expect_route": ""}
    print("[逻辑] 人造反例判分:")
    for text, actual, want, note in SELFTEST:
        r = dict(fake, expect_route=text)
        got = judge_case(r, actual)["category"]
        hit = got == want
        ok &= hit
        print(f"  [{'PASS' if hit else 'FAIL'}] {note}: got={got!r} want={want!r}")
    print("[逻辑] 真实模板+结构化字段判分:")
    for rec, actual, want, note in STRUCT_SELFTEST:
        r = dict({"id": "ST", "domain": "自检"}, **rec)
        got = judge_case(r, actual)["category"]
        hit = got == want
        ok &= hit
        print(f"  [{'PASS' if hit else 'FAIL'}] {note}: got={got!r} want={want!r}")
    return ok

def check_summary() -> bool:
    rows = load_rows()
    actuals = {r["id"]: dict(l1_hit=False, l2_label="professional", fail_open=False,
                             target="local", confirm=False, latency_ms=0) for r in rows}
    demo = judge_all(rows, actuals)
    tbl = summarize(demo)
    print("[汇总] 按域命中表(全 L2 假想路由):")
    for d, s in tbl["by_domain"].items():
        print(f"  {d}: {s}")
    print(f"[汇总] 总命中 {tbl['matched']}/{tbl['total']} L3放行率 {tbl['l3_rate']}")
    return tbl["total"] == 150

# ── 全量判分与汇总 ───────────────────────────────────────
def load_rows() -> list[dict]:
    return [json.loads(l) for l in open(TESTSET, encoding="utf-8") if l.strip()]

def judge_all(rows: list[dict], actuals: dict[str, dict]) -> list[dict]:
    return [judge_case(r, actuals.get(r["id"], {"l1_hit": False, "target": "未跑"}))
            for r in rows]

def summarize(details: list[dict]) -> dict:
    by_domain = defaultdict(Counter)
    for d in details:
        by_domain[d["domain"]][d["category"]] += 1
    matched = sum(1 for d in details if d["matched"])
    exp_l3 = [d for d in details if "L3" in d["expect_layers"] and d["actual_layer"] != "未跑"]
    l3_hit = sum(1 for d in exp_l3 if d["actual_layer"] == "L3")
    return {"total": len(details), "matched": matched,
            "by_domain": {d: dict(c) for d, c in by_domain.items()},
            "l3_rate": f"{l3_hit}/{len(exp_l3)}" if exp_l3 else "n/a",
            "categories": dict(Counter(d["category"] for d in details))}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--actual", help="真跑:实际路由观察 jsonl")
    ap.add_argument("--out", default=os.path.join(BASE, "judge_detail.jsonl"))
    a = ap.parse_args()
    rows = load_rows()
    if a.dry_run:
        ok1 = check_readin(rows)
        ok2 = check_logic()
        ok3 = check_summary()
        print(f"\n[结论] 读入自检={'PASS' if ok1 else 'FAIL'} "
              f"判定逻辑自检={'PASS' if ok2 else 'FAIL'} 汇总表={'PASS' if ok3 else 'FAIL'}")
        return 0 if (ok1 and ok2 and ok3) else 1
    if a.actual:
        actuals = {json.loads(l)["id"]: json.loads(l)
                   for l in open(a.actual, encoding="utf-8") if l.strip()}
        details = judge_all(rows, actuals)
        with open(a.out, "w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        tbl = summarize(details)
        print(json.dumps(tbl, ensure_ascii=False, indent=1))
        print(f"明细 → {a.out}")
        return 0
    ap.error("给 --dry-run 或 --actual")
    return 2

if __name__ == "__main__":
    sys.exit(main())
