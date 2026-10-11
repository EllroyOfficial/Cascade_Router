"""ablate_l2_tiers.py —— L2 阈值分档消融(#0131 用户拍板:甲乙都测)

档① k=3/0.67(现状) 档② k=3/0.60(甲,只动 ratio) 档③ k=5/0.67(乙,只动 k) 档④ k=5/0.60(乙′,双变量,仅参考)
口径:
- routing.yaml 零改动:参数在内存 deepcopy 里改,一次只动一个变量。
- 留出集:出手数/出手正确/出手错判(逐条 S 编号)/fail-open;错误按类分桶。
- 甲轮 150:复用 run_route_half 的观察字段(l1_hit 带规则名),每档落 observations_<档>.jsonl,
  判分复用 judge_testset(--actual/--out),汇总四档 × {命中,漏拦,放行问题,L1误杀,该确认未确认,fail-open率,l3_rate,出手类分布}。
- 不调 L3、零云端开销;跑完不定稿,等用户拍。
跑法:.venv\\Scripts\\python.exe ablate_l2_tiers.py
"""
from __future__ import annotations
# ── 路径去标识（不再硬编码用户名；可用环境变量 HR_DATA_ROOT 覆盖数据根）
import os as _hr_os
_HR_HOME = _hr_os.path.expanduser("~")
_HR_DESKTOP = _hr_os.environ.get("HR_DATA_ROOT") or _hr_os.path.join(_HR_HOME, "Desktop")
_HR_DIR = _hr_os.path.dirname(_hr_os.path.abspath(__file__))

import copy
import io
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from config_loader import load_config
from l2_vector import load_tsv, split_train_holdout, build_from_config
from embed_local import Embedder
from router import HybridRouter

TESTSET = os.path.join(_hr_os.path.join(_HR_DESKTOP, r"send"), "测试集_合并_150.jsonl")
YAML_PATH = os.path.join(BASE, "routing.yaml")
RE_L1 = re.compile(r"L1 单命中 (\S+)")
RE_L2 = re.compile(r"L2=(\w+)")
TIERS = [("档1_k3_0.67", 3, 0.67), ("档2_k3_0.60", 3, 0.60),
         ("档3_k5_0.67", 5, 0.67), ("档4_k5_0.60", 5, 0.60)]


def holdout_eval(cfg, base_cfg_seed_rows):
    rows = load_tsv(cfg["intent_classifier"]["knn"]["train_tsv"]
                    if os.path.isabs(cfg["intent_classifier"]["knn"]["train_tsv"])
                    else os.path.normpath(os.path.join(BASE, cfg["intent_classifier"]["knn"]["train_tsv"])))
    train, hold = split_train_holdout(rows, 0.2, 42)
    vec = build_from_config(cfg, Embedder(), BASE)
    fired = right = wrong = 0
    wrong_ids, wrong_by_class = [], Counter()
    per_class = Counter()
    for sid, text, gold in hold:
        pred = vec.classify(text)
        if not pred:
            continue                       # fail-open:不出手
        fired += 1
        per_class[pred] += 1
        if pred == gold:
            right += 1
        else:
            wrong += 1
            wrong_ids.append(f"{sid}({gold}→{pred})")
            wrong_by_class[f"{gold}→{pred}"] += 1
    return {"fired": fired, "right": right, "wrong": wrong,
            "fail_open": len(hold) - fired, "wrong_ids": wrong_ids,
            "wrong_pairs": dict(wrong_by_class), "fired_per_class": dict(per_class)}


def route_half(cfg_tier, tag):
    vec = build_from_config(cfg_tier, Embedder(), BASE)
    rt = HybridRouter(config_path=YAML_PATH, classifier=vec)
    rows = [json.loads(l) for l in open(TESTSET, encoding="utf-8") if l.strip()]
    out_path = os.path.join(BASE, f"observations_{tag}.jsonl")
    errors = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            try:
                out = rt.route(r["question"])
                d = out.decision
                reasons = d.reasons
                m1 = next((RE_L1.search(x) for x in reasons if RE_L1.search(x)), None)
                m2 = next((RE_L2.search(x) for x in reasons if RE_L2.search(x)), None)
                obs = {"id": r["id"], "l1_rule": m1.group(1) if m1 else "",
                       "l1_hit": m1.group(1) if m1 else False,
                       "l2_label": m2.group(1) if m2 else "",
                       "fail_open": any(x.startswith("L2 fail-open") for x in reasons),
                       "target": d.target.name.lower(), "confirm": bool(d.tool_needs_confirm),
                       "latency_ms": 0}
            except Exception as e:
                errors += 1
                obs = {"id": r["id"], "l1_rule": "", "l1_hit": False, "l2_label": "",
                       "fail_open": False, "target": "error", "confirm": False,
                       "latency_ms": 0, "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(obs, ensure_ascii=False) + "\n")
    return out_path, errors


def judge(tag):
    out_path = os.path.join(BASE, f"observations_{tag}.jsonl")
    detail = os.path.join(BASE, f"judge_{tag}.jsonl")
    p = subprocess.run([sys.executable, os.path.join(BASE, "judge_testset.py"),
                        "--actual", out_path, "--out", detail],
                       capture_output=True, text=True)
    det = [json.loads(l) for l in io.open(detail, encoding="utf-8") if l.strip()]
    cats = Counter(d["category"] for d in det if not d["matched"])
    obs = [json.loads(l) for l in io.open(out_path, encoding="utf-8") if l.strip()]
    n = len(obs)
    fo = sum(1 for o in obs if o.get("fail_open"))
    fired = sum(1 for o in obs if o.get("l2_label"))
    per_class = Counter(o["l2_label"] for o in obs if o.get("l2_label"))
    return {"命中": n - len(cats), "分类明细": dict(cats), "fail_open": fo,
            "fail_open率": f"{fo/n:.1%}", "l2出手": fired,
            "出手类分布": dict(per_class), "errors": sum(1 for o in obs if o.get("target") == "error")}


def main() -> None:
    base_cfg = load_config(YAML_PATH)
    summary = {}
    for tag, k, ratio in TIERS:
        cfg = copy.deepcopy(base_cfg)
        ic = cfg["intent_classifier"]
        knn = ic.setdefault("knn", {})
        knn["k"], knn["min_vote_ratio"] = k, ratio
        print(f"\n===== {tag} (k={k}, min_vote_ratio={ratio}) =====", flush=True)
        h = holdout_eval(cfg, None)
        print(f"[留出] 出手 {h['fired']}/97 对 {h['right']} 错 {h['wrong']} fail-open {h['fail_open']}")
        print(f"[留出] 错判逐条: {', '.join(h['wrong_ids']) or '无'}")
        print(f"[留出] 错判类型: {h['wrong_pairs']} | 出手类分布: {h['fired_per_class']}")
        out_path, errors = route_half(cfg, tag)
        print(f"[甲轮] 150 条 errors={errors} → {os.path.basename(out_path)}", flush=True)
        j = judge(tag)
        print(f"[甲轮] {json.dumps(j, ensure_ascii=False)}")
        summary[tag] = {"k": k, "min_vote_ratio": ratio, "holdout": h, "jiarlun": j}
    out = os.path.join(BASE, "L2_TIER_ABLATION.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\n[落盘] {out}\n[纪律] routing.yaml 全程零改动;跑完不定稿,等用户拍。")


if __name__ == "__main__":
    main()
