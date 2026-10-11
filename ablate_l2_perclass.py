"""ablate_l2_perclass.py —— ②甲 vs ⑤按类分档 消融(#0136 后用户拍板)

档① k=3/0.67 基线;档② k=3/0.60(甲);档⑤ k=3 默认 0.67、tool/emotion 覆盖 0.60(按类)。
- routing.yaml 零改动(内存注入);期望列=改后版(测试集 md5 51f37caf)。
- ⑤ 实现:包装 KNN 件,按头名标签选阈值,不改 l2_vector 源码。
- 报:留出集 出手/对/错(逐条 id)/fail-open + **危险方向错判**(chat→tool、professional→tool);
  甲轮 150:命中/L1误杀/漏拦/放行问题/fail-open率/l2出手分布。不定稿,等用户拍。
跑法:.venv\\Scripts\\python.exe ablate_l2_perclass.py
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
DANGER = ("chat→tool", "professional→tool")   # 危险方向错判


class PerClassRatioClassifier:
    """⑤按类阈值:包装 KNN 件,头名标签决定用哪个出手阈值;接口兼容 SmallClassifier。"""

    def __init__(self, inner, ratios: dict) -> None:
        self.inner = inner
        self.ratios = dict(ratios)            # {label: threshold};未列出的用 inner._min_conf

    def classify(self, text: str, temperature: float = 0.0,
                 max_tokens: int = 10, timeout_ms: int = 2000) -> str:
        try:
            ranked = self.inner._rank(text)
        except Exception:
            self.inner.last_confidence, self.inner.last_detail = 0.0, "embed_error"
            return ""
        if not ranked:
            return ""
        label, conf, _detail = ranked
        thr = self.ratios.get(label, self.inner._min_conf)
        self.inner.last_confidence, self.inner.last_detail = conf, _detail
        if label not in self.inner._enum or conf < thr:
            return ""                          # 低置信/枚举外 → fail-open(红线1)
        return label


TIERS = [("①k3_0.67_基线", 3, None), ("②k3_0.60_甲", 3, 0.60),
         ("⑤k3_按类_tool_emo_0.60", 3, {"tool": 0.60, "emotion": 0.60})]


def build_tier_classifier(base_cfg, k, ratio):
    cfg = copy.deepcopy(base_cfg)
    knn = cfg["intent_classifier"].setdefault("knn", {})
    knn["k"] = k
    inner = build_from_config(cfg, Embedder(), BASE)
    if isinstance(ratio, dict):
        inner._min_conf = 0.67                 # 默认档维持 0.67
        return PerClassRatioClassifier(inner, ratio)
    inner._min_conf = float(ratio) if ratio is not None else inner._min_conf
    return inner


def holdout_eval(clf):
    cfg = load_config(YAML_PATH)
    tsv = cfg["intent_classifier"]["knn"]["train_tsv"]
    rows = load_tsv(tsv if os.path.isabs(tsv) else os.path.normpath(os.path.join(BASE, tsv)))
    train, hold = split_train_holdout(rows, 0.2, 42)
    fired = right = wrong = 0
    wrong_ids, wrong_pairs = [], Counter()
    per_class = Counter()
    for sid, text, gold in hold:
        pred = clf.classify(text)
        if not pred:
            continue
        fired += 1
        per_class[pred] += 1
        if pred == gold:
            right += 1
        else:
            wrong += 1
            wrong_ids.append(f"{sid}({gold}→{pred})")
            wrong_pairs[f"{gold}→{pred}"] += 1
    return {"fired": fired, "right": right, "wrong": wrong, "fail_open": len(hold) - fired,
            "wrong_ids": wrong_ids, "wrong_pairs": dict(wrong_pairs),
            "危险方向错判": {d: wrong_pairs.get(d, 0) for d in DANGER},
            "fired_per_class": dict(per_class)}


def route_half(clf, tag):
    rt = HybridRouter(config_path=YAML_PATH, classifier=clf)
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
    subprocess.run([sys.executable, os.path.join(BASE, "judge_testset.py"),
                    "--actual", out_path, "--out", detail], capture_output=True, text=True)
    det = [json.loads(l) for l in io.open(detail, encoding="utf-8") if l.strip()]
    cats = Counter(d["category"] for d in det if not d["matched"])
    obs = [json.loads(l) for l in io.open(out_path, encoding="utf-8") if l.strip()]
    n = len(obs)
    fo = sum(1 for o in obs if o.get("fail_open"))
    fired = sum(1 for o in obs if o.get("l2_label"))
    return {"命中": n - sum(cats.values()), "分类明细": dict(cats), "fail_open": fo,
            "fail_open率": f"{fo/n:.1%}", "l2出手": fired,
            "出手类分布": dict(Counter(o["l2_label"] for o in obs if o.get("l2_label"))),
            "errors": sum(1 for o in obs if o.get("target") == "error")}


def main() -> None:
    base_cfg = load_config(YAML_PATH)
    summary = {}
    for tag, k, ratio in TIERS:
        print(f"\n===== {tag} =====", flush=True)
        clf = build_tier_classifier(base_cfg, k, ratio)
        h = holdout_eval(clf)
        print(f"[留出] 出手 {h['fired']}/97 对 {h['right']} 错 {h['wrong']} fail-open {h['fail_open']}")
        print(f"[留出] 错判逐条: {', '.join(h['wrong_ids']) or '无'}")
        print(f"[留出] 错判类型: {h['wrong_pairs']} | 危险方向: {h['危险方向错判']} | 出手类分布: {h['fired_per_class']}")
        out_path, errors = route_half(clf, tag)
        print(f"[甲轮] errors={errors} → {os.path.basename(out_path)}", flush=True)
        j = judge(tag)
        print(f"[甲轮] {json.dumps(j, ensure_ascii=False)}")
        summary[tag] = {"k": k, "ratio": str(ratio), "holdout": h, "jiarlun": j}
    out = os.path.join(BASE, "L2_PERCLASS_ABLATION.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\n[落盘] {out}\n[纪律] routing.yaml 零改动;期望列=改后版(md5 51f37caf);不定稿等用户拍。")


if __name__ == "__main__":
    main()
