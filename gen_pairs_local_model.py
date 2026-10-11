"""gen_pairs_local_model —— 真生成对标定 念稿两档+兜底 min_sim(#0143 件2)

数据:题库抽 30 题,材料=题干+选项+解析原文;本地 [LocalModelName] 串行生成:
  正例①忠读念稿(接龙式逐字) ②口语改写(事实锚点保留)
  反例①数字被改(程序化替换,确定性可复跑) ②方向说反(模型改写) ③同域无关(另一题解析,零生成)
每对算:cos(材料,答案)整段 / cos(答案 vs 材料 top-2 命中句) / answer_guard 三项(语义/数字/长度)。
纪律:纯本机不调 L3;先预热;OLLAMA_NUM_PARALLEL=1 串行;生成物+打分落盘可复跑;
模型名/参数走命令行(换 LoRA 直接重跑);结论不落 yaml。
跑法:.venv\\Scripts\\python.exe gen_pairs_local_model --model [LocalModelName] --temperature 0.4 --n 30
"""
from __future__ import annotations
# ── 路径去标识（不再硬编码用户名；可用环境变量 HR_DATA_ROOT 覆盖数据根）
import os as _hr_os
_HR_HOME = _hr_os.path.expanduser("~")
_HR_DESKTOP = _hr_os.environ.get("HR_DATA_ROOT") or _hr_os.path.join(_HR_HOME, "Desktop")
_HR_DIR = _hr_os.path.dirname(_hr_os.path.abspath(__file__))

import argparse
import glob
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from answer_guard import AnswerGuard
from config_loader import load_config
from metrics import Metrics
from ollama_runtime import OllamaEmbedder

PROBE = _hr_os.path.join(_HR_DESKTOP, r"send", r"题库_probe")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def gen(model: str, prompt: str, temperature: float, num_predict: int, timeout: float = 120) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": temperature, "num_predict": num_predict}}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="[LocalModelName]")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--num-predict", type=int, default=400)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--skip-gen", action="store_true", help="只对已落盘生成物重新打分")
    args = ap.parse_args()
    print(f"[参数] model={args.model} temperature={args.temperature} num_predict={args.num_predict} n={args.n}")

    # 材料:题干+选项+解析(配对单选,抽 n 条)
    pool = []
    for p in sorted(glob.glob(os.path.join(PROBE, "中级会计实务*", "押题班-客观题（?）.pdf"))):
        text = open(os.path.join(PROBE, os.path.basename(p).replace(".pdf", ".full.txt")),
                    encoding="utf-8").read()
        body, _, answers = text.partition("参考答案")
        qs = {}
        for m in re.finditer(r"(\d+)\.【(单选题|多选题|判断题)】", body):
            qs[int(m.group(1))] = m.start()
        items = sorted(qs.items())
        for i, (no, start) in enumerate(items):
            end = items[i + 1][1] if i + 1 < len(items) else len(body)
            stem = re.sub(r"\s+", "", body[start:end])
            am = re.search(rf"\n\s*{no}\.【答案】\s*([A-D]+)", answers)
            pm = re.search(rf"\n\s*{no}\.【答案】[^【]*【解析】(.+?)(?=\n\s*\d+\.【答案】|\Z)", answers, re.S)
            om = re.search(r"A\.(.+?)B\.(.+?)C\.(.+?)D\.(.+)", stem)   # 标记切分,短选项也能收
            if not (am and pm and om):
                continue
            key = am.group(1)
            opts = dict(zip("ABCD", [g.strip() for g in om.groups()]))
            if len(key) != 1 or len(opts) != 4:      # 只配单选:正答/错答配对才干净
                continue
            pool.append({"src": os.path.basename(p), "no": no, "材料": stem + pm.group(1).strip(),
                         "正确选项": opts[key], "错误选项": [v for k, v in opts.items() if k != key]})
    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n, len(pool)))
    print(f"[题池] 可配对 {len(pool)},抽 {len(sample)}")
    json.dump(sample, open(os.path.join(PROBE, "生成对_样本.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # 生成(或复用落盘)
    gen_path = os.path.join(PROBE, f"生成物_{args.model.replace(':', '_')}_t{args.temperature}.jsonl")
    if args.skip_gen and os.path.exists(gen_path):
        rows = [json.loads(l) for l in open(gen_path, encoding="utf-8") if l.strip()]
        print(f"[复用] {gen_path} {len(rows)} 条")
    else:
        # 预热(冷启动 36-60s)
        t0 = time.time()
        gen(args.model, "预热:只回两个字。", 0.0, 10, timeout=180)
        print(f"[预热] {time.time()-t0:.0f}s", flush=True)
        rows = []
        for i, q in enumerate(sample, 1):
            other = sample[(i * 7) % len(sample)]
            while other is q:
                other = sample[(i * 7 + 1) % len(sample)]
            def alt_num(t):   # 数字被改:每个数字 ±20%~80%,确定性
                def rep(m):
                    v = float(m.group(0)); nv = v * (1.5 if int(v) % 2 == 0 else 0.5)
                    nv = nv if nv >= 1 else v + 7
                    return str(int(nv)) if v == int(v) else f"{nv:.2f}"
                return NUM_RE.sub(rep, t)
            prompts = {
                "正例_忠读": f"把下面的会计知识逐字接龙念一遍,一字不许改,只输出原文:\n{q['材料'][:600]}",
                "正例_口语改写": f"用口语把下面的会计知识讲一遍,事实、数字、结论必须与原文完全一致,只输出口语版:\n{q['材料'][:600]}",
                "反例_方向说反": f"把下面的会计表述故意说反(资产↔负债、'不得'↔'可以'、增↔减),保持语句通顺,只输出改写结果:\n{q['材料'][:600]}",
            }
            gens = {
                "正例_忠读": gen(args.model, prompts["正例_忠读"], args.temperature, args.num_predict),
                "正例_口语改写": gen(args.model, prompts["正例_口语改写"], args.temperature, args.num_predict),
                "反例_方向说反": gen(args.model, prompts["反例_方向说反"], args.temperature, args.num_predict),
                "反例_数字被改": alt_num(q["材料"]),
                "反例_同域无关": other["材料"],
            }
            rows.append({**q, "generations": gens})
            json.dump(rows, open(gen_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"  [{i}/{len(sample)}] {q['src'][:18]} 题{q['no']} 落盘", flush=True)
        print(f"[生成] {len(rows)} 题 × 5 类 → {gen_path}")

    # 打分
    emb = OllamaEmbedder("bge-m3")
    cfg = load_config(os.path.join(BASE, "routing.yaml"))
    guard = AnswerGuard(cfg, emb, Metrics())
    import math
    def cos(a, b):
        va, vb = emb._embed(a), emb._embed(b)
        dot = sum(x * y for x, y in zip(va, vb))
        na = math.sqrt(sum(x * x for x in va)) or 1.0
        nb = math.sqrt(sum(x * x for x in vb)) or 1.0
        return dot / (na * nb)
    scores = []
    for q in rows:
        mat = q["材料"]
        sents = [s for s in re.split(r"(?<=[。;;!!])", mat) if len(s) > 4] or [mat]
        for kind, ans in q["generations"].items():
            if not ans.strip():
                continue
            c_all = cos(mat, ans)
            sent_sims = sorted(cos(ans, s) for s in sents)          # 答案 vs 材料各句
            top1s, top2s = sent_sims[-1], (sent_sims[-2] if len(sent_sims) > 1 else sent_sims[-1])
            g = guard.check(mat, mat, ans)                          # 三项(语义/数字/长度)
            lens = len(ans) / max(len(mat), 1)
            scores.append({"src": q["src"], "no": q["no"], "kind": kind,
                           "cos_all": round(c_all, 3), "top2句": [round(top1s, 3), round(top2s, 3)],
                           "guard_pass": g.passed, "guard_fail": g.failed_criterion,
                           "length_ratio": round(lens, 2)})
    json.dump(scores, open(os.path.join(PROBE, "生成对_打分.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # 汇总
    def st(xs):
        xs = sorted(xs)
        return f"min {xs[0]:.3f} / 中位 {xs[len(xs)//2]:.3f} / max {xs[-1]:.3f}"
    print("\n[分布] cos(材料,答案)整段")
    pos1 = [s["cos_all"] for s in scores if s["kind"] == "正例_忠读"]
    pos2 = [s["cos_all"] for s in scores if s["kind"] == "正例_口语改写"]
    n1 = [s["cos_all"] for s in scores if s["kind"] == "反例_数字被改"]
    n2 = [s["cos_all"] for s in scores if s["kind"] == "反例_方向说反"]
    n3 = [s["cos_all"] for s in scores if s["kind"] == "反例_同域无关"]
    print(f"  正例_忠读   n={len(pos1)}: {st(pos1)}")
    print(f"  正例_口语改写 n={len(pos2)}: {st(pos2)}")
    print(f"  反例_数字被改 n={len(n1)}: {st(n1)}  ← 验证'语义抓不住事实错'")
    print(f"  反例_方向说反 n={len(n2)}: {st(n2)}")
    print(f"  反例_同域无关 n={len(n3)}: {st(n3)}")
    for kind in ("正例_忠读", "正例_口语改写", "反例_数字被改", "反例_方向说反", "反例_同域无关"):
        rows_k = [s for s in scores if s["kind"] == kind]
        gp = sum(1 for s in rows_k if s["guard_pass"])
        fc = dict(Counter(s["guard_fail"] for s in rows_k if not s["guard_pass"]))
        print(f"  {kind}: guard 通过 {gp}/{len(rows_k)} 拦截原因 {fc}")
    print("[落盘] 生成物+打分已存,可 --skip-gen 复跑打分;结论不落 yaml。")


if __name__ == "__main__":
    main()
