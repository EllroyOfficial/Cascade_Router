# -*- coding: utf-8 -*-
"""stresstest_ctx.py —— num_ctx 压力测试（本地[LocalModelName]，串行，只开一路）

口径（DESIGN.md §（待补）/ #0191/#0192）：
  输入：材料 300 / 1800 / 3000 字 ＋ 注入 300 字 ＋ SYSTEM ≈394 token
  输出：num_predict 512（长输出才是真爆点）
  窗口：2048（基线/阳性对照）/ 4096（目标）
  爆的五条判据：①加载失败/OOM ②显存逼近 6144 ③prompt_eval_count < 实际 token（截断保尾）
               ④内容级——材料"头部锚点"是否出现在答案里（缺失＝被截断）⑤耗时

用法：python scripts\stresstest_ctx.py [--model [LocalModelName]] [--np 512] [--cells idx,idx]
输出：send\\题库_probe\\ctx_stress.jsonl ＋ 屏幕表格
"""
# _hr_bootstrap_v1: 本脚本在 scripts\ 下，需要把仓库根加进 sys.path 才能 import 核心模块
import os as _hr_os, sys as _hr_sys
_hr_sys.path.insert(0, _hr_os.path.dirname(_hr_os.path.dirname(_hr_os.path.abspath(__file__))))
import argparse, json, os, subprocess, sys, threading, time, urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(BASE), "send", "题库_probe", "ctx_stress.jsonl")
sys.path.insert(0, BASE)
try:
    from anchors import extract_numbers, extract_clauses
except Exception:                                     # 兜底：锚点件不可用就退化成数字正则
    import re
    def extract_numbers(t): return re.findall(r"\d+(?:\.\d+)?", t)
    def extract_clauses(t): return re.findall(r"(?:CAS|IFRS|IAS)\s*\d+", t)

SEED = (
    "增值税一般纳税人销售货物适用13%的税率，小规模纳税人征收率减按1%征收，月销售额10万元以下免征增值税；"
    "非正常损失不得抵扣进项税额；可抵扣暂时性差异应确认递延所得税资产；折旧按年限平均法计提；"
    "企业会计准则第18号规定暂时性差异分为应纳税暂时性差异和可抵扣暂时性差异两类。"
)
INJECT = "【知识库】" + ("材料片段：一般计税方法适用13%的税率，小规模纳税人按征收率计算。 " * 12)   # ≈300 字
SYSTEM = ("你是[LocalModelName]，一只傲娇[LocalModelName]。回答用户关于财会的问题。" * 14)                     # ≈394 token 量级


def make_material(chars: int) -> str:
    """把 SEED 重复/截断到目标字数（头部放'第十条'这类可核锚点，便于判截断）。"""
    head = "《增值税法实施条例》第十条："
    body = (head + SEED) * (chars // len(head + SEED) + 1)
    return body[:chars]


def build_prompt(material: str) -> str:
    return f"{SYSTEM}\n\n{INJECT}\n\n材料：{material}\n\n请把上面材料逐字念一遍，不要改写。"


def vram_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


class VramPeak(threading.Thread):
    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval, self.peak, self.stop_flag = interval, 0, False

    def run(self):
        while not self.stop_flag:
            v = vram_mb()
            if v > self.peak:
                self.peak = v
            time.sleep(self.interval)


def ollama_generate(model: str, prompt: str, num_ctx: int, num_predict: int, timeout=600) -> dict:
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps({
            "model": model, "prompt": prompt, "stream": False,
            "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": 0.4},
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    d["_wall_s"] = round(time.time() - t0, 1)
    return d


def head_anchor_check(material: str, answer: str) -> dict:
    """头部锚点（材料前 240 字里的条款号/数字）是否都出现在答案里 ⇒ 缺失＝被截断。"""
    head = material[:240]
    anchors = list(dict.fromkeys(extract_clauses(head) + extract_numbers(head)))
    missing = [a for a in anchors if a not in answer]
    return {"anchors": anchors[:12], "missing": missing, "missing_n": len(missing)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="[LocalModelName]")
    ap.add_argument("--np", type=int, default=512)
    ap.add_argument("--cells", default="")           # 只跑指定序号，如 "1,2"
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    cells = [
        (1, 2048, 1800, "阳性对照(基线窗口+长材料+长输出)"),
        (2, 4096, 1800, "目标档"),
        (3, 4096, 3000, "探边界"),
        (4, 4096, 300, "常规档"),
    ]
    if args.cells:
        want = {int(x) for x in args.cells.split(",") if x.strip()}
        cells = [c for c in cells if c[0] in want]

    rows = []
    print(f"[pre] VRAM 现状 = {vram_mb()} MB")
    print("[warmup] 预热一条废问…")
    try:
        ollama_generate(args.model, "1+1等于几？", 2048, 8, timeout=300)
        print("[warmup] ok")
    except Exception as e:
        print("[warmup] 失败：", e)

    for idx, ctx, chars, tag in cells:
        mat = make_material(chars)
        prompt = build_prompt(mat)
        est_tokens = int(len(prompt) / 1.6)          # 粗估（中文≈1.6 字/token）
        peak = VramPeak(); peak.start()
        rec = {"cell": idx, "num_ctx": ctx, "material_chars": chars, "tag": tag,
               "num_predict": args.np, "est_tokens": est_tokens, "model": args.model}
        t0 = time.time()
        try:
            d = ollama_generate(args.model, prompt, ctx, args.np, timeout=args.timeout)
            ans = (d.get("response") or "").strip()
            rec.update({
                "ok": True,
                "prompt_eval_count": d.get("prompt_eval_count"),
                "eval_count": d.get("eval_count"),
                "load_ms": round((d.get("load_duration") or 0) / 1e6),
                "total_s": round((d.get("total_duration") or 0) / 1e9, 1),
                "wall_s": round(time.time() - t0, 1),
                "answer_chars": len(ans),
                "answer_head": ans[:120],
                "answer_tail": ans[-80:],
            })
            rec.update(head_anchor_check(mat, ans))
            # 判据
            rec["truncated_by_count"] = (rec.get("prompt_eval_count") or 0) < est_tokens * 0.9
            rec["head_anchor_missing"] = rec["missing_n"] > 0
            rec["boom"] = bool(rec["truncated_by_count"] or rec["head_anchor_missing"])
        except Exception as e:
            rec.update({"ok": False, "error": str(e)[:200], "boom": True})
        peak.stop_flag = True
        rec["vram_peak_mb"] = peak.peak
        rows.append(rec)
        print(json.dumps({k: rec.get(k) for k in ("cell", "num_ctx", "material_chars",
              "prompt_eval_count", "est_tokens", "vram_peak_mb", "wall_s",
              "truncated_by_count", "head_anchor_missing", "boom", "error")}, ensure_ascii=False))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[落盘]", OUT)
    print("[一句话] " + ("；".join(f"格{r['cell']}(ctx{r['num_ctx']}/{r['material_chars']}字): " +
          ("爆" if r.get("boom") else "不爆") + f" peak={r.get('vram_peak_mb')}MB" for r in rows)))


if __name__ == "__main__":
    main()
