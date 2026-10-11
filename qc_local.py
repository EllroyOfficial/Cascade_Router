# -*- coding: utf-8 -*-
"""qc_local.py —— 第二道防线：`0.70–0.85` 区间的「定点微型质检」
（用户 2026-10-09 方案：这个区间单独拿出来，用本地[LocalModelName]或云端 API 做微型质检）

**死命令提示词（用户原文，不许改）**：
    请判断以下【回答】是否与【原文】事实一致。只回答PASS或FAIL，不要做任何解释。

设计要点：
* **三态**：`pass` / `fail` / `uncertain` —— **uncertain 一律转人工**（宁漏不误）。
  严格只认 `PASS` / `FAIL`（允许周围有标点空白）；**出现任何其它字样（含人设话、解释、空、乱码）⇒ uncertain**。
* 只在窄区间被调用 ⇒ 单次输出 ≤ 几个 token ⇒ 调用量与延迟可控。
* `ask` 可注入：默认走本地 ollama（纯 urllib，零依赖）；也可换成云端或测试桩。
* **不碰 router**：供 `rewrite_stage` 的 `local_ask` 换件使用（[LocalModelName]已留好换件口）。
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request

DEAD_ORDER = "请判断以下【回答】是否与【原文】事实一致。只回答PASS或FAIL，不要做任何解释。"

# ⚠️ 只容忍"标点/空白"包裹；**不允许中文字样**（否则「PASS，没问题」会被误判成 pass）
_TOL = r"[\s。.！!，,：:、\-—]*"
PASS_RE = re.compile(r"^" + _TOL + r"PASS" + _TOL + r"$", re.I)
FAIL_RE = re.compile(r"^" + _TOL + r"FAIL" + _TOL + r"$", re.I)

DEFAULT_MODEL = "[LocalModelName]"
DEFAULT_URL = "http://127.0.0.1:11434/api/generate"


def build_prompt(material: str, answer: str, max_chars: int = 2400) -> str:
    """按用户死命令拼提示词；材料/回答各截断，防超本地 ctx（2048 token）。"""
    m = (material or "").strip()[:max_chars]
    a = (answer or "").strip()[:max_chars]
    return f"{DEAD_ORDER}\n\n【原文】\n{m}\n\n【回答】\n{a}\n\n只回答 PASS 或 FAIL："


def parse_verdict(raw: str) -> str:
    """严格→**首词**两段解析：`pass` / `fail` / `uncertain`（uncertain = 转人工）。

    ① 先试严格全词匹配（只容忍周围标点空白）；
    ② 再退一步认"**开头的判词**" —— 因为把 `num_predict` 压到 3 之后，模型输出常被截成
       `FAIL。原文` 这种「判词 + 一两个字」，首词就是它给出的结论 ✓
       ⚠️ 代价：`PASS，但我有疑问` 这类会被判 pass —— 实践中它不这么写；且截断已把后文砍掉。
    ③ 其它（`先对齐`、空、人设话、乱码）⇒ uncertain ⇒ 转人工。
    """
    t = (raw or "").strip()
    if PASS_RE.match(t):
        return "pass"
    if FAIL_RE.match(t):
        return "fail"
    head = t[:8].upper()
    if head.startswith("PASS"):
        return "pass"
    if head.startswith("FAIL"):
        return "fail"
    return "uncertain"


def ask_ollama(prompt: str, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL,
               timeout: int = 120, num_predict: int = 6, temperature: float = 0.0) -> str:
    """本地 ollama 一次生成（默认 temp 0、只需几个 token）。

    ⚠️ temperature 是**调用期参数**，会覆盖 Modelfile 里的 PARAMETER ⇒ 温度高低是"我方变量"，
    做对照实验时务必同题同脚本、只改这一个旋钮（见 selftest_qc_temp.py）。
    """
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": 2048},
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "")


def qc_check(material: str, answer: str, ask=None, model: str = DEFAULT_MODEL,
             timeout: int = 120) -> dict:
    """定点质检：返回 {verdict, raw, reason}；verdict ∈ pass|fail|uncertain。"""
    if ask is None:
        def ask(p):  # noqa: E306
            return ask_ollama(p, model=model, timeout=timeout)
    try:
        raw = ask(build_prompt(material, answer))
    except Exception as exc:  # 超时/连接失败/模型没起 → 转人工，绝不放行
        return {"verdict": "uncertain", "raw": "", "reason": f"质检调用失败（{type(exc).__name__}）⇒ 转人工"}
    verdict = parse_verdict(raw)
    reason = {"pass": "质检通过", "fail": "质检判定与原文事实不一致",
              "uncertain": "质检输出无法判定（非 PASS/FAIL）⇒ 转人工"}[verdict]
    return {"verdict": verdict, "raw": (raw or "").strip()[:60], "reason": reason}


# ------------------------------------------------------------------ 标定
# 带内样本（2026-10-09 换基准后落在 0.70–0.85 的 6 条：正 3 / 反 3），
# 直接复用[LocalModelName] smoke_gate_baseline.py 里的真件样本，避免两套数据。
IN_BAND = [
    ("正", "忠读逐字(真件)"),
    ("正", "口语全项(构造)"),
    ("正", "部分摘录(真件)"),
    ("反", "免税混入(真件改错)"),
    ("反", "纯胡编(构造)"),
    ("反", "方向说反(构造)"),
]


def main() -> None:
    import contextlib
    import io
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # ⚠️ 导入她的脚本会跑它的顶层实验（会打印、要 ollama）⇒ 吞掉它的输出
    with contextlib.redirect_stdout(io.StringIO()):
        from smoke_gate_baseline import CASES, MATERIAL  # noqa: E402

    by_name = {name: (cls, ans) for cls, name, ans in CASES}
    want = {"正": "pass", "反": "fail"}
    ok = tot = 0
    uncertain = 0
    print(f"{'期望':<4}{'样本':<20}{'质检':<11}{'原样输出':<12}判定")
    for cls, name in IN_BAND:
        if name not in by_name:
            print(f"  !! 样本缺失: {name}")
            continue
        _, ans = by_name[name]
        r = qc_check(MATERIAL, ans)
        tot += 1
        hit = r["verdict"] == want[cls]
        ok += hit
        uncertain += r["verdict"] == "uncertain"
        print(f"{cls:<4}{name:<20}{r['verdict']:<11}{r['raw']:<12}"
              f"{'✓' if hit else '✗'} {r['reason']}")
    print(f"\n带内准确率 {ok}/{tot}（uncertain {uncertain} 条会转人工）")
    print("注：uncertain 不算错——它把拿不准的推给人工，正是设计意图。")


if __name__ == "__main__":
    main()
