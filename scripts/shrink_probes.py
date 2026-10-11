"""shrink_probes.py —— 缩版探针(#0154 A:卡空重跑,快 5-10 倍)

范围(照 #0154 规格):任务1 = v3 12 条 + v3-orig 4 代表;2A = 8192 档补 中/长/超长 + 去人设档;
2B 温度 2 条×2(0.2/0.8);2C 提示词形态 6 条;2D 引用 #0151 已出的统计。
num_predict=160、串行一路、先预热。只测不改;产物落 send\\题库_probe\\。
跑法:.venv\\Scripts\\python.exe scripts\shrink_probes.py
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

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ifollow_probes import (PROBES, M300, M600, M1800, JIELONG, api, norm, jielong_check, json_ok)

PROBE = _hr_os.path.join(_HR_DESKTOP, r"send", r"题库_probe")
NO_PERSONA = "你是本地会计知识助手。严格按用户指令输出，不要任何人设台词。"


def main() -> None:
    model = "[LocalModelName]"
    NP = 160
    log = open(os.path.join(PROBE, "shrink_probes.jsonl"), "w", encoding="utf-8")

    def run(tag, prompt, temp=0.4, num_ctx=None, system=None):
        t0 = time.time()
        out = api(model, prompt, temp, NP, num_ctx, timeout=300, system=system)
        log.write(json.dumps({"tag": tag, "temp": temp, "num_ctx": num_ctx or 2048,
                              "system_override": system is not None, "output": out[:600]},
                             ensure_ascii=False) + "\n")
        log.flush()
        print(f"  [{tag}] {time.time()-t0:.1f}s | {out[:36]!r}", flush=True)
        return out

    t0 = time.time()
    run("预热", "预热:只回两个字。", temp=0.0)
    print(f"[预热] {time.time()-t0:.1f}s(卡空,应秒回)", flush=True)

    result = {}

    # 任务1:v3 全 12 条
    print("\n== 任务1 v3 全 12 条 ==", flush=True)
    rows = []
    for tag, prompt, checker, _ in PROBES:
        v = checker(run(f"T1-{tag}", prompt))
        rows.append((tag, v))
    v = jielong_check(M300, 200)(run("T1-接龙-短", JIELONG.format(mat=M300)))
    rows.append(("接龙-短", v))
    v = jielong_check(M600, 350)(run("T1-接龙-中", JIELONG.format(mat=M600)))
    rows.append(("接龙-中", v))
    rates = {k: sum(1 for _, v in rows if v.startswith(k)) for k in ("服从", "部分服从", "不服从")}
    result["任务1_v3_12条"] = {"rows": rows, "rates": rates}
    print(f"  → v3: {rates}")

    # 任务1:v3-orig 4 代表
    print("\n== 任务1 v3-orig 4 代表 ==", flush=True)
    rep = ["重复-逐字复述", "格式-JSON", "长度-20字", "内容-禁白米饭"]
    rows_o = []
    for tag, prompt, checker, _ in PROBES:
        if tag in rep:
            rows_o.append((tag, checker(run(f"T1O-{tag}", prompt))))
    result["任务1_orig_4代表"] = rows_o
    print(f"  → {rows_o}")

    # 2A:8192 档 中/长/超长(人设) + 去人设档
    print("\n== 2A 8192 档:中/长/超长 + 去人设 ==", flush=True)
    r2a = {}
    for tag, mat in (("中300", M300), ("长600", M600), ("超长1800", M1800)):
        r2a[f"8192|人设|{tag}"] = jielong_check(mat, max(100, len(norm(mat)) // 2))(
            run(f"2A-8192-{tag}", JIELONG.format(mat=mat), num_ctx=8192))
        r2a[f"8192|去人设|{tag}"] = jielong_check(mat, max(100, len(norm(mat)) // 2))(
            run(f"2A-8192-去人设-{tag}", JIELONG.format(mat=mat), num_ctx=8192, system=NO_PERSONA))
    result["2A_8192"] = r2a

    # 2B 温度 2 条 × 2
    print("\n== 2B 温度 2 条×2 ==", flush=True)
    sub = [("格式-JSON", next(p for p in PROBES if p[0] == "格式-JSON")[1], next(p for p in PROBES if p[0] == "格式-JSON")[2]),
           ("长度-20字", next(p for p in PROBES if p[0] == "长度-20字")[1], next(p for p in PROBES if p[0] == "长度-20字")[2])]
    r2b = {}
    for temp in (0.2, 0.8):
        for tag, prompt, checker in sub:
            r2b[f"{tag}|t{temp}"] = checker(run(f"2B-{tag}-t{temp}", prompt, temp=temp))
    result["2B_温度"] = r2b

    # 2C 提示词形态 6 条(接龙-中300)
    print("\n== 2C 提示词形态 6 条 ==", flush=True)
    forms = {
        "自然语言": JIELONG.format(mat=M300),
        "结构化+few-shot": ("任务：逐字复述材料。\n规则：\n1. 从第一字念到最后一字\n2. 不许改动、不许总结\n"
                            "示例：\n材料：『今天天气不错。』\n输出：今天天气不错。\n"
                            f"材料：『{M300}』\n输出："),
        "指令放尾部": f"材料：\n{M300}\n\n任务：逐字接龙念上面的材料，只输出材料原文。",
        "首尾各一次": f"任务：逐字接龙念下面的材料。\n材料：\n{M300}\n任务重申：逐字接龙念上面的材料。",
        "显式标签分隔": f"### 材料 ###\n{M300}\n### 任务 ###\n逐字复述上面材料原文。",
        "编号清单化": "1. 读下面的材料\n2. 逐字复述，一字不改\n3. 不输出任何别的字\n材料：" + M300,
    }
    chk = jielong_check(M300, 200)
    r2c = {name: chk(run(f"2C-{name}", prompt)) for name, prompt in forms.items()}
    result["2C_提示词形态"] = r2c

    # 2D:引用 #0151(零生成)
    result["2D_语料多约束"] = "1290 条中多约束 1 条(0.1%)——见 #0151,本轮零生成引用"

    json.dump(result, open(os.path.join(PROBE, "shrink_probes_汇总.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log.close()
    print(f"\n[汇总] {json.dumps(result, ensure_ascii=False, indent=1)}")
    print("[落盘] shrink_probes.jsonl + shrink_probes_汇总.json | 只测不改")


if __name__ == "__main__":
    main()
