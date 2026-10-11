"""assembly_ablate.py —— A 主线:传话层关键指令强调(#0147 件2,#0152 后用户拍)

第一步:num_ctx 三档(2048/4096/8192)× 材料四档(100/300/600/1800字)——定材料能给多少。
第二步:组装层候选各自单独测(a 指令放尾部/b 首尾各一次/c 显式标签/d 编号清单化/e few-shot/
       f 材料预压缩/g 出口后处理核对),同一套接龙+复述探针。
第三步:把单测里翻成"服从"的候选组合成一个提示词,测组合版。
纪律:纯本机零 L3、预热、串行、OLLAMA_NUM_PARALLEL=1、模型/参数全走命令行;不改生产文件。
跑法:.venv\\Scripts\\python.exe scripts\assembly_ablate.py --model [LocalModelName]
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

import argparse
import json
import os
import re
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
PROBE = _hr_os.path.join(_HR_DESKTOP, r"send", r"题库_probe")


def api(model, prompt, temperature, num_predict, num_ctx=None, timeout=600):
    opts = {"temperature": temperature, "num_predict": num_predict}
    if num_ctx:
        opts["num_ctx"] = num_ctx
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": opts}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8")).get("response", "").strip()
        except (TimeoutError, OSError) as e:
            if attempt == 2:
                print(f"    [失败] {e}", flush=True)
                return ""
            print(f"    [超时重试] {e}", flush=True)
    return ""


def norm(s):
    return re.sub(r"\s+", "", s)


M300 = ("企业应当在资产负债表日判断资产是否存在可能发生减值的迹象。资产存在减值迹象的，应当进行减值测试，"
        "估计资产的可收回金额。可收回金额应当根据资产的公允价值减去处置费用后的净额与资产预计未来现金流量的"
        "现值两者之间较高者确定。减值测试时应当考虑货币时间价值和资产特定风险。")
M600 = M300 + ("已计提减值准备的资产价值又得以恢复的，不得转回。企业合并所形成的商誉和使用寿命不确定的无形资产，"
               "无论是否存在减值迹象，每年都应当进行减值测试。处置资产时，所计提的减值准备应当一并结转。"
               "对于尚未达到可使用状态的无形资产，由于其价值具有较大的不确定性，也应当每年进行减值测试。")
M1800 = (M600 + "资产负债表日后事项表明资产减值损失应当重新估计的，应当按照资产负债表日后事项的处理原则进行处理。"
         "企业在估计资产可收回金额时，应当以资产当前状况为基础，不应当包括与将来可能会发生的、尚未作出承诺的"
         "重组事项有关的预计未来现金流量。预计资产未来现金流量，应当以资产的当前状况为基础，不应当包括筹资活动"
         "和所得税收付产生的现金流量。折现率应当是反映当前市场货币时间价值和资产特定风险的税前利率。"
         "资产的公允价值减去处置费用后的净额，应当根据公平交易中销售协议价格减去可直接归属于该资产处置费用的"
         "金额确定。不存在销售协议但存在资产活跃市场的，应当按照该资产的市场价格减去处置费用后的金额确定。"
         "企业以资产组为基础进行减值测试的，应当按照资产组账面价值与可收回金额进行比较。")[:1800]
JIELONG = "逐字接龙念下面这段话，一个字不许改，从第一个字念到最后一个字：『{mat}』"


def jielong_ok(mat, out):
    return norm(mat)[:60] in norm(out) and len(norm(out)) >= max(100, len(norm(mat)) // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="[LocalModelName]")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--num-predict", type=int, default=256)
    args = ap.parse_args()
    log_path = os.path.join(PROBE, "assembly_ablate.jsonl")
    log = open(log_path, "w", encoding="utf-8")

    def run(tag, prompt, num_ctx=None):
        t0 = time.time()
        out = api(args.model, prompt, args.temperature, args.num_predict, num_ctx)
        log.write(json.dumps({"tag": tag, "num_ctx": num_ctx or 2048, "output": out[:600]},
                             ensure_ascii=False) + "\n")
        log.flush()
        print(f"  [{tag}] {time.time()-t0:.0f}s 出{len(norm(out))}字 | {out[:32]!r}", flush=True)
        return out

    print(f"[参数] model={args.model} temp={args.temperature} num_predict={args.num_predict}", flush=True)
    t0 = time.time()
    run("预热", "预热:只回两个字。", None)
    print(f"[预热] {time.time()-t0:.0f}s", flush=True)

    mats = (("短100", M300[:100]), ("中300", M300), ("长600", M600), ("超长1800", M1800))
    result = {"第一步_num_ctx": {}, "第二步_单候选": {}, "第三步_组合": {}}

    print("\n== 第一步 num_ctx × 材料四档(接龙) ==", flush=True)
    for nctx in (2048, 4096, 8192):
        for tag, mat in mats:
            out = run(f"numctx{nctx}-{tag}", JIELONG.format(mat=mat), num_ctx=nctx)
            result["第一步_num_ctx"][f"{tag}|ctx{nctx}"] = jielong_ok(mat, out)

    print("\n== 第二步 组装层候选单独测(接龙-中300,默认ctx) ==", flush=True)
    variants = {
        "基线_指令在首": JIELONG.format(mat=M300),
        "a_指令放尾部": f"材料：\n{M300}\n\n任务：逐字接龙念上面的材料，一个字不许改，只输出材料原文。",
        "b_首尾各一次": f"任务：逐字接龙念下面的材料，一个字不许改。\n材料：\n{M300}\n任务重申：逐字接龙念上面的材料，只输出材料原文。",
        "c_显式标签分隔": f"### 材料 ###\n{M300}\n### 任务 ###\n逐字复述上面材料原文，一字不改。",
        "d_编号清单化": "1. 读下面的材料\n2. 逐字复述，一字不改\n3. 除材料外不输出任何字\n材料：" + M300,
    }
    for name, prompt in variants.items():
        result["第二步_单候选"][name] = jielong_ok(M300, run(name, prompt))

    print("\n== 第二步 复述类候选单独测(重复-逐字复述) ==", flush=True)
    echo_prompt = "把下面这句话一个字不差地重复一遍，别的什么都不要说：『今天天气不错，适合散步。』"
    echo_ok = lambda o: "今天天气不错，适合散步。" in o and len(norm(o)) < 40
    result["第二步_单候选"]["基线_复述"] = echo_ok(run("基线_复述", echo_prompt))
    result["第二步_单候选"]["e_few-shot_复述"] = echo_ok(run("e_few-shot_复述",
        "示例：\n指令：把下面这句话一个字不差地重复一遍。\n材料：『账簿要日清月结。』\n输出：账簿要日清月结。\n"
        "现在：\n" + echo_prompt))

    print("\n== 第二步 g 出口后处理核对(对失败输出强化重试) ==", flush=True)
    fails = [(t, o) for t, o in
             [(v[0], None) for v in []]]  # 占位
    recovered, tried = 0, 0
    for name, prompt in variants.items():
        pass
    # 对基线接龙与基线复述各做一次"后处理重试"
    for label, bad_prompt, check in (
            ("接龙", JIELONG.format(mat=M300), lambda o: jielong_ok(M300, o)),
            ("复述", echo_prompt, echo_ok)):
        first = api(args.model, bad_prompt, args.temperature, args.num_predict)
        if check(first):
            continue
        tried += 1
        retry = api(args.model, f"你刚才的输出不符合要求(没有逐字复述材料)。请严格只输出材料原文,不要任何人设台词:\n{bad_prompt}",
                    args.temperature, args.num_predict)
        ok = check(retry)
        recovered += ok
        print(f"  g_{label}: 首次失败→重试{'成功' if ok else '仍失败'}")
    result["第二步_单候选"]["g_出口后处理核对"] = f"{recovered}/{tried} 翻修"

    print("\n== 第三步 组合版(单测里翻身的候选合并) ==", flush=True)
    combined = ("### 任务 ###\n1. 逐字复述下面的材料\n2. 一字不改、不总结、不回应\n3. 除材料原文外不输出任何字\n"
                "### 示例 ###\n材料：『今天天气不错。』\n输出：今天天气不错。\n"
                f"### 材料 ###\n{M300}\n### 任务重申 ###\n逐字复述上面的材料原文，一字不改。")
    result["第三步_组合"]["接龙-中300_组合版"] = jielong_ok(M300, run("组合版-接龙", combined))
    combined_echo = ("### 任务 ###\n把下面这句话一个字不差地重复一遍，别的什么都不要说。\n"
                     "### 示例 ###\n材料：『账簿要日清月结。』\n输出：账簿要日清月结。\n"
                     "### 本题 ###\n把下面这句话一个字不差地重复一遍：『今天天气不错，适合散步。』")
    result["第三步_组合"]["复述_组合版"] = echo_ok(run("组合版-复述", combined_echo))

    json.dump(result, open(os.path.join(PROBE, "assembly_ablate_汇总.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log.close()
    print(f"\n[汇总] {json.dumps(result, ensure_ascii=False, indent=1)}")
    print(f"[落盘] {log_path} + assembly_ablate_汇总.json | 不落生产,等用户拍")


if __name__ == "__main__":
    main()
