"""ifollow_probes.py —— 指令跟随专题:探针集+四档消融(#0146,只测不改)

任务1 探针集:五类约束 12 条(重复/格式/长度/内容/接龙),[LocalModelName] 与 v3-orig 对照,
     每条程序化判 服从/部分服从/不服从 + 失败形态。
任务2 消融(A窗口:材料 100/300/600/1800 字 × num_ctx 默认2048/4096;B温度 0.2/0.4/0.8;
     C提示词形态:自然语言 vs 结构化+few-shot;D语料侧多约束统计——零生成)。
任务3 参数三处清单见回执(本脚本只测)。
纪律:纯本机、零 L3、预热、串行、OLLAMA_NUM_PARALLEL=1;--model/--num-context/--temperature 全参数化。
跑法:.venv\\Scripts\\python.exe scripts\ifollow_probes.py --model [LocalModelName]
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


def api(model: str, prompt: str, temperature: float, num_predict: int,
        num_ctx: int | None = None, timeout: float = 600, system: str | None = None) -> str:
    opts: dict = {"temperature": temperature, "num_predict": num_predict}
    if num_ctx:
        opts["num_ctx"] = num_ctx
    body: dict = {"model": model, "prompt": prompt, "stream": False, "options": opts}
    if system is not None:
        body["system"] = system          # 覆盖 Modelfile 的人设系统提示(去人设消融用)
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    for attempt in (1, 2):                   # 超时重试一次(换 system/ctx 时偶发 KV 重分配慢)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8")).get("response", "").strip()
        except (TimeoutError, urllib.error.URLError) as e:
            if attempt == 2:
                raise
            print(f"    [超时重试] {e}", flush=True)
    return ""


def norm(s: str) -> str:
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


def json_ok(o: str) -> bool:
    m = re.search(r"\{[^{}]*\}", o)
    if not m:
        return False
    try:
        return json.loads(m.group(0)) == {"rating": 5}
    except Exception:
        return False


# (tag, prompt 模板或固定文本, checker, 失败形态备注)
PROBES = [
    ("重复-逐字复述", "把下面这句话一个字不差地重复一遍，别的什么都不要说：『今天天气不错，适合散步。』",
     lambda o: "服从" if ("今天天气不错，适合散步。" in o and len(norm(o)) < 40) else "不服从", "漏做/内容扭曲"),
    ("重复-说三遍", "把『小鱼干真好吃』连说三遍。",
     lambda o: "服从" if norm(o).count("小鱼干真好吃") == 3 else ("部分服从" if norm(o).count("小鱼干真好吃") >= 1 else "不服从"), "漏做/次数错"),
    ("格式-JSON", "只输出一个 JSON 对象，键 rating 值为数字 5，别的什么都不要输出。",
     lambda o: "服从" if json_ok(o) else "不服从", "格式丢/多话"),
    ("格式-列表", "用列表回答，共三行，每行一个词：猫、狗、鸟。",
     lambda o: "服从" if (all(w in o for w in ("猫", "狗", "鸟")) and len(norm(o)) < 40) else ("部分服从" if all(w in o for w in ("猫", "狗", "鸟")) else "不服从"), "约束丢/多话"),
    ("格式-每行一句", "把下面的三件事每行一句输出：吃饭；睡觉；吃小鱼干。",
     lambda o: "服从" if (len([l for l in o.splitlines() if l.strip()]) >= 3 and all(w in o for w in ("吃饭", "睡觉", "小鱼干"))) else "部分服从", "格式丢"),
    ("长度-两字", "只回答两个字：你好。别的一个字都不要。",
     lambda o: "服从" if (len(norm(o)) <= 6 and "你好" in o) else "不服从", "约束丢/多话"),
    ("长度-20字", "用不超过 20 个字介绍什么是折旧。",
     lambda o: "服从" if len(norm(o)) <= 24 else ("部分服从" if len(norm(o)) <= 40 else "不服从"), "超长"),
    ("内容-禁英文", "介绍什么是折旧，全文禁止出现任何英文字母。",
     lambda o: "服从" if (o and not re.search(r"[A-Za-z]", o)) else "不服从", "约束丢"),
    ("内容-带年份", "回答必须带上年份『2×23 年』：无形资产摊销从什么时候开始？",
     lambda o: "服从" if ("2×23" in o or "2X23" in norm(o)) else "不服从", "约束丢"),
    ("内容-禁白米饭", "谈谈你爱吃的食物，回答里禁止出现『白米饭』三个字。",
     lambda o: "服从" if (o and "白米饭" not in o) else "不服从", "约束丢(人设词陷阱)"),
]
JIELONG = "逐字接龙念下面这段话，一个字不许改，从第一个字念到最后一个字：『{mat}』"


def jielong_check(mat: str, min_out: int):
    def chk(o: str) -> str:
        head = norm(mat)[:60]
        if head in norm(o) and len(norm(o)) >= min_out:
            return "服从"
        if len(norm(o)) > min_out // 3:
            return "部分服从"
        return "不服从"
    return chk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="[LocalModelName]")
    ap.add_argument("--orig-model", default="[LocalModelName]")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--num-context", type=int, default=None)
    args = ap.parse_args()
    print(f"[参数] model={args.model} orig={args.orig_model} temp={args.temperature} "
          f"num_predict={args.num_predict} num_ctx={args.num_context or 'Modelfile默认(2048)'}")

    t0 = time.time()
    api(args.model, "预热:只回两个字。", 0.0, 8, timeout=180)
    print(f"[预热] {time.time()-t0:.0f}s", flush=True)

    log_path = os.path.join(PROBE, f"ifollow_{args.model.replace(':', '_')}.jsonl")
    log = open(log_path, "w", encoding="utf-8")
    all_rows = []

    def run(model, tag, prompt, temp=None, num_ctx=None, system=None):
        out = api(model, prompt, args.temperature if temp is None else temp,
                  args.num_predict, num_ctx, system=system)
        rec = {"model": model, "probe": tag, "temp": temp or args.temperature,
               "num_ctx": num_ctx or 2048, "system_override": system is not None,
               "output": out[:500]}
        log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()
        all_rows.append(rec)
        return out

    # 任务1:探针集 × 三臂(默认人设 / 去人设 system 覆盖 / v3-orig)
    print("\n== 任务1 探针集(五类,v3人设 / v3去人设 / v3-orig) ==", flush=True)
    t1 = {}
    NO_PERSONA = "你是本地会计知识助手。严格按用户指令输出，不要任何人设台词。"
    arms = [(args.model, "人设默认", None), (args.model, "去人设", NO_PERSONA),
            (args.orig_model, "v3-orig默认", None)]
    for model, arm_name, sysv in arms:
        rows = []
        for tag, prompt, checker, failmode in PROBES:
            v = checker(run(model, f"{arm_name}|{tag}", prompt, system=sysv))
            rows.append((tag, v, failmode))
            print(f"  [{arm_name}] {tag:<9} {v:<5} | {all_rows[-1]['output'][:34]!r}")
        vj = jielong_check(M300, 200)(run(model, "接龙-短", JIELONG.format(mat=M300)))
        rows.append(("接龙-短", vj, "复读指令/漏做/截断"))
        print(f"  [{model.split(':')[0]}] 接龙-短   {vj:<5}")
        rates = {k: sum(1 for _, v, _ in rows if v.startswith(k)) for k in ("服从", "部分服从", "不服从")}
        t1[arm_name] = {"rates": rates, "rows": rows}
        print(f"  → {arm_name}: {rates}")

    # 任务2A 窗口:接龙材料四档 × num_ctx 默认/4096/8192(nctx 外层循环,避免 KV 重分配抖动)
    print("\n== 任务2A 窗口(材料长度 × num_ctx) ==", flush=True)
    t2a = {}
    mats = (("短100字", M300[:100]), ("中300字", M300), ("长600字", M600), ("超长1800字", M1800))
    for nctx in (None, 4096, 8192):
        for mat_tag, mat in mats:
            v = jielong_check(mat, max(100, len(norm(mat)) // 2))(run(args.model, f"接龙-{mat_tag}", JIELONG.format(mat=mat), num_ctx=nctx))
            t2a[f"{mat_tag}|ctx{nctx or 2048}"] = v
            print(f"  ctx={nctx or 2048} {mat_tag}: {v} | 出 {len(norm(all_rows[-1]['output']))} 字")

    # 任务2B 温度(6 条代表探针)
    print("\n== 任务2B 温度 ==", flush=True)
    sub = [p for p in PROBES if p[0] in ("格式-JSON", "长度-20字", "内容-禁英文", "重复-说三遍", "格式-列表", "内容-带年份")]
    t2b = {}
    for temp in (0.2, 0.4, 0.8):
        ok = sum(1 for tag, prompt, checker, _ in sub if checker(run(args.model, tag, prompt, temp=temp)) == "服从")
        t2b[temp] = f"{ok}/{len(sub)}"
        print(f"  temp={temp}: 服从 {ok}/{len(sub)}")

    # 任务2C 提示词形态(接龙-中)
    print("\n== 任务2C 提示词形态 ==", flush=True)
    chk300 = jielong_check(M300, 200)
    v1 = chk300(run(args.model, "接龙-自然语言", JIELONG.format(mat=M300)))
    structured = ("任务：逐字复述材料。\n规则：\n1. 从第一字念到最后一字\n2. 不许改动、不许总结、不许回应\n"
                  "3. 输出只有材料本身\n示例：\n材料：『今天天气不错。』\n输出：今天天气不错。\n"
                  f"材料：『{M300}』\n输出：")
    v2 = chk300(run(args.model, "接龙-结构化", structured))
    t2c = {"自然语言": v1, "结构化+few-shot": v2}
    print(f"  自然语言: {v1} | 结构化+few-shot: {v2}")

    # 任务2E 组装层候选(#0147 件2):在失败探针上逐个试,报哪几条翻成服从
    print("\n== 任务2E 组装层候选(指令位置/标签/清单/few-shot/后处理) ==", flush=True)
    t2e = {}
    # 候选 a 指令放尾部 / b 首尾各一次 / c 显式标签分隔 / d 编号清单化 —— 都打"接龙-短"这条已知失败探针
    variants = {
        "a_指令放尾部": f"材料：\n{M300}\n\n任务：逐字接龙念上面的材料，一个字不许改，只输出材料原文。",
        "b_首尾各一次": f"任务：逐字接龙念下面的材料，一个字不许改。\n材料：\n{M300}\n任务重申：逐字接龙念上面的材料，只输出材料原文。",
        "c_显式标签分隔": f"### 材料 ###\n{M300}\n### 任务 ###\n逐字复述上面材料原文，一字不改。",
        "d_编号清单化": "1. 读下面的材料\n2. 逐字复述，一字不改\n3. 除材料外不输出任何字\n材料：" + M300,
    }
    for name, prompt in variants.items():
        v = jielong_check(M300, 200)(run(args.model, f"2E-{name}", prompt))
        t2e[name] = v
        print(f"  {name}: {v}")
    # 候选 e 一条 few-shot(打"重复-逐字复述"失败探针)
    fs = ("示例：\n指令：把下面这句话一个字不差地重复一遍。\n材料：『账簿要日清月结。』\n输出：账簿要日清月结。\n"
          "现在：\n把下面这句话一个字不差地重复一遍，别的什么都不要说：『今天天气不错，适合散步。』")
    v = ("服从" if "今天天气不错，适合散步。" in run(args.model, "2E-few-shot", fs) and len(norm(all_rows[-1]['output'])) < 40
         else "不服从")
    t2e["e_few-shot_重复复述"] = v
    print(f"  e_few-shot_重复复述: {v}")
    # 候选 f 材料预压缩(打"忠读长材料"任务:压缩后再逐字,任务本身改写,仅参考)
    comp = M1800[:600]
    v = ("服从" if norm(comp)[:60] in norm(run(args.model, "2E-材料预压缩", f"逐字接龙念：『{comp}』"))
         else "不服从")
    t2e["f_材料预压缩(600字)"] = v
    print(f"  f_材料预压缩(600字): {v}")
    # 候选 g 出口后处理核对(程序化:对本次已记录的失败输出做检测+一次强化重试,数翻修率)
    fails = [r for r in all_rows if r["probe"] in ("重复-逐字复述", "重复-说三遍", "格式-JSON", "长度-两字")]
    recovered = 0
    for r in fails:
        retry = api(args.model, f"你刚才的输出不符合指令要求（{r['probe']}）。请严格只按指令输出，不要任何人设台词：\n"
                                + next(p for t, p, _, _ in PROBES if t == r["probe"]),
                    args.temperature, args.num_predict, num_ctx=None)
        ok = {"重复-逐字复述": lambda o: "今天天气不错，适合散步。" in o and len(norm(o)) < 40,
              "重复-说三遍": lambda o: norm(o).count("小鱼干真好吃") == 3,
              "格式-JSON": json_ok,
              "长度-两字": lambda o: len(norm(o)) <= 6 and "你好" in o}[r["probe"]](retry)
        recovered += ok
        print(f"  g_后处理重试[{r['probe']}]: {'翻修成功' if ok else '仍失败'}")
    t2e["g_后处理核对翻修"] = f"{recovered}/{len(fails)}"
    print(f"  后处理核对翻修率: {recovered}/{len(fails)}")

    # 任务2D 语料侧多约束统计(零生成)
    print("\n== 任务2D 语料侧多约束样本统计 ==", flush=True)
    corpus = [json.loads(l) for l in open(_hr_os.path.join(_HR_DESKTOP, r"工程", r"[LocalModelName]"),
                                          encoding="utf-8") if l.strip()]
    kinds = [r"不超过|以内|最多", r"只输出|只回答|只说", r"必须|一定|务必", r"禁止|不许|别|不要",
             r"每行|列表|一条|分点", r"三遍|逐字|一个字不差"]
    multi = sum(1 for r in corpus if sum(1 for k in kinds if re.search(k, r["instruction"])) >= 2)
    single = sum(1 for r in corpus if sum(1 for k in kinds if re.search(k, r["instruction"])) == 1)
    print(f"  语料 {len(corpus)} 条:含≥2类约束指令 {multi} 条({multi/len(corpus):.1%}),仅1类 {single} 条,"
          f"纯问答 {len(corpus)-multi-single} 条")

    json.dump({"t1": {m: {"rates": v["rates"], "rows": [list(r) for r in v["rows"]]} for m, v in t1.items()},
               "t2a": t2a, "t2b": {str(k): v for k, v in t2b.items()}, "t2c": t2c, "t2e": t2e,
               "t2d": {"总数": len(corpus), "多约束": multi, "单约束": single}},
              open(os.path.join(PROBE, f"ifollow_汇总_{args.model.replace(':', '_')}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log.close()
    print(f"\n[落盘] {log_path} + ifollow_汇总_*.json | 只测不改,零生产配置,零 L3")


if __name__ == "__main__":
    main()
