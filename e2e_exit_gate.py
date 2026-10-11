"""e2e_exit_gate.py —— 出口硬兜真链路端到端验收(#0178 用户拍:在线路径首跑)

走 router 在线路径(auto_wire+真生成+真嵌入+真检索),不直调闸;纯本机零 L3;
本地生成([LocalModelName])串行一路、先预热;六类样本各一条,逐条留
"输入→线上判定→动作→前端最终文本",存"在线首跑基线"。
E2E 配置 = 生产 routing.yaml 副本 + rewrite.enabled + format_checks(样本3用);生产文件零改动。
跑法:.venv\\Scripts\\python.exe e2e_exit_gate.py
"""
from __future__ import annotations
# ── 路径去标识（不再硬编码用户名；可用环境变量 HR_DATA_ROOT 覆盖数据根）
import os as _hr_os
_HR_HOME = _hr_os.path.expanduser("~")
_HR_DESKTOP = _hr_os.environ.get("HR_DATA_ROOT") or _hr_os.path.join(_HR_HOME, "Desktop")
_HR_DIR = _hr_os.path.dirname(_hr_os.path.abspath(__file__))

import json
import os
import re
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# ── E2E 配置:生产副本 + rewrite 开 + format_checks(样本3) ──
import yaml
_base = open(os.path.join(BASE, "routing.yaml"), encoding="utf-8").read()
_cfg = yaml.safe_load(_base)
_cfg["rewrite"]["enabled"] = True
_cfg["exit_gate"]["format_checks"] = [{"intent": "knowledge_qa", "sections_min": 3}]
_CFG_PATH = os.path.join(BASE, "e2e_config.yaml")
with open(_CFG_PATH, "w", encoding="utf-8") as f:
    yaml.dump(_cfg, f, allow_unicode=True, sort_keys=False)

GEN_MODEL = "[LocalModelName]"


def local_gen(prompt: str, num_predict: int = 256, temperature: float = 0.4) -> str:
    body = json.dumps({"model": GEN_MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": temperature, "num_predict": num_predict}}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "").strip()


class LocalExecutor:
    """真本地生成端:ollama [LocalModelName],execute(req, rag_ctx) 走 router 生成契约。"""

    def __init__(self, name: str = "local-[LocalModelName]") -> None:
        self.name = name
        self.call_count = 0

    def execute(self, req, rag_ctx) -> str:
        self.call_count += 1
        prompt = ("你是本地会计知识助手。基于材料回答问题；材料里没有的就直说没有。\n"
                  + (f"[材料] {rag_ctx}\n" if rag_ctx else "")
                  + f"[问题] {req.text}")
        return local_gen(prompt)

    def is_available(self) -> bool:
        return True


class LocalL3Stub:
    """l3 替身(纯本机):rewrite_stage 的改写模式用它,避免连发 DSH/L3 通道。"""

    def __init__(self) -> None:
        self.call_count = 0

    def ask(self, prompt: str) -> str:
        self.call_count += 1
        return local_gen(prompt, num_predict=300)


class E2ERetriever:
    """真检索:E2E 语料(题库材料+条例卡)嵌入后按 cos 取 top-k。"""

    def __init__(self, chunks, embedder) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.vecs = []
        for c in chunks:
            try:
                self.vecs.append((c, embedder.embed(c)))
            except Exception as e:
                body = ""
                if hasattr(e, "read"):
                    try:
                        body = e.read().decode("utf-8", "replace")[:200]
                    except Exception:
                        pass
                print(f"  [检索器跳过块 {len(self.vecs)}] {type(e).__name__}: {body or e}", flush=True)
        self.calls: list[str] = []

    def retrieve(self, query: str, top_k: int = 5):
        self.calls.append(query)
        from interfaces import RetrievedChunk
        qv = self.embedder.embed(query)
        scored = sorted(((sum(x * y for x, y in zip(qv, v)), c) for c, v in self.vecs), reverse=True)
        return [RetrievedChunk(text=c, source="e2e_corpus", backend_rank=i)
                for i, (s, c) in enumerate(scored[:top_k])]



def _clean(text: str) -> str:
    """去控制字符(pdftotext 换页符  等会让 ollama /api/embed 报 400)。"""
    import re as _re
    bad = "[" + "".join(chr(c) for c in list(range(0, 9)) + [11] + list(range(14, 32))) + "]"
    return _re.sub(bad, "", text).strip()


def main() -> None:
    from config_loader import load_config
    from embed_local import Embedder
    from router import HybridRouter

    emb = Embedder(consumer="exit_gate")
    # 语料:30 份题库材料 + 增值税条例卡
    corpus = []
    gen = json.load(open(_hr_os.path.join(_HR_DESKTOP, r"send", r"题库_probe", r"生成物_[LocalModelName]"),
                         encoding="utf-8"))
    for q in gen:
        chunk = _clean(q["材料"][:600])
        if chunk:
            corpus.append(chunk)
    tdir = _hr_os.path.join(_HR_HOME, r"tiddlywiki", r"main", r"tiddlers")
    for fn in os.listdir(tdir):
        if fn.startswith("《增值税法实施条例》") and fn.endswith(".md"):
            t = open(os.path.join(tdir, fn), encoding="utf-8").read()
            chunk = _clean(t.split("## 原文（逐字）", 1)[1].split("## 锚点", 1)[0].strip())
            if chunk:
                corpus.append(chunk)

    # 预热(本地生成模型)
    t0 = time.time()
    local_gen("预热:只回两个字。", num_predict=8)
    print(f"[预热] {time.time()-t0:.0f}s", flush=True)

    rt = HybridRouter(config_path=_CFG_PATH, auto_wire_l2=True,
                      retrievers=[E2ERetriever(corpus, emb)], l3_provider=LocalL3Stub(),
                      executors={"local": LocalExecutor(), "cloud": LocalExecutor(name="local-[LocalModelName]位")})
    m_before = rt.metrics.snapshot()

    SAMPLES = [
        ("1_正常念稿(应PASS)", "把下面这段增值税规定原文照念：『增值税税率是13%，一般纳税人销售货物适用该税率。』"),
        ("2_纯壳(应Fail→重试→直出原文)", "把下面这段会计准则原文逐字念一遍：『企业前后各期采用的会计政策应保持一致，不得随意变更。』"),
        ("3_格式错(sections_min=3,应Fail)", "帮我讲讲递延所得税资产的确认。"),
        ("4_整跑题(应[答非所问])", "把下面这段增值税规定逐字念一遍：『增值税税率为13%。』"),
        ("5_如实没查到(应PASS)", "企业合并商誉的减值测试在增值税条例里是怎么规定的？"),
        ("6_数字被改(应数字闸拦)", "销售价款100万元的货物，按增值税条例帮我算销项税额是多少？"),
    ]
    baseline = []
    for tag, query in SAMPLES:
        m0 = rt.metrics.snapshot()
        out = rt.route(query)
        m1 = rt.metrics.snapshot()
        delta = {k: m1[k] - m0[k] for k in m1 if m1[k] != m0[k]}
        d = out.decision
        ans = out.answer
        header_leak = "<<答>>" in ans
        rec = {"样本": tag, "query": query[:40], "intent": d.intent,
               "target": d.target.value, "answer_head": ans[:60], "answer_len": len(ans),
               "head_leak": header_leak, "reasons": d.reasons[-3:], "计数器增量": delta,
               "guard": (out.guard.failed_criterion if out.guard else None)}
        baseline.append(rec)
        print(f"[{tag}]\n  intent={d.intent} target={d.target.value} 前端{len(ans)}字 头泄漏={header_leak}")
        print(f"  增量计数器: {delta}")
        print(f"  前端文本: {ans[:90]!r}\n", flush=True)

    # 三条单独确认
    print("== 单独确认 ==")
    leaks = sum(1 for r in baseline if r["head_leak"])
    print(f"①头泄漏: {leaks} 条(必须 0)")
    fl = os.path.join(BASE, "exit_gate_fails.jsonl")
    fl_rows = [json.loads(l) for l in open(fl, encoding="utf-8")] if os.path.exists(fl) else []
    two_col = all(("output" in r and "output_raw" in r) for r in fl_rows[-10:])
    print(f"②Fail 留档: {len(fl_rows)} 条,最近 10 条两栏齐全={two_col}")
    print(f"③计数器终值: { {k: v for k, v in rt.metrics.snapshot().items() if v and k.startswith(('exit_gate', 'direction'))} }")

    json.dump({"样本": baseline, "fail_log条数": len(fl_rows)},
              open(os.path.join(PROBE_DIR(), "在线首跑基线.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[基线落盘] send\\题库_probe\\在线首跑基线.json")


def PROBE_DIR():
    return _hr_os.path.join(_HR_DESKTOP, r"send", r"题库_probe")


if __name__ == "__main__":
    main()
