# -*- coding: utf-8 -*-
"""anchors.py —— 财会文本的「事实锚点」抽取与比对（纯 re，零依赖）。

用户 2026-10-09 定的口径：财会科目、条款号、数字，全部用 Python `re` 去死磕；
不指望模型算数（底层结构使然，换更简单的方式）。

用途（两边共用，避免两套判据）：
  ①router 的 L3 出口质量门「事实闸」：答案 vs 注入的知识库原文
  ②判分脚本的客观分：抽样比对 + 缩小人工复核范围

设计原则（吸取 2026-09-29「机械 QC 假阳性」的教训）：
  * **宁漏不误**：机械判据只用来缩小人工复核范围，报出来的条目仍要回原文核；
  * **先归一化再比**：全角/半角、中文数字/阿拉伯数字、千分位、百分号都要先归一；
  * **只报"缺什么"，不判"对不对"**：事实错由人来定性，程序只负责指出不一致的位置。
"""
from __future__ import annotations

import re
from typing import Iterable

# ---------------------------------------------------------------- 归一化

_FULLWIDTH = {c: c - 0xFEE0 for c in range(0xFF01, 0xFF5F)}  # 全角 ASCII → 半角
_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}


def to_halfwidth(text: str) -> str:
    return text.translate(_FULLWIDTH)


def cn_to_int(token: str) -> int | None:
    """把「三」「十五」「二十万」「一亿二千万」这类中文数字转成 int；失败返回 None。"""
    token = token.strip()
    if not token or any(ch not in _CN_DIGIT and ch not in _CN_UNIT for ch in token):
        return None
    total, section, number = 0, 0, 0
    for ch in token:
        if ch in _CN_DIGIT:
            number = _CN_DIGIT[ch]
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            if unit >= 10000:                      # 万/亿：结算当前段
                section = (section + number) * unit
                total += section
                section, number = 0, 0
            else:                                   # 十/百/千
                if number == 0:
                    number = 1                      # 「十五」= 15
                section += number * unit
                number = 0
    return total + section + number


_NUM_RE = re.compile(
    r"""
    (?P<cn>[零〇一二两三四五六七八九十百千万亿]{1,12})
    |
    (?P<ar>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    """,
    re.VERBOSE,
)


def extract_numbers(text: str) -> list[str]:
    """抽出所有数字锚点（归一化为「十进制字符串」形态，保留小数；百分号另记为 %）。"""
    out: list[str] = []
    half = to_halfwidth(text)
    for m in _NUM_RE.finditer(half):
        cn, ar = m.group("cn"), m.group("ar")
        if ar:
            out.append(ar.replace(",", ""))
        else:
            v = cn_to_int(cn)
            if v is not None:
                out.append(str(v))
    # 百分号紧跟数字时，补一个「N%」形态，便于和没有 % 的写法区分
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", half):
        out.append(m.group(1) + "%")
    return out


# ---------------------------------------------------------------- 条款号

_CLAUSE_RES = [
    re.compile(r"第\s*[零〇一二两三四五六七八九十百千0-9]+\s*条"),
    re.compile(r"第\s*[零〇一二两三四五六七八九十百千0-9]+\s*号"),
    re.compile(r"\bCAS\s*\d+", re.I),
    re.compile(r"\bIFRS\s*\d+", re.I),
    re.compile(r"\bIAS\s*\d+", re.I),
    re.compile(r"企业会计准则第\s*\d+\s*号"),
]


def extract_clauses(text: str) -> list[str]:
    """抽条款号：第N条／第N号／CAS N／IFRS N／IAS N／企业会计准则第N号（去空格、统一大写）。"""
    half = to_halfwidth(text)
    out: list[str] = []
    for rx in _CLAUSE_RES:
        for m in rx.finditer(half):
            out.append(re.sub(r"\s+", "", m.group(0)).upper())
    return out


# ---------------------------------------------------------------- 财会科目

# 最小内置词表（够冒烟用；正式标定时再从 wiki 的 60 张财会卡里自动抽）
DEFAULT_ACCOUNTS = [
    "递延所得税资产", "递延所得税负债", "使用权资产", "租赁负债", "合同资产", "合同负债",
    "投资性房地产", "固定资产", "在建工程", "无形资产", "商誉", "长期股权投资",
    "应收票据", "应收账款", "预付款项", "其他应收款", "存货", "交易性金融资产",
    "其他债权投资", "其他权益工具投资", "债权投资", "长期待摊费用",
    "短期借款", "应付票据", "应付账款", "预收款项", "应付职工薪酬", "应交税费",
    "其他应付款", "长期借款", "应付债券", "预计负债", "实收资本", "资本公积",
    "盈余公积", "未分配利润", "其他综合收益", "营业收入", "营业成本", "税金及附加",
    "销售费用", "管理费用", "研发费用", "财务费用", "公允价值变动收益", "投资收益",
    "信用减值损失", "资产减值损失", "资产处置收益", "其他收益", "所得税费用",
    "递延收益", "营业收入（主营业务收入）", "营业利润", "利润总额", "净利润",
]


def extract_accounts(text: str, vocab: Iterable[str] | None = None) -> list[str]:
    """在文本里找词表命中的财会科目（长词优先，避免「应付账款」被「账款」重复计）。"""
    words = sorted(set(vocab or DEFAULT_ACCOUNTS), key=len, reverse=True)
    hits: list[str] = []
    for w in words:
        if w in text and not any(w in h for h in hits):
            hits.append(w)
    return hits


# ---------------------------------------------------------------- 比对

def _set(items: Iterable[str]) -> set[str]:
    return {s for s in items if s}


def anchor_compare(material: str, answer: str, vocab: Iterable[str] | None = None) -> dict:
    """把「材料里有的锚点」和「答案里出现的锚点」对照。

    返回各维度的 覆盖（答案盖住材料多少）/ 新增（答案里有、材料里没有 = 可疑）。
    ⚠️ 只报事实，不判对错：`missing` 里可能只是说法不同（人工核），`extra` 里也可能是合理补充。
    """
    res: dict = {}
    for name, fn in (("num", extract_numbers),
                     ("clause", extract_clauses),
                     ("account", lambda t: extract_accounts(t, vocab))):
        m, a = _set(fn(material)), _set(fn(answer))
        covered = m & a
        res[name] = {
            "material": sorted(m),
            "answer": sorted(a),
            "missing_in_answer": sorted(m - a),   # 材料有、答案没提 → 可能漏念
            "extra_in_answer": sorted(a - m),     # 答案有、材料没有 → 可能胡编
            "coverage": (len(covered) / len(m)) if m else None,
        }
    return res


def anchors_text(material: str, answer: str, vocab: Iterable[str] | None = None) -> str:
    """给人看的一行摘要（判分脚本/日志用）。"""
    r = anchor_compare(material, answer, vocab)
    parts = []
    for k in ("num", "clause", "account"):
        v = r[k]
        cov = "n/a" if v["coverage"] is None else f"{v['coverage']:.2f}"
        parts.append(f"{k}:覆盖{cov} 漏{len(v['missing_in_answer'])} 多{len(v['extra_in_answer'])}")
    return " | ".join(parts)


# ---------------------------------------------------------------- 用户的三条"截杀"规则
# 用户 2026-10-09 定（写死在 router 出口，命中即拒）：
#   ①否定词一致性：回答里有否定词 ⇒ 原文对应处必须有否定；对不上 ⇒ 拒
#   ②财务方向词：资产/负债、借方/贷方 等方向对不上 ⇒ 拒
#   ③数字：回答里"新增"的数字（原文没有）⇒ 拒（只查新增，不查没提到的）
# 设计取舍：三条都要"配对"——不能只做集合比对；配对用最简办法（按句切 + 关键词重叠），
# 零嵌入、零依赖。⚠️ 有假阳性风险 ⇒ action 可配 deny|review，证据全部落下来备查。

_SENT = re.compile(r"[。！？!?\n；;]")
_SENT_SOFT = re.compile(r"[，,、]")
# 否定词：⚠️ 只用多字词，**不要单字 `不/无/非`** —— 否则「不动产/非正常损失/无形资产」这类
# 财会术语会被当成否定词（实测踩过）。
NEG_RE = re.compile(r"不计提|不得|不能|不应|不予|无需|无须|不属于|并非|没有|不会|未曾|未")
# 方向对（前一个出现、原文对应句却只有后一个 ⇒ 方向反了）
DIR_PAIRS = [("资产", "负债"), ("借方", "贷方"), ("增加", "减少"), ("确认", "冲销")]
_STOP = {"我们", "可以", "如果", "这个", "那个", "他们", "因为", "所以", "以及", "并且", "但是", "其中"}


def _sentences(text: str) -> list[str]:
    """先按句末切；**长句再用逗号切细** —— 否则「A 不得抵扣，B 减按 1%」这种
    多主题长句会整段当一个 token，配对必然错（实测假阳性就是这么来的）。"""
    out: list[str] = []
    for s in _SENT.split(text):
        s = s.strip()
        if not s:
            continue
        if len(s) > 20:
            out += [x.strip() for x in _SENT_SOFT.split(s) if x.strip()]
        else:
            out.append(s)
    return out


def _key_tokens(s: str) -> set[str]:
    """句子的关键词：整段词 + **2-gram 滑窗**（滑窗是关键，否则同义改写两边对不上）。"""
    toks = set(re.findall(r"[\u4e00-\u9fa5]{2,}", s))
    grams: set[str] = set()
    for t in toks:
        grams |= {t[i:i + 2] for i in range(len(t) - 1)}
    toks |= grams
    toks |= set(extract_numbers(s)) | set(extract_accounts(s))
    return {t for t in toks if t not in _STOP}


def _anchors_of(s: str) -> set[str]:
    """**配对锚**：3-gram 滑窗 + 数字 + 科目 + 条款号。

    为什么用 3-gram：2-gram 太泛（"抵扣"这种会把两个不相干的主题拉在一起 ⇒ 错配），
    3-gram 既能容同义改写（"非正常损失的进项税额" 与 "非正常损失对应的进项税额" 共享
    "非正常/正常损/常损失/进项税/项税额"），又不会跨主题乱配。
    """
    out: set[str] = set()
    for t in re.findall(r"[\u4e00-\u9fa5]{3,}", s):
        out |= {t[i:i + 3] for i in range(len(t) - 2)}
    out |= set(extract_numbers(s)) | set(extract_accounts(s)) | set(extract_clauses(s))
    return out


def _best_match(sent: str, pool: list[str], min_overlap: int = 1):
    """给一句话，在 pool 里找对应句：**必须共享至少一个配对锚**，否则返回 None。

    （错配正是上一版假阳性的主因：只按"重叠最多"选 ⇒ 长句的另一半也会被选中。）
    """
    sa = _anchors_of(sent)
    st = _key_tokens(sent)
    best, score = None, 0
    for other in pool:
        if not (sa & _anchors_of(other)):
            continue
        sc = len(st & _key_tokens(other))
        if sc > score:
            best, score = other, sc
    return best if score >= min_overlap else None


def check_negation(material: str, answer: str) -> dict:
    """①否定词一致性（**双向**）。

    正向：回答有否定 ⇒ 材料对应句也得有（防"编个不存在的否定"）。
    反向：材料有否定、而回答**谈到了同一主题**却没否定 ⇒ 也拒
          （防"不得→可以 / 不计提→计提"这类致命错；⚠️ **只在回答确实谈到该主题时判**，
           回答合法省略的主题不算违规 —— 这是我的判断，可改）。
    """
    ms, hits = _sentences(material), []
    for s in _sentences(answer):
        m = _best_match(s, ms)
        if NEG_RE.search(s):
            if m is None:
                hits.append({"side": "answer→material", "answer_sentence": s, "matched": None,
                             "reason": "回答里有否定，但材料里找不到对应句"})
            elif not NEG_RE.search(m):
                hits.append({"side": "answer→material", "answer_sentence": s, "matched": m,
                             "reason": "材料对应句里没有否定词"})
    for m in ms:
        if not NEG_RE.search(m):
            continue
        a = _best_match(m, _sentences(answer))
        if a is not None and not NEG_RE.search(a):
            hits.append({"side": "material→answer", "material_sentence": m, "matched": a,
                         "reason": "材料有否定、回答谈到同一主题却把否定丢了（不得→可以 类）"})
    return {"rule": "negation", "violations": hits, "deny": bool(hits)}


def check_direction(material: str, answer: str, pairs=None) -> dict:
    """pairs: 方向词对列表(可配,#0186);缺省用内置 DIR_PAIRS。"""
    pairs = pairs or DIR_PAIRS
    """②财务方向词一致性（**双向**）。

    回答说是「资产」而材料对应句只说「负债」（或反之）⇒ 拒。
    ⚠️ 上一版只查了单向（"回答说资产、材料只有负债"），漏掉了最常见的"把资产说成负债"。
    """
    ms, hits = _sentences(material), []
    for a, b in pairs:
        for s in _sentences(answer):
            m = _best_match(s, ms)
            if m is None:
                continue
            if a in s and b in m and a not in m:
                hits.append({"pair": [a, b], "answer_sentence": s, "matched": m,
                             "reason": f"回答说是「{a}」，材料对应句只说「{b}」"})
            elif b in s and a in m and b not in m:
                hits.append({"pair": [b, a], "answer_sentence": s, "matched": m,
                             "reason": f"回答说是「{b}」，材料对应句只说「{a}」"})
    return {"rule": "direction", "violations": hits, "deny": bool(hits)}


def check_numbers(material: str, answer: str) -> dict:
    """③数字只许有出处：回答里**新增**的数字 ⇒ 拒。

    ⚠️ 只查"新增"（extra），**不查"没提到"** —— 合法的部分摘录本来就少提。
    允许 `1` 与 `1%` 互认（百分号形态折算后再比一次）。
    """
    mn, an = set(extract_numbers(material)), set(extract_numbers(answer))
    strip_pct = lambda x: x[:-1] if x.endswith("%") else x  # noqa: E731
    mn_core = {strip_pct(x) for x in mn}
    extra = sorted(x for x in (an - mn) if strip_pct(x) not in mn_core)
    return {"rule": "numbers", "extra_numbers": extra, "deny": bool(extra)}


def guard_check(material: str, answer: str, action: str = "deny") -> dict:
    """三条截杀规则的总入口（给 router 出口用）。

    action="deny"（用户要的默认）⇒ 命中即拒；action="review" ⇒ 只标记不拒（先统计假阳性用）。
    返回 {"deny", "action", "fired", "violations", "summary"}。
    """
    res = {"negation": check_negation(material, answer),
           "direction": check_direction(material, answer),
           "numbers": check_numbers(material, answer)}
    fired = [k for k, v in res.items() if v["deny"]]
    counts = []
    for k in fired:
        v = res[k]
        counts.append(f"{k}×{len(v.get('violations') or v.get('extra_numbers') or [])}")
    return {"deny": bool(fired), "action": action, "fired": fired,
            "violations": res, "summary": ("; ".join(counts) or "clean")}


if __name__ == "__main__":
    material = "《企业会计准则第18号》规定，可抵扣暂时性差异应确认递延所得税资产。"
    good = "递延所得税资产确认要满足条件，《企业会计准则第18号》里有规定。"
    bad = "递延所得税负债按规定确认即可，第 22 号准则。"
    print("to_halfwidth :", to_halfwidth("１２３％ＡＢＣ"))
    print("cn_to_int    :", cn_to_int("十五"), cn_to_int("二十万"), cn_to_int("一亿二千万"))
    print("numbers      :", extract_numbers("税率按 20% 计算，共 1,200.50 元，另有十四五条"))
    print("clauses      :", extract_clauses("见《企业会计准则第18号》与 CAS 14、ifrs16 第 3 条"))
    print("accounts     :", extract_accounts("递延所得税资产和递延所得税负债都要看使用权资产"))
    print("GOOD  :", anchors_text(material, good))
    print("BAD   :", anchors_text(material, bad))
    # 冒烟断言：好答案不该比坏答案差
    g = anchor_compare(material, good)["num"]["coverage"]
    b = anchor_compare(material, bad)["num"]["coverage"]
    print("num coverage good/bad =", g, "/", b)
    assert (g or 0) >= (b or 0), "好答案的数字覆盖不该低于坏答案"

    # ---- 三条截杀规则的自检（用户 2026-10-09 定的那三条）----
    m2 = ("小规模纳税人增值税征收率减按 1% 征收，月销售额 10 万元以下免征；"
          "非正常损失不得抵扣进项税额；可抵扣暂时性差异应确认递延所得税资产。")
    cases = [
        ("正-照抄", "征收率减按 1%，月销售额 10 万元以下免征；非正常损失不得抵扣进项税额。"),
        ("反-改数字", "征收率减按 3%，月销售额 30 万元以下免征。"),
        ("反-否定丢了", "非正常损失可以抵扣进项税额。"),
        ("反-方向反", "该可抵扣暂时性差异应确认递延所得税负债。"),
        ("正-合法省略", "征收率减按 1%。"),
        ("正-口语改写", "按现行规定：非正常损失不得抵扣进项税额，征收率减按 1%。"),
    ]
    print()
    for tag, a in cases:
        gg = guard_check(m2, a)
        print(f"  {tag:10s} deny={str(gg['deny']):5s} fired={gg['fired']}  {gg['summary']}")
    # 三条反例必须全被拒；三条正例必须放行（本轮规则的验收线）
    assert guard_check(m2, cases[0][1])["deny"] is False, "照抄正例不该被拒"
    assert guard_check(m2, cases[1][1])["deny"] is True, "改数字必须被拒"
    assert guard_check(m2, cases[2][1])["deny"] is True, "否定丢了必须被拒"
    assert guard_check(m2, cases[3][1])["deny"] is True, "方向反了必须被拒"
    assert guard_check(m2, cases[4][1])["deny"] is False, "合法省略不该被拒"
    assert guard_check(m2, cases[5][1])["deny"] is False, "口语改写不该被拒"
    print()
    print("OK")


# ---------------------------------------------------------------- 可推导数集(#0186 用户拍)
# (b) 换刀口:答案数字"可推导即放行"——三选一:(i)在问题里 (ii)在材料里
#     (iii)可由问题/材料里的数字推导(× ÷ 百分比 ±,容差相对 1%)。
# 规则表(ops)进配置;本函数零嵌入、零外部依赖。

DERIVABLE_OPS_DEFAULT = ("add", "sub", "mul", "div", "pct")


def _num_floats(text: str) -> list[float]:
    """extract_numbers 的浮点视图:'13%' 记 13 与 0.13 两形;'万'位归一化后的 0 值丢弃。"""
    out = []
    for tok in extract_numbers(text):
        v = None
        if tok.endswith("%"):
            try:
                v = float(tok[:-1])
            except ValueError:
                continue
            out.append(v)
            out.append(v / 100.0)          # 百分比两形:13 与 0.13
        else:
            try:
                v = float(tok)
            except ValueError:
                continue
            if v != 0:
                out.append(v)
    return out


def derivable_numbers(query: str, material: str, ops: Iterable[str] | None = None) -> set[float]:
    """可由 问题/材料 里的数字经 × ÷ 百分比 ± 推出的数集(浮点)。

    一轮 pairwise(不递归,防爆炸);0 不作操作数(0 只会产 0/重复)。
    单位归一(#0188):每个结果附 千元/万元 两形(×1000/×10000 及其反),
    「0.85 万元」与「8500 元」才能在集合里相遇。
    """
    use = tuple(ops) if ops else DERIVABLE_OPS_DEFAULT
    vals = [v for v in (_num_floats(query) + _num_floats(material)) if v != 0]
    out: set[float] = set(vals)
    for i, a in enumerate(vals):
        for b in vals[i + 1:]:
            if "add" in use:
                out.add(a + b)
                out.add(abs(a - b))
            if "mul" in use:
                out.add(a * b)
            if "div" in use and b != 0:
                out.add(a / b)
            if "pct" in use:
                out.add(a * b / 100.0)
    # 单位两形(#0188):元 ↔ 千元/万元
    scaled: set[float] = set()
    for v in out:
        for s in (1000.0, 10000.0):
            scaled.add(v * s)
            scaled.add(v / s)
    out |= scaled
    return out


def num_matches_derived(value: float, derived: set[float], tol_abs: float = 0.01) -> bool:
    """绝对 0.01 容差命中(#0187 用户口径,覆盖 #0186 相对 1%;单位统一后比较)。

    ⚠️ 加 1e-9 机器精度余量:0.36-0.35=0.010000000000000009 > 0.01,
    「该 0.35 答 0.36 ⇒ 放行+标注」这条验收就是被这个浮点边界打掉的。
    """
    return any(abs(value - d) <= tol_abs + 1e-9 for d in derived)


# ---------------------------------------------------------------- 安全计算与金额规范(#0187/#0188)
import ast as _ast

_CALC_MAX_LEN = 200
_CALC_MAX_OPS = 16
_CALC_FUNCS = {"round": round}


def _calc_node_ok(node) -> bool:
    """AST 白名单:+ - * / ** % 括号、数字常量、round();其余一律拒绝。"""
    if isinstance(node, _ast.Expression):
        return _calc_node_ok(node.body)
    if isinstance(node, _ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, _ast.BinOp):
        return (isinstance(node.op, (_ast.Add, _ast.Sub, _ast.Mult, _ast.Div,
                                     _ast.Pow, _ast.Mod))
                and _calc_node_ok(node.left) and _calc_node_ok(node.right))
    if isinstance(node, _ast.UnaryOp):
        return isinstance(node.op, (_ast.USub, _ast.UAdd)) and _calc_node_ok(node.operand)
    if isinstance(node, _ast.Call):
        return (isinstance(node.func, _ast.Name) and node.func.id in _CALC_FUNCS
                and not node.keywords
                and all(_calc_node_ok(a) for a in node.args))
    return False


def safe_calc(expression: str) -> float | None:
    """AST 白名单安全求值(#0188:计算一律走 Python,禁裸 eval)。

    限表达式长度与算子数;任何非法/超限/除零 ⇒ None(调用方如实报"算不了")。
    """
    expr = (expression or "").strip()
    if not expr or len(expr) > _CALC_MAX_LEN:
        return None
    try:
        tree = _ast.parse(expr, mode="eval")
    except Exception:
        return None
    if not _calc_node_ok(tree):
        return None
    ops = sum(1 for n in _ast.walk(tree)
              if isinstance(n, (_ast.BinOp, _ast.UnaryOp, _ast.Call)))
    if ops > _CALC_MAX_OPS:
        return None
    try:
        return float(eval(compile(tree, "<safe_calc>", "eval"),
                          {"__builtins__": {}}, dict(_CALC_FUNCS)))
    except Exception:
        return None


_AMT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(万元|千元|元)")


def amounts_yuan(text: str) -> list[float]:
    """带单位金额 → 元(浮点),统一单位后比较(#0188);裸数字不在此列。"""
    out = []
    scale = {"万元": 10000.0, "千元": 1000.0, "元": 1.0}
    for v, u in _AMT_RE.findall(text or ""):
        out.append(float(v) * scale[u])
    return out


def format_amount(value_yuan: float) -> str:
    """金额规范输出:千元/万元/元 + 两位小数(#0188)。"""
    v = abs(value_yuan)
    if v >= 10000:
        return f"{value_yuan / 10000:.2f} 万元"
    if v >= 1000:
        return f"{value_yuan / 1000:.2f} 千元"
    return f"{value_yuan:.2f} 元"
