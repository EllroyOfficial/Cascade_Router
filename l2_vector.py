"""l2_vector.py —— L2 向量匹配分类(kNN3 主选 + 原型法备选,#0081 派活①)

选型依据([LocalModelName]四档评估,2026-10-08):bge-m3 kNN3 = 88.9% / 21ms;
bge-small 原型法 = 85.6% / 14ms(备选,可配置切换)。
[LocalModelName]人设模型零样本分类弱(professional 全判 chat)⇒ L2 不再让 LLM 出标签。

规约:
- 训练集 = send\意图分类样本_结果.tsv 的 8 成;留出 2 成绝不参与投票/原型/调参
  (口径照旧,只有评测脚本 smoke_l2_knn.py 碰留出集,只报指标、不回灌)。
- 置信度字段 confidence_kind 明确标注口径:
    knn3       → "vote_ratio"(k 票里头名标签的票数比,⚠️ 不是概率)
    prototype  → "cosine"(对原型向量的余弦,⚠️ 也不是校准概率)
- 枚举外/低置信 → classify 返回 "" ⇒ L2Classifier 判枚举外 → fail-open 上 L3,
  禁止重试(红线1)。阈值一律"待标定、可配",等用户题库正式标定,不许私调。
- 零硬编码枚举(红线5):标签集合来自 config["intent_classifier"]["enum"]。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random


def load_tsv(path: str) -> list[tuple[str, str, str]]:
    """读「编号\t原句\t标签」TSV,返回 [(编号, 原句, 标签)];空行/短行跳过。"""
    rows: list[tuple[str, str, str]] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)                      # 表头
        for r in reader:
            if len(r) >= 3 and r[1].strip() and r[2].strip():
                rows.append((r[0].strip(), r[1].strip(), r[2].strip()))
    return rows


def split_train_holdout(rows, holdout_ratio: float, seed: int):
    """8:2 **分层**切分(2026-10-09 口径):按标签分组,组内确定性洗牌后各按比例
    留出;同一 seed 下每次切分逐条一致(可复跑)。留出集绝不参与投票/原型/调参。
    分层保证 emotion/tool 这类小类不被整体挤进训练集或留出集。"""
    by_label: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_label.setdefault(row[2], []).append(i)
    hold_idx: set[int] = set()
    rng = random.Random(seed)
    for label in sorted(by_label):
        idx = by_label[label]
        rng.shuffle(idx)
        n_hold = round(len(idx) * holdout_ratio)
        hold_idx.update(idx[:n_hold])
    train = [rows[i] for i in range(len(rows)) if i not in hold_idx]
    holdout = [rows[i] for i in sorted(hold_idx)]
    return train, holdout


class _VectorBase:
    """公共件:enum 校验、向量缓存、余弦。子类实现 _rank(text)。"""

    def __init__(self, enum, vector_cache: str | None = None):
        self._enum = tuple(enum)
        self._cache_path = vector_cache
        self._vectors: dict[str, list[float]] = {}   # text -> vector(缓存)
        self.last_confidence: float = 0.0
        self.confidence_kind: str = ""
        self.last_detail: str = ""

    # ── 向量缓存:训练向量按文本 md5 键存 JSON,重启不重算 ──
    def ensure_dim(self, embedder) -> None:
        """维度哨兵:缓存向量与当前嵌入件维度不符 ⇒ 整表作废(防 zip 截断静默算错)。"""
        probe = embedder.embed("维度探测")
        if not self._vectors and self._cache_path and os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, encoding="utf-8") as f:
                    self._vectors = json.load(f)      # 先加载,否则哨兵看的是空表
            except (ValueError, OSError):
                self._vectors = {}
        if self._vectors and len(next(iter(self._vectors.values()))) != len(probe):
            self._vectors = {}                        # 维度不符 ⇒ 作废重建

    def _embed_cached(self, embedder, text: str) -> list[float]:
        if text in self._vectors:
            return self._vectors[text]
        if self._cache_path and os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, encoding="utf-8") as f:
                    self._vectors = json.load(f)
            except (ValueError, OSError):
                self._vectors = {}                   # 坏缓存:重建,不静默当有效
        if text not in self._vectors:
            vec = embedder.embed(text)
            if self._vectors and len(next(iter(self._vectors.values()))) != len(vec):
                self._vectors = {}                   # 维度不匹配(换模型/后端):缓存作废重建
            self._vectors[text] = vec
        return self._vectors[text]

    def _flush_cache(self) -> None:
        if not self._cache_path:
            return
        with open(self._cache_path, "w", encoding="utf-8") as f:
            json.dump(self._vectors, f)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (na * nb)

    def _rank(self, text: str):
        raise NotImplementedError

    def classify(self, text: str, temperature: float = 0.0,
                 max_tokens: int = 10, timeout_ms: int = 2000) -> str:
        """接口兼容 SmallClassifier(温度/tokens/超时参数收下但向量匹配用不上)。

        返回枚举内标签,或 ""(低置信/异常 ⇒ 调用方按枚举外 fail-open,红线1)。
        """
        try:
            ranked = self._rank(text)
        except Exception:
            self.last_confidence, self.last_detail = 0.0, "embed_error"
            return ""
        if not ranked:
            return ""
        label, conf, detail = ranked
        if label not in self._enum or conf < self._min_conf:
            self.last_confidence = conf
            self.last_detail = detail
            return ""                                # 低置信/枚举外 → fail-open
        self.last_confidence = conf
        self.last_detail = detail
        return label


class KNN3Classifier(_VectorBase):
    """kNN 投票分类(主选):对训练向量取 top-k,头名票数比过阈值才出标签。"""

    def __init__(self, embedder, train_rows, enum, k: int = 3,
                 min_vote_ratio: float = 0.67, vector_cache: str | None = None):
        super().__init__(enum, vector_cache)
        self._embedder = embedder
        self._k = max(1, int(k))
        self._min_conf = float(min_vote_ratio)
        self._train = [(t, lab) for _, t, lab in train_rows]
        self._train_vecs: list[tuple[list[float], str]] = []
        self.ensure_dim(embedder)                    # 哨兵先探维度,防全命中缓存绕过比对
        for text, lab in self._train:
            self._train_vecs.append((self._embed_cached(embedder, text), lab))
        self._flush_cache()
        self.confidence_kind = "vote_ratio"          # ⚠️ 票数比,不是概率

    def _rank(self, text: str):
        q = self._embed_cached(self._embedder, text)
        sims = sorted(
            ((self._cosine(q, v), lab) for v, lab in self._train_vecs),
            key=lambda x: -x[0])
        top = sims[:self._k]
        votes: dict[str, int] = {}
        for _, lab in top:
            votes[lab] = votes.get(lab, 0) + 1
        best = max(votes, key=lambda k: votes[k])
        ratio = votes[best] / len(top)
        detail = (f"knn(k={self._k}): votes={dict(sorted(votes.items(), key=lambda x: -x[1]))} "
                  f"top1_sim={top[0][0]:.3f}")
        return best, ratio, detail


class PrototypeClassifier(_VectorBase):
    """原型法(备选,配 bge-small):每类训练向量取均值,对原型算余弦。"""

    def __init__(self, embedder, train_rows, enum,
                 min_sim: float = 0.55, vector_cache: str | None = None):
        super().__init__(enum, vector_cache)
        self._embedder = embedder
        self._min_conf = float(min_sim)
        acc: dict[str, list[list[float]]] = {}
        self.ensure_dim(embedder)                    # 哨兵先探维度,防全命中缓存绕过比对
        for _, text, lab in train_rows:
            acc.setdefault(lab, []).append(self._embed_cached(embedder, text))
        self._prototypes: dict[str, list[float]] = {}
        for lab, vecs in acc.items():
            dim = len(vecs[0])
            self._prototypes[lab] = [sum(v[i] for v in vecs) / len(vecs)
                                     for i in range(dim)]
        self._flush_cache()
        self.confidence_kind = "cosine"              # ⚠️ 余弦,不是校准概率

    def _rank(self, text: str):
        q = self._embed_cached(self._embedder, text)
        sims = sorted(((self._cosine(q, p), lab) for lab, p in self._prototypes.items()),
                      key=lambda x: -x[0])
        top_sim, top_lab = sims[0]
        margin = top_sim - sims[1][0] if len(sims) > 1 else top_sim
        detail = f"prototype: sims={{{', '.join(f'{l}:{s:.3f}' for s, l in sims)}}} margin={margin:.3f}"
        return top_lab, top_sim, detail


def build_from_config(config: dict, embedder, base_dir: str) -> _VectorBase:
    """按 config["intent_classifier"] 组装向量分类件(选型/路径/阈值全走配置,红线5)。"""
    ic = config["intent_classifier"]
    method = ic.get("method", "knn3")
    knn_cfg = ic.get("knn", {})
    tsv = knn_cfg.get("train_tsv", "")
    if tsv and not os.path.isabs(tsv):
        tsv = os.path.normpath(os.path.join(base_dir, tsv))
    rows = load_tsv(tsv)
    train, _holdout = split_train_holdout(
        rows, float(knn_cfg.get("holdout_ratio", 0.2)), int(knn_cfg.get("seed", 42)))
    # _holdout 就地丢弃:留出集绝不参与投票/原型/调参,只有评测脚本单独切。
    cache = knn_cfg.get("vector_cache", "")
    cache_path = os.path.normpath(os.path.join(base_dir, cache)) if cache else None
    if method == "prototype":
        proto_cfg = ic.get("prototype", {})
        return PrototypeClassifier(
            embedder, train, ic["enum"],
            min_sim=float(proto_cfg.get("min_sim", 0.55)),
            vector_cache=cache_path)
    return KNN3Classifier(
        embedder, train, ic["enum"],
        k=int(knn_cfg.get("k", 3)),
        min_vote_ratio=float(knn_cfg.get("min_vote_ratio", 0.67)),
        vector_cache=cache_path)
