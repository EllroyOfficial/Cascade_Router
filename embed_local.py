# -*- coding: utf-8 -*-
"""embed_local.py —— 嵌入的**统一入口**（可更换：模型 / 后端都不写死）

设计要点（用户 2026-10-10 的接口要求）：
  * 模型名、后端、端点、线程 **全部走配置**（embedding.json > 环境变量 > 代码默认）
  * 换模型 / 换后端 = **改一个键，零代码改动**
  * 两种后端都支持：
      - backend="inproc" ：进程内 torch(CPU) 跑 HF 权重（**0 显存**）
      - backend="ollama" ：走 http://127.0.0.1:11434/api/embed（现状，占显存）
  * 只暴露 embed(texts) / embed_one(text) / dim() / info()，调用方不关心后端

安全默认：**backend 默认 "ollama"** —— 未重标阈值前不得静默切换（阈值/缓存都是按旧模型分布调的）。

自检:  python embed_local.py --selftest
单问:  python embed_local.py "一句话"
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_CFG_FILE = os.path.join(_HERE, "embedding.json")

_DEFAULTS = {
    "backend": "ollama",                                  # ollama | inproc
    "inproc_model": "BAAI/bge-small-zh-v1.5",             # HF 仓库 id（进程内用）
    "ollama_model": "qllama/bge-small-zh-v1.5:latest",    # ollama tag（写完整 tag，别用模糊名）
    "ollama_url": "http://127.0.0.1:11434",
    "threads": 8,
    "max_length": 512,
}


def _load_config() -> dict:
    cfg = dict(_DEFAULTS)
    if os.path.isfile(_CFG_FILE):
        try:
            with open(_CFG_FILE, "r", encoding="utf-8") as f:
                cfg.update({k: v for k, v in json.load(f).items() if v is not None})
        except Exception as e:
            print("[embed_local] 读 %s 失败(%r)，用默认配置" % (_CFG_FILE, e), file=sys.stderr)
    for k in list(_DEFAULTS):
        env = os.environ.get("EMBED_" + k.upper())
        if env:
            cfg[k] = int(env) if isinstance(_DEFAULTS[k], int) else env
    return cfg


CFG = _load_config()

_INPROC = {}        # {model: (tokenizer, model)}——按模型缓存,支持分消费者不同模型
_DIM = {}


# ---------------- inproc (torch CPU) ----------------

def _load_inproc(model: str | None = None):
    model = model or CFG["inproc_model"]
    if model in _INPROC:
        return _INPROC[model]
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(int(CFG["threads"]))
    m = CFG["inproc_model"]
    try:                                    # 离线优先：权重在本地缓存时绝不联网（联网会卡 100s 起）
        tok = AutoTokenizer.from_pretrained(m, local_files_only=True)
        mdl = AutoModel.from_pretrained(m, local_files_only=True)
    except Exception:                       # 缓存里没有才联网（默认走镜像）
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        tok = AutoTokenizer.from_pretrained(m)
        mdl = AutoModel.from_pretrained(m)
    mdl.eval()
    _INPROC[model] = (tok, mdl)
    _DIM[model] = int(mdl.config.hidden_size)
    return _INPROC[model]


def _embed_inproc(texts, cfg=None):
    cfg = cfg or CFG
    tok, mdl = _load_inproc(cfg["inproc_model"])
    import torch
    with torch.no_grad():
        b = tok(list(texts), padding=True, truncation=True,
                max_length=int(cfg["max_length"]), return_tensors="pt")
        out = mdl(**b)
        emb = out.last_hidden_state[:, 0]                 # bge 官方：cls pooling
        emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.tolist()


# ---------------- ollama ----------------

def _embed_ollama(texts, cfg=None):
    cfg = cfg or CFG
    body = json.dumps({"model": cfg["ollama_model"], "input": list(texts)}).encode("utf-8")
    req = urllib.request.Request(cfg["ollama_url"].rstrip("/") + "/api/embed",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        j = json.loads(r.read().decode("utf-8"))
    return j["embeddings"]


# ---------------- 对外 ----------------

def _cfg_for(consumer: str | None) -> dict:
    """分消费者配置(#0173 B):embedding.json 的 consumers 节按消费者覆盖全局;无覆盖=全局。"""
    if not consumer:
        return CFG
    ov = (CFG.get("consumers") or {}).get(consumer) or {}
    return {**CFG, **ov}


def embed(texts, consumer: str | None = None):
    """texts: list[str] -> list[list[float]]（已 L2 归一化）
    consumer: 消费者名("l2"/"exit_gate"),命中 embedding.json.consumers 覆盖节时按消费者配置;"""
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)
    cfg = _cfg_for(consumer)
    backend = str(cfg["backend"]).lower()
    if backend == "inproc":
        return _embed_inproc(texts, cfg)
    if backend == "ollama":
        return _embed_ollama(texts, cfg)
    raise ValueError("未知 backend: %r（只支持 inproc/ollama）" % cfg["backend"])


def embed_one(text):
    return embed([text])[0]


def dim(consumer: str | None = None):
    cfg = _cfg_for(consumer)
    key = (cfg["inproc_model"] if cfg["backend"] == "inproc" else cfg["ollama_model"]) + "|" + str(consumer)
    if key not in _DIM:
        _DIM[key] = len(embed(["维度探测"], consumer=consumer)[0])
    return _DIM[key]


def info(consumer: str | None = None):
    cfg = _cfg_for(consumer)
    return {"backend": cfg["backend"],
            "model": cfg["inproc_model"] if cfg["backend"] == "inproc" else cfg["ollama_model"],
            "dim": dim(consumer), "config_file": _CFG_FILE,
            "consumer": consumer or "(全局)"}


class Embedder:
    is_semantic = True   # 语义嵌入件标志:cos 类判据只在真语义件上生效
    """OllamaEmbedder 形状的适配壳(.embed 单文本/.similarity),router 等组件无缝换件。
    consumer: 分消费者配置名(#0173 B),命中 embedding.json.consumers 节;"""

    def __init__(self, consumer: str | None = None):
        self.consumer = consumer

    def embed(self, text: str) -> list[float]:
        return embed([text], consumer=self.consumer)[0]

    def embed_batch(self, texts) -> list:
        return embed(texts, consumer=self.consumer)

    def similarity(self, a: str, b: str) -> float:
        va, vb = embed([a, b], consumer=self.consumer)   # 向量已归一化,点积即余弦
        return sum(x * y for x, y in zip(va, vb))


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))


def _selftest():
    print("配置: " + json.dumps(CFG, ensure_ascii=False))
    print("生效: " + json.dumps(info(), ensure_ascii=False))
    q = "公允价值模式的投资性房地产是不是不计提减值？"
    t0 = time.perf_counter(); v = embed_one(q); ms = (time.perf_counter() - t0) * 1000
    print("单条短问: %.1f ms  维度=%d  前3维=%s" % (ms, len(v), [round(x, 4) for x in v[:3]]))
    # 与另一后端对比（若可用）
    other = "inproc" if str(CFG["backend"]).lower() == "ollama" else "ollama"
    saved = CFG["backend"]
    try:
        CFG["backend"] = other
        t0 = time.perf_counter(); v2 = embed_one(q); ms2 = (time.perf_counter() - t0) * 1000
        print("另一后端(%s): %.1f ms  维度=%d  cos(两者)=%.4f" % (other, ms2, len(v2), _cos(v, v2)))
    except Exception as e:
        print("另一后端(%s) 不可用: %r" % (other, e))
    finally:
        CFG["backend"] = saved
    print("OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif len(sys.argv) > 1:
        vec = embed_one(sys.argv[1])
        print(json.dumps({"dim": len(vec), "vec_head": [round(x, 5) for x in vec[:5]]}, ensure_ascii=False))
    else:
        print(__doc__)
