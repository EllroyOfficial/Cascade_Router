"""l3_provider.py —— L3 云端大脑的可替换 provider 层

当前实现:走 dsh-astrbot-ingress 的 POST /inbound(SSE),把 DSH 侧 L3 专用会话当云端。
以后换别家(智谱/DeepSeek 直连)只换本层的 L3Provider 实现,router.py 不动。

坑位备忘([LocalModelName]实测):
① 命令必须发裸词(compact),带 /dsh 前缀会被当任务空转;② 新会话第一条被人设
提示词吃掉,需先暖场(L3 专用会话 session-11519cfd 已暖过,勿新建)。

超时双重闸:总超时 timeout_ms + 流空闲 idle_ms;失败抛 L3UnavailableError,
由调用方如实报错并降级,禁止静默吞(用户原话)。
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request

DEFAULT_UMO = "ncd-app_l3:PrivateMessage:l3-local"
DEFAULT_IDLE_MS = 8_000
DEFAULT_TIMEOUT_MS = 180_000
DEFAULT_BEACON = os.path.join(os.path.expanduser("~"), ".dsh", "astrbot-ingress.json")

# 会话分隔符只在**尾部**剥(2026-10-09 派活②):正则锚定字符串末尾($),
# 正文里正常的破折号/类似措辞一律不碰;允许可变的空格与换行包夹。
_SESSION_DELIMITER_RE = re.compile(r"——\s*本回合结束\s*$")


def strip_session_delimiter(text: str) -> str:
    """剥掉答复尾部的会话分隔符「—— 本回合结束」(精确锚定行尾,不误伤正文)。"""
    return _SESSION_DELIMITER_RE.sub("", text).rstrip()


class L3UnavailableError(Exception):
    """L3 不可用(连不上/HTTP 错/流断/空答复/错误事件)。"""


def _read_beacon(path: str) -> dict:
    """读信标文件取 token/port;没有/坏 → L3UnavailableError(不静默)。"""
    if not os.path.exists(path):
        raise L3UnavailableError(f"找不到信标文件: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
    except ValueError as e:
        raise L3UnavailableError(f"信标文件解析失败: {e}") from None
    token = j.get("token") or j.get("bearer") or j.get("authToken")
    if not token:
        raise L3UnavailableError("信标文件里没有 token 字段")
    return {"token": token, "port": int(j.get("port") or 3188)}


class DSHIngressL3Provider:
    """L3Provider 现役实现:ingress SSE → text 事件聚合为答复。"""

    def __init__(self, umo: str = DEFAULT_UMO, beacon_path: str = DEFAULT_BEACON) -> None:
        self.umo = umo
        self.beacon_path = beacon_path

    def ask(self, text: str, timeout_ms: int = DEFAULT_TIMEOUT_MS,
            idle_ms: int = DEFAULT_IDLE_MS) -> str:
        beacon = _read_beacon(self.beacon_path)
        url = f"http://127.0.0.1:{beacon['port']}/inbound"
        body = json.dumps({"umo": self.umo, "text": text,
                           "senderId": "l3", "chatType": "private"}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + beacon["token"],
            "Accept": "text/event-stream",
        })
        deadline = time.monotonic() + timeout_ms / 1000.0
        try:
            resp = urllib.request.urlopen(req, timeout=max(1.0, deadline - time.monotonic()))
        except Exception as e:
            raise L3UnavailableError(f"请求失败: {e}") from None
        if resp.status != 200:
            raise L3UnavailableError(f"HTTP {resp.status}")

        replies: list[str] = []
        errors: list[str] = []
        buf: str = ""
        ev: str | None = None
        data_lines: list[str] = []
        done = False
        last_data = time.monotonic()
        idle_s = idle_ms / 1000.0

        def flush() -> None:
            nonlocal ev, data_lines, done, last_data
            if not ev and not data_lines:
                return
            name = ev or "message"
            raw = "\n".join(data_lines)
            try:
                payload = json.loads(raw) if raw else None
            except ValueError:
                payload = {"raw": raw}
            last_data = time.monotonic()
            if name in ("text", "message", "delta", "reply"):
                t = (payload or {}).get("text") or (payload or {}).get("message") \
                    or (payload or {}).get("delta") or (payload or {}).get("content") or ""
                if t:
                    replies.append(str(t))
            elif name == "error":
                errors.append(str((payload or {}).get("message") or raw))
            elif name == "done":
                done = True
            ev, data_lines = None, []

        try:
            while not done:
                if time.monotonic() - last_data > idle_s:
                    break                      # 静默超时:拿到什么算什么
                if time.monotonic() > deadline:
                    break                      # 总超时
                chunk = resp.read1(4096) if hasattr(resp, "read1") else resp.read(4096)
                if not chunk:
                    break
                buf += chunk.decode("utf-8", errors="replace")
                idx = buf.find("\n")
                while idx >= 0:
                    line, buf = buf[:idx].rstrip("\r"), buf[idx + 1:]
                    idx = buf.find("\n")
                    if line == "":
                        flush()
                    elif line.startswith(":"):
                        last_data = time.monotonic()
                    elif line.startswith("event:"):
                        ev = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].strip())
                if done:
                    break
        finally:
            try:
                resp.close()
            except Exception:
                pass

        if errors:
            raise L3UnavailableError("; ".join(errors))
        reply = "".join(replies).strip()
        if not reply:
            raise L3UnavailableError("流结束但无答复文本")
        return strip_session_delimiter(reply)
