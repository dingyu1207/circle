"""
对话会话管理模块
存储到 sessions/sessions.json，按 session_id 索引，支持熔断和摘要。
"""

import json
import os
from datetime import datetime
import config

_SESSIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions", "sessions.json")


def _ensure():
    os.makedirs(os.path.dirname(_SESSIONS_FILE), exist_ok=True)


def _load() -> dict:
    _ensure()
    if not os.path.exists(_SESSIONS_FILE):
        return {"sessions": {}}
    with open(_SESSIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict):
    _ensure()
    with open(_SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_session(title: str = "新对话") -> str:
    """创建新会话，返回 session_id。"""
    sid = "s_" + datetime.now().strftime("%Y%m%d%H%M%S") + "_" + os.urandom(3).hex()
    data = _load()
    data["sessions"][sid] = {
        "id": sid,
        "title": title[:50],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "messages": [],
        "summary": "",
    }
    _save(data)
    return sid


def get_session(sid: str) -> dict | None:
    """获取会话数据，不存在返回 None。"""
    return _load()["sessions"].get(sid)


def add_message(sid: str, role: str, content: str):
    """向会话添加一条消息，自动处理熔断和摘要。"""
    data = _load()
    if sid not in data["sessions"]:
        # 会话不存在时直接用 sid 注册（此前误用 create_session(sid) 会生成随机 id 导致 KeyError）
        now = datetime.now().isoformat()
        data["sessions"][sid] = {
            "id": sid,
            "title": "新对话",
            "created_at": now,
            "updated_at": now,
            "messages": [],
            "summary": "",
        }
    sess = data["sessions"][sid]
    sess["messages"].append(
        {
            "role": role,
            "content": content[: config.MAX_MSG_LEN],
            "time": datetime.now().isoformat(),
        }
    )

    # 熔断保护：超出 MAX_ROUNDS 轮（每轮 = user+assistant）时丢弃最旧的
    rounds = len([m for m in sess["messages"] if m["role"] == "user"])
    if rounds > config.MAX_ROUNDS:
        # 找到并删除最旧的一轮（1 user + 1 assistant）
        # 简化处理：如果第一条是 user，刪前2条；如果第一条是 assistant，刪前3条直到碰到下一个 user
        over = rounds - config.MAX_ROUNDS
        idx = 0
        for _ in range(over):
            # 跳过前面的 assistant（孤立的），删除一对 user+assistant
            while idx < len(sess["messages"]) and sess["messages"][idx]["role"] == "assistant":
                idx += 1
            if idx < len(sess["messages"]) and sess["messages"][idx]["role"] == "user":
                idx += 1  # 删 user
                if idx < len(sess["messages"]) and sess["messages"][idx]["role"] == "assistant":
                    idx += 1  # 删 assistant
        sess["messages"] = sess["messages"][idx:]

    # 摘要生成：超出阈值时，把前半段压缩为摘要
    if rounds > config.SUMMARY_THRESHOLD and not sess.get("summary"):
        early_msgs = sess["messages"][: config.SUMMARY_THRESHOLD * 2]  # 前20轮的问答
        user_msgs = [m["content"][:60] for m in early_msgs if m["role"] == "user"]
        sess["summary"] = "对话要旨：" + " | ".join(user_msgs[-10:])

    sess["updated_at"] = datetime.now().isoformat()
    # 自动更新标题（用第一条用户消息的前20字）
    if sess["title"] == "新对话":
        for m in sess["messages"]:
            if m["role"] == "user":
                sess["title"] = m["content"][:20] + ("…" if len(m["content"]) > 20 else "")
                break
    _save(data)


def get_context(sid: str) -> tuple[list, str]:
    """
    返回 (最近 N 轮消息列表, 摘要文本)。
    消息列表可直接发给 API。
    """
    sess = get_session(sid)
    if not sess:
        return [], ""
    msgs = sess["messages"]
    # 取最近 CONTEXT_ROUNDS 轮
    recent = []
    user_count = 0
    for m in reversed(msgs):
        recent.insert(0, {"role": m["role"], "content": m["content"]})
        if m["role"] == "user":
            user_count += 1
            if user_count >= config.CONTEXT_ROUNDS:
                break
    return recent, sess.get("summary", "")


def list_sessions() -> list:
    """返回所有会话摘要列表（按更新时间倒序）。"""
    sessions = _load()["sessions"]
    items = [
        {
            "id": s["id"],
            "title": s["title"],
            "updated_at": s["updated_at"],
            "msg_count": len(s["messages"]),
        }
        for s in sessions.values()
    ]
    items.sort(key=lambda x: x["updated_at"], reverse=True)
    return items


def delete_session(sid: str):
    """删除一个会话。"""
    data = _load()
    data["sessions"].pop(sid, None)
    _save(data)
