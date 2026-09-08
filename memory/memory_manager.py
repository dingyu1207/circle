"""
用户记忆管理模块
使用本地 JSON 文件存储（memory/user_memory.json），支持多用户。
"""

import json
import os
import re
from datetime import datetime

_MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_memory.json")


def _load_all() -> dict:
    """加载全部用户数据。"""
    if not os.path.exists(_MEMORY_FILE):
        return {}
    with open(_MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_all(data: dict):
    """保存全部用户数据。"""
    with open(_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_or_create_user(user_id: str) -> dict:
    """获取用户记录，不存在则创建。返回用户 dict。"""
    all_data = _load_all()
    if user_id not in all_data:
        now = datetime.now().isoformat()
        all_data[user_id] = {
            "user_id": user_id,
            "first_seen": now,
            "last_seen": now,
            "name": "",
            "preferences": "",
            "health_info": "",
            "conversation_history": [],
        }
        _save_all(all_data)
    return all_data[user_id]


def update_user(user_id: str, **kwargs) -> dict:
    """更新用户字段（name / preferences / health_info 等）。"""
    all_data = _load_all()
    if user_id not in all_data:
        # 避免 .get() 默认参数被急切求值导致新建用户后写回旧快照（清空记录）
        get_or_create_user(user_id)
        all_data = _load_all()
    user = all_data[user_id]
    for k, v in kwargs.items():
        if k in user:
            if v:  # 非空才更新
                user[k] = v
    user["last_seen"] = datetime.now().isoformat()
    _save_all(all_data)
    return user


def add_conversation(user_id: str, summary: str):
    """添加对话摘要，保留最近 5 条。"""
    all_data = _load_all()
    if user_id not in all_data:
        get_or_create_user(user_id)
        all_data = _load_all()
    user = all_data[user_id]
    user["conversation_history"].append(summary)
    user["conversation_history"] = user["conversation_history"][-5:]
    user["last_seen"] = datetime.now().isoformat()
    _save_all(all_data)


def delete_user_field(user_id: str, key: str) -> bool:
    """删除单条记忆字段。字段不存在或为空时返回 False。"""
    all_data = _load_all()
    user = all_data.get(user_id)
    if not user or key not in user or not user.get(key):
        return False
    user[key] = [] if key == "conversation_history" else ""
    user["last_seen"] = datetime.now().isoformat()
    _save_all(all_data)
    return True


def clear_user(user_id: str) -> bool:
    """清空用户全部记忆。用户不存在返回 False。"""
    all_data = _load_all()
    user = all_data.get(user_id)
    if not user:
        return False
    user["name"] = ""
    user["preferences"] = ""
    user["health_info"] = ""
    user["conversation_history"] = []
    user["last_seen"] = datetime.now().isoformat()
    _save_all(all_data)
    return True


def extract_info_from_message(user_message: str) -> dict:
    """用规则从用户消息中提取个人信息（纯代码，0 token）。返回需更新的字段。"""
    updates = {}

    # 称呼：我叫X / 我是X / 叫我X / 换名字叫X
    m = re.search(r'(?:我叫|我是|叫我|换名.*?叫|改名.*?叫)\s*[“"「]?(\S{1,10})[”"」]?', user_message)
    if m:
        updates["name"] = m.group(1).rstrip("，,。.")

    # 饮食偏好：我不吃X / 我忌口X / 我喜欢吃X / 我爱吃X
    prefs = []
    for pat in [
        r"我不吃\s*(.+?)(?:[，,。\.;；]|$)",
        r"我忌口\s*(.+?)(?:[，,。\.;；]|$)",
        r"我喜欢吃\s*(.+?)(?:[，,。\.;；]|$)",
        r"我爱吃\s*(.+?)(?:[，,。\.;；]|$)",
    ]:
        m = re.search(pat, user_message)
        if m:
            prefs.append(m.group(1).rstrip("，,。."))
    if prefs:
        updates["preferences"] = "；".join(prefs)

    # 健康：我痛经 / 生理期会X / 我有X / 我患有X / 我膝盖/腰/肩/颈椎/胃X
    health = []
    m = re.search(r"我痛经", user_message)
    if m:
        health.append("痛经")
    m = re.search(r"(?:生理期|经期|大姨妈).*?(?:会|都|就|容易)\s*(.+?)(?:[，,。\.;；]|$)", user_message)
    if m:
        health.append(m.group(1).rstrip("，,。."))
    m = re.search(r"我(?:有|患有?|得了?)\s*(.+?)(?:[，,。\.;；]|$)", user_message)
    if m:
        health.append(m.group(1).rstrip("，,。."))
    for part in ["膝盖", "腰", "肩", "颈椎", "胃", "头", "心脏", "血压", "血糖", "甲状腺", "卵巢", "乳腺"]:
        m = re.search(rf"我{part}(.+?)(?:[，,。\.;；]|$)", user_message)
        if m:
            health.append(f"{part}{m.group(1).rstrip('，,。.')}")
    if health:
        updates["health_info"] = "；".join(health)

    return updates


def is_new_user(user_id: str) -> bool:
    """新用户 = 还没告诉 Circle 自己的称呼。"""
    user = get_or_create_user(user_id)
    return not user.get("name")


def format_memory_for_prompt(user_id: str) -> str:
    """将用户记忆格式化为 System Prompt 补充文本。"""
    user = get_or_create_user(user_id)
    parts = []
    if user.get("name"):
        parts.append(f"- 称呼：{user['name']}")
    if user.get("preferences"):
        parts.append(f"- 饮食偏好：{user['preferences']}")
    if user.get("health_info"):
        parts.append(f"- 健康信息：{user['health_info']}")
    if user.get("conversation_history"):
        parts.append(f"- 最近对话摘要：{' | '.join(user['conversation_history'][-3:])}")

    if not parts:
        return ""
    return (
        "\n\n【你记得关于当前用户的信息】\n"
        + "\n".join(parts)
        + '\n请在回答时自然地调用这些信息，不要刻意说"根据记录"。'
    )


def get_user_status(user_id: str) -> dict:
    """返回前端用的用户状态。"""
    user = get_or_create_user(user_id)
    return {
        "is_new": not user.get("name"),
        "name": user.get("name", ""),
        "has_info": bool(user.get("preferences") or user.get("health_info")),
    }
