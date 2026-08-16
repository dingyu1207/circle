"""
用户反馈管理模块
存储到 feedback/feedback.json，纯数据逻辑。
"""

import json
import os
from datetime import datetime

_FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feedback.json')


def _load() -> list:
    if not os.path.exists(_FEEDBACK_FILE):
        return []
    with open(_FEEDBACK_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save(data: list):
    with open(_FEEDBACK_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_feedback(user_id: str, question: str, answer: str,
                  fb_type: str, comment: str = '') -> dict:
    """
    存储一条反馈。
    fb_type: 'like' | 'dislike'
    """
    entry = {
        'user_id': user_id,
        'question': question[:500],
        'answer': answer[:500],
        'type': fb_type,
        'comment': comment[:200],
        'timestamp': datetime.now().isoformat(),
    }
    data = _load()
    data.append(entry)
    _save(data)
    return entry