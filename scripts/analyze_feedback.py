"""
反馈数据分析脚本
读取 feedback/feedback.json，统计点赞/点踩比例、评论数、每日反馈量，
以易读表格输出（tabulate），有评论时显示前 10 条。

用法：python scripts/analyze_feedback.py
"""

import json
import os
from collections import Counter
from datetime import datetime

from tabulate import tabulate

FEEDBACK_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "feedback",
    "feedback.json",
)


def load_feedback(path: str) -> list:
    """读取反馈数据；文件缺失或损坏时返回空列表。"""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []
    return data if isinstance(data, list) else []


def _day_of(record: dict) -> str:
    """从时间戳提取日期字符串；解析失败时回退为前缀截取。"""
    ts = record.get("timestamp", "")
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return str(ts)[:10]


def main() -> None:
    records = load_feedback(FEEDBACK_FILE)
    if not records:
        print("暂无反馈数据（feedback/feedback.json 为空或不存在）。")
        return

    total = len(records)
    likes = sum(1 for r in records if r.get("type") == "like")
    dislikes = total - likes
    like_ratio = likes / total if total else 0
    with_comments = sum(1 for r in records if (r.get("comment") or "").strip())

    daily = Counter(_day_of(r) for r in records)

    print(f"=== Circle 用户反馈分析（共 {total} 条）===")
    print()

    summary = [
        ["总反馈数", total],
        ["点赞", f"{likes}（{like_ratio:.1%}）"],
        ["点踩", f"{dislikes}（{1 - like_ratio:.1%}）"],
        ["有评论", with_comments],
        ["无评论", total - with_comments],
    ]
    print(tabulate(summary, headers=["指标", "数值"], tablefmt="grid"))
    print()

    daily_rows = [[day, count] for day, count in sorted(daily.items())]
    print("每日反馈量：")
    print(tabulate(daily_rows, headers=["日期", "反馈量"], tablefmt="grid"))
    print()

    comments = [r.get("comment", "").strip() for r in records if (r.get("comment") or "").strip()]
    if comments:
        top_n = min(10, len(comments))
        print(f"=== 前 {top_n} 条评论 ===")
        for i, comment in enumerate(comments[:10], 1):
            print(f"{i}. {comment}")
    else:
        print("暂无评论。")


if __name__ == "__main__":
    main()
