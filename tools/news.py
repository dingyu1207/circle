"""每日新闻：抓取官方媒体 RSS，解析标题 / 链接 / 时间。

为什么是 RSS 而不是搜索 API：
- DeepSeek 原生 ``web_search`` 在部分账号上不生效——``tools`` 声明被服务端收下并回显，
  却从不产生 ``web_search_call``，模型只会回答"我无法联网"；
- DuckDuckGo Instant Answer 在国内网络不可达（ConnectTimeout）。

官方媒体 RSS 国内直连、1 秒内返回，且 **0 token、0 key、零第三方依赖**（只用标准库）。

**为什么必须有新鲜度闸门**：中文媒体的 RSS 大量是「僵尸源」——HTTP 200、XML 结构完全正常，
只是内容停更在两三年前。若无闸门，模型会拿 2022 年的新闻当今天的头条播报，
而且从任何一层都看不出异常。抓取本身不花钱，只有把头条送进模型组织语言时才消耗 token。
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests

from config import (
    FETCH_TIMEOUT,
    NEWS_CACHE_TTL,
    NEWS_FEEDS,
    NEWS_MAX_AGE_DAYS,
    NEWS_MAX_PER_FEED,
    NEWS_MAX_TOTAL,
)

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CircleBot/1.0)"}

logger = logging.getLogger(__name__)

# 从正文 URL 里兜底抠日期，形如 /2026/09-15/ 或 /2026-09/15/
_URL_DATE = re.compile(r"/(20\d{2})[-/]?(\d{2})[-/]?(\d{2})?/")

# 时间一律归一到北京时间（naive，不带 tzinfo），不跟随运行机器的本地时区。
#
# 为什么必须显式钉死：NEWS_FEEDS 全是中文媒体，时间只在「北京时间」这一种读法下才对。
# 部署形态是 Docker + gunicorn，而容器默认时区是 UTC——若用 astimezone() 取本机时区，
# 本机（UTC+8）测试全绿，线上却把 15:38 的头条显示成 07:38，且新鲜度闸门也跟着偏 8 小时。
# 这正是「本机绿 / CI 红」的典型形状：CI 跑在 UTC 上才暴露出来。
_BJ_TZ = timezone(timedelta(hours=8))


def _now_bj() -> datetime:
    """当前北京时间（naive，与 _parse_pubdate 的归一结果同一口径）。"""
    return datetime.now(_BJ_TZ).replace(tzinfo=None)


# 进程内缓存：{时间戳, 头条列表}。进程重启即清零，与限流计数器同一风格。
_cache = {"ts": 0.0, "items": []}


def _parse_pubdate(raw: str):
    """RFC822 时间字符串 → 北京时间 datetime（naive）；解析不出来返回 None。"""
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:  # 少数源不带时区，按北京时间理解
        dt = dt.replace(tzinfo=_BJ_TZ)
    return dt.astimezone(_BJ_TZ).replace(tzinfo=None)


def _date_from_url(url: str):
    """从 URL 路径里兜底抠日期（不少媒体只在 URL 里带时间，pubDate 是空的）。"""
    match = _URL_DATE.search(url or "")
    if not match:
        return None
    year, month, day = match.group(1), match.group(2), match.group(3) or "01"
    try:
        return datetime(int(year), int(month), int(day))
    except ValueError:
        return None


def _parse_rss(content: bytes, source: str) -> list:
    """解析 RSS 2.0 的 <item> 节点，产出统一结构的头条列表。

    时间优先取 ``pubDate``，取不到再从 URL 兜底；两处都拿不到则 ``ts`` 为 None
    （调用方的新鲜度闸门会把它判为「无法验证」并丢弃）。
    """
    root = ET.fromstring(content)
    items = []
    for node in root.iter("item"):
        title = (node.findtext("title") or "").strip()
        url = (node.findtext("link") or "").strip()
        if not title or not url:
            continue
        ts = _parse_pubdate((node.findtext("pubDate") or "").strip()) or _date_from_url(url)
        items.append(
            {
                "title": title,
                "url": url,
                "source": source,
                "ts": ts,
                "published": ts.strftime("%m-%d %H:%M") if ts else "",
            }
        )
    return items


def _interleave(per_feed: list, limit: int) -> list:
    """轮流从各源取头条，避免榜单被单一大源（单个源一次可返回 300+ 条）占满。"""
    merged = []
    idx = 0
    while len(merged) < limit:
        advanced = False
        for feed in per_feed:
            if idx < len(feed):
                merged.append(feed[idx])
                advanced = True
                if len(merged) >= limit:
                    break
        if not advanced:
            break
        idx += 1
    return merged


def _keep_fresh(items: list, now: datetime) -> list:
    """新鲜度闸门：丢弃过期条目，以及**时间无法验证**的条目（拿不准就不采用）。"""
    cutoff = now - timedelta(days=NEWS_MAX_AGE_DAYS)
    return [it for it in items if it["ts"] is not None and it["ts"] >= cutoff]


def fetch_news(limit: int = NEWS_MAX_TOTAL, use_cache: bool = True, now=None) -> list:
    """抓取各源头条并合并。全部源都失败（或全部过期）时返回 []，由调用方降级。

    ``now`` 可注入北京时间（naive），供测试固定时钟用。
    """
    epoch = time.time()
    if use_cache and _cache["items"] and epoch - _cache["ts"] < NEWS_CACHE_TTL:
        return _cache["items"][:limit]

    now_dt = now or _now_bj()
    per_feed = []
    for feed in NEWS_FEEDS:
        try:
            resp = requests.get(feed["url"], timeout=FETCH_TIMEOUT, headers=_HEADERS)
            resp.raise_for_status()
            parsed = _parse_rss(resp.content, feed["name"])
        except Exception:
            continue  # 单源失败不影响其它源
        fresh = _keep_fresh(parsed, now_dt)[:NEWS_MAX_PER_FEED]
        if not fresh and parsed:
            # 僵尸源特征：抓得到内容，但一条都过不了新鲜度闸门
            logger.warning(
                "源「%s」%s 条全部超过 %s 天，已丢弃（疑似僵尸源）",
                feed["name"],
                len(parsed),
                NEWS_MAX_AGE_DAYS,
            )
        per_feed.append(fresh)

    items = _interleave(per_feed, limit)
    if items:  # 全失败时不覆盖旧缓存，下次还能用上
        _cache["ts"] = epoch
        _cache["items"] = items
    return items


def format_news(items: list) -> str:
    """把头条格式化成注入 system prompt 的文本块。"""
    if not items:
        return ""
    lines = []
    for i, item in enumerate(items, 1):
        when = f"（{item['published']}）" if item.get("published") else ""
        lines.append(f"{i}. 【{item['source']}】{when} {item['title']}\n   {item['url']}")
    return "\n".join(lines)


def _tokens(text: str) -> set:
    """粗切词：英文/数字按整词，中文按 2-gram。够用即可，不引入分词依赖。"""
    tokens = set(re.findall(r"[A-Za-z0-9]{2,}", text))
    han = re.findall(r"[一-鿿]", text)
    tokens.update("".join(han[i : i + 2]) for i in range(len(han) - 1))
    return tokens


def search_news(query: str, limit: int = 5) -> list:
    """在已抓取的头条里做本地关键词匹配（0 网络 / 0 token）。

    注意能力边界：这只覆盖**当前抓到的十来条官方媒体头条**，不是通用网页搜索。
    查不到时返回 []，调用方必须能优雅降级——不要把它当搜索引擎用。
    """
    items = fetch_news()
    if not items or not query:
        return []
    wanted = _tokens(query)
    if not wanted:
        return []
    return [it for it in items if wanted & _tokens(it["title"])][:limit]
