"""新闻模块离线测试：RSS 解析、新鲜度闸门、僵尸源丢弃、本地关键词检索。

全部不触网——``requests.get`` 一律被 monkeypatch 成假响应。
"""

from datetime import datetime, timedelta

import pytest

from tools import news


def _item(title: str, link: str, pub: str = "") -> str:
    pub_xml = f"<pubDate>{pub}</pubDate>" if pub else ""
    return f"<item><title>{title}</title><link>{link}</link>{pub_xml}</item>"


def _rss(*items: str) -> bytes:
    body = "".join(items)
    return (
        f'<?xml version="1.0" encoding="utf-8"?><rss version="2.0"><channel><title>测试源</title>{body}</channel></rss>'
    ).encode("utf-8")


def _rfc822(dt: datetime) -> str:
    """把北京时间（naive）格式化成 +0800 的 RFC822。

    注意：**不要**用 datetime.now() 造「现在」——那取的是运行机器的本地时区，
    在 CI（UTC）与本机（UTC+8）会得到不同结果。统一用 news._now_bj()。
    """
    return dt.strftime("%a, %d %b %Y %H:%M:%S +0800")


class FakeResponse:
    """最小可用的 requests 响应替身。"""

    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def _clear_cache():
    """每个用例前后清空进程内缓存，避免用例相互污染。"""
    news._cache["ts"] = 0.0
    news._cache["items"] = []
    yield
    news._cache["ts"] = 0.0
    news._cache["items"] = []


def _patch_source(monkeypatch, name: str, url: str, content: bytes, status_code: int = 200):
    """把 news 的数据源与网络访问都换成可控的假实现。"""
    monkeypatch.setattr(news, "NEWS_FEEDS", [{"name": name, "url": url}])
    monkeypatch.setattr(news.requests, "get", lambda *a, **kw: FakeResponse(content, status_code))


# ── 解析 ────────────────────────────────────────────────────────


def test_parse_rss_reads_title_link_and_pubdate():
    content = _rss(_item("今日头条", "https://example.com/a", "Mon, 15 Sep 2026 10:00:00 +0800"))
    items = news._parse_rss(content, "测试网")
    assert len(items) == 1
    assert items[0]["title"] == "今日头条"
    assert items[0]["url"] == "https://example.com/a"
    assert items[0]["source"] == "测试网"
    assert items[0]["ts"] is not None
    assert items[0]["published"].endswith("10:00")


def test_parse_rss_falls_back_to_date_in_url():
    """不少媒体 pubDate 是空的，时间只藏在 URL 里。"""
    content = _rss(_item("无 pubDate", "https://example.com/2026/09-15/c_1.htm"))
    items = news._parse_rss(content, "测试网")
    assert items[0]["ts"] == datetime(2026, 9, 15)


def test_parse_rss_skips_items_missing_title_or_link():
    content = _rss("<item><title>没有链接</title></item>", _item("正常", "https://example.com/ok"))
    items = news._parse_rss(content, "测试网")
    assert [it["title"] for it in items] == ["正常"]


# ── 新鲜度闸门（本轮核心） ────────────────────────────────────────


def test_parse_pubdate_normalizes_to_beijing_regardless_of_server_tz():
    """回归：时间必须换算到**北京时间**，而不是运行机器的本地时区。

    原先写的是 ``dt.astimezone()``（无参 = 取本机时区）：本机 UTC+8 全绿，
    CI（UTC）上同一条「+0800 的 10:00」变成 02:00。部署形态是 Docker + gunicorn，
    容器默认时区同样是 UTC——照原样上线会把中新网 15:38 的头条显示成 07:38。
    用两个等价时刻断言，任何时区的机器上结果都必须一致。
    """
    assert news._parse_pubdate("Mon, 15 Sep 2026 10:00:00 +0800") == datetime(2026, 9, 15, 10, 0)
    assert news._parse_pubdate("Mon, 15 Sep 2026 02:00:00 +0000") == datetime(2026, 9, 15, 10, 0)


def test_parse_pubdate_treats_naive_time_as_beijing():
    """少数源不带时区，按北京时间理解（而不是当成本机时间）。"""
    assert news._parse_pubdate("Mon, 15 Sep 2026 10:00:00") == datetime(2026, 9, 15, 10, 0)


def test_keep_fresh_drops_stale_and_undated_items():
    """过期条目丢弃；**时间无法验证的也丢弃**——拿不准就不采用。"""
    now = datetime(2026, 9, 15, 12, 0)
    items = [
        {"title": "今天", "ts": now - timedelta(hours=1)},
        {"title": "很旧", "ts": now - timedelta(days=1371)},
        {"title": "时间未知", "ts": None},
    ]
    kept = news._keep_fresh(items, now)
    assert [it["title"] for it in kept] == ["今天"]


def test_fetch_news_discards_zombie_feed(monkeypatch):
    """回归：僵尸源 HTTP 200、XML 完整，但内容停更三年——必须一条都不留。

    这正是新华网/人民网 RSS 的真实状态（实测停更 1371 / 467 天），
    没有闸门的话模型会拿旧闻当今天的头条播报，且任何一层都看不出异常。
    """
    stale = _rss(_item("三年前的新闻", "https://example.com/2022-12/14/c_1.htm"))
    _patch_source(monkeypatch, "僵尸网", "https://example.com/rss.xml", stale)
    assert news.fetch_news() == []


def test_fetch_news_keeps_fresh_feed(monkeypatch):
    now = news._now_bj()
    fresh = _rss(_item("刚刚发布", "https://example.com/today", _rfc822(now - timedelta(hours=2))))
    _patch_source(monkeypatch, "活跃网", "https://example.com/rss.xml", fresh)
    items = news.fetch_news()
    assert [it["title"] for it in items] == ["刚刚发布"]


def test_fetch_news_returns_empty_when_all_sources_fail(monkeypatch):
    _patch_source(monkeypatch, "挂掉的网", "https://example.com/rss.xml", b"", status_code=500)
    assert news.fetch_news() == []


def test_fetch_news_returns_empty_on_broken_xml(monkeypatch):
    _patch_source(monkeypatch, "坏 XML", "https://example.com/rss.xml", b"<rss><channel>")
    assert news.fetch_news() == []


def test_fetch_news_uses_cache_within_ttl(monkeypatch):
    """抓取免费但没必要每次提问都抓——缓存期内不再发第二次请求。"""
    now = news._now_bj()
    fresh = _rss(_item("缓存命中", "https://example.com/today", _rfc822(now - timedelta(hours=1))))
    calls = []

    def _counting_get(*args, **kwargs):
        calls.append(1)
        return FakeResponse(fresh)

    monkeypatch.setattr(news, "NEWS_FEEDS", [{"name": "活跃网", "url": "https://example.com/rss.xml"}])
    monkeypatch.setattr(news.requests, "get", _counting_get)

    assert news.fetch_news()
    assert news.fetch_news()
    assert len(calls) == 1


# ── 格式化与检索 ────────────────────────────────────────────────


def test_format_news_includes_source_time_title_and_url():
    items = [{"title": "标题一", "url": "https://example.com/1", "source": "中新网", "published": "09-15 15:04"}]
    text = news.format_news(items)
    assert "中新网" in text and "标题一" in text and "https://example.com/1" in text and "09-15 15:04" in text


def test_format_news_empty_returns_empty_string():
    assert news.format_news([]) == ""


def test_search_news_matches_chinese_by_bigram(monkeypatch):
    now = news._now_bj()
    content = _rss(
        _item("运载火箭发射成功", "https://example.com/a", _rfc822(now)),
        _item("农产品价格指数下降", "https://example.com/b", _rfc822(now)),
    )
    _patch_source(monkeypatch, "活跃网", "https://example.com/rss.xml", content)
    hits = news.search_news("火箭发射")
    assert [it["title"] for it in hits] == ["运载火箭发射成功"]


def test_search_news_returns_empty_for_unrelated_query(monkeypatch):
    now = news._now_bj()
    _patch_source(
        monkeypatch, "活跃网", "https://example.com/rss.xml", _rss(_item("火箭发射", "https://e.com/a", _rfc822(now)))
    )
    assert news.search_news("量子计算") == []
