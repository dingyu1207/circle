"""web_fetch 单测：URL 提取、抓取/正文抽取、段落切片、缓存、SSRF 防护。"""

from tools import web_fetch as wf


class FakeGet:
    """模拟 requests.get 的响应对象（流式读取正文字节）。"""

    def __init__(self, body: bytes, content_type="text/html; charset=utf-8"):
        self._body = body
        self._content_type = content_type
        self.closed = False

    def raise_for_status(self):
        pass

    @property
    def headers(self):
        return {"Content-Type": self._content_type}

    def iter_content(self, chunk_size=65536):
        body = self._body
        for i in range(0, len(body), chunk_size):
            yield body[i : i + chunk_size]

    def close(self):
        self.closed = True


HTML = """<html><head><title>  经期 护理与热敷指南  </title></head>
<body><nav>导航垃圾站</nav>
<article><h1>经期护理指南</h1>
<p>经期痛经时，用热水袋热敷小腹可以明显缓解不适。</p>
<p>注意多休息、少喝冰饮，保持规律作息。</p>
<p>这篇文章和你毫无关系的占位段落。</p>
</article><footer>版权页脚</footer></body></html>"""


# ── extract_urls ──


def test_extract_urls_basic():
    urls = wf.extract_urls("看这个 https://example.com/a 和 https://sub.example.org/b?a=1 两篇")
    assert urls == ["https://example.com/a", "https://sub.example.org/b?a=1"]


def test_extract_urls_ignores_bad_schemes_and_chinese_punct():
    urls = wf.extract_urls("ftp://example.com/x javascript:void(0) http://example.com/好。 结尾")
    assert urls == ["http://example.com/好"]


def test_extract_urls_dedupe():
    urls = wf.extract_urls("https://example.com/a 重复一次 https://example.com/a")
    assert urls == ["https://example.com/a"]


def test_extract_urls_blocks_internal():
    assert wf.extract_urls("http://localhost/admin") == []
    assert wf.extract_urls("http://127.0.0.1/x") == []
    assert wf.extract_urls("http://192.168.1.10/x") == []


def test_is_safe_url():
    assert wf._is_safe_url("https://example.com/a")
    assert wf._is_safe_url("http://example.com.cn")
    assert not wf._is_safe_url("ftp://example.com/x")
    assert not wf._is_safe_url("http://localhost/a")
    assert not wf._is_safe_url("http://127.0.0.1/a")
    assert not wf._is_safe_url("http://192.168.0.1/a")


# ── fetch_title / fetch_article（mock 网络） ──


def _patch_get(monkeypatch, body=HTML.encode("utf-8")):
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        return FakeGet(body)

    monkeypatch.setattr(wf.requests, "get", fake_get)
    return calls


def test_fetch_title_returns_stripped_title(monkeypatch):
    _patch_get(monkeypatch)
    info = wf.fetch_title("https://example.com/health")
    assert info["url"] == "https://example.com/health"
    assert info["title"] == "经期 护理与热敷指南"  # 首尾空白被清理


def test_fetch_title_failure_returns_empty(monkeypatch):
    def boom(url, **kw):
        raise RuntimeError("timeout")

    monkeypatch.setattr(wf.requests, "get", boom)
    assert wf.fetch_title("https://example.com/x") == {}


def test_fetch_article_strips_noise(monkeypatch):
    _patch_get(monkeypatch)
    info = wf.fetch_article("https://example.com/health")
    text = info["text"]
    assert "经期痛经时" in text  # 正文保留
    assert "导航垃圾站" not in text  # nav 被剥
    assert "版权页脚" not in text  # footer 被剥


def test_fetch_article_failure_returns_empty(monkeypatch):
    def boom(url, **kw):
        raise RuntimeError("timeout")

    monkeypatch.setattr(wf.requests, "get", boom)
    assert wf.fetch_article("https://example.com/x") == {}


def test_fetch_caches_and_does_not_refetch(monkeypatch):
    url = "https://example.com/cache-check"
    calls = _patch_get(monkeypatch)
    assert wf.fetch_article(url)
    assert wf.fetch_article(url)
    assert len(calls) == 1  # 第二次命中缓存，不再发请求
    # 清理缓存，避免影响其他用例
    wf._CACHE.pop(url, None)


# ── pick_passages（省 token：只送相关片段） ──


def test_pick_passages_returns_only_related():
    # 每段是独立长行 → 各自成块，便于验证「只挑相关段、裁掉无关段」
    text = (
        "第一节：天气晴朗适合出门跑步散步呼吸新鲜空气。\n"
        + "第二节：经期痛经时可以热敷小腹，也可以多喝温水缓解，注意休息。" * 20
        + "\n第三节：旅行攻略推荐京都奈良与小众景点打卡拍照。"
    )
    picked = wf.pick_passages(text, "经期痛经怎么缓解", max_chars=1200)
    assert "热敷小腹" in picked
    assert "小众景点" not in picked  # 无关段落被裁掉


def test_pick_passages_falls_back_to_head_when_unrelated():
    text = "第一段内容。\n" + "第二段内容。" * 30
    picked = wf.pick_passages(text, "完全不相关的话题甲", max_chars=200)
    assert picked  # 不空答
    assert len(picked) <= 200


def test_pick_passages_respects_max_chars():
    text = ("痛经怎么缓解。" * 20 + "\n") * 5
    picked = wf.pick_passages(text, "痛经缓解", max_chars=100)
    assert len(picked) <= 100
