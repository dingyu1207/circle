"""
网页抓取模块：Circle「读网页」的底层能力（纯 HTTP，0 LLM token）。

用法：
    from tools import web_fetch

    web_fetch.extract_urls("看这个 https://example.com/a")   # -> ["https://example.com/a"]
    web_fetch.fetch_title(url)        # 只抓 <title>（0 token 预览用）
    info = web_fetch.fetch_article(url)         # 抓正文纯文本（按需读时使用）
    web_fetch.pick_passages(info["text"], "痛经怎么办", 1200)  # 只留最相关片段

要点：
- 解析逻辑与 file_parser 的 HTML 分支一致（bs4 + lxml），剥掉 script/style/nav/footer。
- 抓取结果缓存在内存（key = url），重复问同一链接不再发网络请求，也不重复烧读 token。
- 仅允许 http/https，且拒绝本机/内网地址（SSRF 基础防护）。
- 绝不把整篇正文送模型：pick_passages 只取与关注点最相关的段落，
  这是「读网页」唯一烧 token 的环节里省 token 的关键。
"""

import ipaddress
import re
from urllib.parse import urlsplit

import requests

from config import FETCH_MAX_BYTES, FETCH_TIMEOUT

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 Circle/1.0"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# 抓取缓存：url -> {"title": str, "text": str}（仅内存，不落盘）
_CACHE = {}
_MAX_CACHE = 100

_URL_RE = re.compile(r"https?://[^\s<>\"'()\[\]{}，。；：！？、」』）】]+")


def _is_safe_url(url: str) -> bool:
    """仅允许 http/https，且拒绝本机/内网回环地址（SSRF 基础防护）。"""
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.scheme not in ("http", "https"):
        return False
    host = (parts.hostname or "").lower()
    if not host:
        return False
    if host == "localhost":
        return False
    stripped = host.strip("[]")
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", stripped):
        try:
            ip = ipaddress.ip_address(stripped)
        except ValueError:
            return True
        return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)
    return True


def extract_urls(text: str) -> list:
    """从文本中提取去重后的 http(s) 链接。"""
    if not text:
        return []
    seen, out = set(), []
    for m in _URL_RE.finditer(text):
        url = m.group(0).rstrip(".,;:!?")
        if url in seen or not _is_safe_url(url):
            continue
        seen.add(url)
        out.append(url)
    return out[:10]  # 上限保护，防极端输入


def _get(url: str) -> bytes:
    """GET 并返回正文字节（带体积上限，防超大页面拖垮进程）。"""
    resp = requests.get(url, headers=_HEADERS, timeout=FETCH_TIMEOUT, stream=True)
    resp.raise_for_status()
    chunks = []
    size = 0
    for chunk in resp.iter_content(chunk_size=65536):
        chunks.append(chunk)
        size += len(chunk)
        if size > FETCH_MAX_BYTES:
            break
    resp.close()
    return b"".join(chunks)


def _parse_html(raw: bytes) -> tuple:
    """解析 HTML：返回 (title, 正文纯文本)。解析库延迟导入，保持模块轻量。"""
    from bs4 import BeautifulSoup  # 已列入 requirements

    html = raw.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    # 剥掉导航/脚本/页脚等非正文噪音
    for tag in soup(["script", "style", "noscript", "iframe", "nav", "footer", "header", "aside"]):
        tag.decompose()
    node = soup.find("article") or soup.find("main") or soup.body or soup
    title = ""
    if soup.title and soup.title.string:
        title = " ".join(soup.title.string.split())
    text = node.get_text("\n", strip=True) if node else ""
    return title, text


def _remember(url: str, *, title: str, text: str):
    """写入内存缓存，超出上限时淘汰最早一条。"""
    if len(_CACHE) >= _MAX_CACHE:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[url] = {"title": title, "text": text}


def fetch_title(url: str) -> dict:
    """只抓 <title>（0 token 预览）。成功返回 {"url","title"}，失败返回 {}。"""
    cache = _CACHE.get(url)
    if cache and cache.get("title"):
        return {"url": url, "title": cache["title"]}
    if not _is_safe_url(url):
        return {}
    try:
        raw = _get(url)
        title, text = _parse_html(raw)
    except Exception:
        return {}
    if not title:
        return {}
    # 顺带缓存正文：用户稍后说「读一下」时无需再次联网
    _remember(url, title=title, text=text)
    return {"url": url, "title": title}


def fetch_article(url: str) -> dict:
    """抓取正文纯文本。成功返回 {"url","title","text"}，失败返回 {}。"""
    cache = _CACHE.get(url)
    if cache:
        return {"url": url, "title": cache.get("title") or url, "text": cache.get("text") or ""}
    if not _is_safe_url(url):
        return {}
    try:
        raw = _get(url)
        title, text = _parse_html(raw)
    except Exception:
        return {}
    if not text.strip():
        return {}
    _remember(url, title=title, text=text)
    return {"url": url, "title": title, "text": text}


def pick_passages(text: str, query: str, max_chars: int = 1200) -> str:
    """把正文切成小段，用 N-gram 打分只留与 query 最相关的片段（省 token 关键）。

    没有任何段落明显相关时（如只是要一篇的整体概括），退回正文开头，
    保证模型有内容可答。
    """
    if not text:
        return ""
    chunks = _chunk_lines(text, size=300)
    if not chunks:
        return ""
    scored = []
    for i, ch in enumerate(chunks):
        s = _score_chunk(ch, query)
        if s > 0:
            scored.append((s, i, ch))
    if not scored:
        return text[:max_chars]
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = []
    budget = max_chars
    for _s, i, ch in scored:
        if budget <= 0:
            break
        selected.append((i, ch))
        budget -= len(ch)
    selected.sort(key=lambda x: x[0])  # 恢复原文顺序
    return "\n".join(ch for _, ch in selected)[:max_chars]


def _chunk_lines(text: str, size: int = 300) -> list:
    """按行聚成大小约为 size 字符的段落块。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chunks = []
    cur, cur_len = [], 0
    for ln in lines:
        cur.append(ln)
        cur_len += len(ln) + 1
        if cur_len >= size:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _score_chunk(chunk: str, query: str) -> float:
    """字符 N-gram 相关度打分（与知识库检索 retrieve 同思路，更轻量）。"""
    c = re.sub(r"\s+", "", chunk)
    q = re.sub(r"\s+", "", query)
    if not c or not q:
        return 0.0
    score = 0.0
    if q in c:
        score += 3.0
    for i in range(len(q) - 1):
        if q[i : i + 2] in c:
            score += 1.0
    if len(q) >= 3:
        for i in range(len(q) - 2):
            if q[i : i + 3] in c:
                score += 0.5
    return score
