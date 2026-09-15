"""
Circle 全局配置
所有可调参数集中管理，修改后重启即生效。
"""

# ── 对话上下文 ──
CONTEXT_ROUNDS = 8  # 发送给 API 的最近对话轮数（1轮 = 1问1答）
MAX_ROUNDS = 50  # 单会话熔断上限（超出自动丢弃最旧轮次）
SUMMARY_THRESHOLD = 20  # 触发摘要的轮数阈值（超过后自动生成摘要压缩历史）

# ── 模型参数 ──
MODEL_NAME = "deepseek-chat"
TEMPERATURE = 0.7
MAX_TOKENS = 1200  # 单次回答最大 token 数
MAX_CONTINUE_ROUNDS = 2  # 回答被 MAX_TOKENS 截断时，自动续写的最大轮数（每轮仍按 MAX_TOKENS）
API_TIMEOUT = 30  # API 调用超时（秒）
API_BASE_URL = "https://api.deepseek.com/v1/chat/completions"

# 检索触发词，分强弱两档（判定逻辑见 app.py 的 _should_search）：
# 强意图——本身就是「我要找信息」的明确信号，不受消息长度限制。
# 中文四个字已是完整语义（如「今日新闻」），统一套用长度阈值会误杀短查询。
SEARCH_STRONG_KEYWORDS = ["新闻", "查询", "搜索", "搜一下", "查一下"]
# 弱意图——可能只是闲聊里的顺带词（"我今天很累"），需配合长度兜底，避免误触发
SEARCH_WEAK_KEYWORDS = ["最新", "今天", "最近"]

# ── 每日新闻（官方媒体 RSS，0 token / 0 key / 零第三方依赖） ──
# 为什么不用搜索 API：DeepSeek 原生 web_search 在部分账号上不生效——tools 声明被服务端
# 收下，却从不产生 web_search_call；DuckDuckGo 在国内网络不可达。
#
# 实测（2026-09-15）：多数中文媒体 RSS 已是「僵尸源」——HTTP 200、XML 正常，内容却是几年前的。
# 下列源实测停更，故未启用；若日后恢复，加回 NEWS_FEEDS 即可，新鲜度闸门会自动放行：
#   新华网 时政 http://www.xinhuanet.com/politics/news_politics.xml  停更 1371 天
#   人民网 时政 http://www.people.com.cn/rss/politics.xml            停更 467 天
#   新浪 国内要闻 http://rss.sina.com.cn/news/china/focus15.xml      停更 2913 天
NEWS_FEEDS = [
    {"name": "中新网", "url": "https://www.chinanews.com.cn/rss/scroll-news.xml"},
]
NEWS_MAX_AGE_DAYS = 3  # 新鲜度闸门：超过这个天数的条目一律丢弃（僵尸源靠它兜住）
NEWS_CACHE_TTL = 600  # 头条缓存秒数：抓取虽然免费，但没必要每次提问都抓一遍
NEWS_MAX_PER_FEED = 8  # 每个源最多取几条
NEWS_MAX_TOTAL = 10  # 合并后最多给模型几条
# ── 网页抓取 / 权威检索（0 token 优先；「读正文」是唯一烧 token 的自愿环节） ──
# 用户贴链接后，命中任一「读」词才抓正文送模型；否则只给标题 + 可点链接（0 token）
READ_TRIGGER_KEYWORDS = [
    "读一下",
    "读一读",
    "帮我读",
    "读了",
    "读这篇",
    "总结",
    "摘要",
    "讲重点",
    "概括",
    "提取要点",
    "讲了什么",
    "内容是什么",
    "说下这篇",
    "说说这篇",
    "这篇文章",
]
# 命中任一权威词 → 走免费来源检索并给可点出处（0 token），不读正文
AUTHORITY_TRIGGER_KEYWORDS = ["权威", "官方", "可靠", "可信", "出处", "正规", "权威性"]
READ_MAX_CHARS = 1200  # 每次读一篇时，最多送入模型的正文片段字符数（切片，非整篇）
FETCH_TIMEOUT = 8  # 抓取超时（秒）
FETCH_MAX_BYTES = 2 * 1024 * 1024  # 单次抓取体积上限 2MB
MAX_FETCH_URLS = 2  # 一次最多抓取/预览几个链接

# ── 输入校验 ──
MAX_MSG_LEN = 2000  # 单条消息最大字符数
MAX_MSG_COUNT = 50  # 单次请求最多携带消息数

# ── 文件上传 ──
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_TEXT_LEN = 8000  # 提取文本截断长度

# ── 限流与成本控制 ──
RATE_LIMIT_PER_MINUTE = 10  # 每分钟最大请求数（按 IP，60 秒滑动窗口）
DAILY_TOKEN_BUDGET = 1000000  # 每日 token 预算（超出后返回 429）
