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

# ── 联网搜索（DeepSeek Responses API） ──
SEARCH_MODEL = "deepseek-v4-flash"  # 目前仅该模型支持原生联网搜索
SEARCH_API_BASE = "https://api.deepseek.com"  # openai SDK 会自动追加 /responses
# 命中任一关键词即走联网搜索（过短的闲聊消息不会触发，见 app.py 的 _should_search）
SEARCH_TRIGGER_KEYWORDS = ["最新", "今天", "最近", "新闻", "查询", "搜一下", "查一下"]
# 联网搜索的系统指令：优先呈现女性视角（可在 config 中直接调整措辞）
SEARCH_INSTRUCTIONS = (
    "你是一个关注女性视角的搜索助手。在搜索和回答问题时，请优先关注和呈现："
    "1. 涉及女性成就、女性领导力、女性创业的新闻；"
    "2. 女性健康、权益保护、教育就业相关内容；"
    "3. 女性在科技、经济、文化等领域的贡献；"
    "4. 反映女性生活状态、职场发展、社会地位变化的报道。"
    "如果搜索结果中缺乏女性视角的内容，请如实告知用户，"
    "并尽可能从已有信息中提取与女性相关的部分。"
    "当用户询问理财、投资、保险、养老等金融问题时，优先搜索面向女性用户的"
    "金融知识和建议，关注资金安全、稳健理财、长期规划等方向。"
    "无论搜索到什么内容，回答都不得出现低俗、色情、性暗示或贬低性的玩笑与措辞，保持客观、体贴与尊重。"
    "\n\n关于日期：请严格遵守附加 instructions 中给出的【当前日期】，"
    "不要编造或猜测今天是几号。新闻中的「昨天」「今天」「本周」等相对时间词，"
    "都应以【当前日期】为准进行转换。"
)

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
