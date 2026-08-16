"""
Circle 全局配置
所有可调参数集中管理，修改后重启即生效。
"""

# ── 对话上下文 ──
CONTEXT_ROUNDS = 8       # 发送给 API 的最近对话轮数（1轮 = 1问1答）
MAX_ROUNDS = 50          # 单会话熔断上限（超出自动丢弃最旧轮次）
SUMMARY_THRESHOLD = 20   # 触发摘要的轮数阈值（超过后自动生成摘要压缩历史）

# ── 模型参数 ──
MODEL_NAME = 'deepseek-chat'
TEMPERATURE = 0.7
MAX_TOKENS = 800
API_TIMEOUT = 30         # API 调用超时（秒）
API_BASE_URL = 'https://api.deepseek.com/v1/chat/completions'

# ── 输入校验 ──
MAX_MSG_LEN = 2000       # 单条消息最大字符数
MAX_MSG_COUNT = 50       # 单次请求最多携带消息数

# ── 文件上传 ──
MAX_FILE_SIZE = 5 * 1024 * 1024   # 5MB
MAX_TEXT_LEN = 8000               # 提取文本截断长度