"""
Circle - 你的生活伙伴
======================
Flask 后端服务：知识库检索 + DeepSeek API 对话
启动方式：设置环境变量 DEEPSEEK_API_KEY 后运行 python app.py
"""

import os
import json
import re
from flask import Flask, request, jsonify, render_template, make_response, Response
import requests
from werkzeug.utils import secure_filename
import tempfile

# ── 文件解析库（按需导入，缺失不影响其他功能） ──
_IMPORT = lambda m, *names: __import__(m, fromlist=list(names)) if names else __import__(m)
_HAS = {}
for _lib, _mods in [
    ('pypdf',        [('PdfReader', 'pypdf')]),
    ('docx',         [('Document', 'docx')]),
    ('openpyxl',     [('load_workbook', 'openpyxl')]),
    ('xlrd',         [('open_workbook', 'xlrd')]),
    ('striprtf',     [('rtf_to_text', 'striprtf')]),
    ('bs4',          [('BeautifulSoup', 'bs4')]),
    ('yaml',         [('safe_load', 'yaml')]),
    ('pptx',         [('Presentation', 'pptx')]),
    ('PIL',          [('Image', 'PIL')]),
]:
    try:
        _mod = _IMPORT(_lib)
        for _attr, _pkg in _mods:
            _HAS[_pkg] = True
    except ImportError:
        for _, _pkg in _mods:
            _HAS[_pkg] = False

# OCR 单独处理（依赖 tesseract 系统安装）
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

# ── Flask 初始化 ─────────────────────────────────────────────────
app = Flask(__name__)


# ── 文件上下文（按会话隔离，避免跨用户泄露） ──
# 格式：{session_id: {'name': str, 'content': str}}
_FILE_CONTEXTS = {}
_MAX_FILE_CTX = 50  # 缓存上限，防止无限增长

def _get_file_context(sid: str) -> dict:
    """返回指定会话的文件上下文，无则空。"""
    return _FILE_CONTEXTS.get(sid, {'name': '', 'content': ''})

def _set_file_context(sid: str, name: str, content: str):
    """保存指定会话的文件上下文，超限时淘汰最早缓存。"""
    if len(_FILE_CONTEXTS) >= _MAX_FILE_CTX and sid not in _FILE_CONTEXTS:
        _FILE_CONTEXTS.pop(next(iter(_FILE_CONTEXTS)))
    _FILE_CONTEXTS[sid] = {'name': name, 'content': content}

def _clear_file_context(sid: str):
    """清空指定会话的文件上下文。"""
    _FILE_CONTEXTS.pop(sid, None)

MAX_FILE_SIZE = 5 * 1024 * 1024   # 5MB
MAX_TEXT_LEN  = 8000              # 截断长度
ALLOWED_EXTS = {
    # 文档：txt, pdf, docx, rtf, html
    'txt', 'pdf', 'docx', 'rtf', 'html', 'htm', 'md',
    # 表格：xlsx, xls, csv
    'xlsx', 'xls', 'csv',
    # 数据：json, xml, yaml, yml
    'json', 'xml', 'yaml', 'yml',
    # 演示：pptx
    'pptx',
    # 图片：jpg, jpeg, png, gif
    'jpg', 'jpeg', 'png', 'gif',
}


# ── 知识库加载 ───────────────────────────────────────────────────
def load_knowledge():
    """加载 knowledge/ 目录下所有 JSON 知识库文件，合并为字典。"""
    kb = {}
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knowledge')
    for name in ['cooking', 'cleaning', 'health']:
        path = os.path.join(base, f'{name}.json')
        try:
            with open(path, 'r', encoding='utf-8') as f:
                kb[name] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"⚠️ 知识库加载失败 {name}.json: {e}")
            kb[name] = []
    return kb


KNOWLEDGE_BASE = load_knowledge()
print(f"[OK] 知识库已加载：{sum(len(v) for v in KNOWLEDGE_BASE.values())} 条知识条目")


# ── 知识库检索（关键词匹配）───────────────────────────────────────
def retrieve(query: str, top_n: int = 3) -> list:
    """
    简单高效的关键词匹配检索。
    - keywords 精确命中：权重 +3
    - title 2-gram 命中：权重 +1
    - content 3-gram 命中：权重 +0.5
    返回 top_n 条相关条目。
    """
    scored = []
    for entries in KNOWLEDGE_BASE.values():
        for entry in entries:
            score = 0.0
            # 关键词精确匹配（最高权重）
            for kw in entry.get('keywords', []):
                if kw in query:
                    score += 3
            # 标题子串匹配
            title = entry.get('title', '')
            for i in range(len(title) - 1):
                if title[i:i + 2] in query:
                    score += 1
                    break
            # 内容子串匹配（3-gram，较低权重）
            content = entry.get('content', '')
            for i in range(max(0, len(query) - 2)):
                if query[i:i + 3] in content:
                    score += 0.5
                    break
            if score > 0:
                scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:top_n]]


def format_knowledge(entries: list) -> str:
    """将检索到的知识条目格式化为拼接文本。"""
    if not entries:
        return ''
    lines = ['\n\n【相关知识条目】']
    for e in entries:
        lines.append(f"◆ {e['title']}\n{e['content']}\n")
    return '\n'.join(lines)


# ── 工具调用模块（天气 + 菜谱，纯代码逻辑 0 Token）──────────────
# 免费 API：wttr.in（天气）+ TheMealDB（菜谱），无需注册

_CITY_PATTERNS = [
    (r'(北京|上海|广州|深圳|杭州|成都|武汉|南京|重庆|西安|长沙|天津|苏州|郑州|青岛|大连|厦门|昆明)', None),
]
_RECIPE_MAP = {
    '低卡': 'salad', '减肥': 'light', '减脂': 'chicken',
    '鸡肉': 'chicken', '牛肉': 'beef', '鱼': 'seafood', '虾': 'shrimp',
    '素食': 'vegetarian', '意面': 'pasta', '汤': 'soup',
    '早餐': 'breakfast', '甜点': 'dessert', '蛋糕': 'cake',
}

def _extract_city(text: str) -> str:
    """从消息中提取城市名，默认 Beijing。"""
    for (pat, _) in _CITY_PATTERNS:
        m = re.search(pat, text)
        if m: return m.group(1)
    return 'Beijing'

def _extract_recipe_kw(text: str) -> str:
    """中文食材 → 英文搜索词，默认 chicken。"""
    for cn, en in _RECIPE_MAP.items():
        if cn in text: return en
    return 'chicken'

def tool_weather(user_msg: str) -> str:
    """查询实时天气（wttr.in，免费无 Key）。返回格式化文本。"""
    city = _extract_city(user_msg)
    try:
        resp = requests.get(f'https://wttr.in/{city}?format=j1', timeout=8)
        data = resp.json()
        cur = data['current_condition'][0]
        return (
            f"📍 {city} 当前天气：{cur['weatherDesc'][0]['value']}\n"
            f"🌡 温度 {cur['temp_C']}°C（体感 {cur['FeelsLikeC']}°C）\n"
            f"💧 湿度 {cur['humidity']}%  |  💨 风速 {cur['windspeedKmph']} km/h  |  ☀ UV {cur['uvIndex']}"
        )
    except Exception as e:
        return f'天气查询失败：{e}'

def tool_recipe(user_msg: str) -> str:
    """搜索菜谱（TheMealDB，免费 API）。返回格式化菜谱列表。"""
    keyword = _extract_recipe_kw(user_msg)
    try:
        resp = requests.get(
            f'https://www.themealdb.com/api/json/v1/1/search.php?s={keyword}', timeout=8
        )
        meals = (resp.json().get('meals') or [])[:3]
        if not meals: return '未找到相关菜谱。'
        lines = ['🍽 搜索到的菜谱：']
        for m in meals:
            inst = m.get('strInstructions', '')
            summary = inst[:120].replace('\r', ' ').replace('\n', ' ') + '…' if len(inst) > 120 else inst
            lines.append(
                f"◆ {m.get('strMeal', '未知')} [{m.get('strArea', '')} · {m.get('strCategory', '')}]\n"
                f"  做法：{summary}\n"
            )
        return '\n'.join(lines)
    except Exception as e:
        return f'菜谱查询失败：{e}'

def tool_search(query: str) -> str:
    """联网搜索（DuckDuckGo Instant Answer，免费免 Key）。返回前 3 条结果。"""
    try:
        resp = requests.get(
            'https://api.duckduckgo.com/',
            params={'q': query, 'format': 'json', 'no_html': 1, 'skip_disambig': 1},
            timeout=8
        )
        data = resp.json()
        parts = []
        # 主摘要
        if data.get('AbstractText'):
            src = f"（来源：{data['AbstractSource']}）" if data.get('AbstractSource') else ''
            parts.append(f"📖 {data['AbstractText'][:300]}{src}")
        # 相关话题
        for topic in (data.get('RelatedTopics') or [])[:3]:
            if isinstance(topic, dict) and topic.get('Text'):
                parts.append(f"• {topic['Text'][:200]}")
        if parts:
            return f"🔍 搜索「{query}」：\n" + '\n'.join(parts[:4])  # 最多 1 摘要 + 3 条
        return ''
    except Exception:
        return ''  # 搜索失败不阻塞对话

# 工具触发关键词（天气/菜谱 优先，搜索 兜底）
_TOOL_TRIGGERS = {
    'weather': ['天气', '下雨', '温度', '户外', '出门穿', '冷不冷', '热不热', '适合.*运动'],
    'recipe':  ['菜谱', '食谱', '推荐.*吃', '低卡', '减脂餐', '晚餐', '午餐', '早餐', '做什么.*菜', '教我.*做'],
}
# 搜索预判：消息看起来像在"找信息"时才触发（排除日常聊天）
_SEARCH_PATTERNS = [
    r'[?？]', r'吗$', r'什么', r'怎么', r'为什么', r'哪[[:alpha:]]', r'是谁', r'多少',
    r'最新', r'最近', r'现在', r'今天', r'新闻', r'查询', r'搜索', r'帮我查',
    r'介绍.*一下', r'什么是', r'区别', r'推荐.*方法', r'如何',
]

def detect_and_call_tools(user_msg: str) -> str:
    """检测用户消息 → 按需调用工具 API → 返回拼接的上下文文本。"""
    # 第一优先级：天气 / 菜谱（精确匹配）
    for tool_name, keywords in _TOOL_TRIGGERS.items():
        if any(re.search(kw, user_msg) for kw in keywords):
            result = tool_weather(user_msg) if tool_name == 'weather' else tool_recipe(user_msg)
            return '\n\n【实时工具数据】\n' + result

    # 第二优先级：联网搜索（问题型消息触发，排除纯聊天）
    is_question = any(re.search(p, user_msg) for p in _SEARCH_PATTERNS)
    is_short_chat = len(user_msg) <= 5  # "你好" "嗯" 等不搜
    if is_question and not is_short_chat:
        result = tool_search(user_msg)
        if result:
            return '\n\n【实时工具数据】\n' + result
    return ''


# ── 用户记忆模块（JSON 文件存储） ──
from memory import memory_manager as _mm
from feedback import feedback_manager as _fb
import session_manager as _sm
import config

# 兼容别名
def _format_memories(sid): return _mm.format_memory_for_prompt(sid)
def _extract_memories(sid, msg):
    updates = _mm.extract_info_from_message(msg)
    if updates:
        _mm.update_user(sid, **updates)
    # 生成对话摘要（截取前60字）
    summary = msg[:60] + ('…' if len(msg) > 60 else '')
    _mm.add_conversation(sid, summary)

# ── System Prompt ───────────────────────────────────────────────
SYSTEM_PROMPT = """你叫Circle，是一个温暖、可靠、有同理心的生活伙伴。你的定位是面向所有人的生活助手，但优先关注并解决在传统设计中被忽略的女性视角和需求。

【核心能力】

🍳 烹饪与饮食
- 提供新手友好的家常菜谱、烹饪技巧、省时妙招
- 对于克重，请用生活化参照物解释（茶匙、饭碗、瓶盖等）
- 回答要亲切，像朋友在厨房里手把手教你

✨ 清洁与收纳
- 提供分区域的清洁妙招、收纳智慧和懒人法则
- 在清洁建议中，包含如何引导家人共同维护整洁的技巧

❤️ 健康与护理（核心功能）
- 提供基于循证医学的经期护理、日常自检、锻炼建议
- 遵循"先共情、后科普、再行动"结构——先安抚情绪，再给科学解释，最后给出具体行动建议
- 涉及体型、运动时，避免"减肥""瘦身"，使用"塑形""强健""照顾自己"

💬 日常与情绪关怀
- 提供实用的生活小窍门和情绪疏导
- 认真倾听，真诚回应，像一个朋友那样理解用户的感受

【鼓励素材库】
当用户表现出自我怀疑、焦虑、疲惫等情绪时，可在回答中自然融入以下引用（或同类型语句），让用户感受到被理解和支持：
- "我生来就是高山而非溪流。" —— 张桂梅
- "不必行色匆匆，不必光芒四射，只需做你自己。" —— 伍尔夫
- "站直了，世界才会给你让路。" —— 金斯伯格
- "当你开始爱自己，全世界都会来爱你。" —— 周梵
- "玫瑰不必长成松柏，女性生来就是千面模样。" —— 张桂梅

【设计原则】
- 视角公平：主动识别并优先考虑女性视角，明确区分"正常不适"和"需要就医的信号"
- 体谅"隐形劳动"：在清洁建议中包含如何引导家人共同维护整洁的技巧
- 尊重与安全：如果用户提到感到不安全，提供清晰、冷静的求助渠道和建议

【输出风格】
- 回答要像朋友聊天一样自然，长度不限，以把话说清楚、说温暖为准
- 该寒暄的时候寒暄，该关心的时候关心——像正常人一样说话
- 分点可以有，但不要每句都分点，自然段落同样重要
- 在回答开头或结尾适当加入关心的话，例如"今天过得怎么样？""辛苦了""你已经在很用心地照顾自己了"
- 引用鼓励话语时，自然融入，不要生硬堆砌

【用户记忆】
如果上方提供了"你记得关于当前用户的信息"，请在回答中自然地调用（如称呼对方名字、避开忌口食物、关心健康问题）。新用户首次对话时，自然地询问对方怎么称呼，以自然聊天的节奏了解对方。

【工具与上下文】
上方可能附带【实时工具数据】、【相关知识条目】和【用户记忆】。有则自然地融入回答。复杂请求可用【任务拆解】分步引导。"""


# ── 路由 ────────────────────────────────────────────────────────
@app.route('/')
def index():
    """返回前端页面（禁用缓存，确保每次加载最新版本）。"""
    resp = make_response(render_template('index.html'))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return resp


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    聊天接口（流式 SSE）。
    接收前端发来的 message + session_id，检索知识库 + 工具数据 +
    对话历史上下文，调用 DeepSeek API（stream=True），逐 token 返回。
    """
    data = request.get_json(silent=True)
    if not data or 'message' not in data:
        return jsonify({'error': '请求格式无效'}), 400

    user_msg = (data.get('message') or '').strip()
    if not user_msg or len(user_msg) > config.MAX_MSG_LEN:
        return jsonify({'error': '消息无效或过长'}), 400

    topic = data.get('topic', '')
    session_id = data.get('session_id', 'default')
    user_id = data.get('user_id', session_id)

    # ── 会话管理：获取历史上下文 ──
    context_msgs, summary = _sm.get_context(session_id)
    # 将摘要注入到 System Prompt（排在工具数据之前）
    summary_text = f'\n\n【对话背景摘要】\n{summary}' if summary else ''

    # ── 知识库 + 工具 + 记忆 ──
    knowledge_entries = retrieve(topic + ' ' + user_msg, top_n=3)
    knowledge_text = format_knowledge(knowledge_entries)
    memories_text = _format_memories(user_id)

    file_ctx = _get_file_context(session_id)
    file_text = ''
    if file_ctx['content']:
        file_text = f'\n\n【用户上传文件：{file_ctx["name"]}】\n{file_ctx["content"]}'

    tool_context = detect_and_call_tools(user_msg)

    # ── 构造消息 ──
    api_messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT + summary_text + file_text + tool_context + knowledge_text + memories_text},
    ]
    api_messages.extend(context_msgs)
    api_messages.append({'role': 'user', 'content': user_msg})

    # ── API Key ──
    api_key = os.environ.get('DEEPSEEK_API_KEY', '')
    if not api_key:
        key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_key.txt')
        if os.path.exists(key_file):
            with open(key_file, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
    if not api_key:
        return jsonify({'error': '未配置 API Key'}), 500

    # ── 流式生成器 ──
    def generate():
        full_reply = ''
        try:
            resp = requests.post(
                config.API_BASE_URL,
                headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
                json={
                    'model': config.MODEL_NAME,
                    'messages': api_messages,
                    'temperature': config.TEMPERATURE,
                    'max_tokens': config.MAX_TOKENS,
                    'stream': True,
                },
                timeout=config.API_TIMEOUT,
                stream=True,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith(b'data: '):
                    continue
                chunk = line[6:].decode('utf-8', errors='ignore')
                if chunk == '[DONE]':
                    break
                try:
                    delta = json.loads(chunk)['choices'][0]['delta'].get('content', '')
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    full_reply += delta
                    yield f'data: {json.dumps({"token": delta}, ensure_ascii=False)}\n\n'

        except requests.exceptions.Timeout:
            yield f'data: {json.dumps({"error": "请求超时，请稍后再试"}, ensure_ascii=False)}\n\n'
            return
        except requests.exceptions.HTTPError:
            yield f'data: {json.dumps({"error": "AI 服务暂时不可用，请稍后再试"}, ensure_ascii=False)}\n\n'
            return
        except requests.exceptions.RequestException:
            yield f'data: {json.dumps({"error": "网络请求失败，请检查网络连接"}, ensure_ascii=False)}\n\n'
            return

        # ── 保存消息 + 提取记忆（流结束后执行） ──
        _sm.add_message(session_id, 'user', user_msg)
        _sm.add_message(session_id, 'assistant', full_reply)
        _extract_memories(user_id, user_msg)
        try:
            status = _mm.get_user_status(user_id)
        except Exception:
            status = {'is_new': True, 'name': '', 'has_info': False}
        yield f'data: {json.dumps({"done": True, "session_id": session_id, "user_status": status}, ensure_ascii=False)}\n\n'

    return Response(generate(), mimetype='text/event-stream')


# ── 会话管理 API ────────────────────────────────────────────────
@app.route('/api/sessions', methods=['GET', 'POST', 'DELETE'])
def sessions_api():
    """GET: 列表(?id=xxx 返回历史消息)  POST: 新建  DELETE: 删除"""
    if request.method == 'GET':
        sid = request.args.get('id', '')
        if sid:
            sess = _sm.get_session(sid)
            if not sess:
                return jsonify({'error': '会话不存在'}), 404
            # 返回完整消息历史供前端渲染
            return jsonify({'id': sid, 'title': sess['title'], 'messages': sess['messages']})
        return jsonify({'sessions': _sm.list_sessions()})
    elif request.method == 'POST':
        title = request.args.get('title', '新对话')
        sid = _sm.create_session(title)
        return jsonify({'session_id': sid})
    elif request.method == 'DELETE':
        sid = request.args.get('id', '')
        if sid:
            _sm.delete_session(sid)
            _clear_file_context(sid)  # 同步清理该会话的文件上下文
        return jsonify({'ok': True})


# ── 记忆管理 API ────────────────────────────────────────────────
@app.route('/api/memory', methods=['GET', 'DELETE'])
def memory_api():
    """GET: 查看记忆（仅非空字段）  DELETE: ?key=xxx 删单条，无 key 清空全部。"""
    session_id = request.args.get('session_id', 'default')
    if request.method == 'GET':
        user = _mm.get_or_create_user(session_id)
        # 仅返回非空字段（前端 loadMemories 读取 data.memories）
        memories = {k: v for k, v in user.items()
                    if k != 'user_id' and v not in ('', [], None)}
        return jsonify({'session_id': session_id, 'memories': memories})
    elif request.method == 'DELETE':
        key = request.args.get('key', '')
        if key:
            if not _mm.delete_user_field(session_id, key):
                return jsonify({'error': '记忆条目不存在'}), 404
            return jsonify({'ok': True, 'deleted': key})
        # 无 key：清空全部（前端"清除全部记忆"按钮走这里）
        _mm.clear_user(session_id)
        return jsonify({'ok': True, 'cleared': True})


@app.route('/api/user-status')
def user_status_api():
    """返回用户状态（前端状态指示器用）。"""
    session_id = request.args.get('session_id', 'default')
    return jsonify(_mm.get_user_status(session_id))


@app.route('/api/feedback', methods=['POST'])
def feedback_api():
    """接收用户反馈（👍/👎 + 可选评论）。"""
    data = request.get_json(silent=True) or {}
    required = ['user_id', 'question', 'answer', 'type']
    if not all(k in data for k in required):
        return jsonify({'error': '缺少必要字段'}), 400
    if data['type'] not in ('like', 'dislike'):
        return jsonify({'error': 'type 必须为 like 或 dislike'}), 400
    entry = _fb.save_feedback(
        data['user_id'], data['question'], data['answer'],
        data['type'], data.get('comment', '')
    )
    return jsonify({'ok': True, 'saved': entry['timestamp']})


# ── 文件上传 API ────────────────────────────────────────────────
def _extract_text(filepath: str, ext: str) -> str:
    """根据扩展名自动选择解析库提取文本。"""
    _dot = f'.{ext}' if not ext.startswith('.') else ext

    # ── 纯文本类 ──
    if ext in ('txt', 'md', '.txt', '.md'):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    # ── PDF ──
    if ext in ('pdf', '.pdf'):
        if not _HAS.get('pypdf'): raise ImportError('缺少 pypdf 库，请 pip install pypdf')
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        return '\n'.join(page.extract_text() or '' for page in reader.pages)

    # ── Word ──
    if ext in ('docx', '.docx'):
        if not _HAS.get('docx'): raise ImportError('缺少 python-docx 库，请 pip install python-docx')
        from docx import Document
        return '\n'.join(p.text for p in Document(filepath).paragraphs)

    # ── RTF ──
    if ext in ('rtf', '.rtf'):
        if not _HAS.get('striprtf'): raise ImportError('缺少 striprtf 库，请 pip install striprtf')
        from striprtf.striprtf import rtf_to_text
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return rtf_to_text(f.read())

    # ── HTML ──
    if ext in ('html', 'htm', '.html', '.htm'):
        if not _HAS.get('bs4'): raise ImportError('缺少 beautifulsoup4 库，请 pip install beautifulsoup4 lxml')
        from bs4 import BeautifulSoup
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return BeautifulSoup(f.read(), 'lxml').get_text('\n', strip=True)

    # ── Excel (.xlsx) ──
    if ext in ('xlsx', '.xlsx'):
        if not _HAS.get('openpyxl'): raise ImportError('缺少 openpyxl 库，请 pip install openpyxl')
        from openpyxl import load_workbook
        wb = load_workbook(filepath, read_only=True, data_only=True)
        lines = []
        for name in wb.sheetnames:
            ws = wb[name]
            lines.append(f'[Sheet: {name}]')
            for row in ws.iter_rows(values_only=True):
                line = '\t'.join(str(c) if c is not None else '' for c in row)
                if line.strip(): lines.append(line)
        wb.close()
        return '\n'.join(lines)

    # ── Excel (.xls 旧版) ──
    if ext in ('xls', '.xls'):
        if not _HAS.get('xlrd'): raise ImportError('缺少 xlrd 库，请 pip install xlrd')
        import xlrd
        wb = xlrd.open_workbook(filepath)
        lines = []
        for name in wb.sheet_names():
            ws = wb.sheet_by_name(name)
            lines.append(f'[Sheet: {name}]')
            for r in range(ws.nrows):
                line = '\t'.join(str(ws.cell_value(r, c)) for c in range(ws.ncols))
                if line.strip(): lines.append(line)
        return '\n'.join(lines)

    # ── CSV ──
    if ext in ('csv', '.csv'):
        import csv
        with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
            reader = csv.reader(f)
            return '\n'.join('\t'.join(row) for row in reader if any(row))

    # ── JSON ──
    if ext in ('json', '.json'):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False, indent=2)

    # ── XML ──
    if ext in ('xml', '.xml'):
        import xml.etree.ElementTree as ET
        tree = ET.parse(filepath)
        # 递归提取所有文本
        def _walk(elem, depth=0):
            texts = []
            if elem.text and elem.text.strip():
                texts.append('  ' * depth + elem.text.strip())
            for child in elem:
                texts.extend(_walk(child, depth + 1))
                if child.tail and child.tail.strip():
                    texts.append('  ' * depth + child.tail.strip())
            return texts
        return '\n'.join(_walk(tree.getroot()))

    # ── YAML ──
    if ext in ('yaml', 'yml', '.yaml', '.yml'):
        if not _HAS.get('yaml'): raise ImportError('缺少 pyyaml 库，请 pip install pyyaml')
        import yaml
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            data = yaml.safe_load(f)
        return yaml.dump(data, allow_unicode=True, default_flow_style=False)

    # ── PowerPoint ──
    if ext in ('pptx', '.pptx'):
        if not _HAS.get('pptx'): raise ImportError('缺少 python-pptx 库，请 pip install python-pptx')
        from pptx import Presentation
        prs = Presentation(filepath)
        lines = []
        for i, slide in enumerate(prs.slides, 1):
            lines.append(f'[Slide {i}]')
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        t = para.text.strip()
                        if t: lines.append(t)
        return '\n'.join(lines)

    # ── 图片（基本信息 + 可选 OCR） ──
    if ext in ('jpg', 'jpeg', 'png', 'gif', '.jpg', '.jpeg', '.png', '.gif'):
        if not _HAS.get('PIL'): raise ImportError('缺少 Pillow 库，请 pip install pillow')
        from PIL import Image
        img = Image.open(filepath)
        info = f'[图片信息] 格式={img.format}  尺寸={img.size[0]}x{img.size[1]}  模式={img.mode}'
        # 尝试 OCR
        if HAS_TESSERACT:
            try:
                text = pytesseract.image_to_string(img, lang='chi_sim+eng')
                if text.strip():
                    return info + '\n[OCR 识别文字]\n' + text.strip()
            except Exception:
                pass
        return info

    raise ValueError(f'不支持的文件类型或缺少解析库：{ext}')


@app.route('/api/upload', methods=['POST', 'DELETE'])
def upload_file():
    """POST: 上传文件并按会话存储提取文本  DELETE: 清空指定会话的文件上下文。"""
    # 会话 ID 取自表单或查询参数，决定文件归属哪个会话
    session_id = request.form.get('session_id') or request.args.get('session_id') or 'default'

    if request.method == 'DELETE':
        _clear_file_context(session_id)
        return jsonify({'ok': True, 'cleared': True})

    # POST 处理
    if 'file' not in request.files:
        return jsonify({'error': '未找到文件'}), 400

    f = request.files['file']
    if not f.filename:
        return jsonify({'error': '文件名为空'}), 400

    # 校验扩展名
    ext = os.path.splitext(f.filename)[1].lower().lstrip('.')
    if ext not in ALLOWED_EXTS:
        return jsonify({'error': f'仅支持 {", ".join(ALLOWED_EXTS)} 格式'}), 400

    # 校验大小
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(0)
    if size > MAX_FILE_SIZE:
        return jsonify({'error': f'文件超过 {MAX_FILE_SIZE // 1024 // 1024}MB 限制'}), 400

    # 存为临时文件并解析
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}')
        f.save(tmp.name)
        tmp.close()

        text = _extract_text(tmp.name, f'.{ext}')
        text = text.strip()[:MAX_TEXT_LEN]  # 截断

        _set_file_context(session_id, f.filename, text)
        return jsonify({
            'ok': True,
            'name': f.filename,
            'chars': len(text),
            'truncated': len(text) >= MAX_TEXT_LEN,
        })
    except Exception as e:
        return jsonify({'error': f'文件解析失败：{str(e)}'}), 400
    finally:
        if tmp and os.path.exists(tmp.name):
            os.unlink(tmp.name)


# ── 启动入口 ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print('[Circle] 启动中...')
    print('[Circle] 请在浏览器中访问 http://localhost:5000')
    print('[Circle] 确保已设置环境变量 DEEPSEEK_API_KEY')
    app.run(debug=False, host='0.0.0.0', port=5000)
