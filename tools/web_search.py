"""
DeepSeek 原生联网搜索（Responses API）
调用 DeepSeek Responses API 并开启 web_search 工具，让 Circle 获取实时信息。

用法：
    from tools import web_search

    web_search.search_deepseek("今天有什么新闻")          # 非流式，一次返回完整回答
    for delta in web_search.stream_search("今天有什么新闻"):  # 流式，逐 token 返回
        ...

说明：
- API Key 从环境变量 DEEPSEEK_API_KEY 读取，未设置时回退到项目根目录的 api_key.txt。
- openai SDK 采用延迟导入，未安装 openai 时不影响 app 其他功能。
"""

import os

from config import API_TIMEOUT, SEARCH_API_BASE, SEARCH_INSTRUCTIONS, SEARCH_MODEL


def _get_api_key() -> str:
    """从环境变量读取 API Key；未设置时回退到项目根目录的 api_key.txt。"""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        key_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api_key.txt")
        if os.path.exists(key_file):
            with open(key_file, "r", encoding="utf-8") as f:
                key = f.read().strip()
    return key


def _build_client():
    """构造 openai SDK 客户端（延迟导入，保持轻量，超时对齐 config.API_TIMEOUT）。"""
    from openai import OpenAI

    return OpenAI(api_key=_get_api_key(), base_url=SEARCH_API_BASE, timeout=API_TIMEOUT)


def _request_kwargs(query: str, instructions: str) -> dict:
    """
    构造 Responses API 请求参数。
    instructions 恒为：女性视角搜索指令（config.SEARCH_INSTRUCTIONS）拼接调用方上下文（如有）。
    """
    full_instructions = SEARCH_INSTRUCTIONS
    if instructions:
        full_instructions += "\n\n" + instructions
    return {
        "model": SEARCH_MODEL,
        "input": query,
        "instructions": full_instructions,
        "tools": [{"type": "web_search"}],
    }


def search_deepseek(query: str, instructions: str = "") -> str:
    """
    调用 DeepSeek Responses API 进行联网搜索（非流式，一次返回完整回答）。

    Args:
        query: 用户问题
        instructions: 系统指令（可选，会拼接在女性视角指令之后）

    Returns:
        str: 包含搜索结果的回答
    """
    client = _build_client()
    response = client.responses.create(**_request_kwargs(query, instructions))
    return response.output_text


def stream_search_events(query: str, instructions: str = ""):
    """
    流式联网搜索（低层接口）：逐事件产出，供需要读取用量等元数据的调用方使用。

    主要事件类型（openai SDK）：
    - response.output_text.delta：回答增量文本，字段 delta
    - response.completed：流结束事件，字段 response.usage.total_tokens
    """
    client = _build_client()
    stream = client.responses.create(**_request_kwargs(query, instructions), stream=True)
    yield from stream


def stream_search(query: str, instructions: str = ""):
    """流式联网搜索：逐 token 产出回答文本（str）。"""
    for event in stream_search_events(query, instructions):
        if getattr(event, "type", "") == "response.output_text.delta":
            text = getattr(event, "delta", "")
            if text:
                yield text
