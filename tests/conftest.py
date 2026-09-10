"""
pytest 全局配置：确保项目根目录可导入，供 tests/ 各用例 import app 等模块。
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    """给测试兜底一个假 key，使用例不依赖机器上是否配置了真实 DEEPSEEK_API_KEY。

    否则会出现「本地机器设了 key → 全绿；CI 没设 → 12 个用例失败」的环境依赖问题
    （应用在调模型前会先校验 key，缺 key 时测试根本走不到被 monkeypatch 的网络调用）。
    需要验证「未配置 API Key」分支的用例，可在用例内 monkeypatch.delenv 自行覆盖。
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-not-real")
