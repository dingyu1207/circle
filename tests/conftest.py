"""
pytest 全局配置：确保项目根目录可导入，供 tests/ 各用例 import app 等模块。
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
