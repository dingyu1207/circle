"""用户记忆模块测试：读写、更新、规则提取、格式化、单条/全部删除。"""

import pytest

from memory import memory_manager as mm


@pytest.fixture
def isolated_mem(monkeypatch, tmp_path):
    """将记忆文件重定向到临时目录，避免污染真实用户数据。"""
    monkeypatch.setattr(mm, "_MEMORY_FILE", str(tmp_path / "test_memory.json"))


def test_get_or_create_user(isolated_mem):
    user = mm.get_or_create_user("u1")
    assert user["name"] == ""
    assert user["preferences"] == ""
    assert user["health_info"] == ""
    assert user["conversation_history"] == []


def test_update_user_persists(isolated_mem):
    mm.update_user("u1", name="小美", preferences="不吃香菜")
    user = mm.get_or_create_user("u1")
    assert user["name"] == "小美"
    assert user["preferences"] == "不吃香菜"


def test_update_user_ignores_empty(isolated_mem):
    mm.update_user("u1", name="小美")
    mm.update_user("u1", name="")  # 空值不应覆盖已有记忆
    assert mm.get_or_create_user("u1")["name"] == "小美"


def test_extract_info_from_message():
    assert mm.extract_info_from_message("我叫小美")["name"] == "小美"
    assert "香菜" in mm.extract_info_from_message("我不吃香菜")["preferences"]
    assert "痛经" in mm.extract_info_from_message("我痛经")["health_info"]


def test_format_memory_for_prompt(isolated_mem):
    assert mm.format_memory_for_prompt("u1") == ""
    mm.update_user("u1", name="小美")
    out = mm.format_memory_for_prompt("u1")
    assert "小美" in out


def test_delete_user_field(isolated_mem):
    mm.update_user("u1", name="小美", preferences="不吃香菜")
    assert mm.delete_user_field("u1", "preferences") is True
    user = mm.get_or_create_user("u1")
    assert user["preferences"] == ""
    assert user["name"] == "小美"  # 其他字段保留
    assert mm.delete_user_field("u1", "health_info") is False  # 空字段 → False


def test_clear_user(isolated_mem):
    mm.update_user("u1", name="小美", health_info="痛经")
    assert mm.clear_user("u1") is True
    user = mm.get_or_create_user("u1")
    assert user["name"] == ""
    assert user["health_info"] == ""
