"""多会话管理测试：创建、追加、上下文窗口、自动标题、列表排序、删除。"""

import pytest

import session_manager as sm


@pytest.fixture
def isolated_sessions(monkeypatch, tmp_path):
    """将会话文件重定向到临时目录，避免污染真实数据。"""
    monkeypatch.setattr(sm, '_SESSIONS_FILE', str(tmp_path / 'test_sessions.json'))


def test_create_session(isolated_sessions):
    sid = sm.create_session('测试会话')
    sess = sm.get_session(sid)
    assert sess is not None
    assert sess['title'] == '测试会话'
    assert sess['messages'] == []


def test_add_message_and_get_context(isolated_sessions):
    sid = sm.create_session()
    sm.add_message(sid, 'user', '你好')
    sm.add_message(sid, 'assistant', '你好呀，今天过得怎么样？')
    ctx, summary = sm.get_context(sid)
    assert len(ctx) == 2
    assert ctx[0]['role'] == 'user'
    assert ctx[1]['role'] == 'assistant'


def test_context_window_limits_rounds(isolated_sessions):
    """超过 CONTEXT_ROUNDS 轮时，只返回最近 N 轮。"""
    sid = sm.create_session()
    for i in range(12):  # 12 轮 > 阈值
        sm.add_message(sid, 'user', f'问题{i}')
        sm.add_message(sid, 'assistant', f'回答{i}')
    ctx, _ = sm.get_context(sid)
    user_rounds = len([m for m in ctx if m['role'] == 'user'])
    assert user_rounds == sm.config.CONTEXT_ROUNDS


def test_auto_title_from_first_message(isolated_sessions):
    sid = sm.create_session()
    sm.add_message(sid, 'user', '今天晚餐吃什么比较好呢')
    sess = sm.get_session(sid)
    assert sess['title'] == '今天晚餐吃什么比较好呢'


def test_list_and_delete(isolated_sessions):
    sid = sm.create_session('A')
    sm.add_message(sid, 'user', '第一条')
    lst = sm.list_sessions()
    assert any(s['id'] == sid for s in lst)
    assert any(s['msg_count'] == 1 for s in lst)
    sm.delete_session(sid)
    assert sm.get_session(sid) is None