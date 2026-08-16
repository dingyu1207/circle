"""API 冒烟测试：状态码、响应格式、会话流、文件会话隔离、记忆删除。"""

import json
from io import BytesIO

import pytest

import app as app_module
from memory import memory_manager as mm


class FakeStreamResponse:
    """模拟 DeepSeek SSE 流式响应，避免测试真实调用 API。"""

    def __init__(self, chunks=('你', '好', '今天', '过得', '怎么样')):
        self._lines = []
        for c in chunks:
            payload = json.dumps({'choices': [{'delta': {'content': c}}]}, ensure_ascii=False)
            self._lines.append(b'data: ' + payload.encode('utf-8'))
        self._lines.append(b'data: [DONE]')

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self._lines)


@pytest.fixture
def client(monkeypatch, tmp_path):
    """隔离数据文件 + 返回 Flask 测试客户端。"""
    monkeypatch.setattr(app_module._sm, '_SESSIONS_FILE', str(tmp_path / 'sessions.json'))
    monkeypatch.setattr(app_module._mm, '_MEMORY_FILE', str(tmp_path / 'memory.json'))
    # 测试中不触发真实联网搜索
    monkeypatch.setattr(app_module, 'detect_and_call_tools', lambda msg: '')
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def test_index_ok(client):
    resp = client.get('/')
    assert resp.status_code == 200
    assert 'Circle' in resp.get_data(as_text=True)


def test_chat_missing_message_400(client):
    resp = client.post('/api/chat', json={})
    assert resp.status_code == 400


def test_chat_blank_message_400(client):
    resp = client.post('/api/chat', json={'message': '   '})
    assert resp.status_code == 400


def test_chat_streams_tokens(monkeypatch, client):
    """模拟 API 正常返回，SSE 流应包含 token 与 done 标记。"""
    monkeypatch.setattr(app_module.requests, 'post', lambda *a, **k: FakeStreamResponse())
    resp = client.post('/api/chat', json={'message': '你好', 'session_id': 'sA'})
    assert resp.status_code == 200
    assert 'text/event-stream' in resp.content_type
    data = resp.get_data(as_text=True)
    assert '你' in data
    assert 'done' in data


def test_chat_timeout_yields_error(monkeypatch, client):
    """API 超时时应返回友好错误，而非 5xx。"""
    def boom(*a, **k):
        raise app_module.requests.exceptions.Timeout()
    monkeypatch.setattr(app_module.requests, 'post', boom)
    resp = client.post('/api/chat', json={'message': '你好', 'session_id': 'sA'})
    data = resp.get_data(as_text=True)
    assert '请求超时' in data


def test_sessions_crud_flow(client):
    resp = client.post('/api/sessions')
    assert resp.status_code == 200
    sid = resp.get_json()['session_id']
    assert client.get('/api/sessions').get_json()
    assert any(s['id'] == sid for s in client.get('/api/sessions').get_json()['sessions'])
    resp = client.get('/api/sessions?id=' + sid)
    assert resp.status_code == 200
    assert resp.get_json()['id'] == sid
    resp = client.delete('/api/sessions?id=' + sid)
    assert resp.status_code == 200


def test_upload_requires_file(client):
    resp = client.post('/api/upload', data={})
    assert resp.status_code == 400


def test_upload_file_is_session_isolated(monkeypatch, client):
    """核心回归：A 会话上传的文件不能出现在 B 会话。"""
    captured = {}

    def fake_post(url, headers=None, json=None, **kw):
        captured['messages'] = json['messages']
        return FakeStreamResponse(('ok',))
    monkeypatch.setattr(app_module.requests, 'post', fake_post)

    # 会话 A 上传文件
    resp = client.post('/api/upload', data={
        'file': (BytesIO('我的简历里写了三年后端经验'.encode('utf-8')), 'resume.txt'),
        'session_id': 'sessA',
    })
    assert resp.status_code == 200

    # 会话 A 对话 → system prompt 应包含 A 的文件内容
    client.post('/api/chat', json={'message': '文件内容', 'session_id': 'sessA'})
    sys_a = captured['messages'][0]['content']
    assert '我的简历里写了三年后端经验' in sys_a

    # 会话 B 对话 → 不应包含 A 的文件内容（修复跨用户泄露）
    client.post('/api/chat', json={'message': '你好', 'session_id': 'sessB'})
    sys_b = captured['messages'][0]['content']
    assert '我的简历里写了三年后端经验' not in sys_b


def test_memory_delete_single_vs_clear(client):
    mm.update_user('u1', name='小美', preferences='不吃香菜')
    # 单条删除：只删 preferences，保留 name
    resp = client.delete('/api/memory?session_id=u1&key=preferences')
    assert resp.status_code == 200
    user = mm.get_or_create_user('u1')
    assert user['preferences'] == ''
    assert user['name'] == '小美'
    # 删除不存在的字段 → 404
    resp = client.delete('/api/memory?session_id=u1&key=health_info')
    assert resp.status_code == 404
    # 无 key → 清空全部
    resp = client.delete('/api/memory?session_id=u1')
    assert resp.status_code == 200
    user = mm.get_or_create_user('u1')
    assert user['name'] == ''


def test_memory_get_returns_memories(client):
    mm.update_user('u1', name='小美')
    resp = client.get('/api/memory?session_id=u1')
    data = resp.get_json()
    assert 'memories' in data
    assert data['memories']['name'] == '小美'