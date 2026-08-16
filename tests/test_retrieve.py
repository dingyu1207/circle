"""知识库检索逻辑测试：关键词精确匹配 + 加权打分 + top_n 限制。"""

import app


def test_keyword_exact_match_returns_relevant_entry():
    """'痛经' 应命中健康知识库中的经期护理相关条目。"""
    results = app.retrieve('痛经应该怎么护理', top_n=3)
    assert results, '应检索到经期护理相关知识'
    hit = results[0]
    keywords = hit.get('keywords', [])
    assert ('痛经' in keywords or '经期' in keywords or '痛经' in hit['title']), \
        f'首条结果应含痛经/经期关键词，实际：{keywords} / {hit["title"]}'


def test_top_n_limits_results():
    """top_n 应严格限制返回条数。"""
    results = app.retrieve('家常菜 红烧肉 怎么做好吃', top_n=2)
    assert len(results) <= 2


def test_no_match_returns_empty():
    """无任何关键词命中的查询应返回空列表。"""
    results = app.retrieve('zzzzqqqq11112222', top_n=3)
    assert results == []


def test_entry_shape():
    """检索结果条目应包含 title / content / keywords 三个字段。"""
    results = app.retrieve('痛经', top_n=1)
    assert results
    for e in results:
        assert 'title' in e
        assert 'content' in e
        assert 'keywords' in e