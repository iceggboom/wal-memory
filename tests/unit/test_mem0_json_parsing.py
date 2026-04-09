"""测试 mem0 fork 的 JSON 解析增强 — 验证 GLM 兼容性"""
import json

import pytest

from mem0.memory.utils import remove_code_blocks, extract_json


class TestRemoveCodeBlocks:
    """测试 remove_code_blocks 对各种 LLM 输出格式的处理"""

    def test_standard_json_code_block(self):
        """标准 markdown JSON 代码块"""
        content = '```json\n{"facts": ["a", "b"]}\n```'
        result = remove_code_blocks(content)
        assert result == '{"facts": ["a", "b"]}'

    def test_json_code_block_no_language(self):
        """无语言标记的代码块"""
        content = '```\n{"facts": ["a"]}\n```'
        result = remove_code_blocks(content)
        assert result == '{"facts": ["a"]}'

    def test_leading_whitespace(self):
        """开头有空白"""
        content = '  ```json\n{"facts": ["a"]}\n```'
        result = remove_code_blocks(content)
        assert result == '{"facts": ["a"]}'

    def test_carriage_return(self):
        """Windows 换行 \\r\\n"""
        content = '```json\r\n{"facts": ["a"]}\r\n```'
        result = remove_code_blocks(content)
        assert result == '{"facts": ["a"]}'

    def test_no_code_block(self):
        """无代码块包裹 — 纯 JSON"""
        content = '{"facts": ["a", "b"]}'
        result = remove_code_blocks(content)
        assert result == '{"facts": ["a", "b"]}'

    def test_trailing_text_after_code_block(self):
        """代码块后有尾部文字（GLM 有时在 JSON 后添加解释）"""
        content = '```json\n{"facts": ["a"]}\n```\n以上是提取的记忆。'
        result = remove_code_blocks(content)
        assert '{"facts": ["a"]}' in result

    def test_think_tags_removed(self):
        """移除 <think ...> 标签"""
        content = '{"facts": ["a"]}'
        result = remove_code_blocks(content)
        assert "<think" not in result


class TestExtractJson:
    """测试 extract_json 的括号平衡算法"""

    def test_pure_json(self):
        """纯 JSON 输入"""
        result = extract_json('{"facts": ["a", "b"]}')
        assert '"facts"' in result

    def test_markdown_wrapped_json(self):
        """markdown 包裹的 JSON"""
        result = extract_json('```json\n{"facts": ["a"]}\n```')
        assert '"facts"' in result

    def test_json_with_trailing_text(self):
        """JSON 后有附加文字 — 括号平衡提取"""
        result = extract_json('{"facts": ["a"]}\n这是提取的记忆。')
        assert result == '{"facts": ["a"]}'

    def test_nested_json(self):
        """嵌套 JSON — 括号平衡"""
        result = extract_json('{"memory": [{"text": "a", "event": "ADD"}]}')
        parsed = json.loads(result)
        assert len(parsed["memory"]) == 1

    def test_json_with_nested_braces_in_values(self):
        """值中包含嵌套花括号"""
        result = extract_json('{"data": {"inner": "val"}, "key": "value"}')
        parsed = json.loads(result)
        assert parsed["data"]["inner"] == "val"

    def test_no_json_returns_as_is(self):
        """无 JSON 内容 — 返回原文"""
        result = extract_json("no json here")
        assert result == "no json here"

    def test_glm_real_output(self):
        """GLM 真实输出格式"""
        glm_output = '```json\n{\n  "facts": ["用户刚从北京搬到上海", "用户在一家AI创业公司做全栈开发"]\n}\n```'
        result = extract_json(glm_output)
        parsed = json.loads(result)
        assert len(parsed["facts"]) == 2
        assert "北京" in parsed["facts"][0]
