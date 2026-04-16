# tests/unit/test_layer.py
import pytest
from unittest.mock import MagicMock, patch
from memory_platform.ext.layer import (
    MemoryLayer,
    classify_layer,
    classify_layer_with_llm,
    parse_layer_filter,
)


class TestClassifyLayer:
    def test_profile_keywords(self):
        assert classify_layer("他是一名Java工程师") == MemoryLayer.L1
        assert classify_layer("担任项目经理") == MemoryLayer.L1
        assert classify_layer("职位是产品经理") == MemoryLayer.L1

    def test_preference_keywords(self):
        assert classify_layer("喜欢简洁的沟通风格") == MemoryLayer.L2
        assert classify_layer("偏好下午开会") == MemoryLayer.L2
        assert classify_layer("习惯用键盘快捷键") == MemoryLayer.L2

    def test_episodic_keywords(self):
        assert classify_layer("上周参加了Java培训") == MemoryLayer.L3
        assert classify_layer("昨天去了医院") == MemoryLayer.L3
        assert classify_layer("那次项目评审很有收获") == MemoryLayer.L3

    def test_relational_keywords(self):
        assert classify_layer("和Alice同在项目组") == MemoryLayer.L4
        assert classify_layer("直属领导是Bob") == MemoryLayer.L4
        assert classify_layer("团队一起完成了任务") == MemoryLayer.L4

    def test_explicit_layer_override(self):
        assert classify_layer("任意文本", explicit_layer="L2") == MemoryLayer.L2
        assert classify_layer("他是工程师", explicit_layer="L3") == MemoryLayer.L3

    def test_unknown_text_returns_default(self):
        result = classify_layer("今天天气不错")
        assert result == MemoryLayer.L1  # 默认 L1


class TestParseLayerFilter:
    def test_single_layer(self):
        assert parse_layer_filter("L1") == [MemoryLayer.L1]

    def test_multiple_layers(self):
        result = parse_layer_filter("L1,L3")
        assert result == [MemoryLayer.L1, MemoryLayer.L3]

    def test_none_returns_all(self):
        result = parse_layer_filter(None)
        assert result == [MemoryLayer.L1, MemoryLayer.L2, MemoryLayer.L3, MemoryLayer.L4]


class TestClassifyLayerWithLLM:
    """LLM 辅助层级分类测试"""

    def _mock_llm(self, content: str):
        """构建 mock LLM 实例，generate_response 返回指定内容"""
        from mem0.llms.base import LLMBase
        llm = MagicMock(spec=LLMBase)
        llm.generate_response.return_value = content
        return llm

    def test_llm_classifies_l1_profile(self):
        """LLM 正确识别 L1 Profile 类型"""
        llm = self._mock_llm('{"layer": "L1", "reason": "描述了职业身份"}')

        result = classify_layer_with_llm(
            text="张三在腾讯担任高级工程师",
            llm=llm,
            use_keyword_first=False,
        )

        assert result == MemoryLayer.L1
        llm.generate_response.assert_called_once()

    def test_llm_classifies_l2_preference(self):
        """LLM 正确识别 L2 Preference 类型"""
        llm = self._mock_llm('{"layer": "L2", "reason": "描述了偏好"}')

        result = classify_layer_with_llm(
            text="倾向于使用简洁的代码风格",
            llm=llm,
        )

        assert result == MemoryLayer.L2

    def test_llm_classifies_l3_episodic(self):
        """LLM 正确识别 L3 Episodic 类型"""
        llm = self._mock_llm('{"layer": "L3", "reason": "描述了具体事件"}')

        result = classify_layer_with_llm(
            text="上周五参加了团队的技术分享会",
            llm=llm,
        )

        assert result == MemoryLayer.L3

    def test_llm_classifies_l4_relational(self):
        """LLM 正确识别 L4 Relational 类型"""
        llm = self._mock_llm('{"layer": "L4", "reason": "描述了团队关系"}')

        result = classify_layer_with_llm(
            text="小李和我在同一个项目组",
            llm=llm,
        )

        assert result == MemoryLayer.L4

    def test_llm_returns_default_on_parse_error(self):
        """LLM 返回格式错误时返回默认 L1"""
        llm = self._mock_llm("invalid json")

        result = classify_layer_with_llm(
            text="一些文本",
            llm=llm,
        )

        assert result == MemoryLayer.L1

    def test_llm_returns_default_on_api_error(self):
        """LLM API 调用失败时返回默认 L1"""
        from mem0.llms.base import LLMBase
        llm = MagicMock(spec=LLMBase)
        llm.generate_response.side_effect = Exception("API Error")

        result = classify_layer_with_llm(
            text="一些文本",
            llm=llm,
        )

        assert result == MemoryLayer.L1

    def test_hybrid_classification_keyword_first(self):
        """混合分类：关键词优先，不调用 LLM"""
        from mem0.llms.base import LLMBase
        llm = MagicMock(spec=LLMBase)

        result = classify_layer_with_llm(
            text="我是Python工程师",
            llm=llm,
            use_keyword_first=True,
        )

        assert result == MemoryLayer.L1
        llm.generate_response.assert_not_called()

    def test_hybrid_classification_llm_fallback(self):
        """混合分类：关键词未匹配时调用 LLM"""
        llm = self._mock_llm('{"layer": "L2", "reason": "描述风格偏好"}')

        result = classify_layer_with_llm(
            text="这个人的代码非常有特色",
            llm=llm,
            use_keyword_first=True,
        )

        assert result == MemoryLayer.L2
        llm.generate_response.assert_called_once()
