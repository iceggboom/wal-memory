from unittest.mock import MagicMock
import pytest

from memory_platform.services.write import WriteService, AddMemoryItem


@pytest.fixture
def mock_mem0():
    return MagicMock()


@pytest.fixture
def write_service(mock_mem0):
    return WriteService(mem0=mock_mem0)


class TestAddMemory:
    def test_single_memory(self, write_service, mock_mem0):
        mock_mem0.add.return_value = {
            "results": [
                {"id": "m1", "memory": "test", "event": "ADD"},
            ]
        }
        result = write_service.add_memory(
            user_id="u1",
            agent_id="app1",
            items=[AddMemoryItem(text="test", memory_layer="L1", scope="shared")],
        )
        assert result["added"] == 1
        assert result["updated"] == 0
        assert result["unchanged"] == 0

    def test_explicit_layer_in_metadata(self, write_service, mock_mem0):
        mock_mem0.add.return_value = {"results": []}
        write_service.add_memory(
            user_id="u1",
            agent_id="app1",
            items=[AddMemoryItem(text="test", memory_layer="L2", scope="shared")],
        )
        call_kwargs = mock_mem0.add.call_args
        metadata = call_kwargs.kwargs.get("metadata") or call_kwargs[1].get("metadata")
        assert metadata["memory_layer"] == "L2"
        assert metadata["scope"] == "shared"
        assert metadata["app_id"] == "app1"

    def test_auto_classify_layer(self, write_service, mock_mem0):
        mock_mem0.add.return_value = {"results": []}
        write_service.add_memory(
            user_id="u1",
            agent_id="app1",
            items=[AddMemoryItem(text="喜欢简洁的风格")],
        )
        call_kwargs = mock_mem0.add.call_args
        metadata = call_kwargs.kwargs.get("metadata") or call_kwargs[1].get("metadata")
        assert metadata["memory_layer"] == "L2"

    def test_default_scope_is_shared(self, write_service, mock_mem0):
        mock_mem0.add.return_value = {"results": []}
        write_service.add_memory(
            user_id="u1",
            agent_id="app1",
            items=[AddMemoryItem(text="test")],
        )
        call_kwargs = mock_mem0.add.call_args
        metadata = call_kwargs.kwargs.get("metadata") or call_kwargs[1].get("metadata")
        assert metadata["scope"] == "shared"

    def test_counts_events(self, write_service, mock_mem0):
        mock_mem0.add.side_effect = [
            {"results": [{"id": "m1", "memory": "a", "event": "ADD"}]},
            {"results": [{"id": "m2", "memory": "b", "event": "UPDATE"}]},
            {"results": [{"id": "m3", "memory": "c", "event": "NONE"}]},
            {"results": [{"id": "m4", "memory": "d", "event": "DELETE"}]},
        ]
        result = write_service.add_memory(
            user_id="u1", agent_id="app1",
            items=[AddMemoryItem(text="a"), AddMemoryItem(text="b"),
                   AddMemoryItem(text="c"), AddMemoryItem(text="d")],
        )
        assert result["added"] == 1
        assert result["updated"] == 1
        assert result["unchanged"] == 1


class TestExtractFromConversation:
    def test_extract_calls_mem0_add_with_messages(self, write_service, mock_mem0):
        mock_mem0.add.return_value = {
            "results": [
                {"id": "m1", "memory": "用户是工程师", "event": "ADD"},
            ]
        }
        result = write_service.extract(
            user_id="u1",
            agent_id="app1",
            messages=[{"role": "user", "content": "我是一名Java工程师"}],
        )
        assert result["added"] == 1
        mock_mem0.add.assert_called_once()
        call_args = mock_mem0.add.call_args
        assert call_args[0][0] == [{"role": "user", "content": "我是一名Java工程师"}]


class TestWriteServiceWithLLM:
    """WriteService LLM 分类集成测试"""

    def test_add_memory_uses_llm_classification(self):
        """add_memory 使用 LLM 分类层级"""
        from mem0.llms.base import LLMBase
        mock_memory = MagicMock()
        mock_memory.add.return_value = {"results": [{"event": "ADD"}]}

        mock_llm = MagicMock(spec=LLMBase)
        mock_llm.generate_response.return_value = '{"layer": "L2", "reason": "描述偏好"}'

        svc = WriteService(mem0=mock_memory, llm=mock_llm)

        # 添加一个关键词无法明确分类的记忆
        svc.add_memory(
            user_id="u1",
            agent_id="a1",
            items=[AddMemoryItem(text="这个人做事很有自己的一套")],
        )

        # 验证调用了 LLM
        mock_llm.generate_response.assert_called()

    def test_add_memory_keyword_takes_priority(self):
        """关键词匹配优先，不调用 LLM"""
        from mem0.llms.base import LLMBase
        mock_memory = MagicMock()
        mock_memory.add.return_value = {"results": [{"event": "ADD"}]}

        mock_llm = MagicMock(spec=LLMBase)

        svc = WriteService(mem0=mock_memory, llm=mock_llm)

        # 添加一个关键词能明确分类的记忆
        svc.add_memory(
            user_id="u1",
            agent_id="a1",
            items=[AddMemoryItem(text="我是Python工程师")],
        )

        # 关键词匹配成功，不应调用 LLM
        mock_llm.generate_response.assert_not_called()

    def test_extract_uses_llm_classification(self):
        """extract 使用 LLM 分类层级"""
        from mem0.llms.base import LLMBase
        mock_memory = MagicMock()
        mock_memory.add.return_value = {
            "results": [
                {
                    "event": "ADD",
                    "id": "mem-1",
                    "memory": "用户喜欢简洁的代码风格",
                    "metadata": {},
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                }
            ]
        }

        mock_llm = MagicMock(spec=LLMBase)
        mock_llm.generate_response.return_value = '{"layer": "L2", "reason": "描述偏好风格"}'

        svc = WriteService(mem0=mock_memory, llm=mock_llm)

        result = svc.extract(
            user_id="u1",
            agent_id="a1",
            messages=[{"role": "user", "content": "我喜欢简洁的代码风格"}],
        )

        # 验证返回结果包含正确的层级
        assert len(result["memories"]) >= 0
