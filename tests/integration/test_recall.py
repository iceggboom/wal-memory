from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
import pytest

from memory_platform.services.recall import RecallService
from memory_platform.services.cross_collection import CrossCollectionSearcher


@pytest.fixture
def mock_mem0():
    return MagicMock()


@pytest.fixture
def recall_service(mock_mem0):
    return RecallService(mem0=mock_mem0)


@pytest.fixture
def recall_with_cross_collection():
    """创建支持跨 Collection 搜索的 RecallService"""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    mock_memory = MagicMock()
    mock_memory.search.return_value = {
        "results": [
            {
                "id": "mem-1",
                "memory": "用户是Python工程师",
                "score": 0.95,
                "metadata": {"scope": "shared", "memory_layer": "L1"},
                "created_at": now,
                "updated_at": now,
            }
        ]
    }

    def get_memory(app_id):
        return mock_memory

    searcher = CrossCollectionSearcher(get_memory_for_app=get_memory)
    return RecallService(mem0=mock_memory, cross_collection_searcher=searcher)


class TestRecallService:
    def test_search_applies_confidence_decay(self, recall_service, mock_mem0):
        now = datetime.now(timezone.utc)
        mock_mem0.search.return_value = {
            "results": [
                {
                    "id": "m1", "memory": "是工程师",
                    "score": 0.9,
                    "metadata": {"memory_layer": "L1", "scope": "shared", "app_id": "app1"},
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                },
                {
                    "id": "m2", "memory": "上周参加了培训",
                    "score": 0.85,
                    "metadata": {"memory_layer": "L3", "scope": "shared", "app_id": "app1"},
                    "created_at": (now - timedelta(hours=500)).isoformat(),
                    "updated_at": (now - timedelta(hours=500)).isoformat(),
                },
            ]
        }
        result = recall_service.search(
            query="工程师",
            user_id="u1",
            agent_id="app1",
            min_confidence=0.5,
            now=now,
        )
        assert len(result) <= 2
        for item in result:
            assert item["confidence"] >= 0.5

    def test_search_filters_by_layer(self, recall_service, mock_mem0):
        now = datetime.now(timezone.utc)
        mock_mem0.search.return_value = {
            "results": [
                {
                    "id": "m1", "memory": "是工程师",
                    "score": 0.9,
                    "metadata": {"memory_layer": "L1", "scope": "shared", "app_id": "app1"},
                    "created_at": now.isoformat(), "updated_at": now.isoformat(),
                },
                {
                    "id": "m2", "memory": "上周培训",
                    "score": 0.85,
                    "metadata": {"memory_layer": "L3", "scope": "shared", "app_id": "app1"},
                    "created_at": now.isoformat(), "updated_at": now.isoformat(),
                },
            ]
        }
        result = recall_service.search(
            query="test",
            user_id="u1",
            agent_id="app1",
            memory_layer="L1",
            now=now,
        )
        assert len(result) == 1
        assert result[0]["memory_layer"] == "L1"

    def test_search_filters_by_scope(self, recall_service, mock_mem0):
        now = datetime.now(timezone.utc)
        mock_mem0.search.return_value = {
            "results": [
                {
                    "id": "m1", "memory": "shared",
                    "score": 0.9,
                    "metadata": {"memory_layer": "L1", "scope": "shared", "app_id": "app1"},
                    "created_at": now.isoformat(), "updated_at": now.isoformat(),
                },
                {
                    "id": "m2", "memory": "private",
                    "score": 0.85,
                    "metadata": {"memory_layer": "L1", "scope": "private", "app_id": "app1"},
                    "created_at": now.isoformat(), "updated_at": now.isoformat(),
                },
            ]
        }
        result = recall_service.search(
            query="test", user_id="u1", agent_id="app1", scope="shared", now=now
        )
        assert len(result) == 1
        assert result[0]["scope"] == "shared"

    def test_return_format_includes_confidence(self, recall_service, mock_mem0):
        now = datetime.now(timezone.utc)
        mock_mem0.search.return_value = {
            "results": [
                {
                    "id": "m1", "memory": "test",
                    "score": 0.9,
                    "metadata": {"memory_layer": "L1", "scope": "shared", "app_id": "app1"},
                    "created_at": now.isoformat(), "updated_at": now.isoformat(),
                },
            ]
        }
        result = recall_service.search(
            query="test", user_id="u1", agent_id="app1", now=now
        )
        assert "id" in result[0]
        assert "text" in result[0]
        assert "memory_layer" in result[0]
        assert "confidence" in result[0]
        assert "similarity" in result[0]
        assert "scope" in result[0]


class TestRecallServiceCrossCollection:
    """RecallService 跨 Collection 搜索集成测试"""

    def test_search_with_scope_shared_searches_cross_collection(
        self, recall_with_cross_collection
    ):
        """scope=shared 时触发跨 Collection 搜索"""
        # 调用 search，scope="shared"，传入多个 app_ids
        results = recall_with_cross_collection.search(
            query="Python",
            user_id="u1",
            agent_id="app-1",
            scope="shared",
            all_app_ids=["app-1", "app-2", "app-3"],  # 传入所有应用 ID
            limit=10,
        )

        # 应返回结果
        assert len(results) >= 1

    def test_search_with_scope_private_only_current_app(
        self, recall_with_cross_collection
    ):
        """scope=private 时只搜索当前应用"""
        results = recall_with_cross_collection.search(
            query="Python",
            user_id="u1",
            agent_id="app-1",
            scope="private",
            all_app_ids=["app-1", "app-2"],
            limit=10,
        )

        # 只搜索当前应用
        recall_with_cross_collection.mem0.search.assert_called_once()

    def test_search_without_all_app_ids_falls_back_to_single(
        self, recall_with_cross_collection
    ):
        """未传入 all_app_ids 时退化为单应用搜索"""
        results = recall_with_cross_collection.search(
            query="Python",
            user_id="u1",
            agent_id="app-1",
            scope="shared",
            # 不传 all_app_ids
            limit=10,
        )

        # 退化为单应用搜索
        recall_with_cross_collection.mem0.search.assert_called_once()
