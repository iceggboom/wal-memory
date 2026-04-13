"""跨 Collection 搜索服务测试"""

import pytest
from unittest.mock import MagicMock
from memory_platform.services.cross_collection import CrossCollectionSearcher


class TestCrossCollectionSearcher:
    """跨 Collection 搜索测试"""

    def test_search_single_app(self):
        """单应用搜索：不需要跨 Collection"""
        mock_memory = MagicMock()
        mock_memory.search.return_value = {
            "results": [
                {
                    "id": "mem-1",
                    "memory": "用户是工程师",
                    "score": 0.9,
                    "metadata": {"scope": "shared", "memory_layer": "L1"},
                }
            ]
        }

        searcher = CrossCollectionSearcher(
            get_memory_for_app=lambda app_id: mock_memory
        )

        results = searcher.search(
            query="工程师",
            user_id="u1",
            app_ids=["app-1"],
            limit=10,
        )

        assert len(results) == 1
        assert results[0]["id"] == "mem-1"
        mock_memory.search.assert_called_once()

    def test_search_multiple_apps_merges_results(self):
        """多应用搜索：合并结果"""
        mock_memory_1 = MagicMock()
        mock_memory_1.search.return_value = {
            "results": [
                {
                    "id": "mem-1",
                    "memory": "用户是Python工程师",
                    "score": 0.95,
                    "metadata": {"scope": "shared", "memory_layer": "L1"},
                }
            ]
        }

        mock_memory_2 = MagicMock()
        mock_memory_2.search.return_value = {
            "results": [
                {
                    "id": "mem-2",
                    "memory": "用户喜欢喝咖啡",
                    "score": 0.85,
                    "metadata": {"scope": "shared", "memory_layer": "L2"},
                }
            ]
        }

        def get_memory(app_id):
            return {"app-1": mock_memory_1, "app-2": mock_memory_2}[app_id]

        searcher = CrossCollectionSearcher(get_memory_for_app=get_memory)

        results = searcher.search(
            query="用户信息",
            user_id="u1",
            app_ids=["app-1", "app-2"],
            limit=10,
        )

        assert len(results) == 2
        # 结果应按 score 降序排列
        assert results[0]["score"] >= results[1]["score"]

    def test_search_filters_private_memories(self):
        """跨应用搜索时过滤掉 private 记忆"""
        mock_memory = MagicMock()
        mock_memory.search.return_value = {
            "results": [
                {
                    "id": "mem-1",
                    "memory": "共享记忆",
                    "score": 0.9,
                    "metadata": {"scope": "shared"},
                },
                {
                    "id": "mem-2",
                    "memory": "私有记忆",
                    "score": 0.95,
                    "metadata": {"scope": "private"},
                },
            ]
        }

        searcher = CrossCollectionSearcher(
            get_memory_for_app=lambda app_id: mock_memory
        )

        results = searcher.search(
            query="记忆",
            user_id="u1",
            app_ids=["app-1"],
            limit=10,
            scope="shared",  # 只搜索共享记忆
        )

        # private 记忆应被过滤
        assert len(results) == 1
        assert results[0]["id"] == "mem-1"

    def test_search_deduplicates_by_hash(self):
        """基于 hash 去重"""
        mock_memory_1 = MagicMock()
        mock_memory_1.search.return_value = {
            "results": [
                {
                    "id": "mem-1",
                    "memory": "重复记忆",
                    "score": 0.9,
                    "hash": "hash-abc",
                    "metadata": {"scope": "shared"},
                }
            ]
        }

        mock_memory_2 = MagicMock()
        mock_memory_2.search.return_value = {
            "results": [
                {
                    "id": "mem-2",
                    "memory": "重复记忆",
                    "score": 0.85,
                    "hash": "hash-abc",  # 相同 hash
                    "metadata": {"scope": "shared"},
                }
            ]
        }

        def get_memory(app_id):
            return {"app-1": mock_memory_1, "app-2": mock_memory_2}[app_id]

        searcher = CrossCollectionSearcher(get_memory_for_app=get_memory)

        results = searcher.search(
            query="记忆",
            user_id="u1",
            app_ids=["app-1", "app-2"],
            limit=10,
        )

        # 相同 hash 的记忆只保留一个（保留 score 高的）
        assert len(results) == 1
        assert results[0]["id"] == "mem-1"

    def test_search_respects_limit(self):
        """结果数量限制"""
        mock_memory = MagicMock()
        mock_memory.search.return_value = {
            "results": [
                {"id": f"mem-{i}", "memory": f"记忆{i}", "score": 0.9 - i * 0.1}
                for i in range(5)
            ]
        }

        searcher = CrossCollectionSearcher(
            get_memory_for_app=lambda app_id: mock_memory
        )

        results = searcher.search(
            query="记忆",
            user_id="u1",
            app_ids=["app-1"],
            limit=3,
        )

        assert len(results) == 3

    def test_search_handles_empty_results(self):
        """处理空结果"""
        mock_memory = MagicMock()
        mock_memory.search.return_value = {"results": []}

        searcher = CrossCollectionSearcher(
            get_memory_for_app=lambda app_id: mock_memory
        )

        results = searcher.search(
            query="不存在的记忆",
            user_id="u1",
            app_ids=["app-1", "app-2"],
            limit=10,
        )

        assert results == []

    def test_search_handles_app_not_found(self):
        """处理应用不存在的情况"""
        mock_memory = MagicMock()
        mock_memory.search.return_value = {"results": []}

        def get_memory(app_id):
            if app_id == "unknown-app":
                return None
            return mock_memory

        searcher = CrossCollectionSearcher(get_memory_for_app=get_memory)

        # 不应抛出异常
        results = searcher.search(
            query="记忆",
            user_id="u1",
            app_ids=["app-1", "unknown-app"],
            limit=10,
        )

        assert results == []
