"""Unit tests for TencentVectorStore adapter.

使用 mock 模式测试所有接口，无需真实腾讯云 VectorDB 服务。
"""

from memory_platform.adapters.tencent_vector import (
    OutputData,
    TencentVectorDBConfig,
    TencentVectorStore,
)


def _make_store(**overrides) -> TencentVectorStore:
    """创建 mock 模式的 TencentVectorStore。"""
    config = TencentVectorDBConfig(
        url="http://mock.local",
        username="root",
        key="mock-key",
        mock=True,
        **overrides,
    )
    return TencentVectorStore(config=config)


# ============================================================================
# OutputData
# ============================================================================


class TestOutputData:
    def test_attributes(self):
        result = OutputData(id="id1", payload={"k": "v"}, score=0.9)
        assert result.id == "id1"
        assert result.payload == {"k": "v"}
        assert result.score == 0.9

    def test_equality(self):
        a = OutputData(id="id1", payload={"k": "v"}, score=0.9)
        b = OutputData(id="id1", payload={"k": "v"}, score=0.9)
        assert a == b

    def test_immutable(self):
        result = OutputData(id="id1", payload={}, score=0.9)
        try:
            result.score = 0.0
            assert False, "Should have raised"
        except AttributeError:
            pass


# ============================================================================
# Config
# ============================================================================


class TestConfig:
    def test_defaults(self):
        config = TencentVectorDBConfig(
            url="http://example.com", username="root", key="secret"
        )
        assert config.collection_name == "mem0"
        assert config.embedding_model_dims == 1536
        assert config.database_name == "memory_platform"
        assert config.timeout == 30
        assert config.mock is False

    def test_custom_values(self):
        config = TencentVectorDBConfig(
            url="http://custom",
            username="admin",
            key="k",
            collection_name="c1",
            embedding_model_dims=768,
            database_name="db1",
            timeout=60,
            mock=True,
        )
        assert config.collection_name == "c1"
        assert config.embedding_model_dims == 768
        assert config.mock is True

    def test_timeout_validation(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TencentVectorDBConfig(
                url="http://x", username="root", key="k", timeout=0
            )


# ============================================================================
# Init
# ============================================================================


class TestInit:
    def test_mock_mode_no_real_client(self):
        store = _make_store()
        assert store.client is None
        assert store.db is None
        assert store._is_mock is True

    def test_stores_collection_name(self):
        store = _make_store(collection_name="test_col")
        assert store.collection_name == "test_col"

    def test_stores_embedding_dims(self):
        store = _make_store(embedding_model_dims=768)
        assert store.embedding_model_dims == 768


# ============================================================================
# Insert + Get
# ============================================================================


class TestInsertAndGet:
    def test_insert_and_get(self):
        store = _make_store()
        vec = [0.1] * 1536
        payload = {"data": "test memory", "user_id": "u1"}

        store.insert(vectors=[vec], payloads=[payload], ids=["id-1"])

        result = store.get(vector_id="id-1")
        assert result is not None
        assert result.id == "id-1"
        assert result.payload == payload

    def test_get_missing_returns_none(self):
        store = _make_store()
        assert store.get(vector_id="nonexistent") is None

    def test_insert_batch(self):
        store = _make_store()
        vec = [0.1] * 1536
        store.insert(
            vectors=[vec, vec],
            payloads=[{"data": "a"}, {"data": "b"}],
            ids=["id-1", "id-2"],
        )
        assert store.get("id-1") is not None
        assert store.get("id-2") is not None


# ============================================================================
# Search
# ============================================================================


class TestSearch:
    def test_search_returns_results(self):
        store = _make_store()
        vec = [0.1] * 1536
        store.insert(vectors=[vec], payloads=[{"data": "test"}], ids=["id-1"])

        results = store.search(query="test", vectors=[vec], limit=5)
        assert isinstance(results, list)
        assert len(results) >= 1
        assert results[0].id == "id-1"

    def test_search_with_filters(self):
        store = _make_store()
        vec = [0.1] * 1536
        store.insert(
            vectors=[vec],
            payloads=[{"data": "mem", "user_id": "u1"}],
            ids=["id-1"],
        )
        store.insert(
            vectors=[vec],
            payloads=[{"data": "mem2", "user_id": "u2"}],
            ids=["id-2"],
        )

        results = store.search(
            query="test", vectors=[vec], limit=5, filters={"user_id": "u1"}
        )
        assert len(results) == 1
        assert results[0].id == "id-1"

    def test_search_empty(self):
        store = _make_store()
        results = store.search(query="test", vectors=[[0.1] * 1536], limit=5)
        assert results == []


# ============================================================================
# Delete
# ============================================================================


class TestDelete:
    def test_delete_removes_document(self):
        store = _make_store()
        vec = [0.1] * 1536
        store.insert(vectors=[vec], payloads=[{"data": "test"}], ids=["id-1"])

        store.delete(vector_id="id-1")
        assert store.get("id-1") is None


# ============================================================================
# Update
# ============================================================================


class TestUpdate:
    def test_update_payload(self):
        store = _make_store()
        vec = [0.1] * 1536
        store.insert(vectors=[vec], payloads=[{"data": "old"}], ids=["id-1"])

        store.update(vector_id="id-1", payload={"data": "new"})
        result = store.get("id-1")
        assert result.payload["data"] == "new"

    def test_update_nonexistent_is_noop(self):
        store = _make_store()
        store.update(vector_id="ghost", payload={"data": "x"})  # should not raise


# ============================================================================
# List
# ============================================================================


class TestList:
    def test_list_returns_nested(self):
        store = _make_store()
        vec = [0.1] * 1536
        store.insert(vectors=[vec], payloads=[{"data": "a"}], ids=["id-1"])

        result = store.list()
        assert isinstance(result, list)
        assert len(result) == 1  # 嵌套列表
        assert isinstance(result[0], list)

    def test_list_empty(self):
        store = _make_store()
        result = store.list()
        assert result == [[]]

    def test_list_with_filters(self):
        store = _make_store()
        vec = [0.1] * 1536
        store.insert(
            vectors=[vec], payloads=[{"data": "a", "user_id": "u1"}], ids=["id-1"]
        )
        store.insert(
            vectors=[vec], payloads=[{"data": "b", "user_id": "u2"}], ids=["id-2"]
        )

        result = store.list(filters={"user_id": "u1"})
        assert len(result[0]) == 1
        assert result[0][0].id == "id-1"


# ============================================================================
# Collection management
# ============================================================================


class TestCollectionManagement:
    def test_list_cols(self):
        store = _make_store()
        assert store.list_cols() == ["mock_collection"]

    def test_col_info(self):
        store = _make_store()
        vec = [0.1] * 1536
        store.insert(vectors=[vec], payloads=[{}], ids=["id-1"])
        info = store.col_info()
        assert info["document_count"] == 1

    def test_delete_col_clears_data(self):
        store = _make_store()
        vec = [0.1] * 1536
        store.insert(vectors=[vec], payloads=[{}], ids=["id-1"])
        store.delete_col()
        assert store.get("id-1") is None

    def test_reset_clears_data(self):
        store = _make_store()
        vec = [0.1] * 1536
        store.insert(vectors=[vec], payloads=[{}], ids=["id-1"])
        store.reset()
        assert store.get("id-1") is None
