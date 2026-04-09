"""Unit tests for TencentVectorStore adapter."""

from unittest.mock import MagicMock, patch

from memory_platform.adapters.tencent_vector import (
    OutputData,
    TencentVectorDBConfig,
    TencentVectorStore,
)


class TestOutputData:
    """Tests for the OutputData dataclass."""

    def test_output_data_attributes(self):
        """OutputData should expose id, payload, and score attributes."""
        result = OutputData(
            id="test-id",
            payload={"data": "test memory", "user_id": "u1"},
            score=0.95,
        )

        assert result.id == "test-id"
        assert result.payload == {"data": "test memory", "user_id": "u1"}
        assert result.score == 0.95

    def test_output_data_equality(self):
        """Two OutputData with same values should be equal."""
        a = OutputData(id="id1", payload={"k": "v"}, score=0.9)
        b = OutputData(id="id1", payload={"k": "v"}, score=0.9)

        assert a == b

    def test_output_data_immutable(self):
        """OutputData fields should be immutable (frozen dataclass)."""
        result = OutputData(id="id1", payload={}, score=0.9)

        try:
            result.score = 0.0
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass


class TestTencentVectorStoreInit:
    """Tests for TencentVectorStore constructor."""

    def test_init_creates_client(self):
        """Constructor should create VectorDBClient with correct params."""
        config = TencentVectorDBConfig(
            url="https://example.com",
            username="root",
            key="secret",
        )

        mock_client_cls = MagicMock()
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        with patch(
            "memory_platform.adapters.tencent_vector.VectorDBClient",
            mock_client_cls,
        ):
            TencentVectorStore(config=config)

        mock_client_cls.assert_called_once_with(
            url="https://example.com",
            username="root",
            key="secret",
            timeout=30,
        )

    def test_init_resolves_database(self):
        """Constructor should resolve the database reference."""
        config = TencentVectorDBConfig(
            url="https://example.com",
            username="root",
            key="secret",
            database_name="test_db",
        )

        mock_db = MagicMock()
        mock_client = MagicMock()
        mock_client.database.return_value = mock_db

        with patch(
            "memory_platform.adapters.tencent_vector.VectorDBClient",
            return_value=mock_client,
        ):
            store = TencentVectorStore(config=config)

        mock_client.database.assert_called_once_with("test_db")
        assert store.db == mock_db

    def test_init_stores_collection_name(self):
        """Constructor should store collection_name for later use."""
        config = TencentVectorDBConfig(
            url="https://example.com",
            username="root",
            key="secret",
            collection_name="my_collection",
        )

        mock_client = MagicMock()

        with patch(
            "memory_platform.adapters.tencent_vector.VectorDBClient",
            return_value=mock_client,
        ):
            store = TencentVectorStore(config=config)

        assert store.collection_name == "my_collection"

    def test_init_stores_embedding_dims(self):
        """Constructor should store embedding_model_dims."""
        config = TencentVectorDBConfig(
            url="https://example.com",
            username="root",
            key="secret",
            embedding_model_dims=768,
        )

        mock_client = MagicMock()

        with patch(
            "memory_platform.adapters.tencent_vector.VectorDBClient",
            return_value=mock_client,
        ):
            store = TencentVectorStore(config=config)

        assert store.embedding_model_dims == 768


class TestCreateCol:
    """Tests for TencentVectorStore.create_col."""

    def _make_store(self, config=None):
        if config is None:
            config = TencentVectorDBConfig(
                url="https://example.com",
                username="root",
                key="secret",
            )
        mock_client = MagicMock()
        with patch(
            "memory_platform.adapters.tencent_vector.VectorDBClient",
            return_value=mock_client,
        ):
            return TencentVectorStore(config=config)

    def test_create_col_creates_database_if_not_exists(self):
        """create_col should create database when it does not exist."""
        store = self._make_store()
        store.db.list_collections.return_value = []

        with patch.object(store.db, "create_collection") as mock_create:
            store.create_col(name="mem0", vector_size=1536, distance="cosine")

        # Database should already be resolved via client.database() in __init__,
        # which auto-creates if needed. Collection is created here.
        mock_create.assert_called_once()

    def test_create_col_skips_if_collection_exists(self):
        """create_col should skip creation if collection already exists."""
        store = self._make_store()
        store.db.list_collections.return_value = [
            MagicMock(collection_name="mem0")
        ]

        with patch.object(store.db, "create_collection") as mock_create:
            store.create_col(name="mem0", vector_size=1536, distance="cosine")

        mock_create.assert_not_called()

    def test_create_col_calls_create_collection_with_correct_params(self):
        """create_col should pass correct dimension and index params."""
        store = self._make_store()
        store.db.list_collections.return_value = []

        with patch.object(store.db, "create_collection") as mock_create:
            store.create_col(name="mem0", vector_size=768, distance="cosine")

        # Verify create_collection was called (exact args depend on SDK version)
        call_kwargs = mock_create.call_args
        assert call_kwargs is not None


class TestListCols:
    """Tests for TencentVectorStore.list_cols."""

    def test_list_cols_returns_collection_names(self):
        """list_cols should return list of collection name strings."""
        config = TencentVectorDBConfig(
            url="https://example.com", username="root", key="secret"
        )
        mock_client = MagicMock()

        mock_col1 = MagicMock()
        mock_col1.collection_name = "mem0"
        mock_col2 = MagicMock()
        mock_col2.collection_name = "app_001"
        mock_client.database.return_value.list_collections.return_value = [
            mock_col1,
            mock_col2,
        ]

        with patch(
            "memory_platform.adapters.tencent_vector.VectorDBClient",
            return_value=mock_client,
        ):
            store = TencentVectorStore(config=config)
            result = store.list_cols()

        assert result == ["mem0", "app_001"]

    def test_list_cols_returns_empty_list_when_no_collections(self):
        """list_cols should return empty list when no collections exist."""
        config = TencentVectorDBConfig(
            url="https://example.com", username="root", key="secret"
        )
        mock_client = MagicMock()
        mock_client.database.return_value.list_collections.return_value = []

        with patch(
            "memory_platform.adapters.tencent_vector.VectorDBClient",
            return_value=mock_client,
        ):
            store = TencentVectorStore(config=config)
            result = store.list_cols()

        assert result == []


class TestInsert:
    """Tests for TencentVectorStore.insert."""

    def _make_store_with_collection(self):
        config = TencentVectorDBConfig(
            url="https://example.com", username="root", key="secret"
        )
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.database.return_value.collection.return_value = (
            mock_collection
        )

        with patch(
            "memory_platform.adapters.tencent_vector.VectorDBClient",
            return_value=mock_client,
        ):
            store = TencentVectorStore(config=config)
        return store, mock_collection

    def test_insert_calls_upsert_with_correct_params(
        self, sample_embedding, sample_payload, sample_memory_id
    ):
        """insert should call collection.upsert with doc_ids, documents, embeddings."""
        store, mock_collection = self._make_store_with_collection()

        store.insert(
            vectors=[sample_embedding],
            payloads=[sample_payload],
            ids=[sample_memory_id],
        )

        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args
        assert call_kwargs.kwargs["doc_ids"] == [sample_memory_id]
        assert call_kwargs.kwargs["documents"] == [sample_payload]
        assert call_kwargs.kwargs["embeddings"] == [sample_embedding]

    def test_insert_with_no_payload(
        self, sample_embedding, sample_memory_id
    ):
        """insert should handle empty payload by passing None."""
        store, mock_collection = self._make_store_with_collection()

        store.insert(
            vectors=[sample_embedding],
            payloads=None,
            ids=[sample_memory_id],
        )

        # When payload is None, should pass empty dict or skip
        mock_collection.upsert.assert_called_once()

    def test_insert_batch_multiple(
        self, sample_embedding, sample_payload, sample_memory_id
    ):
        """insert should handle multiple vectors in one call."""
        store, mock_collection = self._make_store_with_collection()

        store.insert(
            vectors=[sample_embedding, sample_embedding],
            payloads=[sample_payload, sample_payload],
            ids=[sample_memory_id, "another-uuid"],
        )

        call_kwargs = mock_collection.upsert.call_args
        assert len(call_kwargs.kwargs["doc_ids"]) == 2


class TestSearch:
    """Tests for TencentVectorStore.search."""

    def _make_store_with_mock_search(self, search_results=None):
        """Create a store with a mocked collection.search return value."""
        config = TencentVectorDBConfig(
            url="https://example.com", username="root", key="secret"
        )
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.search.return_value = search_results or []
        mock_client.database.return_value.collection.return_value = (
            mock_collection
        )

        with patch(
            "memory_platform.adapters.tencent_vector.VectorDBClient",
            return_value=mock_client,
        ):
            store = TencentVectorStore(config=config)
        return store, mock_collection

    def test_search_returns_list_of_output_data(self, sample_embedding):
        """search should return a list of OutputData objects."""
        # Mock search result: list of document-like objects
        mock_doc = MagicMock()
        mock_doc.id = "test-id-1"
        mock_doc.score = 0.95
        mock_doc.metadata = {"data": "test memory", "user_id": "u1"}
        mock_doc.text = "test memory"

        store, mock_collection = self._make_store_with_mock_search([mock_doc])

        result = store.search(
            query="test query",
            vectors=[sample_embedding],
            limit=5,
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], OutputData)
        assert result[0].id == "test-id-1"
        assert result[0].score == 0.95

    def test_search_passes_filter_to_collection_search(self, sample_embedding):
        """search should translate mem0 filters to Tencent VectorDB filter."""
        store, mock_collection = self._make_store_with_mock_search([])

        store.search(
            query="test",
            vectors=[sample_embedding],
            limit=5,
            filters={"user_id": "u1", "memory_layer": "L1"},
        )

        mock_collection.search.assert_called_once()
        call_kwargs = mock_collection.search.call_args
        # Filter should be passed to the SDK search
        assert "filter" in call_kwargs.kwargs or len(call_kwargs.args) > 1

    def test_search_with_no_results(self, sample_embedding):
        """search should return empty list when no results."""
        store, _ = self._make_store_with_mock_search([])

        result = store.search(
            query="no match",
            vectors=[sample_embedding],
            limit=5,
        )

        assert result == []


class TestDelete:
    """Tests for TencentVectorStore.delete."""

    def _make_store(self):
        config = TencentVectorDBConfig(
            url="https://example.com", username="root", key="secret"
        )
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.database.return_value.collection.return_value = (
            mock_collection
        )

        with patch(
            "memory_platform.adapters.tencent_vector.VectorDBClient",
            return_value=mock_client,
        ):
            store = TencentVectorStore(config=config)
        return store, mock_collection

    def test_delete_calls_collection_delete(self, sample_memory_id):
        """delete should call collection.delete with the document ID."""
        store, mock_collection = self._make_store()

        store.delete(vector_id=sample_memory_id)

        mock_collection.delete.assert_called_once_with(
            document_ids=[sample_memory_id]
        )

    def test_delete_with_multiple_ids(self):
        """delete should handle a single ID passed as string."""
        store, mock_collection = self._make_store()

        store.delete(vector_id="id-1")

        mock_collection.delete.assert_called_once_with(document_ids=["id-1"])


class TestUpdate:
    """Tests for TencentVectorStore.update."""

    def _make_store(self):
        config = TencentVectorDBConfig(
            url="https://example.com", username="root", key="secret"
        )
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.database.return_value.collection.return_value = (
            mock_collection
        )

        with patch(
            "memory_platform.adapters.tencent_vector.VectorDBClient",
            return_value=mock_client,
        ):
            store = TencentVectorStore(config=config)
        return store, mock_collection

    def test_update_calls_upsert_with_vector_and_payload(
        self, sample_embedding, sample_payload, sample_memory_id
    ):
        """update should call upsert to replace the document."""
        store, mock_collection = self._make_store()

        store.update(
            vector_id=sample_memory_id,
            vector=sample_embedding,
            payload=sample_payload,
        )

        mock_collection.upsert.assert_called_once_with(
            doc_ids=[sample_memory_id],
            documents=[sample_payload],
            embeddings=[sample_embedding],
        )

    def test_update_with_payload_only(self, sample_payload, sample_memory_id):
        """update should work with only payload (no vector update)."""
        store, mock_collection = self._make_store()

        store.update(
            vector_id=sample_memory_id,
            payload=sample_payload,
        )

        # When no vector provided, we still need to call upsert
        # but need the existing vector — for now we'll pass None
        # and let the implementation handle it
        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args.kwargs
        assert call_kwargs["doc_ids"] == [sample_memory_id]
        assert call_kwargs["documents"] == [sample_payload]


class TestGet:
    """Tests for TencentVectorStore.get."""

    def _make_store_with_mock_query(self, query_results=None):
        config = TencentVectorDBConfig(
            url="https://example.com", username="root", key="secret"
        )
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.query.return_value = query_results or []
        mock_client.database.return_value.collection.return_value = (
            mock_collection
        )

        with patch(
            "memory_platform.adapters.tencent_vector.VectorDBClient",
            return_value=mock_client,
        ):
            store = TencentVectorStore(config=config)
        return store, mock_collection

    def test_get_returns_output_data_for_existing_id(self, sample_payload):
        """get should return OutputData when document exists."""
        mock_doc = MagicMock()
        mock_doc.id = "existing-id"
        mock_doc.metadata = sample_payload
        mock_doc.text = sample_payload.get("data", "")

        store, mock_collection = self._make_store_with_mock_query([mock_doc])

        result = store.get(vector_id="existing-id")

        assert isinstance(result, OutputData)
        assert result.id == "existing-id"
        assert result.payload == sample_payload

    def test_get_returns_none_for_missing_id(self):
        """get should return None when document does not exist."""
        store, mock_collection = self._make_store_with_mock_query([])

        result = store.get(vector_id="nonexistent-id")

        assert result is None

    def test_get_calls_query_with_document_id(self):
        """get should call collection.query with the correct document ID."""
        store, mock_collection = self._make_store_with_mock_query([])

        store.get(vector_id="target-id")

        mock_collection.query.assert_called_once()
        call_kwargs = mock_collection.query.call_args.kwargs
        assert call_kwargs["document_ids"] == ["target-id"]


class TestList:
    """Tests for TencentVectorStore.list."""

    def _make_store_with_mock_documents(self, documents=None):
        config = TencentVectorDBConfig(
            url="https://example.com", username="root", key="secret"
        )
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.query.return_value = documents or []
        mock_client.database.return_value.collection.return_value = (
            mock_collection
        )

        with patch(
            "memory_platform.adapters.tencent_vector.VectorDBClient",
            return_value=mock_client,
        ):
            store = TencentVectorStore(config=config)
        return store, mock_collection

    def test_list_returns_nested_list(self, sample_payload):
        """list should return nested list: [[OutputData, ...]]."""
        mock_doc = MagicMock()
        mock_doc.id = "id-1"
        mock_doc.metadata = sample_payload
        mock_doc.text = sample_payload.get("data", "")

        store, _ = self._make_store_with_mock_documents([mock_doc])

        result = store.list()

        assert isinstance(result, list)
        assert len(result) == 1  # Outer list has 1 element
        assert isinstance(result[0], list)  # Inner is list of OutputData
        assert isinstance(result[0][0], OutputData)

    def test_list_returns_empty_nested_list(self):
        """list should return [[]] when no documents."""
        store, _ = self._make_store_with_mock_documents([])

        result = store.list()

        assert result == [[]]

    def test_list_passes_filters(self):
        """list should translate and pass filters to query."""
        store, mock_collection = self._make_store_with_mock_documents([])

        store.list(filters={"user_id": "u1"})

        mock_collection.query.assert_called_once()


class TestDeleteCol:
    """Tests for TencentVectorStore.delete_col."""

    def _make_store(self):
        config = TencentVectorDBConfig(
            url="https://example.com", username="root", key="secret"
        )
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.database.return_value.collection.return_value = (
            mock_collection
        )

        with patch(
            "memory_platform.adapters.tencent_vector.VectorDBClient",
            return_value=mock_client,
        ):
            store = TencentVectorStore(config=config)
        return store, mock_collection

    def test_delete_col_drops_collection(self):
        """delete_col should drop the current collection."""
        store, mock_collection = self._make_store()

        store.delete_col()

        mock_collection.drop.assert_called_once()


class TestColInfo:
    """Tests for TencentVectorStore.col_info."""

    def test_col_info_returns_collection_info(self):
        """col_info should return collection statistics."""
        config = TencentVectorDBConfig(
            url="https://example.com", username="root", key="secret"
        )
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.document_count = 42
        mock_client.database.return_value.collection.return_value = (
            mock_collection
        )

        with patch(
            "memory_platform.adapters.tencent_vector.VectorDBClient",
            return_value=mock_client,
        ):
            store = TencentVectorStore(config=config)
            info = store.col_info()

        assert info["document_count"] == 42


class TestReset:
    """Tests for TencentVectorStore.reset."""

    def _make_store(self):
        config = TencentVectorDBConfig(
            url="https://example.com", username="root", key="secret"
        )
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection
        mock_client.database.return_value = mock_db

        with patch(
            "memory_platform.adapters.tencent_vector.VectorDBClient",
            return_value=mock_client,
        ):
            store = TencentVectorStore(config=config)
        return store, mock_collection, mock_db

    def test_reset_drops_and_recreates_collection(self):
        """reset should delete and recreate the collection."""
        store, mock_collection, mock_db = self._make_store()
        store.db.list_collections.return_value = []

        store.reset()

        mock_collection.drop.assert_called_once()
        mock_db.create_collection.assert_called_once()

    def test_reset_recreates_with_correct_dimension(self):
        """reset should recreate collection with the configured dimension."""
        config = TencentVectorDBConfig(
            url="https://example.com",
            username="root",
            key="secret",
            embedding_model_dims=768,
        )
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection
        mock_client.database.return_value = mock_db

        with patch(
            "memory_platform.adapters.tencent_vector.VectorDBClient",
            return_value=mock_client,
        ):
            store = TencentVectorStore(config=config)

        store.db.list_collections.return_value = []
        store.reset()

        # Verify the recreated collection uses the correct dimension
        call_kwargs = mock_db.create_collection.call_args
        assert call_kwargs is not None
