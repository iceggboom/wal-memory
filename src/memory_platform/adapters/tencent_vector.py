"""Tencent Cloud VectorDB adapter for mem0.

Implements mem0's VectorStoreBase interface to use Tencent Cloud VectorDB
as the vector storage backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator
from tcvectordb import VectorDBClient

from mem0.vector_stores.base import VectorStoreBase


@dataclass(frozen=True)
class OutputData:
    """Wraps a search/memory result for mem0 compatibility.

    mem0 expects search results and list/get results to expose
    `.id`, `.payload`, and `.score` attributes. This dataclass
    provides that interface.
    """

    id: str
    payload: dict[str, Any]
    score: float


class TencentVectorDBConfig(BaseModel):
    """Configuration for Tencent Cloud VectorDB connection.

    Attributes:
        url: VectorDB service endpoint URL.
        username: Database username.
        key: API key or password.
        collection_name: Default collection name for mem0.
        embedding_model_dims: Dimension of embedding vectors.
        database_name: Database name in VectorDB.
        timeout: Request timeout in seconds.
    """

    url: str
    username: str
    key: str
    collection_name: str = Field(default="mem0")
    embedding_model_dims: int = Field(default=1536)
    database_name: str = Field(default="memory_platform")
    timeout: int = Field(default=30)

    @field_validator("timeout")
    @classmethod
    def timeout_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("timeout must be a positive integer")
        return v


class TencentVectorStore(VectorStoreBase):
    """Tencent Cloud VectorDB adapter for mem0.

    Connects mem0 to Tencent Cloud VectorDB as the vector storage backend.

    Can be instantiated in two ways:

    1. With a TencentVectorDBConfig object:
        config = TencentVectorDBConfig(url="...", username="root", key="...")
        store = TencentVectorStore(config=config)

    2. With kwargs (for mem0 factory compatibility):
        store = TencentVectorStore(url="...", username="root", key="...")
    """

    def __init__(
        self, config: TencentVectorDBConfig | None = None, **kwargs: Any
    ) -> None:
        if config is not None:
            self.config = config
        else:
            known_fields = TencentVectorDBConfig.model_fields
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_fields}
            self.config = TencentVectorDBConfig(**filtered_kwargs)

        self.collection_name = self.config.collection_name
        self.embedding_model_dims = self.config.embedding_model_dims

        self.client = VectorDBClient(
            url=self.config.url,
            username=self.config.username,
            key=self.config.key,
            timeout=self.config.timeout,
        )
        self.db = self.client.database(self.config.database_name)

    # --- Collection management ---

    def list_cols(self) -> list[str]:
        """List all collections in the database.

        Returns:
            List of collection name strings.
        """
        collections = self.db.list_collections()
        return [c.collection_name for c in collections]

    def create_col(
        self, name: str, vector_size: int, distance: str = "cosine"
    ) -> None:
        """Create a collection if it does not already exist.

        Args:
            name: Collection name.
            vector_size: Embedding vector dimension.
            distance: Distance metric ("cosine", "l2", or "ip").
        """
        existing = self.list_cols()
        if name in existing:
            return

        from tcvectordb.model.enum import IndexType
        from tcvectordb.model.index import VectorIndex, HNSWParams

        # Map distance metric
        index_type = IndexType.HNSW
        if distance == "cosine":
            metric_type = "COSINE"
        elif distance == "l2":
            metric_type = "L2"
        elif distance == "ip":
            metric_type = "IP"
        else:
            metric_type = "COSINE"

        vector_index = VectorIndex(
            field_name="embedding",
            index_type=index_type,
            dimension=vector_size,
            metric_type=metric_type,
            params=HNSWParams(m=16, efconstruction=200),
        )

        self.db.create_collection(
            name=name,
            description=f"mem0 collection: {name}",
            shard=1,
            replica=1,
            index=vector_index,
        )

    # --- Document CRUD ---

    def insert(
        self,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """Insert vectors with optional payloads and IDs.

        Args:
            vectors: List of embedding vectors.
            payloads: List of metadata dicts (same length as vectors).
            ids: List of document IDs (same length as vectors).
        """
        collection = self.db.collection(self.collection_name)

        if payloads is None:
            payloads = [{} for _ in vectors]

        if ids is None:
            import uuid

            ids = [str(uuid.uuid4()) for _ in vectors]

        collection.upsert(
            doc_ids=ids,
            documents=payloads,
            embeddings=vectors,
        )

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> Any:
        """Translate mem0 filter dict to Tencent VectorDB Filter object.

        Args:
            filters: mem0-style filter dict, e.g. {"user_id": "u1", "agent_id": "a1"}.

        Returns:
            Tencent VectorDB Filter object, or None if no filters.
        """
        if not filters:
            return None

        from tcvectordb.model.document import Filter

        result: Filter | None = None
        for key, value in filters.items():
            if isinstance(value, list):
                cond = Filter.Include(key, value)
            else:
                cond = Filter.Include(key, [str(value)])

            if result is None:
                result = Filter(cond)
            else:
                result = result.And(cond)

        return result

    def search(
        self,
        query: str,
        vectors: list[list[float]],
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[OutputData]:
        """Search for similar vectors.

        Args:
            query: Search query text (not used directly, vectors are used).
            vectors: Query embedding vector(s).
            limit: Maximum number of results.
            filters: Optional metadata filters.

        Returns:
            List of OutputData objects sorted by score descending.
        """
        collection = self.db.collection(self.collection_name)
        vector = vectors[0] if vectors else []

        db_filter = self._build_filter(filters)

        results = collection.search(
            embedding=vector,
            limit=limit,
            filter=db_filter,
        )

        output = []
        for doc in results:
            payload = getattr(doc, "metadata", None) or {}
            if not isinstance(payload, dict):
                payload = {}

            doc_text = getattr(doc, "text", None)
            if doc_text and "data" not in payload:
                payload["data"] = doc_text

            output.append(
                OutputData(
                    id=doc.id,
                    payload=payload,
                    score=doc.score,
                )
            )

        return output

    def delete(self, vector_id: str) -> None:
        """Delete a document by ID.

        Args:
            vector_id: Document ID to delete.
        """
        collection = self.db.collection(self.collection_name)
        collection.delete(document_ids=[vector_id])

    def update(
        self,
        vector_id: str,
        vector: list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Update a document's vector and/or payload.

        Tencent VectorDB uses upsert for updates (full replace).

        Args:
            vector_id: Document ID to update.
            vector: New embedding vector (optional).
            payload: New metadata dict (optional).
        """
        collection = self.db.collection(self.collection_name)

        if payload is None and vector is None:
            return

        if payload is None:
            payload = {}

        if vector is not None:
            collection.upsert(
                doc_ids=[vector_id],
                documents=[payload],
                embeddings=[vector],
            )
        else:
            existing = self.get(vector_id)
            if existing is None:
                return
            collection.upsert(
                doc_ids=[vector_id],
                documents=[payload],
                embeddings=None,
            )

    def get(self, vector_id: str) -> OutputData | None:
        """Get a document by ID.

        Args:
            vector_id: Document ID to retrieve.

        Returns:
            OutputData with id, payload, score, or None if not found.
        """
        collection = self.db.collection(self.collection_name)

        docs = collection.query(document_ids=[vector_id])

        if not docs:
            return None

        doc = docs[0]
        payload = getattr(doc, "metadata", None) or {}
        if not isinstance(payload, dict):
            payload = {}

        doc_text = getattr(doc, "text", None)
        if doc_text and "data" not in payload:
            payload["data"] = doc_text

        return OutputData(
            id=doc.id,
            payload=payload,
            score=getattr(doc, "score", 1.0),
        )

    def list(
        self,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[list[OutputData]]:
        """List documents, optionally filtered.

        mem0 expects a nested list: [[OutputData, ...]],
        accessed as memories_result[0].

        Args:
            filters: Optional metadata filters.
            limit: Maximum number of results.

        Returns:
            Nested list of OutputData objects.
        """
        collection = self.db.collection(self.collection_name)

        db_filter = self._build_filter(filters)
        docs = collection.query(
            limit=limit,
            filter=db_filter,
        )

        results = []
        for doc in docs:
            payload = getattr(doc, "metadata", None) or {}
            if not isinstance(payload, dict):
                payload = {}

            doc_text = getattr(doc, "text", None)
            if doc_text and "data" not in payload:
                payload["data"] = doc_text

            results.append(
                OutputData(
                    id=doc.id,
                    payload=payload,
                    score=getattr(doc, "score", 1.0),
                )
            )

        return [results]  # Nested list for mem0 compatibility

    def delete_col(self) -> None:
        """Delete the current collection."""
        collection = self.db.collection(self.collection_name)
        collection.drop()

    def col_info(self) -> dict[str, Any]:
        """Get information about the current collection.

        Returns:
            Dict with collection metadata (document_count, etc.).
        """
        collection = self.db.collection(self.collection_name)
        return {
            "collection_name": self.collection_name,
            "document_count": getattr(collection, "document_count", 0),
        }

    def reset(self) -> None:
        """Reset the collection by deleting and recreating it.

        This is used by mem0 to clear all memories.
        """
        try:
            self.delete_col()
        except Exception:
            pass  # Collection may not exist yet

        self.create_col(
            name=self.collection_name,
            vector_size=self.embedding_model_dims,
            distance="cosine",
        )
