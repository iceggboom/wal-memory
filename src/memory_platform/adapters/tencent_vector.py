"""Tencent Cloud VectorDB adapter for mem0.

实现 mem0 的 VectorStoreBase 接口，对接腾讯云向量数据库。
支持 mock 模式用于开发测试，无需真实 VectorDB 服务。
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutputData:
    """mem0 兼容的搜索/记忆结果封装。

    mem0 期望搜索结果暴露 .id、.payload、.score 属性。
    """

    id: str
    payload: dict[str, Any]
    score: float


class TencentVectorDBConfig(BaseModel):
    """腾讯云向量数据库连接配置。

    Attributes:
        url: VectorDB 服务端地址。
        username: 数据库用户名。
        key: API 密钥。
        collection_name: 默认 Collection 名称。
        embedding_model_dims: 向量维度。
        database_name: 数据库名称。
        timeout: 请求超时时间（秒）。
        mock: 是否使用 mock 模式。
    """

    url: str = ""
    username: str = "root"
    key: str = ""
    collection_name: str = Field(default="mem0")
    embedding_model_dims: int = Field(default=1536)
    database_name: str = Field(default="memory_platform")
    timeout: int = Field(default=30)
    embedding_model: str = Field(default="bge-base-zh")
    mock: bool = False

    @field_validator("timeout")
    @classmethod
    def timeout_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("timeout must be a positive integer")
        return v


# ============================================================================
# Mock 向量存储 — 内存实现，用于开发测试
# ============================================================================


class _MockStore:
    """基于内存的 mock 向量存储，模拟腾讯云 VectorDB 的行为。"""

    def __init__(self, dims: int = 1536):
        self._vectors: dict[str, list[float]] = {}
        self._payloads: dict[str, dict[str, Any]] = {}
        self._dims = dims

    def insert(
        self,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        if payloads is None:
            payloads = [{} for _ in vectors]
        for doc_id, vec, payload in zip(ids, vectors, payloads):
            self._vectors[doc_id] = vec
            self._payloads[doc_id] = payload

    def search(
        self,
        vectors: list[list[float]],
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[OutputData]:
        # mem0 传入的 vectors 可能是 list[float] 或 list[list[float]]
        if vectors and isinstance(vectors[0], (int, float)):
            query = list(map(float, vectors))
        else:
            query = vectors[0] if vectors else [0.0] * self._dims
        results: list[tuple[str, float, dict]] = []

        for doc_id, vec in self._vectors.items():
            payload = self._payloads.get(doc_id, {})
            if not _match_filters(payload, filters):
                continue
            score = _cosine_similarity(query, vec)
            results.append((doc_id, score, payload))

        results.sort(key=lambda x: x[1], reverse=True)
        return [OutputData(id=did, score=sc, payload=pl) for did, sc, pl in results[:limit]]

    def get(self, vector_id: str) -> OutputData | None:
        if vector_id not in self._vectors:
            return None
        return OutputData(
            id=vector_id,
            payload=self._payloads.get(vector_id, {}),
            score=1.0,
        )

    def delete(self, vector_id: str) -> None:
        self._vectors.pop(vector_id, None)
        self._payloads.pop(vector_id, None)

    def update(
        self,
        vector_id: str,
        vector: list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if vector_id not in self._vectors:
            return
        if vector is not None:
            self._vectors[vector_id] = vector
        if payload is not None:
            self._payloads[vector_id] = payload

    def list(
        self,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[OutputData]:
        results = []
        for doc_id, payload in self._payloads.items():
            if not _match_filters(payload, filters):
                continue
            results.append(OutputData(id=doc_id, payload=payload, score=1.0))
        if limit:
            results = results[:limit]
        return results

    def list_cols(self) -> list[str]:
        return ["mock_collection"]

    def create_col(self, name: str, **kwargs: Any) -> None:
        pass

    def delete_col(self) -> None:
        self._vectors.clear()
        self._payloads.clear()

    def col_info(self) -> dict[str, Any]:
        return {
            "collection_name": "mock_collection",
            "document_count": len(self._vectors),
        }

    def reset(self) -> None:
        self.delete_col()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度（纯 Python 实现）。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a > 0 and norm_b > 0:
        return dot / (norm_a * norm_b)
    return 0.0


def _match_filters(payload: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """检查 payload 是否匹配过滤条件。"""
    if not filters:
        return True
    for key, value in filters.items():
        payload_val = payload.get(key)
        if isinstance(value, list):
            if payload_val not in value:
                return False
        elif payload_val != value:
            return False
    return True


# ============================================================================
# TencentVectorStore — mem0 VectorStoreBase 实现
# ============================================================================


class TencentVectorStore(VectorStoreBase):
    """腾讯云向量数据库适配器。

    支持两种运行模式：
    1. 真实模式（mock=False）：连接腾讯云 VectorDB
    2. Mock 模式（mock=True）：使用内存存储，无需真实服务

    实例化方式：
        # 通过配置对象
        config = TencentVectorDBConfig(url="...", username="root", key="...")
        store = TencentVectorStore(config=config)

        # 通过 kwargs（兼容 mem0 工厂）
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

        if self.config.mock:
            logger.info("TencentVectorStore running in MOCK mode")
            self._store = _MockStore(dims=self.embedding_model_dims)
            self.client = None
            self.db = None
        else:
            from tcvectordb import RPCVectorDBClient

            self.client = RPCVectorDBClient(
                url=self.config.url,
                username=self.config.username,
                key=self.config.key,
                timeout=self.config.timeout,
            )
            # 创建数据库（如果不存在）
            self.client.create_database_if_not_exists(self.config.database_name)
            # 获取数据库对象
            self.db = self.client.database(self.config.database_name)
            self._store = None
            # 自动创建 Collection（如果不存在）
            self.create_col(
                name=self.collection_name,
                vector_size=self.embedding_model_dims,
                distance="cosine",
            )

    @property
    def _is_mock(self) -> bool:
        return self._store is not None

    # --- Collection management ---

    def list_cols(self) -> list[str]:
        """列出所有 Collection。"""
        if self._is_mock:
            return self._store.list_cols()

        collections = self.db.list_collections()
        return [c.collection_name for c in collections]

    def create_col(
        self, name: str, vector_size: int, distance: str = "cosine"
    ) -> None:
        """创建 Collection（如不存在）。

        真实模式下使用 VectorDB 内置 Embedding，向量由服务端自动生成。
        """
        if self._is_mock:
            return self._store.create_col(name)

        existing = self.list_cols()
        if name in existing:
            return

        from tcvectordb.model.collection import Embedding
        from tcvectordb.model.enum import FieldType, IndexType
        from tcvectordb.model.index import FilterIndex, VectorIndex, HNSWParams

        metric_type = {"cosine": "COSINE", "l2": "L2", "ip": "IP"}.get(
            distance, "COSINE"
        )

        ebd = Embedding(
            vector_field="vector",
            field="text",
            model_name=self.config.embedding_model,
        )

        index = Index(
            FilterIndex(name="id", field_type=FieldType.String, index_type=IndexType.PRIMARY_KEY),
            VectorIndex(
                name="vector",
                dimension=vector_size,
                index_type=IndexType.HNSW,
                metric_type=metric_type,
                params=HNSWParams(m=16, efconstruction=200),
            ),
        )

        self.db.create_collection(
            name=name,
            description=f"mem0 collection: {name}",
            shard=1,
            replicas=1,
            embedding=ebd,
            index=index,
        )

    # --- Document CRUD ---

    def insert(
        self,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """插入向量文档。

        真实模式：使用 VectorDB 内置 Embedding，传文本由服务端向量化。
        Mock 模式：使用客户端向量直接存入内存。
        """
        if self._is_mock:
            return self._store.insert(vectors, payloads, ids)

        from tcvectordb.model.document import Document

        if payloads is None:
            payloads = [{} for _ in vectors]

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        docs = []
        for doc_id, payload in zip(ids, payloads):
            text = payload.pop("data", "") if "data" in payload else ""
            doc = Document(id=doc_id, text=text, **payload)
            docs.append(doc)

        self.client.upsert(
            database_name=self.config.database_name,
            collection_name=self.collection_name,
            documents=docs,
        )

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> Any:
        """将 mem0 过滤条件转换为腾讯云 VectorDB Filter 对象。"""
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
        """向量相似度搜索。

        真实模式：使用 search_by_text 传文本，服务端自动 Embedding。
        Mock 模式：使用客户端向量进行内存搜索。
        """
        if self._is_mock:
            return self._store.search(vectors, limit, filters)

        db_filter = self._build_filter(filters)

        result = self.client.search_by_text(
            database_name=self.config.database_name,
            collection_name=self.collection_name,
            embedding_items=[query],
            filter=db_filter,
            limit=limit,
            retrieve_vector=False,
            output_fields=["text"],
        )

        output = []
        for doc_list in result.get("documents", []):
            for doc in doc_list:
                payload = {}
                doc_text = getattr(doc, "text", None) or ""
                if doc_text:
                    payload["data"] = doc_text
                # 提取其他字段作为 metadata
                for field in ["user_id", "agent_id", "run_id", "actor_id", "role",
                              "hash", "memory_layer", "scope", "app_id",
                              "created_at", "updated_at"]:
                    val = getattr(doc, field, None) or (getattr(doc, "_data", {}).get(field))
                    if val:
                        payload[field] = val

                output.append(
                    OutputData(
                        id=getattr(doc, "id", ""),
                        payload=payload,
                        score=getattr(doc, "score", 0.0),
                    )
                )

        return output

    def delete(self, vector_id: str) -> None:
        """删除文档。"""
        if self._is_mock:
            return self._store.delete(vector_id)

        collection = self.db.collection(self.collection_name)
        collection.delete(document_ids=[vector_id])

    def update(
        self,
        vector_id: str,
        vector: list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """更新文档。"""
        if self._is_mock:
            return self._store.update(vector_id, vector, payload)

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
        """获取单个文档。"""
        if self._is_mock:
            return self._store.get(vector_id)

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
        """列出文档（mem0 期望嵌套列表格式）。"""
        if self._is_mock:
            results = self._store.list(filters, limit)
            return [results]

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

        return [results]

    def delete_col(self) -> None:
        """删除当前 Collection。"""
        if self._is_mock:
            return self._store.delete_col()

        collection = self.db.collection(self.collection_name)
        collection.drop()

    def col_info(self) -> dict[str, Any]:
        """获取 Collection 信息。"""
        if self._is_mock:
            return self._store.col_info()

        collection = self.db.collection(self.collection_name)
        return {
            "collection_name": self.collection_name,
            "document_count": getattr(collection, "document_count", 0),
        }

    def reset(self) -> None:
        """重置 Collection。"""
        if self._is_mock:
            return self._store.reset()

        try:
            self.delete_col()
        except Exception:
            pass

        self.create_col(
            name=self.collection_name,
            vector_size=self.embedding_model_dims,
            distance="cosine",
        )
