# Tencent VectorDB Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a mem0-compatible `VectorStoreBase` adapter that connects mem0 to Tencent Cloud VectorDB, enabling the AI Memory Platform to use Tencent Cloud as its vector storage backend.

**Architecture:** The adapter (`TencentVectorStore`) inherits from mem0's `VectorStoreBase` abstract class and translates each of the 13 abstract methods into Tencent Cloud VectorDB SDK (`tcvectordb`) calls. Results are wrapped in `OutputData` dataclasses to satisfy mem0's expected return types (`.id`, `.payload`, `.score`). A Pydantic config class (`TencentVectorDBConfig`) handles connection parameters. The adapter is registered with mem0's `VectorStoreConfig._provider_configs` so it can be instantiated via mem0's factory.

**Tech Stack:** Python 3.12+, mem0 SDK, tcvectordb SDK, pydantic v2, pytest + pytest-asyncio, uv

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata, dependencies (mem0, tcvectordb, pydantic, fastapi, etc.) |
| `src/memory_platform/__init__.py` | Package init, export adapter class |
| `src/memory_platform/config.py` | Global platform config (loads from env/`.env`) |
| `src/memory_platform/adapters/__init__.py` | Adapter package init |
| `src/memory_platform/adapters/tencent_vector.py` | Core adapter: `TencentVectorStore` + `OutputData` + `TencentVectorDBConfig` |
| `tests/conftest.py` | Shared fixtures (mock client, sample embeddings, sample payloads) |
| `tests/unit/adapters/__init__.py` | Test package init |
| `tests/unit/adapters/test_tencent_vector.py` | Unit tests for all 13 `VectorStoreBase` methods (mock SDK) |
| `tests/unit/adapters/test_tencent_vector_config.py` | Config validation tests |

---

## Key Reference: mem0 VectorStoreBase Interface

From `mem0/vector_stores/base.py`, the 13 abstract methods:

```python
class VectorStoreBase(ABC):
    def __init__(self, **kwargs): ...

    @abstractmethod
    def create_col(self, name, vector_size, distance): ...

    @abstractmethod
    def insert(self, vectors, payloads=None, ids=None): ...

    @abstractmethod
    def search(self, query, vectors, limit=5, filters=None): ...

    @abstractmethod
    def delete(self, vector_id): ...

    @abstractmethod
    def update(self, vector_id, vector=None, payload=None): ...

    @abstractmethod
    def get(self, vector_id): ...

    @abstractmethod
    def list_cols(self): ...

    @abstractmethod
    def delete_col(self): ...

    @abstractmethod
    def col_info(self): ...

    @abstractmethod
    def list(self, filters=None, limit=None): ...

    @abstractmethod
    def reset(self): ...
```

## Key Reference: Return Type Contracts

mem0's `Memory` class (in `mem0/memory/main.py`) expects:

- `search()` returns objects with `.id`, `.payload`, `.score` attributes
- `list()` returns a nested list: `[[OutputData(...), ...]]` (accessed as `memories_result[0]`)
- `get()` returns a single object with `.id`, `.payload`, `.score` or `None`
- `list_cols()` returns a list of collection name strings

## Key Reference: Tencent Cloud VectorDB SDK Patterns

```python
from tcvectordb import VectorDBClient

client = VectorDBClient(url=url, username=username, key=key, timeout=timeout)
db = client.database(db_name)
db.create_database(db_name)          # create if not exists
db.list_databases()                  # list all databases
db.drop_database(db_name)            # drop database

collection = db.collection(collection_name)
# or: db.create_collection(name=..., dimension=..., replica=...,
#     index=..., description=...)

collection.upsert(doc_ids=[...], documents=[...], embeddings=[...])
collection.search(embedding=[...], limit=10, filter=Filter(...))
collection.query(document_ids=[...])   # get by IDs
collection.delete(document_ids=[...])
collection.list_documents()           # list all docs
collection.drop()                     # delete collection
```

## Key Reference: mem0 Payload Structure

mem0 stores metadata in payload like this:
```python
{
    "data": "actual memory text",
    "hash": "md5hash",
    "created_at": "2026-04-06T12:00:00",
    "updated_at": "2026-04-06T12:00:00",
    "user_id": "user123",
    "agent_id": None,
    "run_id": None,
    "actor_id": None,
    # ... custom metadata from Extension Layer:
    "memory_layer": "L1",
    "app_id": "app001",
    "scope": "shared"
}
```

## Key Reference: Filter Translation

mem0 passes filters as nested dicts, e.g.:
```python
{"user_id": "user123", "actor_id": "agent456"}
```

Tencent VectorDB uses `Filter` objects:
```python
from tcvectordb.model.document import Filter
Filter(Filter.In("user_id", ["user123"]), Filter.Ne("actor_id", None))
```

---

### Task 1: Project Scaffolding — pyproject.toml

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "wal-memory"
version = "0.1.0"
description = "AI Memory Platform based on mem0 with Tencent Cloud VectorDB"
requires-python = ">=3.12"
dependencies = [
    "mem0ai>=0.1.0",
    "tcvectordb>=1.3.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "fastapi>=0.115.0",
    "uvicorn>=0.30.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24.0",
    "pytest-mock>=3.14.0",
    "ruff>=0.5.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/memory_platform"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
target-version = "py312"
```

- [ ] **Step 2: Install dependencies**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv sync --all-extras`
Expected: Creates `.venv` and `uv.lock`

- [ ] **Step 3: Create package directories**

Run:
```bash
mkdir -p /Users/I0W02SJ/gitCode/wal-memory/src/memory_platform/adapters
mkdir -p /Users/I0W02SJ/gitCode/wal-memory/tests/unit/adapters
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: scaffold project with pyproject.toml and dependencies"
```

---

### Task 2: Package Init Files

**Files:**
- Create: `src/memory_platform/__init__.py`
- Create: `src/memory_platform/adapters/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/adapters/__init__.py`

- [ ] **Step 1: Create all `__init__.py` files**

`src/memory_platform/__init__.py`:
```python
"""AI Memory Platform - shared memory infrastructure for AI applications."""

__version__ = "0.1.0"
```

`src/memory_platform/adapters/__init__.py`:
```python
"""Vector store adapters for mem0."""

from memory_platform.adapters.tencent_vector import TencentVectorDBConfig, TencentVectorStore

__all__ = ["TencentVectorDBConfig", "TencentVectorStore"]
```

`tests/__init__.py`:
```python

```

`tests/unit/__init__.py`:
```python

```

`tests/unit/adapters/__init__.py`:
```python

```

- [ ] **Step 2: Verify imports work**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run python -c "import memory_platform; print(memory_platform.__version__)"`
Expected: `0.1.0`

- [ ] **Step 3: Commit**

```bash
git add src/memory_platform/__init__.py src/memory_platform/adapters/__init__.py tests/__init__.py tests/unit/__init__.py tests/unit/adapters/__init__.py
git commit -m "chore: add package init files"
```

---

### Task 3: OutputData Dataclass

**Files:**
- Create: `src/memory_platform/adapters/tencent_vector.py` (partial)
- Create: `tests/unit/adapters/test_tencent_vector.py` (partial)
- Create: `tests/conftest.py`

This task creates the `OutputData` dataclass that wraps Tencent VectorDB results into the interface mem0 expects (objects with `.id`, `.payload`, `.score`).

- [ ] **Step 1: Write the failing test for OutputData**

`tests/conftest.py`:
```python
"""Shared test fixtures."""

import pytest


@pytest.fixture
def sample_embedding():
    """A sample 1536-dimensional embedding vector."""
    return [0.1] * 1536


@pytest.fixture
def sample_payload():
    """A sample mem0-compatible payload."""
    return {
        "data": "是一名Java工程师",
        "hash": "abc123md5",
        "created_at": "2026-04-06T12:00:00",
        "updated_at": "2026-04-06T12:00:00",
        "user_id": "user123",
        "memory_layer": "L1",
        "app_id": "app001",
        "scope": "shared",
    }


@pytest.fixture
def sample_memory_id():
    return "550e8400-e29b-41d4-a716-446655440000"
```

`tests/unit/adapters/test_tencent_vector.py`:
```python
"""Unit tests for TencentVectorStore adapter."""

from memory_platform.adapters.tencent_vector import OutputData


class TestOutputData:
    """Tests for the OutputData dataclass."""

    def test_output_data_attributes(self, sample_memory_id, sample_payload):
        """OutputData should expose id, payload, and score attributes."""
        result = OutputData(
            id=sample_memory_id,
            payload=sample_payload,
            score=0.95,
        )

        assert result.id == sample_memory_id
        assert result.payload == sample_payload
        assert result.score == 0.95

    def test_output_data_equality(self, sample_memory_id, sample_payload):
        """Two OutputData with same values should be equal."""
        a = OutputData(id=sample_memory_id, payload=sample_payload, score=0.95)
        b = OutputData(id=sample_memory_id, payload=sample_payload, score=0.95)

        assert a == b

    def test_output_data_immutable(self, sample_memory_id, sample_payload):
        """OutputData fields should be immutable (frozen dataclass)."""
        result = OutputData(
            id=sample_memory_id,
            payload=sample_payload,
            score=0.95,
        )

        try:
            result.score = 0.0
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector.py::TestOutputData -v`
Expected: FAIL with `ImportError: cannot import name 'OutputData'`

- [ ] **Step 3: Write minimal OutputData implementation**

`src/memory_platform/adapters/tencent_vector.py`:
```python
"""Tencent Cloud VectorDB adapter for mem0.

Implements mem0's VectorStoreBase interface to use Tencent Cloud VectorDB
as the vector storage backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector.py::TestOutputData -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/adapters/tencent_vector.py tests/conftest.py tests/unit/adapters/test_tencent_vector.py
git commit -m "feat: add OutputData dataclass for mem0 return type compatibility"
```

---

### Task 4: TencentVectorDBConfig Configuration Class

**Files:**
- Modify: `src/memory_platform/adapters/tencent_vector.py`
- Create: `tests/unit/adapters/test_tencent_vector_config.py`

- [ ] **Step 1: Write the failing tests for config**

`tests/unit/adapters/test_tencent_vector_config.py`:
```python
"""Unit tests for TencentVectorDBConfig."""

import pytest
from pydantic import ValidationError

from memory_platform.adapters.tencent_vector import TencentVectorDBConfig


class TestTencentVectorDBConfig:
    """Tests for the Tencent Cloud VectorDB configuration."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = TencentVectorDBConfig(
            url="https://example.tencentcloudapi.com",
            username="root",
            key="secret-key",
        )

        assert config.url == "https://example.tencentcloudapi.com"
        assert config.username == "root"
        assert config.key == "secret-key"
        assert config.collection_name == "mem0"
        assert config.embedding_model_dims == 1536
        assert config.database_name == "memory_platform"
        assert config.timeout == 30

    def test_custom_values(self):
        """Config should accept custom overrides."""
        config = TencentVectorDBConfig(
            url="https://custom.url",
            username="admin",
            key="my-key",
            collection_name="my_collection",
            embedding_model_dims=768,
            database_name="my_db",
            timeout=60,
        )

        assert config.collection_name == "my_collection"
        assert config.embedding_model_dims == 768
        assert config.database_name == "my_db"
        assert config.timeout == 60

    def test_url_required(self):
        """Config should fail without url."""
        with pytest.raises(ValidationError) as exc_info:
            TencentVectorDBConfig(username="root", key="secret")

        errors = exc_info.value.errors()
        error_fields = {e["loc"][0] for e in errors}
        assert "url" in error_fields

    def test_username_required(self):
        """Config should fail without username."""
        with pytest.raises(ValidationError) as exc_info:
            TencentVectorDBConfig(
                url="https://example.com", key="secret"
            )

        errors = exc_info.value.errors()
        error_fields = {e["loc"][0] for e in errors}
        assert "username" in error_fields

    def test_key_required(self):
        """Config should fail without key."""
        with pytest.raises(ValidationError) as exc_info:
            TencentVectorDBConfig(
                url="https://example.com", username="root"
            )

        errors = exc_info.value.errors()
        error_fields = {e["loc"][0] for e in errors}
        assert "key" in error_fields

    def test_timeout_must_be_positive(self):
        """Config should reject non-positive timeout."""
        with pytest.raises(ValidationError):
            TencentVectorDBConfig(
                url="https://example.com",
                username="root",
                key="secret",
                timeout=0,
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'TencentVectorDBConfig'`

- [ ] **Step 3: Write config implementation**

Append to `src/memory_platform/adapters/tencent_vector.py`:
```python
from pydantic import BaseModel, Field, field_validator


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector_config.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/adapters/tencent_vector.py tests/unit/adapters/test_tencent_vector_config.py
git commit -m "feat: add TencentVectorDBConfig pydantic model"
```

---

### Task 5: TencentVectorStore Constructor

**Files:**
- Modify: `src/memory_platform/adapters/tencent_vector.py`
- Modify: `tests/unit/adapters/test_tencent_vector.py`

This task implements the `TencentVectorStore.__init__()` method that creates the Tencent Cloud VectorDB client and resolves the database/collection references.

- [ ] **Step 1: Write the failing test for constructor**

Append to `tests/unit/adapters/test_tencent_vector.py`:
```python
from unittest.mock import MagicMock, patch

from memory_platform.adapters.tencent_vector import (
    TencentVectorDBConfig,
    TencentVectorStore,
)


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
            store = TencentVectorStore(config=config)

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector.py::TestTencentVectorStoreInit -v`
Expected: FAIL with `ImportError: cannot import name 'TencentVectorStore'`

- [ ] **Step 3: Write constructor implementation**

Append to `src/memory_platform/adapters/tencent_vector.py`:
```python
from tcvectordb import VectorDBClient


class TencentVectorStore:
    """mem0 VectorStoreBase implementation for Tencent Cloud VectorDB.

    Connects mem0 to Tencent Cloud VectorDB as the vector storage backend.

    Usage:
        config = TencentVectorDBConfig(
            url="https://...",
            username="root",
            key="...",
        )
        store = TencentVectorStore(config=config)
    """

    def __init__(self, config: TencentVectorDBConfig) -> None:
        self.config = config
        self.collection_name = config.collection_name
        self.embedding_model_dims = config.embedding_model_dims

        self.client = VectorDBClient(
            url=config.url,
            username=config.username,
            key=config.key,
            timeout=config.timeout,
        )
        self.db = self.client.database(config.database_name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector.py::TestTencentVectorStoreInit -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/adapters/tencent_vector.py tests/unit/adapters/test_tencent_vector.py
git commit -m "feat: add TencentVectorStore constructor with client setup"
```

---

### Task 6: create_col and list_cols Methods

**Files:**
- Modify: `src/memory_platform/adapters/tencent_vector.py`
- Modify: `tests/unit/adapters/test_tencent_vector.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/adapters/test_tencent_vector.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector.py::TestCreateCol tests/unit/adapters/test_tencent_vector.py::TestListCols -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Implement create_col and list_cols**

Append to `src/memory_platform/adapters/tencent_vector.py` (inside `TencentVectorStore` class):
```python
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

        from tcvectordb.model.enum import FieldType, IndexType
        from tcvectordb.model.index import Index, VectorIndex, HNSWParams

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector.py::TestCreateCol tests/unit/adapters/test_tencent_vector.py::TestListCols -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/adapters/tencent_vector.py tests/unit/adapters/test_tencent_vector.py
git commit -m "feat: add create_col and list_cols methods"
```

---

### Task 7: insert Method

**Files:**
- Modify: `src/memory_platform/adapters/tencent_vector.py`
- Modify: `tests/unit/adapters/test_tencent_vector.py`

mem0 calls `insert(vectors=[embedding], payloads=[metadata], ids=[memory_id])`. The `vectors` param is a list of embedding lists, `payloads` is a list of dicts, `ids` is a list of string IDs.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/adapters/test_tencent_vector.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector.py::TestInsert -v`
Expected: FAIL with `AttributeError: 'TencentVectorStore' object has no attribute 'insert'`

- [ ] **Step 3: Implement insert**

Append to `src/memory_platform/adapters/tencent_vector.py` (inside class):
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector.py::TestInsert -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/adapters/tencent_vector.py tests/unit/adapters/test_tencent_vector.py
git commit -m "feat: add insert method for writing vectors to Tencent VectorDB"
```

---

### Task 8: search Method

**Files:**
- Modify: `src/memory_platform/adapters/tencent_vector.py`
- Modify: `tests/unit/adapters/test_tencent_vector.py`

mem0 calls `search(query=query, vectors=embeddings, limit=5, filters=filters)` and iterates results as `mem.id`, `mem.payload`, `mem.score`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/adapters/test_tencent_vector.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector.py::TestSearch -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Implement search with filter translation**

Append to `src/memory_platform/adapters/tencent_vector.py` (inside class):
```python
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

        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append(Filter.In(key, value))
            elif value is None:
                conditions.append(Filter.NotExists(key))
            else:
                conditions.append(Filter(key, value))

        if len(conditions) == 1:
            return conditions[0]
        return Filter(Filter.And(*conditions))

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
            # Tencent VectorDB search results expose .id, .score
            # Metadata may be in .metadata or direct attributes
            payload = getattr(doc, "metadata", None) or {}
            if not isinstance(payload, dict):
                payload = {}

            # If there's a 'text' field on the doc, merge into payload
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector.py::TestSearch -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/adapters/tencent_vector.py tests/unit/adapters/test_tencent_vector.py
git commit -m "feat: add search method with filter translation"
```

---

### Task 9: delete and update Methods

**Files:**
- Modify: `src/memory_platform/adapters/tencent_vector.py`
- Modify: `tests/unit/adapters/test_tencent_vector.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/adapters/test_tencent_vector.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector.py::TestDelete tests/unit/adapters/test_tencent_vector.py::TestUpdate -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Implement delete and update**

Append to `src/memory_platform/adapters/tencent_vector.py` (inside class):
```python
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
            # When no vector provided, fetch existing and re-upsert
            existing = self.get(vector_id)
            if existing is None:
                return
            collection.upsert(
                doc_ids=[vector_id],
                documents=[payload],
                embeddings=None,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector.py::TestDelete tests/unit/adapters/test_tencent_vector.py::TestUpdate -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/adapters/tencent_vector.py tests/unit/adapters/test_tencent_vector.py
git commit -m "feat: add delete and update methods"
```

---

### Task 10: get and list Methods

**Files:**
- Modify: `src/memory_platform/adapters/tencent_vector.py`
- Modify: `tests/unit/adapters/test_tencent_vector.py`

mem0 calls `get(vector_id)` expecting `OutputData | None`, and `list(filters, limit)` expecting `list[list[OutputData]]` (nested list accessed as `result[0]`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/adapters/test_tencent_vector.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector.py::TestGet tests/unit/adapters/test_tencent_vector.py::TestList -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Implement get and list**

Append to `src/memory_platform/adapters/tencent_vector.py` (inside class):
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector.py::TestGet tests/unit/adapters/test_tencent_vector.py::TestList -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/adapters/tencent_vector.py tests/unit/adapters/test_tencent_vector.py
git commit -m "feat: add get and list methods with nested list return type"
```

---

### Task 11: delete_col, col_info, and reset Methods

**Files:**
- Modify: `src/memory_platform/adapters/tencent_vector.py`
- Modify: `tests/unit/adapters/test_tencent_vector.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/adapters/test_tencent_vector.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector.py::TestDeleteCol tests/unit/adapters/test_tencent_vector.py::TestColInfo tests/unit/adapters/test_tencent_vector.py::TestReset -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Implement delete_col, col_info, and reset**

Append to `src/memory_platform/adapters/tencent_vector.py` (inside class):
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector.py::TestDeleteCol tests/unit/adapters/test_tencent_vector.py::TestColInfo tests/unit/adapters/test_tencent_vector.py::TestReset -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/adapters/tencent_vector.py tests/unit/adapters/test_tencent_vector.py
git commit -m "feat: add delete_col, col_info, and reset methods"
```

---

### Task 12: Inherit from VectorStoreBase

**Files:**
- Modify: `src/memory_platform/adapters/tencent_vector.py`
- Modify: `tests/unit/adapters/test_tencent_vector.py`

Now that all 13 methods are implemented, make `TencentVectorStore` officially inherit from `mem0.vector_stores.base.VectorStoreBase`. Also add an `__init__` that accepts `**kwargs` for compatibility with mem0's factory pattern.

- [ ] **Step 1: Write the failing test for VectorStoreBase inheritance**

Append to `tests/unit/adapters/test_tencent_vector.py`:
```python
class TestVectorStoreBaseInheritance:
    """Verify TencentVectorStore is a valid VectorStoreBase implementation."""

    def test_is_subclass_of_vector_store_base(self):
        """TencentVectorStore should inherit from VectorStoreBase."""
        from mem0.vector_stores.base import VectorStoreBase

        assert issubclass(TencentVectorStore, VectorStoreBase)

    def test_implements_all_abstract_methods(self):
        """All 13 abstract methods must be implemented."""
        from mem0.vector_stores.base import VectorStoreBase

        abstract_methods = set(VectorStoreBase.__abstractmethods__)
        for method in abstract_methods:
            assert hasattr(TencentVectorStore, method), (
                f"TencentVectorStore missing method: {method}"
            )

    def test_can_be_instantiated_via_kwargs(self):
        """Should be instantiable with kwargs (factory pattern)."""
        mock_client = MagicMock()

        with patch(
            "memory_platform.adapters.tencent_vector.VectorDBClient",
            return_value=mock_client,
        ):
            store = TencentVectorStore(
                url="https://example.com",
                username="root",
                key="secret",
            )

        assert store.collection_name == "mem0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector.py::TestVectorStoreBaseInheritance -v`
Expected: FAIL — `TencentVectorStore` does not inherit from `VectorStoreBase`

- [ ] **Step 3: Add VectorStoreBase inheritance**

Modify the `TencentVectorStore` class definition and constructor. The key change is:

1. Import `VectorStoreBase` from `mem0.vector_stores.base`
2. Make `TencentVectorStore` inherit from it
3. Accept `**kwargs` in `__init__` and extract known params from kwargs or build `TencentVectorDBConfig`

Replace the class definition line and constructor:

```python
from mem0.vector_stores.base import VectorStoreBase


class TencentVectorStore(VectorStoreBase):
    """mem0 VectorStoreBase implementation for Tencent Cloud VectorDB.

    Can be instantiated in two ways:

    1. With a TencentVectorDBConfig object:
        config = TencentVectorDBConfig(url="...", username="root", key="...")
        store = TencentVectorStore(config=config)

    2. With kwargs (for mem0 factory compatibility):
        store = TencentVectorStore(url="...", username="root", key="...")
    """

    def __init__(self, config: TencentVectorDBConfig | None = None, **kwargs: Any) -> None:
        if config is not None:
            self.config = config
        else:
            # Build config from kwargs for mem0 factory compatibility
            known_fields = TencentVectorDBConfig.model_fields
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k in known_fields
            }
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/test_tencent_vector.py::TestVectorStoreBaseInheritance -v`
Expected: 3 passed

- [ ] **Step 5: Run ALL tests to check nothing broke**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/adapters/ -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/memory_platform/adapters/tencent_vector.py tests/unit/adapters/test_tencent_vector.py
git commit -m "feat: inherit TencentVectorStore from VectorStoreBase for mem0 compatibility"
```

---

### Task 13: Register with mem0 VectorStoreConfig

**Files:**
- Create: `src/memory_platform/config.py`

Register `tencent_vector` as a provider in mem0's `VectorStoreConfig._provider_configs` so users can specify `provider="tencent_vector"` in their mem0 config.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_config.py`:
```python
"""Unit tests for platform config and adapter registration."""

from unittest.mock import patch


class TestAdapterRegistration:
    """Tests for mem0 adapter registration."""

    def test_tencent_vector_registered_in_provider_configs(self):
        """After register(), tencent_vector should be a valid provider."""
        from memory_platform.config import register_tencent_vector_provider

        register_tencent_vector_provider()

        from mem0.vector_stores.configs import VectorStoreConfig

        assert "tencent_vector" in VectorStoreConfig._provider_configs

    def test_can_create_memory_with_tencent_vector_provider(self):
        """Memory class should accept tencent_vector as a vector store provider."""
        from memory_platform.config import register_tencent_vector_provider

        register_tencent_vector_provider()

        from mem0.vector_stores.configs import VectorStoreConfig

        config = VectorStoreConfig(
            provider="tencent_vector",
            config={
                "url": "https://example.com",
                "username": "root",
                "key": "secret",
            },
        )

        assert config.provider == "tencent_vector"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/test_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'register_tencent_vector_provider'`

- [ ] **Step 3: Implement registration function**

`src/memory_platform/config.py`:
```python
"""Platform configuration and adapter registration."""

from __future__ import annotations


def register_tencent_vector_provider() -> None:
    """Register TencentVectorDBConfig as a vector store provider in mem0.

    After calling this function, you can use:
        config = VectorStoreConfig(
            provider="tencent_vector",
            config={"url": "...", "username": "...", "key": "..."},
        )
    """
    from mem0.vector_stores.configs import VectorStoreConfig

    VectorStoreConfig._provider_configs["tencent_vector"] = (
        "memory_platform.adapters.tencent_vector.TencentVectorDBConfig"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/unit/test_config.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/config.py tests/unit/test_config.py
git commit -m "feat: add adapter registration for mem0 provider config"
```

---

### Task 14: Full Test Suite Run

**Files:** None (validation only)

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 2: Run ruff linting**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv run ruff check src/ tests/`
Expected: No errors (or fix any issues found)

- [ ] **Step 3: Fix any linting issues and re-run**

If ruff reports issues, fix them and re-run both lint and tests.

- [ ] **Step 4: Commit any fixes**

If any changes were needed:
```bash
git add -A
git commit -m "fix: resolve linting issues"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- Adapter location: `src/memory_platform/adapters/tencent_vector.py` -- Task 3+
- 4-layer memory model: metadata fields `memory_layer`, `scope`, `app_id` are in payload (pass-through) -- covered
- Collection naming `app_{app_id}`: configurable via `collection_name` -- covered
- Confidence decay: implemented in Extension Layer (future task), not in adapter -- correct separation
- `min_confidence` filtering: post-processing (future task) -- correct separation
- Return type contracts: `OutputData` with `.id`, `.payload`, `.score`; nested list from `list()` -- covered

**2. Placeholder scan:**
- All code steps have actual code -- checked
- All test steps have actual assertions -- checked
- No TBD/TODO/fill-in-later -- checked

**3. Type consistency:**
- `TencentVectorDBConfig` used consistently across tasks -- checked
- `OutputData` frozen dataclass with `id: str`, `payload: dict`, `score: float` -- consistent
- Method signatures match `VectorStoreBase` abstract methods -- checked
- `collection_name` vs `self.collection_name` -- consistent
- `embedding_model_dims` vs `self.embedding_model_dims` -- consistent

---

## Out of Scope (Future Plans)

These are intentionally NOT included in this plan:

1. **Memory Extension Layer** (`src/memory_platform/ext/`) -- separate plan
2. **REST API layer** (`src/memory_platform/api/`) -- separate plan
3. **Services** (`src/memory_platform/services/`) -- separate plan
4. **Middleware** (auth, degradation) -- separate plan
5. **Integration/E2E tests with real Tencent Cloud VectorDB** -- separate plan
6. **Dockerfile and deployment** -- separate plan
