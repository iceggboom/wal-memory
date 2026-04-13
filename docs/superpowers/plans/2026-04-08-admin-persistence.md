# Admin 应用管理持久化 + mem0 SQLite→MySQL 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Admin 应用管理持久化到 MySQL，同时将 mem0 内置的 SQLite 历史存储替换为 MySQL。

**Architecture:** 新增 `db/` 子包管理 MySQL 连接池和应用注册表。`MySQLManager` 替换 mem0 的 `SQLiteManager`（接口一致，直接修改 fork 的 mem0 源码）。`AppRegistry` 提供 apps 表 CRUD。配置统一到根目录 `config.yaml`。认证中间件改为从数据库查询 API Key。

**Tech Stack:** Python 3.12+, pymysql, pyyaml, FastAPI, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `config.yaml` | MySQL 连接配置 |
| `pyproject.toml` | 新增 pymysql + pyyaml 依赖 |
| `src/memory_platform/db/__init__.py` | DB 包初始化 |
| `src/memory_platform/db/connection.py` | MySQL 连接池管理 |
| `src/memory_platform/db/mysql_manager.py` | MySQLManager — 替换 SQLiteManager |
| `src/memory_platform/db/app_registry.py` | AppRegistry — apps 表 CRUD |
| `src/memory_platform/config.py` | 加载 YAML + MySQL 配置 |
| `src/memory_platform/main.py` | 初始化 DB + 注入依赖 |
| `src/memory_platform/services/admin.py` | 接入 AppRegistry |
| `src/memory_platform/api/admin.py` | 去掉 TODO，接入 AppRegistry |
| `src/memory_platform/middleware/auth.py` | 从数据库验证 API Key |
| `src/mem0/memory/main.py` | 替换 SQLiteManager 为可注入的 storage |
| `src/mem0/memory/storage.py` | 新增 StorageManager 基类 |
| `tests/unit/db/test_connection.py` | 连接池测试 |
| `tests/unit/db/test_app_registry.py` | AppRegistry 测试 |
| `tests/unit/db/test_mysql_manager.py` | MySQLManager 测试 |
| `tests/unit/test_config.py` | 更新配置测试 |
| `tests/integration/test_admin_api.py` | 更新 Admin API 测试 |

---

## Key Reference: mem0 SQLiteManager 接口

```python
# src/mem0/memory/storage.py — 需要匹配的接口
class SQLiteManager:
    def __init__(self, db_path: str = ":memory:"): ...
    def add_history(self, memory_id, old_memory, new_memory, event, *,
                    created_at=None, updated_at=None, is_deleted=0,
                    actor_id=None, role=None) -> None: ...
    def get_history(self, memory_id: str) -> list[dict]: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...
```

mem0 中使用位置：
- `src/mem0/memory/main.py` 第 259 行: `self.db = SQLiteManager(self.config.history_db_path)`
- 同文件 `reset()` 方法中重建

## Key Reference: 现有依赖注入模式

```python
# main.py — 闭包工厂模式
def create_app(mem0=None):
    if mem0 is None:
        config = build_mem0_config()
        mem0 = Memory(config=config)
    app.include_router(create_admin_router(mem0))

# api/admin.py
def create_router(mem0: Memory) -> APIRouter:
    admin_svc = AdminService(mem0=mem0)
    # ...
```

## Key Reference: 现有 config.py Settings 结构

```python
class Settings(BaseSettings):
    api_keys: dict[str, str] = {}
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    # ... 其余字段
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
```

## Key Reference: 测试 mock_env fixture

```python
@pytest.fixture
def mock_env(monkeypatch):
    from memory_platform.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("API_KEYS", '{"test-app":"test-key-123"}')
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    # ...
```

---

### Task 1: 添加依赖 — pymysql + pyyaml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: 添加 pymysql 和 pyyaml 到依赖**

在 `pyproject.toml` 的 `dependencies` 列表中添加 `pymysql>=1.1` 和 `pyyaml>=6.0`：

```toml
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "qdrant-client>=1.9.1",
    "openai>=1.90.0",
    "posthog>=3.5.0",
    "pytz>=2024.1",
    "sqlalchemy>=2.0.31",
    "protobuf>=5.29.6,<7.0.0",
    "httpx>=0.28",
    "pydantic>=2.10",
    "pydantic-settings>=2.7",
    "anthropic>=0.89.0",
    "tcvectordb>=1.3.0",
    "pymysql>=1.1",
    "pyyaml>=6.0",
]
```

- [ ] **Step 2: 安装依赖**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv sync`
Expected: 新增 pymysql 和 pyyaml

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add pymysql and pyyaml dependencies"
```

---

### Task 2: config.yaml 配置文件 + Settings 加载 YAML

**Files:**
- Create: `config.yaml`
- Modify: `src/memory_platform/config.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: 创建 config.yaml**

```yaml
# AI Memory Platform 配置

mysql:
  host: "127.0.0.1"
  port: 3306
  database: "memory_platform"
  username: "root"
  password: ""
  pool_size: 5
```

- [ ] **Step 2: 写 Settings 加载 YAML 的测试**

在 `tests/unit/test_config.py` 末尾追加：

```python
class TestYAMLConfig:
    """Tests for YAML config loading."""

    def test_loads_mysql_config_from_yaml(self, tmp_path, monkeypatch):
        """Settings should load MySQL config from config.yaml."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "mysql:\n"
            "  host: 'db.example.com'\n"
            "  port: 3307\n"
            "  database: 'test_db'\n"
            "  username: 'admin'\n"
            "  password: 'secret'\n"
            "  pool_size: 10\n"
        )
        from memory_platform.config import Settings

        s = Settings(_yaml_path=str(yaml_file))
        assert s.mysql_host == "db.example.com"
        assert s.mysql_port == 3307
        assert s.mysql_database == "test_db"
        assert s.mysql_username == "admin"
        assert s.mysql_password == "secret"
        assert s.mysql_pool_size == 10

    def test_mysql_defaults_when_no_yaml(self, monkeypatch):
        """Settings should use defaults when no YAML config exists."""
        from memory_platform.config import Settings

        s = Settings(_yaml_path="/nonexistent/config.yaml")
        assert s.mysql_host == "127.0.0.1"
        assert s.mysql_port == 3306
        assert s.mysql_pool_size == 5
```

- [ ] **Step 3: 运行测试确认失败**

Run: `.venv/bin/python3 -m pytest tests/unit/test_config.py::TestYAMLConfig -v`
Expected: FAIL — Settings 没有 mysql 字段

- [ ] **Step 4: 修改 Settings 类加载 YAML**

修改 `src/memory_platform/config.py`，在 `Settings` 类中添加 MySQL 字段和 YAML 加载逻辑：

```python
"""配置模块 — 统一管理应用配置和 mem0 SDK 配置"""

from functools import lru_cache
from pathlib import Path

from mem0.configs.base import EmbedderConfig, LlmConfig, MemoryConfig, VectorStoreConfig
from mem0.utils.factory import EmbedderFactory
from pydantic import Field
from pydantic_settings import BaseSettings

EmbedderFactory.provider_to_class["mock"] = "memory_platform.embeddings.mock.MockEmbedder"

_DEFAULT_YAML_PATH = str(Path(__file__).resolve().parent.parent.parent / "config.yaml")


class Settings(BaseSettings):
    # API 认证
    api_keys: dict[str, str] = {}

    # LLM 配置 (mem0 用于记忆提取/去重)
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str | None = None

    # 向量存储配置
    vector_store_provider: str = "qdrant"
    vector_store_config: dict = {}

    # Embedder 配置
    embedder_provider: str = "openai"
    embedder_model: str = "text-embedding-3-small"
    embedder_api_key: str = ""
    embedder_base_url: str | None = None

    # 腾讯云向量 DB（适配器用）
    tcvdb_db_url: str = ""
    tcvdb_db_key: str = ""
    tcvdb_db_name: str = ""
    tcvdb_embedding_model: str = ""

    # MySQL 配置
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_database: str = "memory_platform"
    mysql_username: str = "root"
    mysql_password: str = ""
    mysql_pool_size: int = 5

    # 降级开关
    degradation_enabled: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def __init__(self, _yaml_path: str = _DEFAULT_YAML_PATH, **kwargs):
        yaml_config = self._load_yaml(_yaml_path)
        # YAML 值作为默认，env 值可覆盖
        for key, value in yaml_config.items():
            kwargs.setdefault(key, value)
        super().__init__(**kwargs)

    @staticmethod
    def _load_yaml(path: str) -> dict:
        try:
            import yaml

            with open(path) as f:
                data = yaml.safe_load(f)
        except (FileNotFoundError, TypeError):
            return {}
        if not data or "mysql" not in data:
            return {}
        mysql = data["mysql"]
        return {
            "mysql_host": mysql.get("host", "127.0.0.1"),
            "mysql_port": mysql.get("port", 3306),
            "mysql_database": mysql.get("database", "memory_platform"),
            "mysql_username": mysql.get("username", "root"),
            "mysql_password": mysql.get("password", ""),
            "mysql_pool_size": mysql.get("pool_size", 5),
        }

    def validate_api_key(self, key: str) -> bool:
        return key in self.api_keys.values()

    @property
    def mysql_enabled(self) -> bool:
        """MySQL 是否已配置（host 和 database 非空）。"""
        return bool(self.mysql_host and self.mysql_database)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def build_mem0_config() -> MemoryConfig:
    s = get_settings()

    llm_config: dict = {"model": s.llm_model, "api_key": s.llm_api_key}
    if s.llm_provider == "anthropic" and s.llm_base_url:
        llm_config["anthropic_base_url"] = s.llm_base_url
    elif s.llm_base_url:
        llm_config["openai_base_url"] = s.llm_base_url

    embedder_config: dict = {"model": s.embedder_model, "api_key": s.embedder_api_key}
    if s.embedder_base_url:
        embedder_config["openai_base_url"] = s.embedder_base_url

    embedder_cfg = EmbedderConfig.model_construct(
        provider=s.embedder_provider,
        config=embedder_config,
    )

    return MemoryConfig(
        llm=LlmConfig(provider=s.llm_provider, config=llm_config),
        vector_store=VectorStoreConfig(
            provider=s.vector_store_provider,
            config=s.vector_store_config,
        ),
        embedder=embedder_cfg,
    )
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv/bin/python3 -m pytest tests/unit/test_config.py -v`
Expected: 全部通过（旧测试也通过，因为 `_load_yaml` 对不存在的文件返回空 dict）

- [ ] **Step 6: Commit**

```bash
git add config.yaml src/memory_platform/config.py tests/unit/test_config.py
git commit -m "feat: add config.yaml with MySQL settings and YAML loading in Settings"
```

---

### Task 3: MySQL 连接池

**Files:**
- Create: `src/memory_platform/db/__init__.py`
- Create: `src/memory_platform/db/connection.py`
- Create: `tests/unit/db/__init__.py`
- Create: `tests/unit/db/test_connection.py`

- [ ] **Step 1: 创建目录结构**

Run:
```bash
mkdir -p /Users/I0W02SJ/gitCode/wal-memory/src/memory_platform/db
mkdir -p /Users/I0W02SJ/gitCode/wal-memory/tests/unit/db
```

- [ ] **Step 2: 创建 `__init__.py` 文件**

`src/memory_platform/db/__init__.py`:
```python
"""Database connection and management."""
```

`tests/unit/db/__init__.py`:
```python
```

- [ ] **Step 3: 写连接池测试**

`tests/unit/db/test_connection.py`:
```python
"""Tests for MySQL connection pool."""

from unittest.mock import MagicMock, patch

from memory_platform.db.connection import MySQLConnectionPool


class TestMySQLConnectionPool:
    def test_init_creates_pool(self):
        """Pool should create connections on init."""
        with patch("memory_platform.db.connection.pymysql") as mock_pymysql:
            mock_conn = MagicMock()
            mock_pymysql.connect.return_value = mock_conn

            pool = MySQLConnectionPool(
                host="localhost",
                port=3306,
                database="test_db",
                username="root",
                password="secret",
                pool_size=3,
            )

            assert mock_pymysql.connect.call_count == 3

    def test_get_connection_returns_connection(self):
        """get_connection should return a connection from the pool."""
        with patch("memory_platform.db.connection.pymysql") as mock_pymysql:
            mock_conn = MagicMock()
            mock_pymysql.connect.return_value = mock_conn

            pool = MySQLConnectionPool(
                host="localhost", port=3306, database="test",
                username="root", password="", pool_size=2,
            )

            conn = pool.get_connection()
            assert conn is not None

    def test_return_connection_puts_back(self):
        """return_connection should put connection back to pool."""
        with patch("memory_platform.db.connection.pymysql") as mock_pymysql:
            mock_conn = MagicMock()
            mock_pymysql.connect.return_value = mock_conn

            pool = MySQLConnectionPool(
                host="localhost", port=3306, database="test",
                username="root", password="", pool_size=1,
            )

            conn = pool.get_connection()
            pool.return_connection(conn)

            # Should be able to get it again
            conn2 = pool.get_connection()
            assert conn2 is not None

    def test_close_all(self):
        """close_all should close all connections."""
        with patch("memory_platform.db.connection.pymysql") as mock_pymysql:
            mock_conn = MagicMock()
            mock_pymysql.connect.return_value = mock_conn

            pool = MySQLConnectionPool(
                host="localhost", port=3306, database="test",
                username="root", password="", pool_size=2,
            )

            pool.close_all()

            assert mock_conn.close.call_count == 2
```

- [ ] **Step 4: 运行测试确认失败**

Run: `.venv/bin/python3 -m pytest tests/unit/db/test_connection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memory_platform.db'`

- [ ] **Step 5: 实现连接池**

`src/memory_platform/db/connection.py`:
```python
"""MySQL connection pool management."""

import logging
import queue
from typing import Any

import pymysql
import pymysql.cursors

logger = logging.getLogger(__name__)


class MySQLConnectionPool:
    """Simple MySQL connection pool using queue.Queue."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 3306,
        database: str = "memory_platform",
        username: str = "root",
        password: str = "",
        pool_size: int = 5,
    ) -> None:
        self._pool: queue.Queue[pymysql.Connection] = queue.Queue(maxsize=pool_size)
        self._all_connections: list[pymysql.Connection] = []

        for _ in range(pool_size):
            conn = pymysql.connect(
                host=host,
                port=port,
                user=username,
                password=password,
                database=database,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=True,
            )
            self._pool.put(conn)
            self._all_connections.append(conn)

        logger.info(
            "MySQL connection pool created: %s:%d/%s pool_size=%d",
            host, port, database, pool_size,
        )

    def get_connection(self) -> pymysql.Connection:
        """Get a connection from the pool (blocking)."""
        return self._pool.get()

    def return_connection(self, conn: pymysql.Connection) -> None:
        """Return a connection back to the pool."""
        self._pool.put(conn)

    def close_all(self) -> None:
        """Close all connections in the pool."""
        for conn in self._all_connections:
            try:
                conn.close()
            except Exception:
                pass
        self._all_connections.clear()
        logger.info("MySQL connection pool closed")
```

- [ ] **Step 6: 运行测试确认通过**

Run: `.venv/bin/python3 -m pytest tests/unit/db/test_connection.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add src/memory_platform/db/ tests/unit/db/
git commit -m "feat: add MySQL connection pool with tests"
```

---

### Task 4: MySQLManager — 替换 mem0 SQLiteManager

**Files:**
- Create: `src/memory_platform/db/mysql_manager.py`
- Create: `tests/unit/db/test_mysql_manager.py`

- [ ] **Step 1: 写 MySQLManager 测试**

`tests/unit/db/test_mysql_manager.py`:
```python
"""Tests for MySQLManager — mem0 history storage backed by MySQL."""

from unittest.mock import MagicMock, patch

from memory_platform.db.mysql_manager import MySQLManager


class TestMySQLManager:
    def _make_manager(self):
        """Create a MySQLManager with a mocked connection pool."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.get_connection.return_value = mock_conn
        return MySQLManager(db=mock_pool), mock_conn

    def test_create_history_table_on_init(self):
        """MySQLManager should create history table on init."""
        manager, mock_conn = self._make_manager()
        cursor = mock_conn.cursor.return_value

        # Should have called CREATE TABLE
        create_calls = [
            c for c in cursor.execute.call_args_list
            if "CREATE TABLE" in str(c)
        ]
        assert len(create_calls) >= 1

    def test_add_history_inserts_record(self):
        """add_history should INSERT a record."""
        manager, mock_conn = self._make_manager()
        cursor = mock_conn.cursor.return_value

        manager.add_history(
            memory_id="mem-1",
            old_memory=None,
            new_memory="test memory",
            event="ADD",
            created_at="2026-04-08T00:00:00",
            updated_at="2026-04-08T00:00:00",
        )

        insert_calls = [
            c for c in cursor.execute.call_args_list
            if "INSERT" in str(c)
        ]
        assert len(insert_calls) >= 1

    def test_get_history_returns_list(self):
        """get_history should return list of dicts."""
        manager, mock_conn = self._make_manager()
        cursor = mock_conn.cursor.return_value
        cursor.fetchall.return_value = [
            {
                "id": "h1",
                "memory_id": "mem-1",
                "old_memory": None,
                "new_memory": "test",
                "event": "ADD",
                "created_at": "2026-04-08T00:00:00",
                "updated_at": "2026-04-08T00:00:00",
                "is_deleted": 0,
                "actor_id": None,
                "role": None,
            }
        ]

        result = manager.get_history(memory_id="mem-1")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["memory_id"] == "mem-1"
        assert result[0]["event"] == "ADD"

    def test_reset_drops_and_recreates(self):
        """reset should DROP and re-create the history table."""
        manager, mock_conn = self._make_manager()
        cursor = mock_conn.cursor.return_value

        manager.reset()

        drop_calls = [
            c for c in cursor.execute.call_args_list
            if "DROP TABLE" in str(c)
        ]
        assert len(drop_calls) >= 1

    def test_returns_connection_to_pool(self):
        """Each method should return connection to pool after use."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.get_connection.return_value = mock_conn

        manager = MySQLManager(db=mock_pool)
        mock_conn.cursor.return_value.fetchall.return_value = []

        manager.get_history(memory_id="mem-1")

        mock_pool.return_connection.assert_called_with(mock_conn)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python3 -m pytest tests/unit/db/test_mysql_manager.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memory_platform.db.mysql_manager'`

- [ ] **Step 3: 实现 MySQLManager**

`src/memory_platform/db/mysql_manager.py`:
```python
"""MySQL-backed storage manager — replaces mem0's SQLiteManager."""

import logging
import uuid
from typing import Any

from memory_platform.db.connection import MySQLConnectionPool

logger = logging.getLogger(__name__)


class MySQLManager:
    """Drop-in replacement for mem0's SQLiteManager using MySQL.

    Interface matches SQLiteManager exactly:
    - add_history(memory_id, old_memory, new_memory, event, **kwargs)
    - get_history(memory_id) -> list[dict]
    - reset()
    - close()
    """

    def __init__(self, db: MySQLConnectionPool) -> None:
        self.db = db
        self._create_history_table()

    def _create_history_table(self) -> None:
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS history (
                        id           VARCHAR(36) PRIMARY KEY,
                        memory_id    VARCHAR(36),
                        old_memory   TEXT,
                        new_memory   TEXT,
                        event        VARCHAR(16),
                        created_at   DATETIME,
                        updated_at   DATETIME,
                        is_deleted   TINYINT DEFAULT 0,
                        actor_id     VARCHAR(255),
                        role         VARCHAR(64)
                    )
                    """
                )
        finally:
            self.db.return_connection(conn)

    def add_history(
        self,
        memory_id: str,
        old_memory: str | None,
        new_memory: str | None,
        event: str,
        *,
        created_at: str | None = None,
        updated_at: str | None = None,
        is_deleted: int = 0,
        actor_id: str | None = None,
        role: str | None = None,
    ) -> None:
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO history (
                        id, memory_id, old_memory, new_memory, event,
                        created_at, updated_at, is_deleted, actor_id, role
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid.uuid4()),
                        memory_id,
                        old_memory,
                        new_memory,
                        event,
                        created_at,
                        updated_at,
                        is_deleted,
                        actor_id,
                        role,
                    ),
                )
        except Exception as e:
            logger.error("Failed to add history record: %s", e)
            raise
        finally:
            self.db.return_connection(conn)

    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, memory_id, old_memory, new_memory, event,
                           created_at, updated_at, is_deleted, actor_id, role
                    FROM history
                    WHERE memory_id = %s
                    ORDER BY created_at ASC, updated_at ASC
                    """,
                    (memory_id,),
                )
                rows = cursor.fetchall()
        finally:
            self.db.return_connection(conn)

        return [
            {
                "id": r["id"],
                "memory_id": r["memory_id"],
                "old_memory": r["old_memory"],
                "new_memory": r["new_memory"],
                "event": r["event"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "is_deleted": bool(r["is_deleted"]),
                "actor_id": r["actor_id"],
                "role": r["role"],
            }
            for r in rows
        ]

    def reset(self) -> None:
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("DROP TABLE IF EXISTS history")
            self._create_history_table()
        except Exception as e:
            logger.error("Failed to reset history table: %s", e)
            raise
        finally:
            self.db.return_connection(conn)

    def close(self) -> None:
        """No-op — connection pool manages lifecycle."""
        pass
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python3 -m pytest tests/unit/db/test_mysql_manager.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/db/mysql_manager.py tests/unit/db/test_mysql_manager.py
git commit -m "feat: add MySQLManager replacing mem0 SQLiteManager"
```

---

### Task 5: 修改 mem0 支持注入 storage manager

**Files:**
- Modify: `src/mem0/memory/storage.py`
- Modify: `src/mem0/memory/main.py`

mem0 的 `Memory.__init__` 硬编码 `self.db = SQLiteManager(...)`。我们需要在 fork 的代码中添加注入点。

- [ ] **Step 1: 修改 mem0 的 Memory 类支持 storage 注入**

修改 `src/mem0/memory/main.py`，在 `Memory.__init__` 中（约第 259 行）：

将：
```python
        self.db = SQLiteManager(self.config.history_db_path)
```

改为：
```python
        # Support external storage manager injection (e.g., MySQLManager)
        if hasattr(self.config, "_storage_manager") and self.config._storage_manager is not None:
            self.db = self.config._storage_manager
        else:
            self.db = SQLiteManager(self.config.history_db_path)
```

对 `AsyncMemory.__init__`（约第 1422 行附近）做同样的修改。

对 `Memory.reset()`（约第 1349-1378 行）中的 `self.db = SQLiteManager(...)` 也做同样修改。

对 `AsyncMemory.reset()` 中的 `self.db = SQLiteManager(...)` 也做同样修改。

- [ ] **Step 2: 验证导入不受影响**

Run: `.venv/bin/python3 -c "from mem0 import Memory; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/mem0/memory/main.py
git commit -m "feat: add storage manager injection point in mem0 Memory class"
```

---

### Task 6: AppRegistry — 应用注册表

**Files:**
- Create: `src/memory_platform/db/app_registry.py`
- Create: `tests/unit/db/test_app_registry.py`

- [ ] **Step 1: 写 AppRegistry 测试**

`tests/unit/db/test_app_registry.py`:
```python
"""Tests for AppRegistry — apps table CRUD."""

from datetime import datetime
from unittest.mock import MagicMock

from memory_platform.db.app_registry import AppRegistry


class TestAppRegistry:
    def _make_registry(self):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.get_connection.return_value = mock_conn
        return AppRegistry(db=mock_pool), mock_conn

    def test_register_inserts_app(self):
        """register should INSERT into apps table."""
        registry, mock_conn = self._make_registry()
        cursor = mock_conn.cursor.return_value

        registry.register(
            app_id="hr-assistant",
            name="HR 助手",
            api_key="key-hr-001",
        )

        insert_calls = [
            c for c in cursor.execute.call_args_list
            if "INSERT" in str(c)
        ]
        assert len(insert_calls) >= 1

    def test_list_apps_returns_list(self):
        """list_apps should return list of app dicts."""
        registry, mock_conn = self._make_registry()
        cursor = mock_conn.cursor.return_value
        cursor.fetchall.return_value = [
            {
                "app_id": "hr-assistant",
                "name": "HR 助手",
                "api_key": "key-hr-001",
                "status": "active",
                "created_at": datetime(2026, 4, 8),
                "updated_at": datetime(2026, 4, 8),
            }
        ]

        result = registry.list_apps()

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["app_id"] == "hr-assistant"

    def test_get_by_api_key_returns_app(self):
        """get_by_api_key should return app dict or None."""
        registry, mock_conn = self._make_registry()
        cursor = mock_conn.cursor.return_value
        cursor.fetchone.return_value = {
            "app_id": "hr-assistant",
            "name": "HR 助手",
            "api_key": "key-hr-001",
            "status": "active",
            "created_at": datetime(2026, 4, 8),
            "updated_at": datetime(2026, 4, 8),
        }

        result = registry.get_by_api_key("key-hr-001")

        assert result is not None
        assert result["app_id"] == "hr-assistant"

    def test_get_by_api_key_returns_none_for_unknown(self):
        """get_by_api_key should return None for unknown key."""
        registry, mock_conn = self._make_registry()
        cursor = mock_conn.cursor.return_value
        cursor.fetchone.return_value = None

        result = registry.get_by_api_key("unknown-key")

        assert result is None

    def test_count_returns_number(self):
        """count should return number of active apps."""
        registry, mock_conn = self._make_registry()
        cursor = mock_conn.cursor.return_value
        cursor.fetchone.return_value = {"cnt": 5}

        result = registry.count()

        assert result == 5

    def test_returns_connection_to_pool(self):
        """Each method should return connection to pool after use."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.get_connection.return_value = mock_conn
        registry = AppRegistry(db=mock_pool)
        cursor = mock_conn.cursor.return_value
        cursor.fetchall.return_value = []

        registry.list_apps()

        mock_pool.return_connection.assert_called_with(mock_conn)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python3 -m pytest tests/unit/db/test_app_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memory_platform.db.app_registry'`

- [ ] **Step 3: 实现 AppRegistry**

`src/memory_platform/db/app_registry.py`:
```python
"""AppRegistry — apps table CRUD for application management."""

import logging
from datetime import datetime, timezone
from typing import Any

from memory_platform.db.connection import MySQLConnectionPool

logger = logging.getLogger(__name__)


class AppRegistry:
    """Manages the apps table in MySQL."""

    def __init__(self, db: MySQLConnectionPool) -> None:
        self.db = db
        self._create_table()

    def _create_table(self) -> None:
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS apps (
                        app_id     VARCHAR(64) PRIMARY KEY,
                        name       VARCHAR(255) NOT NULL,
                        api_key    VARCHAR(255) NOT NULL UNIQUE,
                        status     VARCHAR(16) NOT NULL DEFAULT 'active',
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    )
                    """
                )
        finally:
            self.db.return_connection(conn)

    def register(self, app_id: str, name: str, api_key: str) -> dict[str, Any]:
        """Register a new application."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO apps (app_id, name, api_key, status, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (app_id, name, api_key, "active", now, now),
                )
        finally:
            self.db.return_connection(conn)

        return {
            "app_id": app_id,
            "name": name,
            "api_key": api_key,
            "status": "active",
            "created_at": now,
            "updated_at": now,
        }

    def get(self, app_id: str) -> dict[str, Any] | None:
        """Get an app by app_id."""
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT app_id, name, api_key, status, created_at, updated_at FROM apps WHERE app_id = %s",
                    (app_id,),
                )
                return cursor.fetchone()
        finally:
            self.db.return_connection(conn)

    def get_by_api_key(self, api_key: str) -> dict[str, Any] | None:
        """Get an app by its API key."""
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT app_id, name, api_key, status, created_at, updated_at FROM apps WHERE api_key = %s AND status = 'active'",
                    (api_key,),
                )
                return cursor.fetchone()
        finally:
            self.db.return_connection(conn)

    def list_apps(self) -> list[dict[str, Any]]:
        """List all registered applications."""
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT app_id, name, api_key, status, created_at, updated_at FROM apps ORDER BY created_at"
                )
                return cursor.fetchall()
        finally:
            self.db.return_connection(conn)

    def update_status(self, app_id: str, status: str) -> None:
        """Update app status (active/inactive)."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE apps SET status = %s, updated_at = %s WHERE app_id = %s",
                    (status, now, app_id),
                )
        finally:
            self.db.return_connection(conn)

    def count(self) -> int:
        """Count active apps."""
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as cnt FROM apps WHERE status = 'active'")
                row = cursor.fetchone()
                return row["cnt"] if row else 0
        finally:
            self.db.return_connection(conn)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python3 -m pytest tests/unit/db/test_app_registry.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/db/app_registry.py tests/unit/db/test_app_registry.py
git commit -m "feat: add AppRegistry for apps table CRUD"
```

---

### Task 7: 接入 AdminService + Admin API

**Files:**
- Modify: `src/memory_platform/services/admin.py`
- Modify: `src/memory_platform/api/admin.py`

- [ ] **Step 1: 更新 AdminService 接入 AppRegistry**

修改 `src/memory_platform/services/admin.py`：

```python
"""管理服务 — 用户记忆的管理操作"""

from __future__ import annotations

from typing import Any

from mem0 import Memory

from memory_platform.db.app_registry import AppRegistry


class AdminService:
    def __init__(self, mem0: Memory, app_registry: AppRegistry | None = None):
        self.mem0 = mem0
        self.app_registry = app_registry

    def register_app(self, app_id: str, name: str, api_key: str) -> dict[str, Any]:
        """Register a new application."""
        if self.app_registry is None:
            raise RuntimeError("App registry not configured (MySQL not enabled)")
        return self.app_registry.register(app_id=app_id, name=name, api_key=api_key)

    def list_apps(self) -> list[dict[str, Any]]:
        """List all registered applications."""
        if self.app_registry is None:
            return []
        return self.app_registry.list_apps()

    def stats(self) -> dict[str, Any]:
        """Get platform statistics."""
        if self.app_registry is None:
            return {"total_memories": 0, "total_users": 0, "total_apps": 0}
        return {
            "total_apps": self.app_registry.count(),
            "total_memories": 0,
            "total_users": 0,
        }

    def get_user_memories(self, user_id: str, agent_id: str, limit: int = 100) -> list[dict]:
        raw = self.mem0.get_all(user_id=user_id, agent_id=agent_id, limit=limit)
        return raw.get("results", [])

    def delete_user_memories(self, user_id: str, agent_id: str) -> None:
        self.mem0.delete_all(user_id=user_id, agent_id=agent_id)
```

- [ ] **Step 2: 更新 Admin API 路由**

修改 `src/memory_platform/api/admin.py` — 更新 `create_router` 签名和 TODO 端点：

将 `create_router(mem0: Memory)` 改为 `create_router(mem0: Memory, app_registry: AppRegistry | None = None)`。

更新三个 TODO 端点：

`GET /apps`:
```python
    @router.get("/apps")
    def list_apps(request: Request):
        _auth(request)
        apps = admin_svc.list_apps()
        return {"apps": apps}
```

`POST /apps`:
```python
    @router.post("/apps")
    def register_app(req: RegisterAppRequest, request: Request):
        _auth(request)
        import secrets
        api_key = f"mpk-{secrets.token_hex(16)}"
        try:
            result = admin_svc.register_app(app_id=req.app_id, name=req.name, api_key=api_key)
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
```

注意：`RegisterAppRequest` 模型不变（只有 `app_id` 和 `name`）。API Key 由系统自动生成。

`GET /stats`:
```python
    @router.get("/stats")
    def get_stats(request: Request):
        _auth(request)
        return admin_svc.stats()
```

- [ ] **Step 3: 运行现有测试确认不回归**

Run: `.venv/bin/python3 -m pytest tests/unit/ -v --tb=short`
Expected: 全部通过（Admin API 测试使用 mock，app_registry=None 走回退路径）

- [ ] **Step 4: Commit**

```bash
git add src/memory_platform/services/admin.py src/memory_platform/api/admin.py
git commit -m "feat: wire AppRegistry into AdminService and Admin API"
```

---

### Task 8: 认证中间件接入 AppRegistry

**Files:**
- Modify: `src/memory_platform/middleware/auth.py`

- [ ] **Step 1: 修改 require_auth 支持数据库查询**

修改 `src/memory_platform/middleware/auth.py`：

```python
"""Auth middleware — API Key authentication."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from memory_platform.db.app_registry import AppRegistry


def get_api_key(request: Request) -> str | None:
    """Extract Bearer token from Authorization header."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


def require_auth(api_key: str | None, app_registry: AppRegistry | None = None) -> bool:
    """Validate API key. Uses AppRegistry if available, falls back to env config."""
    if not api_key:
        raise ValueError("Missing API key")

    # MySQL mode: check apps table
    if app_registry is not None:
        app = app_registry.get_by_api_key(api_key)
        if app is None:
            raise ValueError("Invalid API key")
        return True

    # Fallback: env-based validation
    from memory_platform.config import get_settings

    settings = get_settings()
    if not settings.validate_api_key(api_key):
        raise ValueError("Invalid API key")
    return True
```

- [ ] **Step 2: 更新 admin.py 的 _auth 调用**

在 `src/memory_platform/api/admin.py` 的 `_auth` 内部函数中传入 `app_registry`：

```python
    def _auth(request: Request) -> None:
        key = get_api_key(request)
        try:
            require_auth(key, app_registry=app_registry)
        except ValueError as e:
            raise HTTPException(status_code=401, detail=str(e))
```

同样需要对 `api/memories.py` 中的 `_auth` 做同样更新（如果它也调用 `require_auth`）。检查 `src/memory_platform/api/memories.py` 中的 `_auth` 函数，将 `require_auth(key)` 改为 `require_auth(key, app_registry=app_registry)`。如果 `create_router` 没有 `app_registry` 参数，需要添加。

- [ ] **Step 3: 运行测试确认不回归**

Run: `.venv/bin/python3 -m pytest tests/unit/ tests/integration/ -v --tb=short`
Expected: 全部通过

- [ ] **Step 4: Commit**

```bash
git add src/memory_platform/middleware/auth.py src/memory_platform/api/admin.py src/memory_platform/api/memories.py
git commit -m "feat: auth middleware supports AppRegistry for API key validation"
```

---

### Task 9: 应用入口接入 MySQL

**Files:**
- Modify: `src/memory_platform/main.py`

- [ ] **Step 1: 修改 create_app 初始化 MySQL 并注入**

修改 `src/memory_platform/main.py`：

```python
"""FastAPI 应用入口 — AI Memory Platform"""

import logging

from fastapi import FastAPI
from mem0 import Memory

from memory_platform.config import build_mem0_config, get_settings

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)


def create_app(mem0: Memory | None = None) -> FastAPI:
    app = FastAPI(title="AI Memory Platform", version="0.1.0", docs_url="/docs")
    settings = get_settings()

    # MySQL 初始化
    db_pool = None
    app_registry = None

    if settings.mysql_enabled:
        from memory_platform.db.connection import MySQLConnectionPool
        from memory_platform.db.app_registry import AppRegistry

        db_pool = MySQLConnectionPool(
            host=settings.mysql_host,
            port=settings.mysql_port,
            database=settings.mysql_database,
            username=settings.mysql_username,
            password=settings.mysql_password,
            pool_size=settings.mysql_pool_size,
        )
        app_registry = AppRegistry(db=db_pool)

    # mem0 初始化
    if mem0 is None:
        config = build_mem0_config()

        # 注入 MySQLManager 替换 SQLiteManager
        if db_pool is not None:
            from memory_platform.db.mysql_manager import MySQLManager
            config._storage_manager = MySQLManager(db=db_pool)

        mem0 = Memory(config=config)

    llm_client = None
    if settings.llm_provider == "anthropic" and settings.llm_api_key:
        from anthropic import Anthropic

        llm_client = Anthropic(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

    from memory_platform.api.memories import create_router as create_memories_router
    from memory_platform.api.admin import create_router as create_admin_router

    app.include_router(
        create_memories_router(
            mem0=mem0,
            llm_client=llm_client,
            cross_collection_searcher=None,
            app_registry=app_registry,
        )
    )
    app.include_router(create_admin_router(mem0, app_registry=app_registry))

    @app.get("/health")
    def health():
        return {"status": "ok", "mysql": "enabled" if settings.mysql_enabled else "disabled"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("memory_platform.main:app", host="0.0.0.0", port=8000)
```

注意：如果 `api/memories.py` 的 `create_router` 签名目前不接受 `app_registry`，需要添加该参数。

- [ ] **Step 2: 验证导入不报错**

Run: `.venv/bin/python3 -c "from memory_platform.config import get_settings; print(get_settings().mysql_enabled)"`
Expected: `False`（无 MySQL 配置时回退）

- [ ] **Step 3: Commit**

```bash
git add src/memory_platform/main.py
git commit -m "feat: initialize MySQL in create_app and inject into mem0 + admin"
```

---

### Task 10: 更新集成测试

**Files:**
- Modify: `tests/integration/test_admin_api.py`

- [ ] **Step 1: 更新测试 fixture 注入 mock AppRegistry**

修改 `tests/integration/test_admin_api.py`：

```python
"""Integration tests for Admin API."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_mem0():
    return MagicMock()


@pytest.fixture
def mock_app_registry():
    registry = MagicMock()
    registry.list_apps.return_value = []
    registry.count.return_value = 0
    registry.get_by_api_key.return_value = None
    return registry


@pytest.fixture
def app_client(mock_mem0, mock_app_registry, mock_env):
    from memory_platform.api.admin import create_router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(create_router(mock_mem0, app_registry=mock_app_registry))
    return TestClient(app)


class TestAdminApps:
    def test_list_apps_empty(self, app_client, mock_env):
        resp = app_client.get("/v1/admin/apps", headers={"authorization": "Bearer test-key-123"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["apps"] == []

    def test_register_app(self, app_client, mock_app_registry, mock_env):
        mock_app_registry.register.return_value = {
            "app_id": "new-app",
            "name": "New App",
            "api_key": "mpk-generated-key",
            "status": "active",
        }

        resp = app_client.post(
            "/v1/admin/apps",
            headers={"authorization": "Bearer test-key-123"},
            json={"app_id": "new-app", "name": "New App"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["app_id"] == "new-app"
        assert data["name"] == "New App"
        assert data["status"] == "active"
        assert "api_key" in data

    def test_register_app_returns_generated_key(self, app_client, mock_app_registry, mock_env):
        """POST /apps should return a system-generated API key."""
        mock_app_registry.register.return_value = {
            "app_id": "test-app",
            "name": "Test",
            "api_key": "mpk-abc123",
            "status": "active",
        }

        resp = app_client.post(
            "/v1/admin/apps",
            headers={"authorization": "Bearer test-key-123"},
            json={"app_id": "test-app", "name": "Test"},
        )
        data = resp.json()
        assert data["api_key"].startswith("mpk-")


class TestAdminUsers:
    def test_get_user_memories(self, app_client, mock_mem0, mock_env):
        mock_mem0.get_all.return_value = {"results": []}
        resp = app_client.get(
            "/v1/admin/users/u1/memories?app_id=app1",
            headers={"authorization": "Bearer test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []
        assert data["total"] == 0

    def test_delete_user_memories(self, app_client, mock_mem0, mock_env):
        mock_mem0.delete_all.return_value = {"message": "ok"}
        resp = app_client.delete(
            "/v1/admin/users/u1/memories?app_id=app1",
            headers={"authorization": "Bearer test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["message"] == "User memories deleted successfully"


class TestAdminStats:
    def test_get_stats(self, app_client, mock_app_registry, mock_env):
        mock_app_registry.count.return_value = 3

        resp = app_client.get("/v1/admin/stats", headers={"authorization": "Bearer test-key-123"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_apps"] == 3


class TestAdminAuth:
    def test_unauthorized_access(self, app_client, mock_env):
        resp = app_client.get("/v1/admin/apps")
        assert resp.status_code == 401

    def test_invalid_api_key(self, app_client, mock_env):
        resp = app_client.get("/v1/admin/apps", headers={"authorization": "Bearer wrong-key"})
        assert resp.status_code == 401
```

- [ ] **Step 2: 运行测试**

Run: `.venv/bin/python3 -m pytest tests/integration/test_admin_api.py -v --tb=short`
Expected: 全部通过

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_admin_api.py
git commit -m "test: update admin API tests with mock AppRegistry"
```

---

### Task 11: 全量测试 + Lint

**Files:** None (validation only)

- [ ] **Step 1: 运行全量测试**

Run: `.venv/bin/python3 -m pytest tests/unit/ tests/integration/ -v --tb=short`
Expected: 全部通过

- [ ] **Step 2: 运行 ruff lint**

Run: `.venv/bin/python3 -m ruff check src/memory_platform/ tests/`
Expected: 无错误（或修复后通过）

- [ ] **Step 3: 修复并提交**

如有 lint 问题，修复后：
```bash
git add -A
git commit -m "fix: resolve linting issues"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- config.yaml + Settings 加载 YAML — Task 2
- MySQL 连接池 — Task 3
- MySQLManager 替换 SQLiteManager — Task 4 + 5
- AppRegistry apps 表 CRUD — Task 6
- AdminService + Admin API 去掉 TODO — Task 7
- Auth 中间件接入 AppRegistry — Task 8
- main.py 初始化注入 — Task 9
- 集成测试更新 — Task 10
- 全量测试 + lint — Task 11

**2. Placeholder scan:**
- 所有代码步骤有实际代码
- 所有测试步骤有实际断言
- 无 TBD/TODO/fill-in-later

**3. Type consistency:**
- `AppRegistry` 在 Task 6 定义，Task 7/8/9/10 中使用一致
- `MySQLConnectionPool` 在 Task 3 定义，后续任务使用一致
- `create_router` 签名变更在 Task 7/9 中保持一致
- `require_auth` 签名变更在 Task 8 中一致
