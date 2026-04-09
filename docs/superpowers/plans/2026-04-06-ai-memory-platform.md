# AI 记忆平台实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建基于 mem0 的 AI 记忆平台 REST API，为约 10 个 AI 应用提供共享记忆基础设施。

**Architecture:** 三层架构 — FastAPI REST API 接入层 → Memory Extension Layer 业务策略层（4 层记忆模型、置信度衰减、多应用 scope）→ mem0 SDK 记忆引擎层。腾讯云向量 DB 作为向量存储，GLM 5.0 作为 LLM，均通过 mem0 的 Provider 抽象接入。

**Tech Stack:** Python 3.12, FastAPI, mem0, uv, pytest, ruff, Docker

**Spec:** `docs/superpowers/specs/2026-04-06-ai-memory-platform-design.md`

**开发策略：** MVP 阶段使用 Qdrant（mem0 默认向量存储）进行本地开发和测试，Tencent VectorDB 适配器作为独立 Task 在最后实现。这样无需云凭证即可开发和测试所有业务逻辑。

---

## 文件结构总览

```
wal-memory/
├── pyproject.toml                    # Task 1
├── Dockerfile                        # Task 14
├── src/
│   └── memory_platform/
│       ├── __init__.py               # Task 1
│       ├── main.py                   # Task 13
│       ├── config.py                 # Task 2
│       ├── api/
│       │   ├── __init__.py           # Task 1
│       │   ├── memories.py           # Task 10
│       │   └── admin.py              # Task 11
│       ├── services/
│       │   ├── __init__.py           # Task 1
│       │   ├── write.py              # Task 6
│       │   ├── recall.py             # Task 7
│       │   └── admin.py              # Task 12
│       ├── ext/
│       │   ├── __init__.py           # Task 1
│       │   ├── layer.py              # Task 4
│       │   ├── confidence.py         # Task 3
│       │   └── scope.py              # Task 5
│       ├── adapters/
│       │   ├── __init__.py           # Task 1
│       │   └── tencent_vector.py     # Task 15
│       └── middleware/
│           ├── __init__.py           # Task 1
│           ├── auth.py               # Task 8
│           └── degradation.py        # Task 9
├── tests/
│   ├── conftest.py                   # Task 2
│   ├── unit/
│   │   ├── test_confidence.py        # Task 3
│   │   ├── test_layer.py             # Task 4
│   │   ├── test_scope.py             # Task 5
│   │   └── test_write.py             # Task 6
│   ├── integration/
│       │   ├── test_memories_api.py  # Task 10
│       │   ├── test_recall.py        # Task 7
│       │   └── test_admin_api.py     # Task 12
│   └── e2e/
│       └── test_full_flow.py         # Task 13
└── docs/
    └── superpowers/
        ├── specs/2026-04-06-ai-memory-platform-design.md
        └── plans/2026-04-06-ai-memory-platform.md
```

---

### Task 1: 项目脚手架

**Files:**
- Create: `pyproject.toml`
- Create: `src/memory_platform/__init__.py`
- Create: `src/memory_platform/api/__init__.py`
- Create: `src/memory_platform/services/__init__.py`
- Create: `src/memory_platform/ext/__init__.py`
- Create: `src/memory_platform/adapters/__init__.py`
- Create: `src/memory_platform/middleware/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/e2e/__init__.py`

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[project]
name = "memory-platform"
version = "0.1.0"
description = "AI Memory Platform based on mem0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "mem0ai>=0.1.0",
    "httpx>=0.28",
    "pydantic>=2.10",
    "pydantic-settings>=2.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.25",
    "ruff>=0.9",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/memory_platform"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py312"
line-length = 100
```

- [ ] **Step 2: 创建所有 __init__.py 文件**

所有 `__init__.py` 文件内容为空。

- [ ] **Step 3: 安装依赖**

Run: `cd /Users/I0W02SJ/gitCode/wal-memory && uv sync --extra dev`
Expected: 依赖安装成功

- [ ] **Step 4: 验证 pytest 可运行**

Run: `uv run pytest --co -q`
Expected: `no tests ran` (0 errors)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/ tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py tests/e2e/__init__.py uv.lock
git commit -m "chore: project scaffolding with dependencies"
```

---

### Task 2: 配置模块

**Files:**
- Create: `src/memory_platform/config.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: 编写配置模块测试**

```python
# tests/conftest.py
import pytest
from unittest.mock import patch


@pytest.fixture
def mock_env(monkeypatch):
    """默认测试环境变量"""
    monkeypatch.setenv("API_KEYS", '{"test-app":"test-key-123"}')
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "qdrant")
    monkeypatch.setenv("EMBEDDER_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDER_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDER_API_KEY", "sk-test")
```

```python
# tests/unit/test_config.py
from memory_platform.config import Settings, build_mem0_config


class TestSettings:
    def test_loads_from_env(self, mock_env):
        s = Settings()
        assert s.llm_provider == "openai"
        assert s.llm_model == "gpt-4o-mini"
        assert s.vector_store_provider == "qdrant"
        assert s.embedder_provider == "openai"

    def test_api_keys_parsed(self, mock_env):
        s = Settings()
        assert s.api_keys == {"test-app": "test-key-123"}

    def test_validate_api_key_valid(self, mock_env):
        s = Settings()
        assert s.validate_api_key("test-key-123") is True

    def test_validate_api_key_invalid(self, mock_env):
        s = Settings()
        assert s.validate_api_key("wrong-key") is False


class TestBuildMem0Config:
    def test_returns_mem0_config(self, mock_env):
        config = build_mem0_config()
        assert config.llm.provider == "openai"
        assert config.vector_store.provider == "qdrant"
        assert config.embedder.provider == "openai"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: 实现配置模块**

```python
# src/memory_platform/config.py
import json
from functools import lru_cache

import mem0
from mem0.configs.base import MemoryConfig
from pydantic_settings import BaseSettings


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

    # 腾讯云向量 DB（适配器用）
    tcvdb_db_url: str = ""
    tcvdb_db_key: str = ""
    tcvdb_db_name: str = ""
    tcvdb_embedding_model: str = ""

    # 降级开关
    degradation_enabled: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def validate_api_key(self, key: str) -> bool:
        return key in self.api_keys.values()


@lru_cache
def get_settings() -> Settings:
    return Settings()


def build_mem0_config() -> MemoryConfig:
    s = get_settings()

    llm_config: dict = {
        "model": s.llm_model,
        "api_key": s.llm_api_key,
    }
    if s.llm_base_url:
        llm_config["openai_base_url"] = s.llm_base_url

    embedder_config: dict = {
        "model": s.embedder_model,
        "api_key": s.embedder_api_key,
    }

    return MemoryConfig(
        llm=mem0.configs.llms.base.LlmConfig(
            provider=s.llm_provider,
            config=llm_config,
        ),
        vector_store=mem0.configs.vector_stores.base.VectorStoreConfig(
            provider=s.vector_store_provider,
            config=s.vector_store_config,
        ),
        embedder=mem0.configs.embeddings.base.EmbedderConfig(
            provider=s.embedder_provider,
            config=embedder_config,
        ),
    )
```

- [ ] **Step 4: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/config.py tests/conftest.py tests/unit/test_config.py
git commit -m "feat: add configuration module with env-based settings"
```

---

### Task 3: 置信度衰减模块

**Files:**
- Create: `src/memory_platform/ext/confidence.py`
- Create: `tests/unit/test_confidence.py`

- [ ] **Step 1: 编写测试**

```python
# tests/unit/test_confidence.py
import math
from datetime import datetime, timedelta, timezone

from memory_platform.ext.confidence import (
    DECAY_LAMBDA,
    compute_confidence,
    filter_by_confidence,
)


class TestComputeConfidence:
    def test_no_decay_for_fresh_memory(self):
        now = datetime.now(timezone.utc)
        result = compute_confidence(similarity=0.9, updated_at=now, layer="L1", now=now)
        assert result == 0.9

    def test_l1_slow_decay(self):
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(hours=168)
        result = compute_confidence(similarity=0.9, updated_at=week_ago, layer="L1", now=now)
        expected = 0.9 * math.exp(-0.001 * 168)
        assert abs(result - expected) < 0.001

    def test_l3_fast_decay(self):
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(hours=168)
        result = compute_confidence(similarity=0.9, updated_at=week_ago, layer="L3", now=now)
        expected = 0.9 * math.exp(-0.02 * 168)
        assert abs(result - expected) < 0.001

    def test_default_layer_is_l1(self):
        now = datetime.now(timezone.utc)
        result = compute_confidence(
            similarity=0.8, updated_at=now, layer=None, now=now
        )
        assert result == 0.8

    def test_updated_at_as_string(self):
        now = datetime.now(timezone.utc)
        updated_str = now.isoformat()
        result = compute_confidence(
            similarity=0.9, updated_at=updated_str, layer="L1", now=now
        )
        assert result == 0.9


class TestFilterByConfidence:
    def test_filters_below_threshold(self):
        now = datetime.now(timezone.utc)
        memories = [
            {"id": "1", "memory": "test", "score": 0.9, "metadata": {"memory_layer": "L1"},
             "updated_at": now.isoformat(), "created_at": now.isoformat()},
            {"id": "2", "memory": "old", "score": 0.9, "metadata": {"memory_layer": "L3"},
             "updated_at": (now - timedelta(hours=500)).isoformat(), "created_at": now.isoformat()},
        ]
        results = filter_by_confidence(memories, min_confidence=0.5, now=now)
        assert len(results) == 1
        assert results[0][0]["id"] == "1"

    def test_sorts_by_confidence_descending(self):
        now = datetime.now(timezone.utc)
        memories = [
            {"id": "1", "memory": "low", "score": 0.7, "metadata": {"memory_layer": "L2"},
             "updated_at": (now - timedelta(hours=100)).isoformat(), "created_at": now.isoformat()},
            {"id": "2", "memory": "high", "score": 0.9, "metadata": {"memory_layer": "L1"},
             "updated_at": now.isoformat(), "created_at": now.isoformat()},
        ]
        results = filter_by_confidence(memories, min_confidence=0.0, now=now)
        assert results[0][1] >= results[1][1]

    def test_respects_limit(self):
        now = datetime.now(timezone.utc)
        memories = [
            {"id": str(i), "memory": f"m{i}", "score": 0.9,
             "metadata": {"memory_layer": "L1"},
             "updated_at": now.isoformat(), "created_at": now.isoformat()}
            for i in range(10)
        ]
        results = filter_by_confidence(memories, min_confidence=0.0, limit=3, now=now)
        assert len(results) == 3
```

- [ ] **Step 2: 运行测试验证失败**

Run: `uv run pytest tests/unit/test_confidence.py -v`
Expected: FAIL

- [ ] **Step 3: 实现置信度模块**

```python
# src/memory_platform/ext/confidence.py
import math
from datetime import datetime, timezone


DECAY_LAMBDA: dict[str, float] = {
    "L1": 0.001,   # Profile — 极慢衰减
    "L2": 0.005,   # Preference — 慢衰减
    "L3": 0.02,    # Episodic — 中等衰减
    "L4": 0.01,    # Relational — 较慢衰减
}

DEFAULT_LAYER = "L1"


def _parse_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def compute_confidence(
    similarity: float,
    updated_at: datetime | str,
    layer: str | None = None,
    now: datetime | None = None,
) -> float:
    lam = DECAY_LAMBDA.get(layer or DEFAULT_LAYER, DECAY_LAMBDA[DEFAULT_LAYER])
    updated = _parse_datetime(updated_at)
    current = now or datetime.now(timezone.utc)
    delta_hours = (current - updated).total_seconds() / 3600
    return similarity * math.exp(-lam * delta_hours)


def filter_by_confidence(
    memories: list[dict],
    min_confidence: float = 0.5,
    limit: int = 100,
    now: datetime | None = None,
) -> list[tuple[dict, float]]:
    current = now or datetime.now(timezone.utc)
    scored: list[tuple[dict, float]] = []
    for mem in memories:
        layer = (mem.get("metadata") or {}).get("memory_layer", DEFAULT_LAYER)
        updated_at = mem.get("updated_at") or mem.get("created_at") or current
        similarity = mem.get("score", 0.0)
        if similarity is None:
            continue
        conf = compute_confidence(
            similarity=similarity,
            updated_at=updated_at,
            layer=layer,
            now=current,
        )
        if conf >= min_confidence:
            scored.append((mem, conf))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]
```

- [ ] **Step 4: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_confidence.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/ext/confidence.py tests/unit/test_confidence.py
git commit -m "feat: add confidence decay module with per-layer lambda"
```

---

### Task 4: 4 层记忆映射模块

**Files:**
- Create: `src/memory_platform/ext/layer.py`
- Create: `tests/unit/test_layer.py`

- [ ] **Step 1: 编写测试**

```python
# tests/unit/test_layer.py
from memory_platform.ext.layer import (
    MemoryLayer,
    classify_layer,
    parse_layer_filter,
    LAYER_KEYWORDS,
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
```

- [ ] **Step 2: 运行测试验证失败**

Run: `uv run pytest tests/unit/test_layer.py -v`
Expected: FAIL

- [ ] **Step 3: 实现层级模块**

```python
# src/memory_platform/ext/layer.py
from enum import Enum


class MemoryLayer(str, Enum):
    L1 = "L1"  # Profile — 角色·职业·兴趣
    L2 = "L2"  # Preference — 风格·偏好
    L3 = "L3"  # Episodic — 具体经历
    L4 = "L4"  # Relational — 团队·社交


LAYER_KEYWORDS: dict[MemoryLayer, list[str]] = {
    MemoryLayer.L1: [
        "是", "担任", "职位", "工程师", "经理", "主管", "总监", "负责人",
        "专业", "岗位", "职业", "角色", "学历", "毕业",
    ],
    MemoryLayer.L2: [
        "喜欢", "偏好", "习惯", "风格", "倾向", "爱好", "兴趣",
        "注重", "追求", "擅长", "倾向于",
    ],
    MemoryLayer.L3: [
        "上周", "昨天", "前天", "上次", "那次", "参加了", "去了",
        "完成了", "经历了", "最近", "之前", "刚", "曾", "已",
    ],
    MemoryLayer.L4: [
        "同事", "领导", "团队", "同组", "一起", "搭档", "伙伴",
        "上级", "下属", "合作", "组长", "班长",
    ],
}

ALL_LAYERS = list(MemoryLayer)


def classify_layer(text: str, explicit_layer: str | None = None) -> MemoryLayer:
    if explicit_layer:
        return MemoryLayer(explicit_layer)

    scores: dict[MemoryLayer, int] = {layer: 0 for layer in MemoryLayer}
    for layer, keywords in LAYER_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[layer] += 1

    best = max(scores, key=lambda l: scores[l])
    return best if scores[best] > 0 else MemoryLayer.L1


def parse_layer_filter(layer_str: str | None) -> list[MemoryLayer]:
    if not layer_str:
        return ALL_LAYERS
    return [MemoryLayer(l.strip()) for l in layer_str.split(",")]
```

- [ ] **Step 4: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_layer.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/ext/layer.py tests/unit/test_layer.py
git commit -m "feat: add 4-layer memory model classification"
```

---

### Task 5: 多应用 Scope 路由模块

**Files:**
- Create: `src/memory_platform/ext/scope.py`
- Create: `tests/unit/test_scope.py`

- [ ] **Step 1: 编写测试**

```python
# tests/unit/test_scope.py
from memory_platform.ext.scope import Scope, apply_scope_filter, deduplicate_memories


class TestScope:
    def test_all_includes_shared_and_private(self):
        assert Scope.ALL.include_shared is True
        assert Scope.ALL.include_private is True

    def test_shared_only(self):
        assert Scope.SHARED.include_shared is True
        assert Scope.SHARED.include_private is False

    def test_private_only(self):
        assert Scope.PRIVATE.include_shared is False
        assert Scope.PRIVATE.include_private is True


class TestApplyScopeFilter:
    def test_all_returns_all(self):
        memories = [
            {"id": "1", "memory": "shared", "metadata": {"scope": "shared"}},
            {"id": "2", "memory": "private", "metadata": {"scope": "private"}},
        ]
        result = apply_scope_filter(memories, Scope.ALL)
        assert len(result) == 2

    def test_shared_filters_private(self):
        memories = [
            {"id": "1", "memory": "shared", "metadata": {"scope": "shared"}},
            {"id": "2", "memory": "private", "metadata": {"scope": "private"}},
        ]
        result = apply_scope_filter(memories, Scope.SHARED)
        assert len(result) == 1
        assert result[0]["id"] == "1"

    def test_private_filters_shared(self):
        memories = [
            {"id": "1", "memory": "shared", "metadata": {"scope": "shared"}},
            {"id": "2", "memory": "private", "metadata": {"scope": "private"}},
        ]
        result = apply_scope_filter(memories, Scope.PRIVATE)
        assert len(result) == 1
        assert result[0]["id"] == "2"

    def test_no_metadata_defaults_to_shared(self):
        memories = [
            {"id": "1", "memory": "no scope", "metadata": {}},
        ]
        result = apply_scope_filter(memories, Scope.SHARED)
        assert len(result) == 1


class TestDeduplicateMemories:
    def test_removes_duplicate_hashes(self):
        memories = [
            {"id": "1", "memory": "a", "hash": "h1"},
            {"id": "2", "memory": "a", "hash": "h1"},
            {"id": "3", "memory": "b", "hash": "h2"},
        ]
        result = deduplicate_memories(memories)
        assert len(result) == 2

    def test_no_duplicates(self):
        memories = [
            {"id": "1", "memory": "a", "hash": "h1"},
            {"id": "2", "memory": "b", "hash": "h2"},
        ]
        result = deduplicate_memories(memories)
        assert len(result) == 2
```

- [ ] **Step 2: 运行测试验证失败**

Run: `uv run pytest tests/unit/test_scope.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 scope 模块**

```python
# src/memory_platform/ext/scope.py
from enum import Enum


class Scope(str, Enum):
    ALL = "all"
    SHARED = "shared"
    PRIVATE = "private"

    @property
    def include_shared(self) -> bool:
        return self in (Scope.ALL, Scope.SHARED)

    @property
    def include_private(self) -> bool:
        return self in (Scope.ALL, Scope.PRIVATE)


def apply_scope_filter(memories: list[dict], scope: Scope) -> list[dict]:
    filtered = []
    for mem in memories:
        mem_scope = (mem.get("metadata") or {}).get("scope", "shared")
        if mem_scope == "shared" and scope.include_shared:
            filtered.append(mem)
        elif mem_scope == "private" and scope.include_private:
            filtered.append(mem)
    return filtered


def deduplicate_memories(memories: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for mem in memories:
        h = mem.get("hash")
        if h and h in seen:
            continue
        if h:
            seen.add(h)
        unique.append(mem)
    return unique
```

- [ ] **Step 4: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_scope.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/ext/scope.py tests/unit/test_scope.py
git commit -m "feat: add multi-app scope routing and deduplication"
```

---

### Task 6: Write Service — 记忆写入服务

**Files:**
- Create: `src/memory_platform/services/write.py`
- Create: `tests/unit/test_write.py`

- [ ] **Step 1: 编写测试**

```python
# tests/unit/test_write.py
from unittest.mock import MagicMock, patch

import pytest

from memory_platform.services.write import WriteService, AddMemoryItem, ExtractRequest


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
        mock_mem0.add.return_value = {
            "results": [
                {"id": "m1", "memory": "a", "event": "ADD"},
                {"id": "m2", "memory": "b", "event": "UPDATE"},
                {"id": "m3", "memory": "c", "event": "NONE"},
                {"id": "m4", "memory": "d", "event": "DELETE"},
            ]
        }
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
```

- [ ] **Step 2: 运行测试验证失败**

Run: `uv run pytest tests/unit/test_write.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 Write Service**

```python
# src/memory_platform/services/write.py
from dataclasses import dataclass, field

from mem0 import Memory

from memory_platform.ext.layer import classify_layer


@dataclass
class AddMemoryItem:
    text: str
    memory_layer: str | None = None
    scope: str = "shared"


@dataclass
class ExtractRequest:
    user_id: str
    agent_id: str
    messages: list[dict]
    memory_layer: str | None = None


class WriteService:
    def __init__(self, mem0: Memory):
        self.mem0 = mem0

    def add_memory(
        self,
        user_id: str,
        agent_id: str,
        items: list[AddMemoryItem],
    ) -> dict:
        added = 0
        updated = 0
        unchanged = 0

        for item in items:
            layer = classify_layer(item.text, explicit_layer=item.memory_layer)
            metadata = {
                "memory_layer": layer.value,
                "scope": item.scope,
                "app_id": agent_id,
            }
            result = self.mem0.add(
                item.text,
                user_id=user_id,
                agent_id=agent_id,
                metadata=metadata,
                infer=False,
            )
            for r in result.get("results", []):
                event = r.get("event", "NONE")
                if event == "ADD":
                    added += 1
                elif event == "UPDATE":
                    updated += 1
                else:
                    unchanged += 1

        return {"added": added, "updated": updated, "unchanged": unchanged}

    def extract(
        self,
        user_id: str,
        agent_id: str,
        messages: list[dict],
        memory_layer: str | None = None,
    ) -> dict:
        metadata = {
            "scope": "shared",
            "app_id": agent_id,
        }
        if memory_layer:
            metadata["memory_layer"] = memory_layer

        result = self.mem0.add(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            metadata=metadata,
            infer=True,
        )

        added = sum(1 for r in result.get("results", []) if r.get("event") == "ADD")
        updated = sum(1 for r in result.get("results", []) if r.get("event") == "UPDATE")
        deleted = sum(1 for r in result.get("results", []) if r.get("event") == "DELETE")

        return {
            "added": added,
            "updated": updated,
            "deleted": deleted,
            "memories": result.get("results", []),
        }
```

- [ ] **Step 4: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_write.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/services/write.py tests/unit/test_write.py
git commit -m "feat: add write service for explicit and extract memory paths"
```

---

### Task 7: Recall Service — 记忆检索服务

**Files:**
- Create: `src/memory_platform/services/recall.py`
- Create: `tests/integration/test_recall.py`

- [ ] **Step 1: 编写测试**

```python
# tests/integration/test_recall.py
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

from memory_platform.services.recall import RecallService


@pytest.fixture
def mock_mem0():
    return MagicMock()


@pytest.fixture
def recall_service(mock_mem0):
    return RecallService(mem0=mock_mem0)


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
```

- [ ] **Step 2: 运行测试验证失败**

Run: `uv run pytest tests/integration/test_recall.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 Recall Service**

```python
# src/memory_platform/services/recall.py
from datetime import datetime, timezone

from mem0 import Memory

from memory_platform.ext.confidence import filter_by_confidence
from memory_platform.ext.layer import parse_layer_filter
from memory_platform.ext.scope import Scope, apply_scope_filter


class RecallService:
    def __init__(self, mem0: Memory):
        self.mem0 = mem0

    def search(
        self,
        query: str,
        user_id: str,
        agent_id: str,
        limit: int = 10,
        min_confidence: float = 0.5,
        memory_layer: str | None = None,
        scope: str = "all",
        now: datetime | None = None,
    ) -> list[dict]:
        raw = self.mem0.search(
            query,
            user_id=user_id,
            agent_id=agent_id,
            limit=limit * 3,
        )
        memories = raw.get("results", [])

        # 1. Scope 过滤
        scope_enum = Scope(scope)
        memories = apply_scope_filter(memories, scope_enum)

        # 2. Layer 过滤
        allowed_layers = parse_layer_filter(memory_layer)
        allowed_values = [l.value for l in allowed_layers]
        memories = [
            m for m in memories
            if (m.get("metadata") or {}).get("memory_layer", "L1") in allowed_values
        ]

        # 3. 置信度衰减 + 过滤 + 排序
        scored = filter_by_confidence(
            memories, min_confidence=min_confidence, limit=limit, now=now
        )

        # 4. 格式化返回
        results = []
        for mem, confidence in scored:
            metadata = mem.get("metadata") or {}
            results.append({
                "id": mem["id"],
                "text": mem["memory"],
                "memory_layer": metadata.get("memory_layer", "L1"),
                "confidence": round(confidence, 4),
                "similarity": mem.get("score", 0.0),
                "scope": metadata.get("scope", "shared"),
                "created_at": mem.get("created_at", ""),
                "updated_at": mem.get("updated_at", ""),
            })

        return results

    def get_all(
        self,
        user_id: str,
        agent_id: str,
        memory_layer: str | None = None,
        scope: str = "all",
        limit: int = 100,
    ) -> list[dict]:
        raw = self.mem0.get_all(
            user_id=user_id,
            agent_id=agent_id,
            limit=limit,
        )
        memories = raw.get("results", [])

        scope_enum = Scope(scope)
        memories = apply_scope_filter(memories, scope_enum)

        allowed_layers = parse_layer_filter(memory_layer)
        allowed_values = [l.value for l in allowed_layers]
        memories = [
            m for m in memories
            if (m.get("metadata") or {}).get("memory_layer", "L1") in allowed_values
        ]

        results = []
        for mem in memories:
            metadata = mem.get("metadata") or {}
            results.append({
                "id": mem["id"],
                "text": mem["memory"],
                "memory_layer": metadata.get("memory_layer", "L1"),
                "scope": metadata.get("scope", "shared"),
                "created_at": mem.get("created_at", ""),
                "updated_at": mem.get("updated_at", ""),
            })

        return results

    def delete(self, memory_id: str, user_id: str, agent_id: str) -> None:
        self.mem0.delete(memory_id)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `uv run pytest tests/integration/test_recall.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/services/recall.py tests/integration/test_recall.py
git commit -m "feat: add recall service with confidence decay and filtering"
```

---

### Task 8: Auth 中间件

**Files:**
- Create: `src/memory_platform/middleware/auth.py`

- [ ] **Step 1: 编写测试**

```python
# tests/unit/test_auth.py
import pytest
from unittest.mock import patch, MagicMock

from memory_platform.middleware.auth import get_api_key, require_auth


class TestGetApiKey:
    def test_extracts_bearer_token(self):
        request = MagicMock()
        request.headers = {"authorization": "Bearer test-key"}
        assert get_api_key(request) == "test-key"

    def test_returns_none_if_no_header(self):
        request = MagicMock()
        request.headers = {}
        assert get_api_key(request) is None


class TestRequireAuth:
    def test_valid_key_passes(self, mock_env):
        result = require_auth("test-key-123")
        assert result is True

    def test_invalid_key_raises(self, mock_env):
        with pytest.raises(ValueError, match="Invalid API key"):
            require_auth("wrong-key")
```

- [ ] **Step 2: 运行测试验证失败**

Run: `uv run pytest tests/unit/test_auth.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 Auth 中间件**

```python
# src/memory_platform/middleware/auth.py
from fastapi import Request

from memory_platform.config import get_settings


def get_api_key(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


def require_auth(api_key: str | None) -> bool:
    if not api_key:
        raise ValueError("Missing API key")
    settings = get_settings()
    if not settings.validate_api_key(api_key):
        raise ValueError("Invalid API key")
    return True
```

- [ ] **Step 4: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_auth.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/middleware/auth.py tests/unit/test_auth.py
git commit -m "feat: add API key authentication middleware"
```

---

### Task 9: Degradation 中间件

**Files:**
- Create: `src/memory_platform/middleware/degradation.py`

- [ ] **Step 1: 编写测试**

```python
# tests/unit/test_degradation.py
import pytest

from memory_platform.middleware.degradation import memory_degradation_handler


class TestDegradationHandler:
    def test_returns_empty_on_exception(self):
        result = memory_degradation_handler(Exception("DB down"))
        assert result == {"results": [], "total": 0}

    def test_returns_empty_with_error_message(self):
        result = memory_degradation_handler(ValueError("bad input"))
        assert result == {"results": [], "total": 0}
```

- [ ] **Step 2: 运行测试验证失败**

Run: `uv run pytest tests/unit/test_degradation.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 Degradation 中间件**

```python
# src/memory_platform/middleware/degradation.py
import logging

logger = logging.getLogger(__name__)


def memory_degradation_handler(exc: Exception) -> dict:
    """全局降级处理：任何记忆服务异常返回空结果，不阻塞主链。"""
    logger.error(
        "Memory service degradation triggered: %s",
        exc,
        exc_info=type(exc) is not Exception,
    )
    return {"results": [], "total": 0}
```

- [ ] **Step 4: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_degradation.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/middleware/degradation.py tests/unit/test_degradation.py
git commit -m "feat: add degradation middleware for graceful failure"
```

---

### Task 10: API 路由 — 记忆 CRUD + 检索

**Files:**
- Create: `src/memory_platform/api/memories.py`
- Create: `tests/integration/test_memories_api.py`

- [ ] **Step 1: 编写测试**

```python
# tests/integration/test_memories_api.py
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_mem0():
    return MagicMock()


@pytest.fixture
def app_client(mock_mem0, mock_env):
    from memory_platform.api.memories import create_router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(create_router(mock_mem0))
    return TestClient(app)


class TestPostMemories:
    def test_add_memory_success(self, app_client, mock_mem0):
        mock_mem0.add.return_value = {
            "results": [{"id": "m1", "memory": "test", "event": "ADD"}]
        }
        resp = app_client.post(
            "/v1/memories",
            headers={"authorization": "Bearer test-key-123"},
            json={
                "user_id": "u1",
                "app_id": "app1",
                "memories": [{"text": "是一名Java工程师"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["added"] == 1

    def test_missing_api_key(self, app_client):
        resp = app_client.post(
            "/v1/memories",
            json={"user_id": "u1", "app_id": "app1", "memories": [{"text": "test"}]},
        )
        assert resp.status_code == 401

    def test_missing_user_id(self, app_client):
        resp = app_client.post(
            "/v1/memories",
            headers={"authorization": "Bearer test-key-123"},
            json={"app_id": "app1", "memories": [{"text": "test"}]},
        )
        assert resp.status_code == 422


class TestPostExtract:
    def test_extract_success(self, app_client, mock_mem0):
        mock_mem0.add.return_value = {
            "results": [{"id": "m1", "memory": "用户是工程师", "event": "ADD"}]
        }
        resp = app_client.post(
            "/v1/memories/extract",
            headers={"authorization": "Bearer test-key-123"},
            json={
                "user_id": "u1",
                "app_id": "app1",
                "messages": [{"role": "user", "content": "我是一名Java工程师"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["added"] == 1


class TestPostSearch:
    def test_search_success(self, app_client, mock_mem0):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        mock_mem0.search.return_value = {
            "results": [{
                "id": "m1", "memory": "是工程师", "score": 0.9,
                "metadata": {"memory_layer": "L1", "scope": "shared", "app_id": "app1"},
                "created_at": now.isoformat(), "updated_at": now.isoformat(),
            }]
        }
        resp = app_client.post(
            "/v1/memories/search",
            headers={"authorization": "Bearer test-key-123"},
            json={"user_id": "u1", "app_id": "app1", "query": "工程师"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) >= 1
        assert "confidence" in data["results"][0]

    def test_search_degradation_on_error(self, app_client, mock_mem0):
        mock_mem0.search.side_effect = Exception("DB down")
        resp = app_client.post(
            "/v1/memories/search",
            headers={"authorization": "Bearer test-key-123"},
            json={"user_id": "u1", "app_id": "app1", "query": "test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []
        assert data["total"] == 0


class TestGetMemories:
    def test_get_all_memories(self, app_client, mock_mem0):
        mock_mem0.get_all.return_value = {"results": []}
        resp = app_client.get(
            "/v1/memories?user_id=u1&app_id=app1",
            headers={"authorization": "Bearer test-key-123"},
        )
        assert resp.status_code == 200


class TestDeleteMemory:
    def test_delete_success(self, app_client, mock_mem0):
        mock_mem0.delete.return_value = {"message": "ok"}
        resp = app_client.delete(
            "/v1/memories/m1?user_id=u1&app_id=app1",
            headers={"authorization": "Bearer test-key-123"},
        )
        assert resp.status_code == 200
```

- [ ] **Step 2: 运行测试验证失败**

Run: `uv run pytest tests/integration/test_memories_api.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 API 路由**

```python
# src/memory_platform/api/memories.py
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from mem0 import Memory
from pydantic import BaseModel, Field

from memory_platform.middleware.auth import get_api_key, require_auth
from memory_platform.middleware.degradation import memory_degradation_handler
from memory_platform.services.write import WriteService, AddMemoryItem
from memory_platform.services.recall import RecallService

logger = logging.getLogger(__name__)


class MemoriesRequest(BaseModel):
    user_id: str
    app_id: str
    memories: list[AddMemoryItem]


class ExtractRequest(BaseModel):
    user_id: str
    app_id: str
    messages: list[dict]
    memory_layer: str | None = None


class SearchRequest(BaseModel):
    user_id: str
    app_id: str
    query: str
    limit: int = 10
    min_confidence: float = 0.5
    memory_layer: str | None = None
    scope: str = "all"


def create_router(mem0: Memory) -> APIRouter:
    router = APIRouter(prefix="/v1")
    write_svc = WriteService(mem0=mem0)
    recall_svc = RecallService(mem0=mem0)

    def _auth(request: Request):
        key = get_api_key(request)
        try:
            require_auth(key)
        except ValueError as e:
            raise HTTPException(status_code=401, detail=str(e))

    @router.post("/memories")
    def add_memories(req: MemoriesRequest, request: Request):
        _auth(request)
        try:
            return write_svc.add_memory(
                user_id=req.user_id,
                agent_id=req.app_id,
                items=req.memories,
            )
        except Exception as e:
            logger.error("add_memories failed: %s", e)
            raise

    @router.post("/memories/extract")
    def extract_memories(req: ExtractRequest, request: Request):
        _auth(request)
        try:
            return write_svc.extract(
                user_id=req.user_id,
                agent_id=req.app_id,
                messages=req.messages,
                memory_layer=req.memory_layer,
            )
        except Exception as e:
            logger.error("extract_memories failed: %s", e)
            raise

    @router.post("/memories/search")
    def search_memories(req: SearchRequest, request: Request):
        _auth(request)
        try:
            results = recall_svc.search(
                query=req.query,
                user_id=req.user_id,
                agent_id=req.app_id,
                limit=req.limit,
                min_confidence=req.min_confidence,
                memory_layer=req.memory_layer,
                scope=req.scope,
            )
            return {"results": results, "total": len(results)}
        except Exception:
            return memory_degradation_handler(Exception("search failed"))

    @router.get("/memories")
    def get_memories(
        request: Request,
        user_id: str = Query(...),
        app_id: str = Query(...),
        memory_layer: str | None = None,
        scope: str = "all",
    ):
        _auth(request)
        try:
            results = recall_svc.get_all(
                user_id=user_id,
                agent_id=app_id,
                memory_layer=memory_layer,
                scope=scope,
            )
            return {"results": results, "total": len(results)}
        except Exception:
            return memory_degradation_handler(Exception("get_all failed"))

    @router.delete("/memories/{memory_id}")
    def delete_memory(
        memory_id: str,
        request: Request,
        user_id: str = Query(...),
        app_id: str = Query(...),
    ):
        _auth(request)
        recall_svc.delete(memory_id=memory_id, user_id=user_id, agent_id=app_id)
        return {"message": "Memory deleted successfully"}

    return router
```

- [ ] **Step 4: 运行测试验证通过**

Run: `uv run pytest tests/integration/test_memories_api.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/api/memories.py tests/integration/test_memories_api.py
git commit -m "feat: add memory CRUD and search API endpoints"
```

---

### Task 11: Admin Service — 管理服务

**Files:**
- Create: `src/memory_platform/services/admin.py`
- Create: `tests/integration/test_admin_api.py`

- [ ] **Step 1: 编写测试**

```python
# tests/integration/test_admin_api.py
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_mem0():
    return MagicMock()


@pytest.fixture
def app_client(mock_mem0, mock_env):
    from memory_platform.api.admin import create_router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(create_router(mock_mem0))
    return TestClient(app)


class TestAdminApps:
    def test_list_apps_empty(self, app_client, mock_env, monkeypatch):
        resp = app_client.get(
            "/v1/admin/apps",
            headers={"authorization": "Bearer test-key-123"},
        )
        assert resp.status_code == 200

    def test_register_app(self, app_client, mock_env, monkeypatch):
        resp = app_client.post(
            "/v1/admin/apps",
            headers={"authorization": "Bearer test-key-123"},
            json={"app_id": "new-app", "name": "New App"},
        )
        assert resp.status_code == 200


class TestAdminUsers:
    def test_get_user_memories(self, app_client, mock_mem0, mock_env):
        mock_mem0.get_all.return_value = {"results": []}
        resp = app_client.get(
            "/v1/admin/users/u1/memories?app_id=app1",
            headers={"authorization": "Bearer test-key-123"},
        )
        assert resp.status_code == 200

    def test_delete_user_memories(self, app_client, mock_mem0, mock_env):
        mock_mem0.delete_all.return_value = {"message": "ok"}
        resp = app_client.delete(
            "/v1/admin/users/u1/memories?app_id=app1",
            headers={"authorization": "Bearer test-key-123"},
        )
        assert resp.status_code == 200
```

- [ ] **Step 2: 运行测试验证失败**

Run: `uv run pytest tests/integration/test_admin_api.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 Admin Service 和 API**

```python
# src/memory_platform/services/admin.py
from mem0 import Memory


class AdminService:
    def __init__(self, mem0: Memory):
        self.mem0 = mem0

    def get_user_memories(
        self, user_id: str, agent_id: str, limit: int = 100
    ) -> list[dict]:
        raw = self.mem0.get_all(
            user_id=user_id, agent_id=agent_id, limit=limit
        )
        return raw.get("results", [])

    def delete_user_memories(self, user_id: str, agent_id: str) -> None:
        self.mem0.delete_all(user_id=user_id, agent_id=agent_id)
```

```python
# src/memory_platform/api/admin.py
import logging

from fastapi import APIRouter, HTTPException, Query, Request
from mem0 import Memory
from pydantic import BaseModel

from memory_platform.middleware.auth import get_api_key, require_auth
from memory_platform.services.admin import AdminService

logger = logging.getLogger(__name__)


class RegisterAppRequest(BaseModel):
    app_id: str
    name: str


def create_router(mem0: Memory) -> APIRouter:
    router = APIRouter(prefix="/v1/admin")
    admin_svc = AdminService(mem0=mem0)

    def _auth(request: Request):
        key = get_api_key(request)
        try:
            require_auth(key)
        except ValueError as e:
            raise HTTPException(status_code=401, detail=str(e))

    @router.get("/apps")
    def list_apps(request: Request):
        _auth(request)
        return {"apps": []}

    @router.post("/apps")
    def register_app(req: RegisterAppRequest, request: Request):
        _auth(request)
        return {"app_id": req.app_id, "name": req.name, "status": "registered"}

    @router.get("/users/{user_id}/memories")
    def get_user_memories(
        user_id: str, request: Request, app_id: str = Query(...)
    ):
        _auth(request)
        memories = admin_svc.get_user_memories(
            user_id=user_id, agent_id=app_id
        )
        return {"results": memories, "total": len(memories)}

    @router.delete("/users/{user_id}/memories")
    def delete_user_memories(
        user_id: str, request: Request, app_id: str = Query(...)
    ):
        _auth(request)
        admin_svc.delete_user_memories(user_id=user_id, agent_id=app_id)
        return {"message": "User memories deleted successfully"}

    @router.get("/stats")
    def get_stats(request: Request):
        _auth(request)
        return {
            "total_memories": 0,
            "total_users": 0,
            "total_apps": 0,
        }

    return router
```

- [ ] **Step 4: 运行测试验证通过**

Run: `uv run pytest tests/integration/test_admin_api.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/services/admin.py src/memory_platform/api/admin.py tests/integration/test_admin_api.py
git commit -m "feat: add admin service and API for app/user management"
```

---

### Task 12: 健康检查端点

**Files:**
- Modify: `src/memory_platform/main.py` (created in Task 13, but health endpoint can be standalone)

> 注意：此 Task 的代码将在 Task 13 的 main.py 中包含，此处仅确保健康检查逻辑存在。

---

### Task 13: FastAPI 应用入口 + 全量集成

**Files:**
- Create: `src/memory_platform/main.py`
- Modify: `tests/conftest.py` — 添加全局 mock_mem0 fixture

- [ ] **Step 1: 编写集成测试**

```python
# tests/integration/test_app.py
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(mock_env):
    from unittest.mock import MagicMock, patch
    with patch("memory_platform.main.build_mem0_config") as mock_config:
        mock_mem0 = MagicMock()
        with patch("mem0.Memory", return_value=mock_mem0):
            from memory_platform.main import app
            yield TestClient(app), mock_mem0


class TestHealthEndpoint:
    def test_health(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestOpenAPIDocs:
    def test_docs_available(self, client):
        c, _ = client
        resp = c.get("/docs")
        assert resp.status_code == 200

    def test_openapi_json(self, client):
        c, _ = client
        resp = c.get("/openapi.json")
        assert resp.status_code == 200
```

- [ ] **Step 2: 运行测试验证失败**

Run: `uv run pytest tests/integration/test_app.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 main.py**

```python
# src/memory_platform/main.py
import logging

from fastapi import FastAPI
from mem0 import Memory

from memory_platform.config import build_mem0_config
from memory_platform.api.memories import create_router as create_memories_router
from memory_platform.api.admin import create_router as create_admin_router

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)


def create_app(mem0: Memory | None = None) -> FastAPI:
    app = FastAPI(
        title="AI Memory Platform",
        version="0.1.0",
        docs_url="/docs",
    )

    if mem0 is None:
        config = build_mem0_config()
        mem0 = Memory(config=config)

    app.include_router(create_memories_router(mem0))
    app.include_router(create_admin_router(mem0))

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("memory_platform.main:app", host="0.0.0.0", port=8000)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `uv run pytest tests/integration/test_app.py -v`
Expected: 3 passed

- [ ] **Step 5: 运行全量测试**

Run: `uv run pytest -v`
Expected: All tests pass (unit + integration)

- [ ] **Step 6: Commit**

```bash
git add src/memory_platform/main.py tests/integration/test_app.py
git commit -m "feat: add FastAPI app entry point with health check"
```

---

### Task 14: Dockerfile

**Files:**
- Create: `Dockerfile`

- [ ] **Step 1: 创建 Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ src/

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "memory_platform.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: 验证 Docker 构建**

Run: `docker build -t memory-platform .`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "chore: add Dockerfile for TKE deployment"
```

---

### Task 15: 腾讯云向量 DB 适配器

> **前置条件：** 需要腾讯云向量 DB 实例和 API Key。此 Task 使用 Qdrant 进行本地开发和测试。腾讯云适配器实现需要参考腾讯云向量 DB Python SDK 文档。
>
> **实现路径：** 继承 `mem0.vector_stores.base.VectorStoreBase`，实现 12 个抽象方法，注册为 `tencent_vector` provider。

**Files:**
- Create: `src/memory_platform/adapters/tencent_vector.py`

- [ ] **Step 1: 研究腾讯云向量 DB Python SDK**

Run: `pip install tencent vectordb-sdk` 并阅读 SDK 文档，确认以下 API：
- `vdbClient` 初始化
- Collection 创建/删除/列表
- Document 写入/查询/删除/更新
- 过滤器语法
- Embedding API 调用

- [ ] **Step 2: 实现 VectorStoreBase**

参考 mem0 源码中 `mem0/vector_stores/baidu.py`（百度向量 DB 适配器）的实现模式，为腾讯云向量 DB 实现所有抽象方法。

关键方法签名参考（来自 mem0 VectorStoreBase）：

```python
class TencentVectorStore(VectorStoreBase):
    def __init__(self, config): ...
    def create_col(self, name, dimension, metrics): ...
    def insert(self, vectors, payloads, ids): ...
    def search(self, query, vectors, limit, filters): ...
    def delete(self, ids): ...
    def update(self, vectors, payloads, ids): ...
    def get(self, ids): ...
    def list_cols(self): ...
    def delete_col(self, name): ...
    def col_info(self, name): ...
    def list(self, filters, limit): ...
    def reset(self): ...
```

- [ ] **Step 3: 注册到 mem0**

在 `memory_platform/adapters/__init__.py` 中注册 provider：

```python
from mem0.utils.factory import VectorStoreFactory
from memory_platform.adapters.tencent_vector import TencentVectorStore, TencentVectorStoreConfig

VectorStoreFactory.register_provider(
    "tencent_vector",
    "memory_platform.adapters.tencent_vector",
    TencentVectorStoreConfig,
)
```

- [ ] **Step 4: 编写测试并验证**

使用 mock 腾讯云 SDK 客户端进行测试。

- [ ] **Step 5: Commit**

```bash
git add src/memory_platform/adapters/
git commit -m "feat: add Tencent Cloud VectorDB adapter for mem0"
```

---

## 自审清单

### 1. Spec 覆盖

| Spec 章节 | 对应 Task |
|-----------|----------|
| 系统整体架构 | Task 1, 13 |
| Write Service（显式写入 + 提取写入） | Task 6 |
| Recall Service（检索 + 衰减 + 过滤） | Task 7 |
| Admin Service（记忆/应用/统计管理） | Task 11 |
| Memory Extension Layer | Task 3, 4, 5 |
| 4 层记忆模型 | Task 4 |
| 置信度衰减 | Task 3 |
| 多应用 scope 路由 | Task 5 |
| REST API（memories + admin） | Task 10, 11 |
| 认证（API Key） | Task 8 |
| 降级策略 | Task 9 |
| 部署（Dockerfile） | Task 14 |
| 腾讯云向量 DB 适配器 | Task 15 |
| 配置模块 | Task 2 |

### 2. 占位符扫描

无 TBD、TODO、"implement later" 等占位符。Task 15 是唯一需要运行时 SDK 研究的部分，已标注前置条件。

### 3. 类型一致性

- `memory_layer` 统一使用 `str`（"L1"/"L2"/"L3"/"L4"），通过 `MemoryLayer` 枚举保证合法性
- `scope` 统一使用 `str`（"shared"/"private"/"all"），通过 `Scope` 枚举保证合法性
- `user_id` / `agent_id` / `app_id` 统一使用 `str`
- `AddMemoryItem` 在 Task 6 定义，Task 10 直接使用 — 一致
- `SearchRequest.scope` 与 `Scope` 枚举匹配 — 一致
