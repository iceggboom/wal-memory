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


@pytest.fixture
def mock_env(monkeypatch):
    """默认测试环境变量"""
    # 清除 get_settings 的缓存
    from memory_platform.config import get_settings

    get_settings.cache_clear()

    monkeypatch.setenv("API_KEYS", '{"test-app":"test-key-123"}')
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "qdrant")
    monkeypatch.setenv("EMBEDDER_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDER_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDER_API_KEY", "sk-test")
