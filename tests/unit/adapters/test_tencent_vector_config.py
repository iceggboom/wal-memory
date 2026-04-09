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
            TencentVectorDBConfig(url="https://example.com", key="secret")

        errors = exc_info.value.errors()
        error_fields = {e["loc"][0] for e in errors}
        assert "username" in error_fields

    def test_key_required(self):
        """Config should fail without key."""
        with pytest.raises(ValidationError) as exc_info:
            TencentVectorDBConfig(url="https://example.com", username="root")

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
