"""Unit tests for TencentVectorDBConfig."""

import pytest
from pydantic import ValidationError

from memory_platform.adapters.tencent_vector import TencentVectorDBConfig


class TestTencentVectorDBConfig:
    def test_default_values(self):
        config = TencentVectorDBConfig(
            url="https://example.com",
            username="root",
            key="secret-key",
        )

        assert config.url == "https://example.com"
        assert config.username == "root"
        assert config.key == "secret-key"
        assert config.collection_name == "mem0"
        assert config.embedding_model_dims == 1536
        assert config.database_name == "memory_platform"
        assert config.timeout == 30
        assert config.mock is False
        assert config.embedding_model == "bge-base-zh"

    def test_custom_values(self):
        config = TencentVectorDBConfig(
            url="https://custom.url",
            username="admin",
            key="my-key",
            collection_name="my_collection",
            embedding_model_dims=768,
            database_name="my_db",
            timeout=60,
            mock=True,
            embedding_model="bge-large-zh",
        )

        assert config.collection_name == "my_collection"
        assert config.embedding_model_dims == 768
        assert config.database_name == "my_db"
        assert config.timeout == 60
        assert config.mock is True
        assert config.embedding_model == "bge-large-zh"

    def test_timeout_must_be_positive(self):
        with pytest.raises(ValidationError):
            TencentVectorDBConfig(
                url="https://example.com",
                username="root",
                key="secret",
                timeout=0,
            )

    def test_mock_flag(self):
        config = TencentVectorDBConfig(mock=True)
        assert config.mock is True
