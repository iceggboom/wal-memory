"""配置模块 — 统一管理应用配置和 mem0 SDK 配置"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from mem0.configs.base import EmbedderConfig, LlmConfig, MemoryConfig, VectorStoreConfig
from mem0.utils.factory import EmbedderFactory
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
