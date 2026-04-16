"""配置模块 — 从 config.yaml 统一管理所有应用配置和 mem0 SDK 配置"""

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

    # Wal LLM 网关配置
    wal_base_url: str = ""
    aloha_app_name: str = ""
    access_token: str = ""

    # 向量存储配置
    vector_store_provider: str = "qdrant"
    vector_store_config: dict = {}

    # Embedder 配置
    embedder_provider: str = "openai"
    embedder_model: str = "text-embedding-3-small"
    embedder_api_key: str = ""
    embedder_base_url: str | None = None

    # MySQL 配置
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_database: str = "memory_platform"
    mysql_username: str = "root"
    mysql_password: str = ""
    mysql_pool_size: int = 5

    # 降级开关
    degradation_enabled: bool = True

    # 兼容旧环境变量（.env 文件仍可用，但优先级低于 config.yaml）
    model_config = {"extra": "ignore"}

    def __init__(self, _yaml_path: str = _DEFAULT_YAML_PATH, **kwargs):
        yaml_config = self._load_yaml(_yaml_path)
        # config.yaml 优先级高于环境变量，但低于代码显式传入的 kwargs
        for key, value in yaml_config.items():
            kwargs.setdefault(key, value)
        super().__init__(**kwargs)

    @staticmethod
    def _load_yaml(path: str) -> dict:
        """从 config.yaml 读取配置并映射到平铺的字段名。"""
        try:
            import yaml

            with open(path) as f:
                data = yaml.safe_load(f)
        except (FileNotFoundError, TypeError):
            return {}

        if not data:
            return {}

        result: dict = {}

        # API 认证
        if "api_keys" in data:
            result["api_keys"] = data["api_keys"]

        # LLM 配置
        llm = data.get("llm", {})
        if llm:
            result["llm_provider"] = llm.get("provider", "openai")
            result["llm_model"] = llm.get("model", "gpt-4o-mini")
            result["llm_api_key"] = llm.get("api_key", "")
            result["llm_base_url"] = llm.get("base_url")
            result["wal_base_url"] = llm.get("wal_base_url", "")
            result["aloha_app_name"] = llm.get("aloha_app_name", "")
            result["access_token"] = llm.get("access_token", "")

        # 向量存储配置
        vs = data.get("vector_store", {})
        if vs:
            result["vector_store_provider"] = vs.get("provider", "qdrant")
            result["vector_store_config"] = vs.get("config", {})

        # Embedder 配置
        emb = data.get("embedder", {})
        if emb:
            result["embedder_provider"] = emb.get("provider", "openai")
            result["embedder_model"] = emb.get("model", "text-embedding-3-small")
            result["embedder_api_key"] = emb.get("api_key", "")
            result["embedder_base_url"] = emb.get("base_url")

        # MySQL 配置
        mysql = data.get("mysql", {})
        if mysql:
            result["mysql_host"] = mysql.get("host", "127.0.0.1")
            result["mysql_port"] = mysql.get("port", 3306)
            result["mysql_database"] = mysql.get("database", "memory_platform")
            result["mysql_username"] = mysql.get("username", "root")
            result["mysql_password"] = mysql.get("password", "")
            result["mysql_pool_size"] = mysql.get("pool_size", 5)

        # 降级开关
        if "degradation_enabled" in data:
            result["degradation_enabled"] = data["degradation_enabled"]

        return result

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
    """构建 mem0 SDK 所需的 MemoryConfig"""
    s = get_settings()

    if s.llm_provider == "wal":
        llm_config: dict = {
            "model": s.llm_model,
            "wal_base_url": s.wal_base_url,
            "aloha_app_name": s.aloha_app_name,
            "access_token": s.access_token,
        }
    else:
        llm_config = {"model": s.llm_model, "api_key": s.llm_api_key}
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
