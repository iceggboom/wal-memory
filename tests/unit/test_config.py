from memory_platform.config import Settings, build_mem0_config, get_settings


class TestSettings:
    def test_loads_from_yaml(self, tmp_path, monkeypatch):
        """从 YAML 文件加载完整配置。"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "api_keys:\n"
            "  test-app: 'test-key-123'\n"
            "llm:\n"
            "  provider: openai\n"
            "  model: gpt-4o-mini\n"
            "  api_key: sk-test\n"
            "vector_store:\n"
            "  provider: qdrant\n"
            "  config: {}\n"
            "embedder:\n"
            "  provider: openai\n"
            "  model: text-embedding-3-small\n"
            "  api_key: sk-test\n"
            "mysql:\n"
            "  host: '127.0.0.1'\n"
            "  port: 3306\n"
            "  database: 'memory_platform'\n"
            "  username: 'root'\n"
            "  password: ''\n"
            "  pool_size: 5\n"
        )
        get_settings.cache_clear()
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        s = Settings(_yaml_path=str(yaml_file))
        assert s.llm_provider == "openai"
        assert s.llm_model == "gpt-4o-mini"
        assert s.vector_store_provider == "qdrant"
        assert s.embedder_provider == "openai"

    def test_api_keys_parsed(self, tmp_path, monkeypatch):
        """API Keys 从 YAML 正确解析。"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "api_keys:\n"
            "  test-app: 'test-key-123'\n"
            "llm:\n"
            "  provider: openai\n"
            "  model: gpt-4o-mini\n"
            "  api_key: sk-test\n"
        )
        get_settings.cache_clear()
        s = Settings(_yaml_path=str(yaml_file))
        assert s.api_keys == {"test-app": "test-key-123"}

    def test_validate_api_key_valid(self, tmp_path):
        """验证有效的 API Key。"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "api_keys:\n"
            "  test-app: 'test-key-123'\n"
        )
        s = Settings(_yaml_path=str(yaml_file))
        assert s.validate_api_key("test-key-123") is True

    def test_validate_api_key_invalid(self, tmp_path):
        """验证无效的 API Key。"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "api_keys:\n"
            "  test-app: 'test-key-123'\n"
        )
        s = Settings(_yaml_path=str(yaml_file))
        assert s.validate_api_key("wrong-key") is False


class TestBuildMem0Config:
    def test_returns_mem0_config(self, tmp_path, monkeypatch):
        """build_mem0_config 从 YAML 配置构建 mem0 配置。"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "llm:\n"
            "  provider: openai\n"
            "  model: gpt-4o-mini\n"
            "  api_key: sk-test\n"
            "vector_store:\n"
            "  provider: qdrant\n"
            "  config: {}\n"
            "embedder:\n"
            "  provider: openai\n"
            "  model: text-embedding-3-small\n"
            "  api_key: sk-test\n"
        )
        get_settings.cache_clear()
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("EMBEDDER_PROVIDER", raising=False)
        monkeypatch.delenv("VECTOR_STORE_PROVIDER", raising=False)

        # 通过临时 YAML 路径构建 Settings
        from memory_platform import config as cfg

        original_settings = cfg.get_settings
        test_settings = Settings(_yaml_path=str(yaml_file))

        # 替换缓存的 settings
        cfg.get_settings = lambda: test_settings

        config = build_mem0_config()
        assert config.llm.provider == "openai"
        assert config.vector_store.provider == "qdrant"
        assert config.embedder.provider == "openai"

        # 恢复
        cfg.get_settings = original_settings


class TestYAMLConfig:
    """YAML 配置加载测试。"""

    def test_loads_mysql_config_from_yaml(self, tmp_path):
        """Settings 从 config.yaml 加载 MySQL 配置。"""
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
        s = Settings(_yaml_path=str(yaml_file))
        assert s.mysql_host == "db.example.com"
        assert s.mysql_port == 3307
        assert s.mysql_database == "test_db"
        assert s.mysql_username == "admin"
        assert s.mysql_password == "secret"
        assert s.mysql_pool_size == 10

    def test_mysql_defaults_when_no_yaml(self):
        """无 YAML 文件时使用默认值。"""
        s = Settings(_yaml_path="/nonexistent/config.yaml")
        assert s.mysql_host == "127.0.0.1"
        assert s.mysql_port == 3306
        assert s.mysql_pool_size == 5

    def test_loads_llm_config_from_yaml(self, tmp_path):
        """从 YAML 加载 LLM 配置。"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "llm:\n"
            "  provider: anthropic\n"
            "  model: glm-5-turbo\n"
            "  api_key: test-key\n"
            "  base_url: 'https://api.example.com'\n"
        )
        s = Settings(_yaml_path=str(yaml_file))
        assert s.llm_provider == "anthropic"
        assert s.llm_model == "glm-5-turbo"
        assert s.llm_api_key == "test-key"
        assert s.llm_base_url == "https://api.example.com"

    def test_loads_full_config(self, tmp_path, monkeypatch):
        """加载包含所有配置项的完整 YAML。"""
        # 清除环境变量，避免 .env 覆盖 YAML 配置
        for key in [
            "API_KEYS", "LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY",
            "LLM_BASE_URL", "VECTOR_STORE_PROVIDER", "EMBEDDER_PROVIDER",
            "EMBEDDER_MODEL", "EMBEDDER_API_KEY", "EMBEDDER_BASE_URL",
        ]:
            monkeypatch.delenv(key, raising=False)
        # 禁用 .env 文件加载，避免 CWD 中的 .env 干扰测试
        monkeypatch.setattr(
            "memory_platform.config.Settings.model_config",
            {"env_file": "/dev/null", "extra": "ignore"},
        )
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "api_keys:\n"
            "  app1: 'key1'\n"
            "  app2: 'key2'\n"
            "llm:\n"
            "  provider: anthropic\n"
            "  model: glm-5-turbo\n"
            "  api_key: llm-key\n"
            "  base_url: 'https://api.example.com'\n"
            "vector_store:\n"
            "  provider: qdrant\n"
            "  config: {collection: test}\n"
            "embedder:\n"
            "  provider: mock\n"
            "  model: mock-embedding\n"
            "  api_key: ''\n"
            "  base_url: ''\n"
            "mysql:\n"
            "  host: 'db.example.com'\n"
            "  port: 3307\n"
            "  database: 'test_db'\n"
            "  username: 'admin'\n"
            "  password: 'secret'\n"
            "  pool_size: 10\n"
            "degradation_enabled: false\n"
        )
        s = Settings(_yaml_path=str(yaml_file))
        assert s.api_keys == {"app1": "key1", "app2": "key2"}
        assert s.llm_provider == "anthropic"
        assert s.llm_model == "glm-5-turbo"
        assert s.llm_api_key == "llm-key"
        assert s.llm_base_url == "https://api.example.com"
        assert s.vector_store_provider == "qdrant"
        assert s.vector_store_config == {"collection": "test"}
        assert s.embedder_provider == "mock"
        assert s.embedder_model == "mock-embedding"
        assert s.mysql_host == "db.example.com"
        assert s.degradation_enabled is False


class TestWalConfig:
    """Wal LLM 网关配置测试"""

    def test_loads_wal_config_from_yaml(self, tmp_path):
        """从 YAML 加载 Wal 网关配置"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "llm:\n"
            "  provider: wal\n"
            "  model: DeepSeekV3.2\n"
            "  wal_base_url: 'https://llm-gateway.test/api'\n"
            "  aloha_app_name: 'my-app'\n"
            "  access_token: 'my-token'\n"
        )
        s = Settings(_yaml_path=str(yaml_file))
        assert s.llm_provider == "wal"
        assert s.llm_model == "DeepSeekV3.2"
        assert s.wal_base_url == "https://llm-gateway.test/api"
        assert s.aloha_app_name == "my-app"
        assert s.access_token == "my-token"

    def test_build_mem0_config_with_wal(self, tmp_path):
        """build_mem0_config 正确构建 wal 配置"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "llm:\n"
            "  provider: wal\n"
            "  model: DeepSeekV3.2\n"
            "  wal_base_url: 'https://llm-gateway.test/api'\n"
            "  aloha_app_name: 'my-app'\n"
            "  access_token: 'my-token'\n"
            "vector_store:\n"
            "  provider: qdrant\n"
            "  config: {}\n"
            "embedder:\n"
            "  provider: mock\n"
            "  model: mock-embedding\n"
            "  api_key: ''\n"
        )
        get_settings.cache_clear()

        from memory_platform import config as cfg
        original_settings = cfg.get_settings
        test_settings = Settings(_yaml_path=str(yaml_file))
        cfg.get_settings = lambda: test_settings

        try:
            mem0_config = build_mem0_config()
            assert mem0_config.llm.provider == "wal"
            assert mem0_config.llm.config["model"] == "DeepSeekV3.2"
            assert mem0_config.llm.config["wal_base_url"] == "https://llm-gateway.test/api"
            assert mem0_config.llm.config["aloha_app_name"] == "my-app"
            assert mem0_config.llm.config["access_token"] == "my-token"
        finally:
            cfg.get_settings = original_settings
