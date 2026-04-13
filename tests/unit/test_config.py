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


class TestYAMLConfig:
    """Tests for YAML config loading."""

    def test_loads_mysql_config_from_yaml(self, tmp_path):
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

    def test_mysql_defaults_when_no_yaml(self):
        """Settings should use defaults when no YAML config exists."""
        from memory_platform.config import Settings

        s = Settings(_yaml_path="/nonexistent/config.yaml")
        assert s.mysql_host == "127.0.0.1"
        assert s.mysql_port == 3306
        assert s.mysql_pool_size == 5
