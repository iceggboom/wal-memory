"""GLM API 端到端集成测试 — 验证 mem0 通过 Anthropic 协议接入 GLM"""
import os

import pytest


GLM_API_KEY = os.environ.get("GLM_API_KEY", "")
GLM_BASE_URL = "https://api.z.ai/api/anthropic"


@pytest.fixture
def glm_settings(monkeypatch):
    """配置 GLM 环境：LLM 走 Anthropic 协议，Embedding 用 Mock"""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "glm-5-turbo")
    monkeypatch.setenv("LLM_API_KEY", GLM_API_KEY)
    monkeypatch.setenv("LLM_BASE_URL", GLM_BASE_URL)
    # Embedding 使用 Mock（GLM 国际版不支持 embedding）
    monkeypatch.setenv("EMBEDDER_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDER_MODEL", "mock-embedding")
    # 设置 Anthropic SDK 环境变量
    monkeypatch.setenv("ANTHROPIC_API_KEY", GLM_API_KEY)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", GLM_BASE_URL)
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "qdrant")


@pytest.mark.skipif(
    not GLM_API_KEY,
    reason="GLM_API_KEY not set, skipping e2e test",
)
class TestGLMIntegration:
    def test_config_loads_glm_settings(self, glm_settings):
        """验证配置正确加载 GLM 参数"""
        from memory_platform.config import get_settings

        s = get_settings()
        assert s.llm_provider == "anthropic"
        assert s.llm_model == "glm-5-turbo"
        assert s.llm_api_key == GLM_API_KEY
        assert s.llm_base_url == GLM_BASE_URL

    def test_mem0_config_uses_anthropic(self, glm_settings):
        """验证 mem0 MemoryConfig 使用 anthropic provider 和 mock embedder"""
        from memory_platform.config import build_mem0_config

        config = build_mem0_config()
        assert config.llm.provider == "anthropic"
        assert config.llm.config["model"] == "glm-5-turbo"
        assert config.llm.config["api_key"] == GLM_API_KEY
        assert config.llm.config["anthropic_base_url"] == GLM_BASE_URL
        assert config.embedder.provider == "mock"

    def test_anthropic_sdk_reads_env(self, glm_settings):
        """验证 anthropic SDK 能从环境变量读取 base_url"""
        import anthropic

        client = anthropic.Anthropic()
        assert str(client.base_url).rstrip("/") == GLM_BASE_URL.rstrip("/")

    def test_mem0_add_with_glm(self, glm_settings):
        """验证 mem0 通过 Anthropic 协议使用 GLM 提取记忆"""
        from memory_platform.config import build_mem0_config
        from mem0 import Memory

        config = build_mem0_config()
        m = Memory(config=config)

        result = m.add(
            [{"role": "user", "content": "我是一名Python后端工程师，喜欢用FastAPI写API"}],
            user_id="test-glm-user",
            agent_id="test-app",
            metadata={"scope": "shared", "app_id": "test-app"},
        )

        events = [r["event"] for r in result.get("results", [])]
        assert "ADD" in events
        assert len(result.get("results", [])) >= 1

        # 清理测试数据
        m.delete_all(user_id="test-glm-user", agent_id="test-app")
