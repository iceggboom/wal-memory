# tests/unit/test_wal_llm.py
"""WalLLM 单元测试 — 验证 HTTP 调用 Walmart LLM 网关"""

import pytest
from unittest.mock import MagicMock, patch

from mem0.configs.llms.wal import WalConfig
from mem0.llms.wal import WalLLM


def _make_config(**overrides):
    """构建测试用 WalConfig"""
    defaults = {
        "model": "DeepSeekV3.2",
        "wal_base_url": "https://llm-gateway.test/api",
        "aloha_app_name": "test-app",
        "access_token": "test-token",
    }
    defaults.update(overrides)
    return WalConfig(**defaults)


def _mock_httpx_response(status_code=200, json_data=None):
    """构建 mock httpx 响应"""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = str(json_data) if json_data else ""
    return resp


class TestWalLLMInit:
    """初始化测试"""

    def test_creates_with_wal_config(self):
        """使用 WalConfig 初始化"""
        config = _make_config()
        llm = WalLLM(config)
        assert llm.config.model == "DeepSeekV3.2"
        assert llm.config.wal_base_url == "https://llm-gateway.test/api"

    def test_creates_with_dict_config(self):
        """使用 dict 初始化，自动转换为 WalConfig"""
        config_dict = {
            "model": "DeepSeekV3.2",
            "wal_base_url": "https://llm-gateway.test/api",
            "aloha_app_name": "test-app",
            "access_token": "test-token",
        }
        llm = WalLLM(config_dict)
        assert isinstance(llm.config, WalConfig)
        assert llm.config.wal_base_url == "https://llm-gateway.test/api"

    def test_creates_with_base_config_conversion(self):
        """使用 BaseLlmConfig 自动转换为 WalConfig"""
        from mem0.configs.llms.base import BaseLlmConfig
        base = BaseLlmConfig(model="test-model")
        llm = WalLLM(base)
        assert isinstance(llm.config, WalConfig)

    def test_default_model(self):
        """无 model 时使用默认值"""
        config = _make_config(model=None)
        llm = WalLLM(config)
        assert llm.config.model == "deepseek-chat"


class TestWalLLMGenerateResponse:
    """generate_response 测试"""

    @patch("mem0.llms.wal.httpx.Client")
    def test_successful_response(self, mock_client_cls):
        """成功调用返回内容字符串"""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_httpx_response(
            json_data={
                "id": "chat-123",
                "model": "DeepSeekV3.2",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello world"},
                        "finishReason": "stop",
                    }
                ],
            }
        )

        llm = WalLLM(_make_config())
        result = llm.generate_response(
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert result == "Hello world"

    @patch("mem0.llms.wal.httpx.Client")
    def test_request_headers(self, mock_client_cls):
        """验证请求头包含认证信息"""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_httpx_response(
            json_data={"choices": [{"message": {"content": "ok"}}]}
        )

        llm = WalLLM(_make_config())
        llm.generate_response(messages=[{"role": "user", "content": "test"}])

        call_args = mock_client.post.call_args
        headers = call_args.kwargs.get("headers", {})
        assert headers.get("alohaAppName") == "test-app"
        assert headers.get("accessToken") == "test-token"
        assert headers.get("model") == "DeepSeekV3.2"
        assert headers.get("supplierType") == "2"
        assert headers.get("Content-Type") == "application/json"

    @patch("mem0.llms.wal.httpx.Client")
    def test_request_url(self, mock_client_cls):
        """验证请求 URL 拼接正确"""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_httpx_response(
            json_data={"choices": [{"message": {"content": "ok"}}]}
        )

        llm = WalLLM(_make_config())
        llm.generate_response(messages=[{"role": "user", "content": "test"}])

        call_args = mock_client.post.call_args
        assert call_args.args[0] == "https://llm-gateway.test/api/chat/completions"

    @patch("mem0.llms.wal.httpx.Client")
    def test_request_body_with_valid_temperature(self, mock_client_cls):
        """temperature 在 (0, 1) 范围内时发送"""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_httpx_response(
            json_data={"choices": [{"message": {"content": "ok"}}]}
        )

        config = _make_config(temperature=0.5)
        llm = WalLLM(config)
        llm.generate_response(messages=[{"role": "user", "content": "test"}])

        call_args = mock_client.post.call_args
        body = call_args.kwargs.get("json", {})
        assert body["temperature"] == 0.5
        assert "model" not in body  # model 仅在 header 中传递

    @patch("mem0.llms.wal.httpx.Client")
    def test_request_body_always_has_temperature(self, mock_client_cls):
        """temperature 始终包含在请求体中"""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_httpx_response(
            json_data={"choices": [{"message": {"content": "ok"}}]}
        )

        # 使用默认 temperature
        llm = WalLLM(_make_config())
        llm.generate_response(messages=[{"role": "user", "content": "test"}])

        call_args = mock_client.post.call_args
        body = call_args.kwargs.get("json", {})
        assert "temperature" in body
        assert body["temperature"] == 0.95

    @patch("mem0.llms.wal.httpx.Client")
    def test_request_body_with_system_message(self, mock_client_cls):
        """system 消息正确包含在请求体中"""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_httpx_response(
            json_data={"choices": [{"message": {"content": "ok"}}]}
        )

        llm = WalLLM(_make_config())
        llm.generate_response(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"},
            ]
        )

        call_args = mock_client.post.call_args
        body = call_args.kwargs.get("json", {})
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"

    @patch("mem0.llms.wal.httpx.Client")
    def test_failsafe_on_http_error(self, mock_client_cls):
        """HTTP 4xx/5xx 时 fail-safe 返回空字符串"""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_httpx_response(
            status_code=500, json_data={"error": "internal error"}
        )

        llm = WalLLM(_make_config())
        result = llm.generate_response(messages=[{"role": "user", "content": "test"}])

        assert result == ""

    @patch("mem0.llms.wal.httpx.Client")
    def test_failsafe_on_network_error(self, mock_client_cls):
        """网络异常时 fail-safe 返回空字符串"""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = Exception("Connection timeout")

        llm = WalLLM(_make_config())
        result = llm.generate_response(messages=[{"role": "user", "content": "test"}])

        assert result == ""

    @patch("mem0.llms.wal.httpx.Client")
    def test_failsafe_on_malformed_response(self, mock_client_cls):
        """响应格式异常时 fail-safe 返回空字符串"""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_httpx_response(
            json_data={"choices": []}
        )

        llm = WalLLM(_make_config())
        result = llm.generate_response(messages=[{"role": "user", "content": "test"}])

        assert result == ""

    @patch("mem0.llms.wal.httpx.Client")
    def test_failsafe_on_json_decode_error(self, mock_client_cls):
        """JSON 解析失败时 fail-safe 返回空字符串"""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("invalid json")
        resp.text = "not json"
        mock_client.post.return_value = resp

        llm = WalLLM(_make_config())
        result = llm.generate_response(messages=[{"role": "user", "content": "test"}])

        assert result == ""
