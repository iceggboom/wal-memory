# src/mem0/llms/wal.py
"""Wal LLM Provider — 通过 HTTP POST 调用 Walmart 内部 LLM 网关

核心职责：
1. 实现 mem0 LLMBase 的 generate_response 接口
2. 使用 httpx 同步模式发送 HTTP POST 请求
3. 处理 Walmart LLM 网关的认证（自定义 Header）
4. fail-safe 错误处理

使用示例：
    >>> from mem0.llms.wal import WalLLM
    >>> from mem0.configs.llms.wal import WalConfig
    >>> config = WalConfig(model="DeepSeekV3.2", wal_base_url="https://...", ...)
    >>> llm = WalLLM(config)
    >>> result = llm.generate_response(messages=[{"role": "user", "content": "Hi"}])
"""

import logging
from typing import Dict, List, Optional, Union

import httpx

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.wal import WalConfig
from mem0.llms.base import LLMBase

logger = logging.getLogger(__name__)


class WalLLM(LLMBase):
    """Walmart 内部 LLM 网关提供者

    通过 HTTP POST 调用 Walmart LLM 网关的 /chat/completions 端点。
    认证通过自定义 Header（alohaAppName, accessToken）实现。
    响应格式兼容 OpenAI Chat Completions API。
    """

    def __init__(self, config: Optional[Union[BaseLlmConfig, WalConfig, Dict]] = None):
        """初始化 WalLLM

        参数:
            config: 配置对象，支持 WalConfig、dict 或 BaseLlmConfig
        """
        if config is None:
            config = WalConfig()
        elif isinstance(config, dict):
            config = WalConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, WalConfig):
            config = WalConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
                http_client_proxies=config.http_client,
            )

        super().__init__(config)

        if not self.config.model:
            self.config.model = "deepseek-chat"

    def _build_headers(self) -> Dict[str, str]:
        """构建请求头

        包含 Walmart LLM 网关所需的认证信息。

        返回:
            请求头字典
        """
        return {
            "alohaAppName": self.config.aloha_app_name or "",
            "accessToken": self.config.access_token or "",
            "model": self.config.model,
            "supplierType": self.config.supplier_type,
            "Content-Type": "application/json",
        }

    def _build_request_body(self, messages: List[Dict[str, str]], **kwargs) -> Dict:
        """构建请求体

        参数:
            messages: 消息列表
            **kwargs: 额外参数

        返回:
            请求体字典
        """
        body: Dict = {
            "messages": messages,
            "model": self.config.model,
            "temperature": self.config.temperature,
        }

        return body

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """调用 LLM 网关生成响应

        参数:
            messages: 消息列表，每条消息包含 role 和 content
            response_format: 响应格式（暂不使用）
            tools: 工具列表（暂不使用）
            tool_choice: 工具选择方式（暂不使用）
            **kwargs: 额外参数

        返回:
            str: LLM 生成的文本内容，失败时返回空字符串
        """
        url = f"{self.config.wal_base_url}/chat/completions"
        headers = self._build_headers()
        body = self._build_request_body(messages, **kwargs)

        try:
            with httpx.Client() as client:
                response = client.post(url, json=body, headers=headers, timeout=60.0)

            if response.status_code != 200:
                logger.warning(
                    "Wal LLM 返回非 200 状态码: %d, 响应: %s",
                    response.status_code,
                    response.text[:200],
                )
                return ""

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return content

        except (KeyError, IndexError) as e:
            logger.warning("Wal LLM 响应格式异常: %s", e)
            return ""
        except ValueError as e:
            logger.warning("Wal LLM JSON 解析失败: %s", e)
            return ""
        except Exception as e:
            logger.error("Wal LLM 调用失败: %s", e)
            return ""
