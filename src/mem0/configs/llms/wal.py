# src/mem0/configs/llms/wal.py
"""Wal LLM 配置类 — Walmart 内部 LLM 网关专用配置"""

from typing import Optional

from mem0.configs.llms.base import BaseLlmConfig


class WalConfig(BaseLlmConfig):
    """Walmart 内部 LLM 网关配置

    继承 BaseLlmConfig 的通用参数（model, temperature, max_tokens 等），
    新增网关地址和认证相关字段。

    属性:
        wal_base_url: LLM 网关基础地址
        aloha_app_name: 应用标识（用于请求头认证）
        access_token: 访问令牌（用于请求头认证）
        supplier_type: 供应商类型标识，默认 "2"
    """

    def __init__(
        self,
        # 基础参数
        model: Optional[str] = None,
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        max_tokens: int = 2000,
        top_p: float = 0.1,
        top_k: int = 1,
        enable_vision: bool = False,
        vision_details: Optional[str] = "auto",
        http_client_proxies: Optional[dict] = None,
        # Wal 专属参数
        wal_base_url: Optional[str] = None,
        aloha_app_name: Optional[str] = None,
        access_token: Optional[str] = None,
        supplier_type: str = "2",
    ):
        """初始化 Wal LLM 配置

        参数:
            model: 模型名称（如 "DeepSeekV3.2"）
            temperature: 控制输出随机性，默认 0.1
            api_key: API 密钥（Wal 网关不使用，保留兼容）
            max_tokens: 最大生成 token 数，默认 2000
            top_p: 核采样参数，默认 0.1
            top_k: Top-k 采样参数，默认 1
            enable_vision: 是否启用视觉能力，默认 False
            vision_details: 视觉处理细节级别，默认 "auto"
            http_client_proxies: HTTP 客户端代理设置
            wal_base_url: LLM 网关基础地址
            aloha_app_name: 应用标识
            access_token: 访问令牌
            supplier_type: 供应商类型，默认 "2"
        """
        super().__init__(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            enable_vision=enable_vision,
            vision_details=vision_details,
            http_client_proxies=http_client_proxies,
        )
        self.wal_base_url = wal_base_url
        self.aloha_app_name = aloha_app_name
        self.access_token = access_token
        self.supplier_type = supplier_type
