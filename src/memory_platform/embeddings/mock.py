"""Mock Embedder — 用于开发和测试的模拟嵌入器

核心职责：
1. 返回固定维度的伪向量，无需调用真实 Embedding API
2. 确保相同文本生成相同向量（基于哈希）
3. 支持 mem0 的 EmbedderFactory 注册机制

使用场景：
- 本地开发测试，避免消耗 API 配额
- GLM 国际版等不支持 Embedding 的 LLM 服务

注册方式：
    from mem0.utils.factory import EmbedderFactory
    EmbedderFactory.provider_to_class["mock"] = "memory_platform.embeddings.mock.MockEmbedder"
"""
from typing import Literal, Optional

from mem0.embeddings.base import EmbeddingBase
from mem0.configs.embeddings.base import BaseEmbedderConfig


class MockEmbedderConfig(BaseEmbedderConfig):
    """Mock Embedder 配置类

    Attributes:
        embedding_dims: 输出向量维度，默认 1536（与 OpenAI text-embedding-3-small 一致）
    """

    def __init__(self, embedding_dims: int = 1536, **kwargs):
        super().__init__(embedding_dims=embedding_dims, **kwargs)
        self.embedding_dims = embedding_dims


class MockEmbedder(EmbeddingBase):
    """Mock Embedder — 返回基于哈希的伪向量

    特点：
    - 相同文本生成相同向量（确定性）
    - 不同文本生成不同向量（基于哈希）
    - 无需网络请求，速度快

    Example:
        >>> config = MockEmbedderConfig(embedding_dims=512)
        >>> embedder = MockEmbedder(config)
        >>> vector = embedder.embed("Hello World")
        >>> len(vector)
        512
    """

    def __init__(self, config: Optional[MockEmbedderConfig] = None):
        """初始化 Mock Embedder

        Args:
            config: 可选的配置实例，未传入时使用默认配置
        """
        if config is None:
            config = MockEmbedderConfig()
        super().__init__(config)
        # 确保 dims 不为 None
        self.dims = getattr(self.config, "embedding_dims", 1536) or 1536

    def embed(self, text: str, memory_action: Optional[Literal["add", "search", "update"]] = None) -> list[float]:
        """生成文本的伪向量

        算法：
        1. 计算文本哈希值
        2. 将哈希值映射到 [0, 1) 区间作为基准值
        3. 生成等差序列，确保维度间有差异

        Args:
            text: 待嵌入的文本
            memory_action: 记忆操作类型（add/search/update），mock 模式下忽略

        Returns:
            固定维度的浮点数列表
        """
        # 使用简单的哈希生成伪向量，确保相同文本得到相同向量
        hash_val = hash(text)
        base = abs(hash_val) % 1000 / 1000.0
        # 生成等差序列，维度间有 0.001 的差异
        return [base + i * 0.001 for i in range(self.dims)]
