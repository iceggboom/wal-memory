from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TencentVectorConfig(BaseModel):
    """mem0 config class for Tencent Cloud VectorDB provider.

    This is the config class registered with mem0's VectorStoreConfig,
    so users can specify provider="tencent_vector" in their mem0 config.
    """

    url: str = Field(default="", description="VectorDB service endpoint URL")
    username: str = Field(default="root", description="Database username")
    key: str = Field(default="", description="API key or password")
    collection_name: str = Field(default="mem0", description="Collection name")
    embedding_model_dims: Optional[int] = Field(
        default=1536, description="Embedding vector dimensions"
    )
    database_name: str = Field(default="memory_platform", description="Database name")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    embedding_model: str = Field(default="bge-base-zh", description="VectorDB built-in embedding model name")
    mock: bool = Field(default=False, description="Use mock mode (no real VectorDB connection)")

    model_config = {"extra": "ignore"}

    @field_validator("timeout")
    @classmethod
    def timeout_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("timeout must be a positive integer")
        return v
