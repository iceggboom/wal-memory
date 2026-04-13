"""记忆写入服务 — 负责记忆的添加、提取、更新

核心职责：
1. 批量添加记忆（直接写入）
2. 从对话中提取记忆（LLM 推理）
3. 为记忆附加元数据（层级、scope）

核心流程：
    add_memory（直接写入）:
    1. 遍历记忆项列表
    2. 分类记忆层级（基于关键词 + LLM 辅助）
    3. 构建元数据（layer + scope + app_id）
    4. 调用 mem0.add 写入向量存储
    5. 统计 ADD/UPDATE/NONE 事件

    extract（从对话提取）:
    1. 构建元数据
    2. 调用 mem0.add(messages, infer=True)
    3. mem0 内部调用 LLM 提取结构化记忆
    4. 去重和更新已有记忆
    5. 统计 ADD/UPDATE/DELETE 事件
    6. 格式化返回结果

使用示例：
    >>> svc = WriteService(mem0)
    >>> # 直接添加
    >>> svc.add_memory("user-1", "app-1", [AddMemoryItem(text="我是工程师")])
    {"added": 1, "updated": 0, "unchanged": 0}
    >>> # 从对话提取
    >>> svc.extract("user-1", "app-1", [{"role": "user", "content": "我喜欢Python"}])
    {"added": 1, "updated": 0, "deleted": 0, "memories": [...]}
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mem0 import Memory

from memory_platform.ext.layer import classify_layer, classify_layer_with_llm

if TYPE_CHECKING:
    from anthropic import Anthropic


@dataclass
class AddMemoryItem:
    """添加记忆项 — 用于批量添加记忆

    Attributes:
        text: 记忆文本内容
        memory_layer: 显式指定的层级（可选，不指定则自动分类）
        scope: 可见性范围，默认 "shared"
    """

    text: str
    memory_layer: str | None = None
    scope: str = "shared"


class WriteService:
    """记忆写入服务

    提供两种写入方式：
    1. add_memory: 直接写入结构化记忆
    2. extract: 从对话中提取记忆（需要 LLM）
    """

    def __init__(
        self,
        mem0: Memory,
        llm_client: "Anthropic | None" = None,
        llm_model: str = "glm-5-turbo",
    ):
        """初始化服务

        Args:
            mem0: mem0 Memory 实例
            llm_client: 可选的 LLM 客户端，用于层级分类
            llm_model: LLM 模型名称
        """
        self.mem0 = mem0
        self.llm_client = llm_client
        self.llm_model = llm_model

    def _classify_layer(
        self,
        text: str,
        explicit_layer: str | None = None,
    ):
        """分类记忆层级

        策略：
        1. 显式指定优先
        2. 关键词匹配
        3. LLM 辅助（如果配置了 llm_client）

        Args:
            text: 记忆文本
            explicit_layer: 显式指定的层级

        Returns:
            MemoryLayer 枚举值
        """
        if self.llm_client is not None:
            # 使用 LLM 辅助分类（内部已实现关键词优先）
            return classify_layer_with_llm(
                text=text,
                llm_client=self.llm_client,
                model=self.llm_model,
                use_keyword_first=True,
                explicit_layer=explicit_layer,
            )
        else:
            # 仅使用关键词分类
            return classify_layer(text, explicit_layer=explicit_layer)

    def add_memory(
        self,
        user_id: str,
        agent_id: str,
        items: list[AddMemoryItem],
    ) -> dict:
        """批量添加记忆

        核心流程：
        1. 遍历记忆项
        2. 分类层级（关键词优先，LLM 辅助）
        3. 构建元数据
        4. 写入 mem0
        5. 统计事件

        Args:
            user_id: 用户唯一标识
            agent_id: 应用唯一标识
            items: 记忆项列表

        Returns:
            {"added": n, "updated": n, "unchanged": n}

        Example:
            >>> svc.add_memory("u1", "a1", [AddMemoryItem(text="我是工程师")])
            {"added": 1, "updated": 0, "unchanged": 0}
        """
        added = 0
        updated = 0
        unchanged = 0

        for item in items:
            # Step 1: 分类记忆层级（关键词优先，LLM 辅助）
            layer = self._classify_layer(item.text, item.memory_layer)

            # Step 2: 构建元数据
            metadata = {
                "memory_layer": layer.value,
                "scope": item.scope,
                "app_id": agent_id,
            }

            # Step 3: 写入 mem0（不调用 LLM 推理）
            result = self.mem0.add(
                item.text,
                user_id=user_id,
                agent_id=agent_id,
                metadata=metadata,
                infer=False,  # 直接写入，不调用 LLM
            )

            # Step 4: 统计事件类型
            for r in result.get("results", []):
                event = r.get("event", "NONE")
                if event == "ADD":
                    added += 1
                elif event == "UPDATE":
                    updated += 1
                elif event == "NONE":
                    unchanged += 1

        return {"added": added, "updated": updated, "unchanged": unchanged}

    def extract(
        self,
        user_id: str,
        agent_id: str,
        messages: list[dict],
        memory_layer: str | None = None,
    ) -> dict:
        """从对话中提取记忆

        核心流程：
        1. 构建元数据
        2. 调用 mem0.add(messages, infer=True)
        3. mem0 内部调用 LLM 提取结构化记忆
        4. 去重和更新已有记忆
        5. 统计 ADD/UPDATE/DELETE 事件

        Args:
            user_id: 用户唯一标识
            agent_id: 应用唯一标识
            messages: 对话消息列表 [{"role": "user", "content": "..."}]
            memory_layer: 显式指定的层级（可选）

        Returns:
            {
                "added": n,
                "updated": n,
                "deleted": n,
                "memories": [{"id", "text", "memory_layer", "scope", ...}]
            }

        Example:
            >>> svc.extract("u1", "a1", [{"role": "user", "content": "我喜欢Python"}])
            {"added": 1, "updated": 0, "deleted": 0, "memories": [...]}
        """
        # Step 1: 构建元数据
        metadata = {
            "scope": "shared",
            "app_id": agent_id,
        }
        if memory_layer:
            metadata["memory_layer"] = memory_layer

        # Step 2: 调用 mem0 提取记忆（infer=True 触发 LLM 推理）
        result = self.mem0.add(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            metadata=metadata,
            infer=True,  # 调用 LLM 提取结构化记忆
        )

        # Step 3: 统计事件类型
        added = sum(1 for r in result.get("results", []) if r.get("event") == "ADD")
        updated = sum(1 for r in result.get("results", []) if r.get("event") == "UPDATE")
        deleted = sum(1 for r in result.get("results", []) if r.get("event") == "DELETE")

        # Step 4: 格式化返回结果
        memories = []
        for r in result.get("results", []):
            metadata = {}
            if "metadata" in r and r["metadata"]:
                metadata = r["metadata"]
            memories.append(
                {
                    "id": r.get("id", ""),
                    "text": r.get("memory", ""),
                    "memory_layer": metadata.get("memory_layer", ""),
                    "scope": metadata.get("scope", "shared"),
                    "created_at": r.get("created_at", ""),
                    "updated_at": r.get("updated_at", ""),
                }
            )

        return {
            "added": added,
            "updated": updated,
            "deleted": deleted,
            "memories": memories,
        }
