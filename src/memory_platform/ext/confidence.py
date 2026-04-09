"""置信度衰减模块 — 基于时间的记忆置信度计算

核心职责：
1. 计算记忆的置信度（基于相似度和时间衰减）
2. 按置信度过滤和排序记忆列表

衰减公式：
    confidence = similarity × e^(-λ × Δt)

    其中：
    - similarity: 向量相似度 (0-1)
    - λ (lambda): 衰减系数，层级越高衰减越慢
    - Δt: 距离上次更新的时间（小时）

各层级衰减系数：
- L1 (Profile): λ=0.001, 极慢衰减，约 29 天衰减到 50%
- L2 (Preference): λ=0.005, 慢衰减，约 6 天衰减到 50%
- L3 (Episodic): λ=0.02, 中等衰减，约 1.4 天衰减到 50%
- L4 (Relational): λ=0.01, 较慢衰减，约 2.9 天衰减到 50%

使用示例：
    >>> from memory_platform.ext.confidence import compute_confidence, filter_by_confidence
    >>> conf = compute_confidence(similarity=0.9, updated_at="2024-01-01", layer="L1")
    >>> filtered = filter_by_confidence(memories, min_confidence=0.5, limit=10)
"""
import math
from datetime import datetime, timezone


# 各层级的衰减系数 λ
# 值越小衰减越慢，记忆保留时间越长
DECAY_LAMBDA: dict[str, float] = {
    "L1": 0.001,   # Profile — 极慢衰减（约29天半衰期）
    "L2": 0.005,   # Preference — 慢衰减（约6天半衰期）
    "L3": 0.02,    # Episodic — 中等衰减（约1.4天半衰期）
    "L4": 0.01,    # Relational — 较慢衰减（约2.9天半衰期）
}

# 默认层级，用于未标记的记忆
DEFAULT_LAYER = "L1"


def _parse_datetime(value: datetime | str) -> datetime:
    """解析日期时间，确保返回带时区的 datetime

    Args:
        value: datetime 对象或 ISO 格式字符串

    Returns:
        带时区的 datetime 对象（默认 UTC）
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def compute_confidence(
    similarity: float,
    updated_at: datetime | str,
    layer: str | None = None,
    now: datetime | None = None,
) -> float:
    """计算记忆的置信度

    公式：confidence = similarity × e^(-λ × Δt)

    Args:
        similarity: 向量相似度 (0-1)
        updated_at: 记忆最后更新时间
        layer: 记忆层级，用于选择衰减系数
        now: 当前时间（测试时可注入）

    Returns:
        置信度值 (0-1)

    Example:
        >>> compute_confidence(0.9, "2024-01-01T00:00:00Z", "L1")
        0.8991...
    """
    # 获取对应层级的衰减系数
    lam = DECAY_LAMBDA.get(layer or DEFAULT_LAYER, DECAY_LAMBDA[DEFAULT_LAYER])

    # 计算时间差（小时）
    updated = _parse_datetime(updated_at)
    current = now or datetime.now(timezone.utc)
    delta_hours = (current - updated).total_seconds() / 3600
    delta_hours = max(0, delta_hours)  # 防止负数

    # 应用衰减公式
    return similarity * math.exp(-lam * delta_hours)


def filter_by_confidence(
    memories: list[dict],
    min_confidence: float = 0.5,
    limit: int = 100,
    now: datetime | None = None,
) -> list[tuple[dict, float]]:
    """按置信度过滤和排序记忆列表

    核心流程：
    1. 遍历记忆，计算每条的置信度
    2. 过滤掉低于阈值的记忆
    3. 按置信度降序排序
    4. 返回前 limit 条

    Args:
        memories: 记忆列表，每条需包含 score, updated_at/created_at, metadata
        min_confidence: 最小置信度阈值
        limit: 最大返回数量
        now: 当前时间（测试时可注入）

    Returns:
        (memory, confidence) 元组列表，按置信度降序排列

    Example:
        >>> memories = [{"id": "1", "memory": "test", "score": 0.9, "updated_at": "..."}]
        >>> filtered = filter_by_confidence(memories, min_confidence=0.5, limit=10)
        >>> len(filtered)
        1
    """
    current = now or datetime.now(timezone.utc)
    scored: list[tuple[dict, float]] = []

    for mem in memories:
        # 提取记忆元数据
        layer = (mem.get("metadata") or {}).get("memory_layer", DEFAULT_LAYER)
        updated_at = mem.get("updated_at") or mem.get("created_at") or current
        similarity = mem.get("score", 0.0)

        if similarity is None:
            continue

        # 计算置信度
        conf = compute_confidence(
            similarity=similarity,
            updated_at=updated_at,
            layer=layer,
            now=current,
        )

        # 过滤低置信度记忆
        if conf >= min_confidence:
            scored.append((mem, conf))

    # 按置信度降序排序
    scored.sort(key=lambda x: x[1], reverse=True)

    # 返回前 limit 条
    return scored[:limit]
