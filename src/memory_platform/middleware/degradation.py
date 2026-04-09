"""降级处理中间件 — 记忆服务异常时的优雅降级

核心职责：
1. 捕获记忆服务异常
2. 返回空结果，不阻塞主业务流程
3. 记录错误日志

设计理念：
- 记忆服务是"增强型"功能，不是核心功能
- 记忆服务异常时，主业务应继续运行
- 降级返回空结果，而非抛出异常

使用示例：
    >>> from memory_platform.middleware.degradation import memory_degradation_handler
    >>> try:
    ...     result = mem0.search(query)
    ... except Exception as e:
    ...     result = memory_degradation_handler(e)
    >>> result
    {"results": [], "total": 0}
"""
import logging

logger = logging.getLogger(__name__)


def memory_degradation_handler(exc: Exception) -> dict:
    """记忆服务降级处理

    当记忆服务发生任何异常时，返回空结果而非抛出异常。
    这确保主业务流程不会被记忆服务故障阻塞。

    Args:
        exc: 捕获的异常对象

    Returns:
        空结果 {"results": [], "total": 0}

    Side Effects:
        记录 ERROR 级别日志，包含异常堆栈（非 BaseException）

    Example:
        >>> try:
        ...     raise ConnectionError("Vector DB unavailable")
        ... except Exception as e:
        ...     memory_degradation_handler(e)
        {"results": [], "total": 0}
    """
    # 记录错误日志
    # 对于非 BaseException（通常是系统级异常），记录完整堆栈
    logger.error(
        "Memory service degradation triggered: %s",
        exc,
        exc_info=type(exc) is not Exception,
    )

    # 返回空结果，确保接口格式一致
    return {"results": [], "total": 0}
