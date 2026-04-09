# AI Memory Platform 编码规范

本文档定义了项目的代码注释和文档规范，确保代码可维护、可理解。

---

## 1. 模块级注释

每个 Python 模块（`.py` 文件）必须以模块级 docstring 开头，说明：

1. **模块用途** — 一句话概括
2. **核心职责** — 列出主要功能点
3. **使用示例** — 展示典型用法

```python
"""记忆检索服务 — 负责记忆的搜索、查询、删除

核心职责：
1. 向量相似度搜索
2. 置信度衰减计算和过滤
3. 按 layer/scope 过滤
4. 获取全部记忆
5. 删除记忆

使用示例：
    >>> svc = RecallService(mem0)
    >>> results = svc.search("Python工程师", user_id="u1", agent_id="a1")
"""
```

---

## 2. 类注释

类必须有 docstring，说明：

1. **类用途** — 一句话概括
2. **核心流程** — 关键步骤（如果是复杂类）
3. **属性说明** — 重要属性的用途

```python
class WriteService:
    """记忆写入服务 — 负责记忆的添加、提取、更新

    核心流程：
    1. 接收对话消息
    2. 调用 LLM 提取记忆
    3. 写入向量存储
    4. 为记忆附加元数据（层级、scope）

    Attributes:
        mem0: mem0 Memory 实例
    """
```

---

## 3. 方法注释

所有公开方法必须有 docstring，使用 Google 风格：

```python
def search(
    self,
    query: str,
    user_id: str,
    limit: int = 10,
    min_confidence: float = 0.5,
) -> list[dict]:
    """搜索记忆

    核心流程：
    1. 向量相似度搜索
    2. Scope/Layer 过滤
    3. 置信度衰减计算
    4. 过滤低置信度记忆

    Args:
        query: 搜索查询文本
        user_id: 用户唯一标识
        limit: 最大返回数量，默认 10
        min_confidence: 最小置信度阈值，默认 0.5

    Returns:
        记忆列表，每项包含 id, text, confidence 等字段

    Raises:
        ValueError: 当 user_id 为空时

    Example:
        >>> results = svc.search("Python", "u1", limit=5)
        >>> len(results)
        3
    """
```

### 必须包含的段落

| 段落 | 必要性 | 说明 |
|------|--------|------|
| 一句话描述 | 必须 | 简洁说明功能 |
| Args | 必须 | 参数说明（类型已在签名中） |
| Returns | 必须 | 返回值说明 |
| Raises | 视情况 | 如果方法可能抛出异常 |
| Example | 推荐 | 典型用法示例 |

---

## 4. 核心逻辑注释

复杂的业务逻辑必须有行内注释，说明 **为什么** 而非 **做什么**：

```python
# Step 1: 获取候选记忆（取 limit * 3，为后续过滤留余量）
raw = self.mem0.search(query, user_id=user_id, limit=limit * 3)

# Step 2: 应用 Scope 过滤
# shared 记忆跨应用可见，private 仅当前应用可见
scope_enum = Scope(scope)
memories = apply_scope_filter(memories, scope_enum)

# Step 3: 置信度衰减计算
# 公式：confidence = similarity × e^(-λ × Δt)
# λ 由记忆层级决定，L1 衰减最慢，L3 衰减最快
scored = filter_by_confidence(memories, min_confidence=min_confidence)
```

### 注释风格

```python
# ✅ 好的注释 — 解释 WHY
# mem0 的 EmbedderConfig 硬编码了支持的 provider 列表
# 使用 model_construct 绕过验证，支持自定义 mock embedder
embedder_cfg = EmbedderConfig.model_construct(...)

# ❌ 坏的注释 — 只说 WHAT（代码本身已说明）
# 创建 embedder 配置
embedder_cfg = EmbedderConfig.model_construct(...)
```

---

## 5. 常量和配置注释

常量定义必须说明用途和取值依据：

```python
# 各层级的衰减系数 λ
# 值越小衰减越慢，记忆保留时间越长
# 半衰期 = ln(2) / λ（小时）
DECAY_LAMBDA: dict[str, float] = {
    "L1": 0.001,   # Profile — 极慢衰减（约 29 天半衰期）
    "L2": 0.005,   # Preference — 慢衰减（约 6 天半衰期）
    "L3": 0.02,    # Episodic — 中等衰减（约 1.4 天半衰期）
    "L4": 0.01,    # Relational — 较慢衰减（约 2.9 天半衰期）
}
```

---

## 6. API 端点注释

API 路由必须有完整的 docstring：

```python
@router.post("/memories/search")
def search_memories(req: SearchRequest, request: Request):
    """搜索记忆

    基于向量相似度搜索，应用置信度衰减和过滤。

    Request Body:
        user_id: 用户 ID（必填）
        app_id: 应用 ID（必填）
        query: 搜索查询（必填）
        limit: 最大返回数量，默认 10
        min_confidence: 最小置信度，默认 0.5

    Returns:
        {"results": [...], "total": n}

    Note:
        搜索失败时触发降级，返回空结果而非报错
    """
```

---

## 7. 类型注解

所有函数参数和返回值必须有类型注解：

```python
# ✅ 好的例子
def compute_confidence(
    similarity: float,
    updated_at: datetime | str,
    layer: str | None = None,
) -> float:
    ...

# ❌ 坏的例子
def compute_confidence(similarity, updated_at, layer=None):
    ...
```

---

## 8. 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 模块 | snake_case | `recall_service.py` |
| 类 | PascalCase | `RecallService` |
| 函数/方法 | snake_case | `search_memories()` |
| 常量 | UPPER_SNAKE_CASE | `DECAY_LAMBDA` |
| 私有方法 | _leading_underscore | `_parse_datetime()` |

---

## 9. 代码审查清单

在提交代码前，检查以下项目：

- [ ] 模块顶部有 docstring
- [ ] 类有 docstring
- [ ] 公开方法有完整 docstring（Args/Returns）
- [ ] 复杂逻辑有注释说明
- [ ] 常量有说明注释
- [ ] 所有函数有类型注解
- [ ] 命名符合规范

---

## 10. 示例：完整的模块

```python
"""置信度衰减模块 — 基于时间的记忆置信度计算

核心职责：
1. 计算记忆的置信度（基于相似度和时间衰减）
2. 按置信度过滤和排序记忆列表

衰减公式：
    confidence = similarity × e^(-λ × Δt)

使用示例：
    >>> from memory_platform.ext.confidence import compute_confidence
    >>> conf = compute_confidence(0.9, "2024-01-01", layer="L1")
"""
import math
from datetime import datetime, timezone


# 各层级的衰减系数 λ
DECAY_LAMBDA: dict[str, float] = {
    "L1": 0.001,   # Profile — 极慢衰减
    "L2": 0.005,   # Preference — 慢衰减
    "L3": 0.02,    # Episodic — 中等衰减
    "L4": 0.01,    # Relational — 较慢衰减
}


def compute_confidence(
    similarity: float,
    updated_at: datetime | str,
    layer: str | None = None,
) -> float:
    """计算记忆的置信度

    公式：confidence = similarity × e^(-λ × Δt)

    Args:
        similarity: 向量相似度 (0-1)
        updated_at: 记忆最后更新时间
        layer: 记忆层级，用于选择衰减系数

    Returns:
        置信度值 (0-1)
    """
    lam = DECAY_LAMBDA.get(layer or "L1", DECAY_LAMBDA["L1"])
    # ... 实现 ...
```

---

## 参考

- [Google Python Style Guide - Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
