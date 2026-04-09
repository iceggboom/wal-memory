# mem0 Fork — GLM 兼容性修复设计

> 将 mem0 从 pip SDK 依赖改为项目内嵌 fork，修复 GLM 模型的 JSON 解析和 Anthropic 适配问题。

## 1. 背景与问题

### 1.1 现状

当前 `wal-memory` 通过 `mem0ai>=0.1.0` pip 包引入 mem0 SDK，用于记忆的提取、去重、向量 CRUD。

### 1.2 问题

GLM-5-Turbo（通过 Anthropic 协议接入）与 mem0 SDK 存在兼容性问题：

| 问题 | 根因 | 影响 |
|------|------|------|
| `anthropic_base_url` 未传入客户端 | `AnthropicLLM.__init__()` 创建 Anthropic 客户端时忽略了 `anthropic_base_url` 配置项 | 依赖环境变量兜底，配置不透明 |
| GLM 返回 markdown 包裹 JSON | GLM 返回 ` ```json {...} ``` ` 格式，mem0 的 `remove_code_blocks()` 正则对边缘情况处理不足 | extract 接口返回空结果 |
| 异常日志不足 | JSON 解析失败时只记录异常类型，不记录 LLM 返回内容 | 无法排查 GLM 兼容性问题 |

### 1.3 为什么 fork

mem0 SDK 内部的 LLM 交互层（prompt 模板、JSON 解析、LLM 参数传递）是封闭的，无法通过外部扩展修复。Fork 后可自由修改这些内部逻辑。

## 2. 方案概述

**策略**：一次性修改 + 版本锁定，不跟随上游更新。

**Fork 方式**：将 mem0 源码内嵌到 `wal-memory/src/mem0/`，作为项目内部包。

**改动范围**：最小改动，仅修改 3 个文件（~50 行）。

## 3. 项目结构变更

### 3.1 Fork 后目录结构

```
wal-memory/
├── pyproject.toml              # 移除 mem0ai，提升外部依赖
├── src/
│   ├── mem0/                   # ← 新增：fork 自 mem0ai 1.0.10
│   │   ├── __init__.py
│   │   ├── configs/            # 配置类（不变）
│   │   ├── embeddings/         # embedding 基类（不变）
│   │   ├── llms/
│   │   │   ├── base.py
│   │   │   └── anthropic.py    #   修改：传递 base_url
│   │   ├── memory/
│   │   │   ├── main.py         #   修改：增强异常日志
│   │   │   └── utils.py        #   修改：增强 JSON 解析
│   │   ├── utils/              # 工厂函数（不变）
│   │   └── vector_stores/      # 向量存储（不变）
│   └── memory_platform/        # 业务层（import 路径不变）
└── ...
```

### 3.2 需要复制的 mem0 目录

- `mem0/__init__.py`
- `mem0/configs/` — 配置类
- `mem0/embeddings/` — embedding 基类（含 mock 注册点）
- `mem0/llms/` — LLM provider
- `mem0/memory/` — 核心记忆逻辑
- `mem0/utils/` — 工厂函数
- `mem0/vector_stores/` — 向量存储

### 3.3 不需要复制的

- `tests/`、`docs/`、`examples/`、`README.md`

### 3.4 依赖管理

`pyproject.toml` 变更：

```diff
 dependencies = [
-    "mem0ai>=0.1.0",
+    "qdrant-client>=1.9.1",
+    "pydantic>=2.7.3",
+    "openai>=1.90.0",
+    "posthog>=3.5.0",
+    "pytz>=2024.1",
+    "sqlalchemy>=2.0.31",
+    "protobuf>=5.29.6,<7.0.0",
 ]
```

将 mem0 的外部依赖提升为 `wal-memory` 的直接依赖，版本与 mem0 1.0.10 的 `pyproject.toml` 一致。

## 4. 代码修改详情

### 4.1 `mem0/llms/anthropic.py` — 传递 base_url

**位置**：`AnthropicLLM.__init__()` 第 41 行

```python
# 原代码
self.client = anthropic.Anthropic(api_key=api_key)

# 修改为
self.client = anthropic.Anthropic(
    api_key=api_key,
    base_url=self.config.anthropic_base_url,
)
```

**Why**：`AnthropicConfig` 定义了 `anthropic_base_url` 字段，但创建客户端时未传入。GLM 通过 `base_url` 路由到 `api.z.ai`，必须显式传递。

### 4.2 `mem0/memory/utils.py` — 增强 JSON 解析

#### 4.2.1 `remove_code_blocks()` 修改

```python
# 原正则
pattern = r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$"

# 修改为
pattern = r"^\s*```[a-zA-Z0-9]*\r?\n([\s\S]*?)\r?\n\s*```"
```

**变更点**：
- `\s*` 允许开头有空白
- `\r?\n` 兼容 Windows 换行
- 移除末尾 `$`，允许 JSON 后有尾部字符

#### 4.2.2 `extract_json()` 修改

增加括号平衡算法，准确提取完整 JSON（处理嵌套 `{}`）：

```python
def extract_json(text):
    # ... 原有 markdown 移除逻辑不变 ...

    # 增加括号平衡算法
    first_brace = text.find("{")
    if first_brace == -1:
        return text

    depth = 0
    for i in range(first_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[first_brace : i + 1]

    return text[first_brace:]
```

**Why**：GLM 可能在 JSON 后附加解释性文字，原实现用 `rfind("}")` 会匹配到最后一个 `}`，可能截取过多内容。括号平衡算法精确定位最外层闭合位置。

### 4.3 `mem0/memory/main.py` — 增强异常日志

**位置**：`_add_to_vector_store()` 中两处 JSON 解析 except 块

#### 第一次解析（facts 提取，第 531-533 行）

```python
# 原代码
except Exception as e:
    logger.error(f"Error in new_retrieved_facts: {e}")
    new_retrieved_facts = []

# 修改为
except Exception as e:
    logger.error(f"Error in new_retrieved_facts: {e}, response[:200]: {response[:200]}")
    new_retrieved_facts = []
```

#### 第二次解析（memory 操作，第 597-599 行）

```python
# 原代码
except Exception as e:
    logger.error(f"Invalid JSON response: {e}")
    new_memories_with_actions = {}

# 修改为
except Exception as e:
    logger.error(f"Invalid JSON response: {e}, response[:200]: {response[:200]}")
    new_memories_with_actions = {}
```

**Why**：排查 GLM 兼容性问题时，需要看到 LLM 实际返回了什么内容。截断到 200 字符避免日志爆炸。

## 5. 迁移步骤

| 步骤 | 操作 | 验证 |
|------|------|------|
| 1 | 从 `/Users/I0W02SJ/gitCode/mem0/` 复制核心目录到 `src/mem0/` | `import mem0` 不报错 |
| 2 | 修改 `pyproject.toml`：移除 mem0ai，提升依赖 | `uv sync` 成功 |
| 3 | 应用三个文件的代码修改 | 代码审查 |
| 4 | 运行全量测试 | 91 个测试通过 |
| 5 | E2E 验证 extract 功能 | 从对话中成功提取记忆 |

## 6. 风险与缓解

| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| mem0 源码复制不完整，缺少隐式依赖 | 低 | 逐步复制 + 测试驱动 |
| JSON 解析增强破坏其他 LLM 的兼容性 | 低 | `remove_code_blocks()` 放宽正则只增加匹配范围，不缩小 |
| 未来需要升级 mem0 版本 | 可接受 | 一次性锁定策略，需要时重新 fork |

## 7. 不做的事情

- 不精简 mem0 的 LLM provider（保留全部，减少风险）
- 不修改 mem0 的 prompt 模板（最小改动原则）
- 不修改 mem0 的向量存储、embedding 模块
- 不修改 `memory_platform` 业务层的 import（包名不变，路径自动解析）
