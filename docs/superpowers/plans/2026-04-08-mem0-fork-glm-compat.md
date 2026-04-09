# mem0 Fork — GLM 兼容性修复 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 mem0 从 pip SDK 依赖改为项目内嵌 fork，修复 GLM-5-Turbo 的 JSON 解析和 Anthropic 适配问题。

**Architecture:** 从 `/Users/I0W02SJ/gitCode/mem0/` 复制 mem0 核心源码到 `src/mem0/`，修改 3 个文件（anthropic.py / utils.py / main.py），移除 `mem0ai` pip 依赖并将外部依赖提升到 `wal-memory` 的 `pyproject.toml`。

**Tech Stack:** Python 3.12+, mem0 1.0.10 (fork), FastAPI, GLM-5-Turbo via Anthropic protocol

---

## File Structure

| 操作 | 文件路径 | 职责 |
|------|---------|------|
| Create（整目录） | `src/mem0/` | mem0 1.0.10 fork 源码 |
| Modify | `src/mem0/__init__.py` | 修复 `importlib.metadata` 对本地包的兼容性 |
| Modify | `src/mem0/llms/anthropic.py:41` | 传递 `anthropic_base_url` 给 Anthropic 客户端 |
| Modify | `src/mem0/memory/utils.py:109-142` | 增强 `remove_code_blocks()` 和 `extract_json()` |
| Modify | `src/mem0/memory/main.py:531,598` | 增强 JSON 解析异常日志 |
| Modify | `pyproject.toml` | 移除 `mem0ai`，提升 mem0 外部依赖 |
| Create | `tests/unit/test_mem0_json_parsing.py` | 测试 JSON 解析增强 |

---

### Task 1: 复制 mem0 源码到项目

**Files:**
- Create: `src/mem0/` 整个目录（从 `/Users/I0W02SJ/gitCode/mem0/mem0/` 复制）

- [ ] **Step 1: 复制 mem0 核心目录**

```bash
# 从已有的 mem0 本地仓库复制核心源码
cp -r /Users/I0W02SJ/gitCode/mem0/mem0/ /Users/I0W02SJ/gitCode/wal-memory/src/mem0/

# 清除 __pycache__
find /Users/I0W02SJ/gitCode/wal-memory/src/mem0/ -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
```

- [ ] **Step 2: 修复 `__init__.py` 中的版本获取方式**

本地包无法通过 `importlib.metadata.version("mem0ai")` 获取版本。修改 `src/mem0/__init__.py`：

```python
try:
    import importlib.metadata
    __version__ = importlib.metadata.version("mem0ai")
except importlib.metadata.PackageNotFoundError:
    __version__ = "1.0.10-fork"

from mem0.client.main import AsyncMemoryClient, MemoryClient  # noqa
from mem0.memory.main import AsyncMemory, Memory  # noqa
```

- [ ] **Step 3: 修改 `pyproject.toml` — 移除 mem0ai，提升依赖，添加 src/mem0 到包路径**

```toml
[project]
name = "memory-platform"
version = "0.1.0"
description = "AI Memory Platform based on mem0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    # mem0 fork — 外部依赖提升为直接依赖
    "qdrant-client>=1.9.1",
    "openai>=1.90.0",
    "posthog>=3.5.0",
    "pytz>=2024.1",
    "sqlalchemy>=2.0.31",
    "protobuf>=5.29.6,<7.0.0",
    # mem0 传递依赖
    "httpx>=0.28",
    "pydantic>=2.10",
    "pydantic-settings>=2.7",
    "anthropic>=0.89.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.25",
    "ruff>=0.9",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/memory_platform", "src/mem0"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py312"
line-length = 100
```

- [ ] **Step 4: 安装依赖并验证 import**

```bash
uv sync
uv run python -c "from mem0 import Memory; print('mem0 import OK')"
uv run python -c "from memory_platform.config import build_mem0_config; print('memory_platform import OK')"
```

Expected: 两个 import 都成功，无报错。

- [ ] **Step 5: 运行全量测试确认无回归**

```bash
uv run pytest tests/ -q
```

Expected: 91 passed, 4 skipped（与 fork 前一致）

- [ ] **Step 6: 提交**

```bash
git add src/mem0/ pyproject.toml uv.lock
git commit -m "chore: embed mem0 1.0.10 fork into project, replace pip dependency"
```

---

### Task 2: 修复 Anthropic LLM 客户端 — 传递 base_url

**Files:**
- Modify: `src/mem0/llms/anthropic.py:41`
- Test: `tests/unit/test_mem0_json_parsing.py`（验证配置传递）

- [ ] **Step 1: 修改 `src/mem0/llms/anthropic.py` 第 40-42 行**

将：
```python
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)
```

改为：
```python
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        base_url = self.config.anthropic_base_url or os.getenv("ANTHROPIC_BASE_URL")
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**client_kwargs)
```

- [ ] **Step 2: 验证修改**

```bash
uv run python -c "
import os
os.environ['ANTHROPIC_API_KEY'] = 'test'
os.environ['ANTHROPIC_BASE_URL'] = 'https://api.z.ai/api/anthropic'
from mem0.llms.anthropic import AnthropicLLM
llm = AnthropicLLM()
print('base_url:', llm.client.base_url)
assert 'z.ai' in str(llm.client.base_url), 'base_url not set'
print('PASS: base_url correctly passed to Anthropic client')
"
```

Expected: `base_url` 包含 `z.ai`，断言通过。

- [ ] **Step 3: 运行全量测试**

```bash
uv run pytest tests/ -q
```

Expected: 91 passed, 4 skipped

- [ ] **Step 4: 提交**

```bash
git add src/mem0/llms/anthropic.py
git commit -m "fix: pass anthropic_base_url to Anthropic client in mem0 fork"
```

---

### Task 3: 增强 JSON 解析 — remove_code_blocks 和 extract_json

**Files:**
- Modify: `src/mem0/memory/utils.py:109-142`
- Create: `tests/unit/test_mem0_json_parsing.py`

- [ ] **Step 1: 编写 JSON 解析测试**

创建 `tests/unit/test_mem0_json_parsing.py`：

```python
"""测试 mem0 fork 的 JSON 解析增强 — 验证 GLM 兼容性"""
import pytest

from mem0.memory.utils import remove_code_blocks, extract_json


class TestRemoveCodeBlocks:
    """测试 remove_code_blocks 对各种 LLM 输出格式的处理"""

    def test_standard_json_code_block(self):
        """标准 markdown JSON 代码块"""
        content = '```json\n{"facts": ["a", "b"]}\n```'
        result = remove_code_blocks(content)
        assert result == '{"facts": ["a", "b"]}'

    def test_json_code_block_no_language(self):
        """无语言标记的代码块"""
        content = '```\n{"facts": ["a"]}\n```'
        result = remove_code_blocks(content)
        assert result == '{"facts": ["a"]}'

    def test_leading_whitespace(self):
        """开头有空白"""
        content = '  ```json\n{"facts": ["a"]}\n```'
        result = remove_code_blocks(content)
        assert result == '{"facts": ["a"]}'

    def test_carriage_return(self):
        """Windows 换行 \\r\\n"""
        content = '```json\r\n{"facts": ["a"]}\r\n```'
        result = remove_code_blocks(content)
        assert result == '{"facts": ["a"]}'

    def test_no_code_block(self):
        """无代码块包裹 — 纯 JSON"""
        content = '{"facts": ["a", "b"]}'
        result = remove_code_blocks(content)
        assert result == '{"facts": ["a", "b"]}'

    def test_trailing_text_after_code_block(self):
        """代码块后有尾部文字（GLM 有时在 JSON 后添加解释）"""
        content = '```json\n{"facts": ["a"]}\n```\n以上是提取的记忆。'
        result = remove_code_blocks(content)
        assert '{"facts": ["a"]}' in result

    def test_think_tags_removed(self):
        """移除 <think ...> 标签"""
        content = '{"facts": ["a"]}'
        result = remove_code_blocks(content)
        assert "<think" not in result


class TestExtractJson:
    """测试 extract_json 的括号平衡算法"""

    def test_pure_json(self):
        """纯 JSON 输入"""
        result = extract_json('{"facts": ["a", "b"]}')
        assert '"facts"' in result

    def test_markdown_wrapped_json(self):
        """markdown 包裹的 JSON"""
        result = extract_json('```json\n{"facts": ["a"]}\n```')
        assert '"facts"' in result

    def test_json_with_trailing_text(self):
        """JSON 后有附加文字 — 括号平衡提取"""
        result = extract_json('{"facts": ["a"]}\n这是提取的记忆。')
        assert result == '{"facts": ["a"]}'

    def test_nested_json(self):
        """嵌套 JSON — 括号平衡"""
        result = extract_json('{"memory": [{"text": "a", "event": "ADD"}]}')
        parsed = __import__("json").loads(result)
        assert len(parsed["memory"]) == 1

    def test_json_with_nested_braces_in_values(self):
        """值中包含嵌套花括号"""
        result = extract_json('{"data": {"inner": "val"}, "key": "value"}')
        parsed = __import__("json").loads(result)
        assert parsed["data"]["inner"] == "val"

    def test_no_json_returns_as_is(self):
        """无 JSON 内容 — 返回原文"""
        result = extract_json("no json here")
        assert result == "no json here"

    def test_glm_real_output(self):
        """GLM 真实输出格式"""
        glm_output = '```json\n{\n  "facts": ["用户刚从北京搬到上海", "用户在一家AI创业公司做全栈开发"]\n}\n```'
        result = extract_json(glm_output)
        parsed = __import__("json").loads(result)
        assert len(parsed["facts"]) == 2
        assert "北京" in parsed["facts"][0]
```

- [ ] **Step 2: 运行测试确认部分失败（TDD 红灯）**

```bash
uv run pytest tests/unit/test_mem0_json_parsing.py -v
```

Expected: 大部分测试 PASS（因为现有实现已能处理部分场景），`test_json_with_trailing_text` 和 `test_trailing_text_after_code_block` 可能 FAIL。

- [ ] **Step 3: 修改 `src/mem0/memory/utils.py` — 增强 `remove_code_blocks()`**

将第 118 行的正则表达式：

```
pattern = r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$"
```

改为：

```
pattern = r"^\s*```[a-zA-Z0-9]*\r?\n([\s\S]*?)\r?\n\s*```"
```

同时在第 120 行将 `match_res=match.group(1)...` 中的 `=` 后加空格为 `match_res = match.group(1)...`。

变更点：
- 正则前加 `\s*` — 允许开头空白
- `\n` 改为 `\r?\n` — 兼容 Windows 换行
- 移除末尾 `$` — 允许 JSON 后有尾部字符

- [ ] **Step 4: 修改 `src/mem0/memory/utils.py` — 增强 `extract_json()` 括号平衡**

将第 125-142 行：
```python
def extract_json(text):
    """
    Extracts JSON content from a string, removing enclosing triple backticks and optional 'json' tag if present.
    If no code block is found, attempts to locate JSON by finding the first '{' and last '}'.
    If that also fails, returns the text as-is.
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx : end_idx + 1]
        else:
            json_str = text
    return json_str
```

改为：
```python
def extract_json(text):
    """
    Extracts JSON content from a string, removing enclosing triple backticks and optional 'json' tag if present.
    If no code block is found, uses bracket-balanced extraction to find the outermost JSON object.
    If that also fails, returns the text as-is.
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        first_brace = text.find("{")
        if first_brace == -1:
            return text
        # Bracket-balanced extraction for accurate JSON boundary
        depth = 0
        in_string = False
        escape = False
        for i in range(first_brace, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[first_brace : i + 1]
        return text[first_brace:]
    return json_str
```

- [ ] **Step 5: 运行 JSON 解析测试**

```bash
uv run pytest tests/unit/test_mem0_json_parsing.py -v
```

Expected: 全部 PASS

- [ ] **Step 6: 运行全量测试确认无回归**

```bash
uv run pytest tests/ -q
```

Expected: 91+ passed, 4 skipped

- [ ] **Step 7: 提交**

```bash
git add src/mem0/memory/utils.py tests/unit/test_mem0_json_parsing.py
git commit -m "fix: enhance JSON parsing in mem0 fork for GLM compatibility"
```

---

### Task 4: 增强异常日志

**Files:**
- Modify: `src/mem0/memory/main.py:531,598`

- [ ] **Step 1: 修改第一次 JSON 解析的 except 块（第 531-533 行）**

将：
```python
        except Exception as e:
            logger.error(f"Error in new_retrieved_facts: {e}")
            new_retrieved_facts = []
```

改为：
```python
        except Exception as e:
            logger.error(f"Error in new_retrieved_facts: {e}, response[:200]: {response[:200]}")
            new_retrieved_facts = []
```

- [ ] **Step 2: 修改第二次 JSON 解析的 except 块（第 597-599 行）**

将：
```python
            except Exception as e:
                logger.error(f"Invalid JSON response: {e}")
                new_memories_with_actions = {}
```

改为：
```python
            except Exception as e:
                logger.error(f"Invalid JSON response: {e}, response[:200]: {response[:200]}")
                new_memories_with_actions = {}
```

- [ ] **Step 3: 运行全量测试**

```bash
uv run pytest tests/ -q
```

Expected: 91+ passed, 4 skipped

- [ ] **Step 4: 提交**

```bash
git add src/mem0/memory/main.py
git commit -m "fix: enhance error logging in mem0 extract JSON parsing"
```

---

### Task 5: E2E 验证

**Files:** 无代码变更，纯验证

- [ ] **Step 1: 启动服务**

```bash
pkill -f "uvicorn memory_platform" 2>/dev/null || true
sleep 1
rm -rf /tmp/qdrant
uv run python -m uvicorn memory_platform.main:app --host 0.0.0.0 --port 8000 &
sleep 3
curl -s http://localhost:8000/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 2: 验证 extract 从对话中成功提取记忆**

```bash
curl -s -X POST http://localhost:8000/v1/memories/extract \
  -H "Authorization: Bearer test-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id":"e2e-test-user","app_id":"test-app",
    "messages":[
      {"role":"user","content":"我刚从北京搬到上海，在一家AI创业公司做全栈开发"},
      {"role":"assistant","content":"好的，记下了"},
      {"role":"user","content":"主要用React和FastAPI"}
    ]
  }'
```

Expected: 返回 `"added": N`（N > 0），且 `memories` 列表非空。

- [ ] **Step 3: 验证搜索召回**

```bash
curl -s -X POST http://localhost:8000/v1/memories/search \
  -H "Authorization: Bearer test-key-123" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"e2e-test-user","app_id":"test-app","query":"全栈开发","scope":"all"}'
```

Expected: 返回结果包含提取的记忆。

- [ ] **Step 4: 停止服务**

```bash
pkill -f "uvicorn memory_platform" 2>/dev/null || true
```

---

### Task 6: 最终提交与清理

- [ ] **Step 1: 确认全量测试通过**

```bash
uv run pytest tests/ -q
```

Expected: 所有测试通过

- [ ] **Step 2: 最终提交（如有未提交的变更）**

```bash
git status
# 如果有未提交的变更，提交
git add -A
git commit -m "chore: finalize mem0 fork integration"
```
