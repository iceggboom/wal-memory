"""记忆层级分类模块 — 4层记忆模型

核心职责：
1. 定义 4 层记忆层级枚举（L1-L4）
2. 基于关键词自动分类记忆文本
3. 解析层级过滤参数

4层记忆模型：
- L1 Profile: 角色、职业、学历等长期稳定属性（衰减最慢）
- L2 Preference: 偏好、风格、习惯等（衰减较慢）
- L3 Episodic: 具体经历、事件（衰减较快）
- L4 Relational: 团队、社交关系（衰减中等）

使用示例：
    >>> from memory_platform.ext.layer import classify_layer, MemoryLayer
    >>> layer = classify_layer("我是Python工程师")
    >>> layer
    <MemoryLayer.L1: 'L1'>
"""
import json
import logging
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mem0.llms.base import LLMBase

logger = logging.getLogger(__name__)


class MemoryLayer(str, Enum):
    """记忆层级枚举 — 定义记忆的衰减速率

    层级越高（L1 最稳定），衰减越慢，记忆保留时间越长。
    """

    L1 = "L1"  # Profile — 角色·职业·兴趣
    L2 = "L2"  # Preference — 风格·偏好
    L3 = "L3"  # Episodic — 具体经历
    L4 = "L4"  # Relational — 团队·社交


# 各层级的关键词映射表
# 用于基于文本内容自动分类记忆层级
LAYER_KEYWORDS: dict[MemoryLayer, list[str]] = {
    MemoryLayer.L1: [
        # 身份、职业相关
        "是", "担任", "职位", "工程师", "经理", "主管", "总监", "负责人",
        "专业", "岗位", "职业", "角色", "学历", "毕业",
    ],
    MemoryLayer.L2: [
        # 偏好、风格相关
        "喜欢", "偏好", "习惯", "风格", "倾向", "爱好", "兴趣",
        "注重", "追求", "擅长", "倾向于",
    ],
    MemoryLayer.L3: [
        # 时间、经历相关
        "上周", "昨天", "前天", "上次", "那次", "参加了", "去了",
        "完成了", "经历了", "最近", "之前", "刚", "曾", "已",
    ],
    MemoryLayer.L4: [
        # 团队、社交相关
        "同事", "领导", "团队", "同组", "项目组", "一起", "搭档", "伙伴",
        "上级", "下属", "合作", "组长", "班长", "直属", "同在",
    ],
}

# 所有层级的列表，用于默认过滤
ALL_LAYERS = list(MemoryLayer)


def classify_layer(text: str, explicit_layer: str | None = None) -> MemoryLayer:
    """根据文本内容自动分类记忆层级

    算法：
    1. 如果指定了 explicit_layer，直接返回该层级
    2. 遍历各层级关键词，统计命中次数
    3. 返回命中次数最多的层级
    4. 如果没有命中任何关键词，默认返回 L1

    Args:
        text: 记忆文本内容
        explicit_layer: 显式指定的层级，优先使用

    Returns:
        分类后的 MemoryLayer 枚举值

    Example:
        >>> classify_layer("我是Python工程师")
        <MemoryLayer.L1: 'L1'>
        >>> classify_layer("我喜欢喝咖啡")
        <MemoryLayer.L2: 'L2'>
        >>> classify_layer("上周参加了项目会议")
        <MemoryLayer.L3: 'L3'>
    """
    # 显式指定优先
    if explicit_layer:
        return MemoryLayer(explicit_layer)

    # 基于关键词匹配计分
    scores: dict[MemoryLayer, int] = {layer: 0 for layer in MemoryLayer}
    for layer, keywords in LAYER_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[layer] += 1

    # 返回最高分的层级
    best = max(scores, key=lambda layer: scores[layer])
    return best if scores[best] > 0 else MemoryLayer.L1


def parse_layer_filter(layer_str: str | None) -> list[MemoryLayer]:
    """解析层级过滤参数

    支持格式：
    - None 或空: 返回所有层级
    - "L1": 单个层级
    - "L1,L2,L3": 多个层级（逗号分隔）

    Args:
        layer_str: 层级过滤字符串

    Returns:
        MemoryLayer 列表

    Example:
        >>> parse_layer_filter(None)
        [<MemoryLayer.L1: 'L1'>, <MemoryLayer.L2: 'L2'>, ...]
        >>> parse_layer_filter("L1,L2")
        [<MemoryLayer.L1: 'L1'>, <MemoryLayer.L2: 'L2'>]
    """
    if not layer_str:
        return ALL_LAYERS
    return [MemoryLayer(layer.strip()) for layer in layer_str.split(",")]


# LLM 分类 Prompt 模板
LLM_CLASSIFY_PROMPT = """你是一个记忆分类专家。请将以下记忆文本分类到最合适的层级。

## 4层记忆模型

| 层级 | 含义 | 典型内容 |
|------|------|---------|
| L1 | Profile — 角色·职业·兴趣 | 职位、身份、学历、长期属性 |
| L2 | Preference — 风格·偏好 | 喜好、习惯、沟通风格 |
| L3 | Episodic — 具体经历 | 时间相关的事件、经历 |
| L4 | Relational — 团队·社交 | 同事关系、团队信息 |

## 记忆文本
{text}

## 输出要求
返回 JSON 格式：{{"layer": "L1|L2|L3|L4", "reason": "分类原因"}}

仅返回 JSON，不要有其他内容。"""


def _has_l1_keywords(text: str) -> bool:
    """检查文本是否包含 L1 关键词

    用于判断关键词匹配是否真正命中（而非默认返回 L1）
    """
    l1_keywords = LAYER_KEYWORDS.get(MemoryLayer.L1, [])
    return any(kw in text for kw in l1_keywords)


def classify_layer_with_llm(
    text: str,
    llm: "LLMBase",
    use_keyword_first: bool = True,
    explicit_layer: str | None = None,
) -> MemoryLayer:
    """使用 LLM 辅助分类记忆层级

    核心流程：
    1. 如果指定了 explicit_layer，直接返回
    2. 如果 use_keyword_first=True，先尝试关键词匹配
    3. 关键词未匹配或 use_keyword_first=False，调用 LLM
    4. LLM 失败时返回默认 L1

    Args:
        text: 记忆文本内容
        llm: mem0 LLM 实例（LLMBase 子类）
        use_keyword_first: 是否优先使用关键词匹配
        explicit_layer: 显式指定的层级

    Returns:
        MemoryLayer 枚举值

    Example:
        >>> from mem0.utils.factory import LlmFactory
        >>> llm = LlmFactory.create("wal", config={...})
        >>> classify_layer_with_llm("张三是工程师", llm)
        <MemoryLayer.L1: 'L1'>
    """
    # 显式指定优先
    if explicit_layer:
        return MemoryLayer(explicit_layer)

    # 关键词优先策略
    if use_keyword_first:
        keyword_result = classify_layer(text)
        # 关键词匹配成功（得分 > 0）直接返回
        if keyword_result != MemoryLayer.L1 or _has_l1_keywords(text):
            return keyword_result

    # 调用 LLM 分类
    try:
        response = llm.generate_response(
            messages=[
                {
                    "role": "user",
                    "content": LLM_CLASSIFY_PROMPT.format(text=text),
                }
            ],
        )

        # generate_response 返回字符串内容
        content = response.strip()
        result = json.loads(content)
        layer_str = result.get("layer", "L1")

        return MemoryLayer(layer_str)

    except json.JSONDecodeError as e:
        logger.warning("LLM classification JSON parse error: %s", e)
        return MemoryLayer.L1
    except Exception as e:
        logger.error("LLM classification failed: %s", e)
        return MemoryLayer.L1
