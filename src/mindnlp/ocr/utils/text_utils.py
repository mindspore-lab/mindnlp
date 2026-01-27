"""
文本工具函数
"""

import re
import json
from typing import Optional, Dict, Any
from .logger import get_logger


logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """
    清理文本

    Args:
        text: 输入文本

    Returns:
        str: 清理后的文本
    """
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)

    # 移除首尾空白
    text = text.strip()

    return text


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    从文本中提取JSON

    Args:
        text: 输入文本

    Returns:
        dict或None: 提取的JSON对象
    """
    try:
        # 尝试直接解析
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取JSON块
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)

    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    logger.warning("Failed to extract JSON from text")
    return None


def format_text_blocks(blocks: list, separator: str = '\n') -> str:
    """
    格式化文本块

    Args:
        blocks: 文本块列表
        separator: 分隔符

    Returns:
        str: 格式化后的文本
    """
    texts = [block.get('text', '') for block in blocks if block.get('text')]
    return separator.join(texts)


def truncate_text(text: str, max_length: int = 1000, suffix: str = '...') -> str:
    """
    截断文本

    Args:
        text: 输入文本
        max_length: 最大长度
        suffix: 后缀

    Returns:
        str: 截断后的文本
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix
