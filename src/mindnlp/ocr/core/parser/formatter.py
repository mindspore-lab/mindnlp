"""
输出格式化器
将解析后的结果格式化为最终输出
"""

import re
from typing import Dict, Any
from mindnlp.ocr.utils.logger import get_logger


logger = get_logger(__name__)


class OutputFormatter:
    """输出格式化器"""

    def __init__(self):
        """初始化输出格式化器"""
        logger.info("OutputFormatter initialized")

    def format(self, parsed_result: Dict[str, Any], output_format: str) -> Dict[str, Any]:
        """
        格式化输出

        Args:
            parsed_result: 解析后的结果
            output_format: 输出格式 (text/json/markdown)

        Returns:
            dict: 格式化后的结果
        """
        logger.debug(f"Formatting output as {output_format}")

        if output_format == "json":
            return self._format_json(parsed_result)
        elif output_format == "markdown":
            return self._format_markdown(parsed_result)
        else:  # text
            return self._format_text(parsed_result)

    def _format_text(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化为纯文本

        输出格式：
        - 纯文本字符串
        - 按行分隔
        - 保持阅读顺序

        Args:
            parsed_result: 解析后的结果

        Returns:
            dict: 包含纯文本的字典
        """
        text = parsed_result.get('text', '')
        blocks = parsed_result.get('blocks')

        # 如果有结构化的块信息，按块组装文本
        if blocks and isinstance(blocks, list):
            lines = []
            for block in blocks:
                if isinstance(block, dict) and 'text' in block:
                    block_text = block['text'].strip()
                    if block_text:
                        lines.append(block_text)
            text = '\n'.join(lines) if lines else text

        return {
            'text': text.strip(),
            'blocks': None  # 纯文本模式不返回块信息
        }

    def _format_json(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化为JSON（结构化格式）

        参考 PaddleOCR 输出结构，包含：
        - 完整文本
        - 文本块列表
        - 每个块的置信度
        - 边界框坐标（如果有）
        - 元数据信息

        Args:
            parsed_result: 解析后的结果

        Returns:
            dict: 包含完整结构化信息的字典
        """
        text = parsed_result.get('text', '')
        blocks = parsed_result.get('blocks', [])

        # 确保 blocks 是列表格式
        if not isinstance(blocks, list):
            blocks = []

        # 构建完整的 JSON 结构
        formatted_blocks = []
        for idx, block in enumerate(blocks):
            if isinstance(block, dict):
                formatted_block = {
                    'id': idx,
                    'text': block.get('text', ''),
                    'confidence': block.get('confidence', 1.0),
                }

                # 添加边界框信息（如果有）
                if 'bounding_box' in block and block['bounding_box']:
                    formatted_block['bounding_box'] = block['bounding_box']

                # 添加其他元数据
                if 'type' in block:
                    formatted_block['type'] = block['type']

                formatted_blocks.append(formatted_block)

        return {
            'text': text.strip(),
            'blocks': formatted_blocks,
            'block_count': len(formatted_blocks),
            'has_coordinates': any('bounding_box' in b for b in formatted_blocks)
        }

    def _format_markdown(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化为Markdown

        功能：
        - 保持文档结构
        - 智能检测表格并格式化
        - 检测标题并添加层级
        - 保持段落分隔

        Args:
            parsed_result: 解析后的结果

        Returns:
            dict: 包含Markdown格式文本的字典
        """
        text = parsed_result.get('text', '')
        blocks = parsed_result.get('blocks')

        # 如果有结构化的块信息，智能格式化
        if blocks and isinstance(blocks, list):
            markdown_lines = []

            for block in blocks:
                if not isinstance(block, dict) or 'text' not in block:
                    continue

                block_text = block['text'].strip()
                if not block_text:
                    continue

                block_type = block.get('type', 'text')

                # 根据块类型格式化
                if block_type == 'title' or self._is_title(block_text):
                    # 标题格式化
                    level = block.get('level', 2)
                    markdown_lines.append(f"{'#' * level} {block_text}")
                    markdown_lines.append('')  # 空行

                elif block_type == 'table' or self._is_table(block_text):
                    # 表格格式化（简单处理）
                    markdown_lines.append(block_text)
                    markdown_lines.append('')

                elif block_type == 'list' or block_text.startswith(('-', '*', '•')):
                    # 列表项
                    if not block_text.startswith(('-', '*')):
                        block_text = f"- {block_text}"
                    markdown_lines.append(block_text)

                else:
                    # 普通段落
                    markdown_lines.append(block_text)
                    markdown_lines.append('')  # 段落间空行

            text = '\n'.join(markdown_lines).strip()
        else:
            # 如果没有块信息，对文本进行基本的Markdown格式化
            text = self._basic_markdown_format(text)

        return {
            'text': text,
            'blocks': None  # Markdown 模式返回格式化文本
        }

    def _is_title(self, text: str) -> bool:
        """
        判断文本是否可能是标题

        Args:
            text: 待判断文本

        Returns:
            bool: 是否是标题
        """
        # 简单启发式判断：短文本、全大写、数字开头等
        text = text.strip()
        if len(text) < 5:
            return False
        if len(text) > 100:
            return False

        # 检查是否以数字和点开头（如 "1. Introduction"）
        if re.match(r'^\d+\.\s+', text):
            return True

        # 检查是否大部分字母是大写
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if upper_ratio > 0.7:
                return True

        return False

    def _is_table(self, text: str) -> bool:
        """
        判断文本是否可能是表格

        Args:
            text: 待判断文本

        Returns:
            bool: 是否是表格
        """
        # 检查是否包含表格分隔符
        if '|' in text and text.count('|') >= 2:
            return True
        if '\t' in text and text.count('\t') >= 2:
            return True
        return False

    def _basic_markdown_format(self, text: str) -> str:
        """
        对文本进行基本的Markdown格式化

        Args:
            text: 原始文本

        Returns:
            str: 格式化后的Markdown文本
        """
        lines = text.split('\n')
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue

            # 检测并格式化标题
            if self._is_title(line):
                formatted_lines.append(f"## {line}")
                formatted_lines.append('')
            else:
                formatted_lines.append(line)

        return '\n'.join(formatted_lines).strip()
