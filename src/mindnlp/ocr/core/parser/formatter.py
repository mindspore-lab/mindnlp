"""
输出格式化器
将解析后的结果格式化为最终输出
"""

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
            output_format: 输出格式

        Returns:
            dict: 格式化后的结果
        """
        if output_format == "json":
            return self._format_json(parsed_result)
        elif output_format == "markdown":
            return self._format_markdown(parsed_result)
        else:
            return self._format_text(parsed_result)

    def _format_text(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化为纯文本"""
        return {
            'text': parsed_result['text'],
            'blocks': None
        }

    def _format_json(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化为JSON"""
        return {
            'text': parsed_result['text'],
            'blocks': parsed_result.get('blocks')
        }

    def _format_markdown(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化为Markdown"""
        return {
            'text': parsed_result['text'],
            'blocks': None
        }
