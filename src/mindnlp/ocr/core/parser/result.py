"""
结果解析器
解析模型输出的文本，提取结构化信息
"""

import json
import re
from typing import List, Dict, Any, Optional
from utils.logger import get_logger


logger = get_logger(__name__)


class ResultParser:
    """结果解析器"""
    
    def __init__(self):
        """初始化结果解析器"""
        logger.info("ResultParser initialized")
    
    def parse(
        self,
        text: str,
        output_format: str = "text",
        confidence_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        解析文本结果
        
        Args:
            text: 解码后的文本
            output_format: 输出格式
            confidence_threshold: 置信度阈值
            
        Returns:
            dict: 解析后的结构化结果
        """
        if output_format == "json":
            return self._parse_json_format(text, confidence_threshold)
        elif output_format == "markdown":
            return self._parse_markdown_format(text)
        else:
            return self._parse_text_format(text)
    
    def _parse_text_format(self, text: str) -> Dict[str, Any]:
        """解析纯文本格式"""
        return {
            'text': text.strip(),
            'blocks': None
        }
    
    def _parse_json_format(self, text: str, confidence_threshold: float) -> Dict[str, Any]:
        """
        解析JSON格式
        期望模型输出JSON格式的结构化数据
        """
        try:
            # 尝试从文本中提取JSON
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                # 解析文本块
                blocks = []
                if 'blocks' in data:
                    for block_data in data['blocks']:
                        if block_data.get('confidence', 1.0) >= confidence_threshold:
                            blocks.append({
                                'text': block_data.get('text', ''),
                                'confidence': block_data.get('confidence', 1.0),
                                'bounding_box': block_data.get('bounding_box')
                            })
                
                # 合并所有文本
                full_text = ' '.join([block['text'] for block in blocks])
                
                return {
                    'text': full_text,
                    'blocks': blocks
                }
            else:
                # 如果没有找到JSON，回退到文本格式
                logger.warning("No JSON found in output, falling back to text format")
                return self._parse_text_format(text)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return self._parse_text_format(text)
    
    def _parse_markdown_format(self, text: str) -> Dict[str, Any]:
        """解析Markdown格式"""
        return {
            'text': text.strip(),
            'blocks': None
        }
