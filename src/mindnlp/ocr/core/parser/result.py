"""
结果解析器
解析模型输出的文本，提取结构化信息
"""

import json
import re
from typing import Dict, Any
from mindnlp.ocr.utils.logger import get_logger


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
        
        增强的JSON解析，支持：
        - 多种JSON格式（包裹在代码块中、纯JSON等）
        - 更好的结构检测
        - 自动提取文本块信息
        - 置信度过滤
        
        Args:
            text: 模型输出文本
            confidence_threshold: 置信度阈值
            
        Returns:
            dict: 包含文本和块信息的字典
        """
        try:
            # 尝试多种方式提取JSON
            json_str = self._extract_json_string(text)
            
            if json_str:
                data = json.loads(json_str)
                
                # 解析文本块
                blocks = []
                if 'blocks' in data and isinstance(data['blocks'], list):
                    for idx, block_data in enumerate(data['blocks']):
                        if not isinstance(block_data, dict):
                            continue
                            
                        confidence = block_data.get('confidence', 1.0)
                        
                        # 过滤低置信度块
                        if confidence >= confidence_threshold:
                            block = {
                                'id': idx,
                                'text': block_data.get('text', ''),
                                'confidence': confidence
                            }
                            
                            # 添加可选字段
                            if 'bounding_box' in block_data:
                                block['bounding_box'] = block_data['bounding_box']
                            if 'type' in block_data:
                                block['type'] = block_data['type']
                            if 'level' in block_data:
                                block['level'] = block_data['level']
                                
                            blocks.append(block)
                
                # 合并所有文本，如果没有blocks则使用text字段
                if blocks:
                    full_text = '\n'.join([block['text'] for block in blocks if block['text']])
                else:
                    full_text = data.get('text', text)
                
                return {
                    'text': full_text.strip(),
                    'blocks': blocks if blocks else None
                }
            else:
                # 如果没有找到JSON，尝试结构化解析文本
                logger.warning("No JSON found in output, attempting structured text parsing")
                return self._parse_structured_text(text)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return self._parse_structured_text(text)
        except Exception as e:
            logger.error(f"Unexpected error during JSON parsing: {e}")
            return self._parse_text_format(text)
    
    def _extract_json_string(self, text: str) -> str:
        """
        从文本中提取JSON字符串
        
        支持多种格式：
        - 纯JSON: {"key": "value"}
        - Markdown代码块: ```json {...} ```
        - 包裹在文本中的JSON
        
        Args:
            text: 原始文本
            
        Returns:
            str: 提取的JSON字符串，如果未找到返回空字符串
        """
        text = text.strip()
        
        # 1. 检查是否在markdown代码块中
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1)
        
        # 2. 尝试匹配完整的JSON对象（贪婪模式）
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        # 3. 尝试匹配JSON数组
        array_match = re.search(r'\[.*\]', text, re.DOTALL)
        if array_match:
            return array_match.group(0)
        
        return ""
    
    def _parse_structured_text(self, text: str) -> Dict[str, Any]:
        """
        解析结构化文本
        
        当无法解析JSON时，尝试从文本中提取结构信息：
        - 按行分割
        - 检测可能的文本块
        - 估计置信度
        
        Args:
            text: 原始文本
            
        Returns:
            dict: 包含文本和可能的块信息
        """
        text = text.strip()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # 如果只有少量行，直接返回文本
        if len(lines) <= 1:
            return self._parse_text_format(text)
        
        # 尝试将每行作为一个块
        blocks = []
        for idx, line in enumerate(lines):
            blocks.append({
                'id': idx,
                'text': line,
                'confidence': 1.0  # 默认置信度
            })
        
        return {
            'text': '\n'.join(lines),
            'blocks': blocks
        }

    def _parse_markdown_format(self, text: str) -> Dict[str, Any]:
        """
        解析Markdown格式
        
        增强的Markdown解析，支持：
        - 检测标题层级
        - 识别列表项
        - 检测表格
        - 提取代码块
        
        Args:
            text: 原始文本
            
        Returns:
            dict: 包含文本和结构化块信息
        """
        text = text.strip()
        lines = text.split('\n')
        
        blocks = []
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            block = {
                'id': len(blocks),
                'text': line,
                'confidence': 1.0
            }
            
            # 检测标题
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                level = len(heading_match.group(1))
                block['type'] = 'title'
                block['level'] = level
                block['text'] = heading_match.group(2)
            
            # 检测列表
            elif re.match(r'^[-*+]\s+', line) or re.match(r'^\d+\.\s+', line):
                block['type'] = 'list'
            
            # 检测代码块标记
            elif line.startswith('```'):
                block['type'] = 'code_fence'
            
            # 检测表格行
            elif '|' in line and line.count('|') >= 2:
                block['type'] = 'table'
            
            else:
                block['type'] = 'text'
            
            blocks.append(block)
        
        return {
            'text': text,
            'blocks': blocks if blocks else None
        }
