"""
MindNLP OCR Module

基于 Vision-Language Model 的 OCR 模块，提供文字识别、文档理解等功能。
"""

__version__ = "0.1.0"
__author__ = "MindNLP Team"

from .core.engine import VLMOCREngine
from .core.mock_engine import MockVLMOCREngine

__all__ = [
    "VLMOCREngine",
    "MockVLMOCREngine",
]
