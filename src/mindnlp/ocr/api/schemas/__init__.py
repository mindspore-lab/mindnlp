"""
请求和响应模型
"""

from .request import OCRRequest, OCRBatchRequest, OCRURLRequest
from .response import OCRResponse, BatchOCRResponse, TextBlock, BoundingBox

__all__ = [
    'OCRRequest',
    'OCRBatchRequest',
    'OCRURLRequest',
    'OCRResponse',
    'BatchOCRResponse',
    'TextBlock',
    'BoundingBox'
]
