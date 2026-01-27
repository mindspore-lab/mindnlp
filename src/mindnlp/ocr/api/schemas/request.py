"""
API请求模型
"""

from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl


class OCRRequest(BaseModel):
    """OCR请求模型"""

    image: bytes = Field(..., description="图像数据 (bytes)")
    output_format: str = Field("text", description="输出格式: text/json/markdown")
    language: str = Field("auto", description="语言设置: auto/zh/en/ja/ko")
    task_type: str = Field("general", description="任务类型: general/document/table/formula")
    confidence_threshold: float = Field(0.0, ge=0.0, le=1.0, description="置信度阈值")
    custom_prompt: Optional[str] = Field(None, description="自定义Prompt")
    model: Optional[str] = Field(None, description="指定模型: 2b/7b 或完整模型名称 (留空使用默认)")

    class Config:
        json_schema_extra = {
            "example": {
                "output_format": "text",
                "language": "auto",
                "task_type": "general",
                "confidence_threshold": 0.5,
                "model": "2b"
            }
        }


class OCRBatchRequest(BaseModel):
    """批量OCR请求模型"""

    images: List[bytes] = Field(..., description="图像数据列表")
    output_format: str = Field("text", description="输出格式: text/json/markdown")
    language: str = Field("auto", description="语言设置: auto/zh/en/ja/ko")
    task_type: str = Field("general", description="任务类型: general/document/table/formula")
    confidence_threshold: float = Field(0.0, ge=0.0, le=1.0, description="置信度阈值")
    custom_prompt: Optional[str] = Field(None, description="自定义Prompt")
    model: Optional[str] = Field(None, description="指定模型: 2b/7b 或完整模型名称 (留空使用默认)")


class OCRURLRequest(BaseModel):
    """URL OCR请求模型"""

    image_url: HttpUrl = Field(..., description="图像URL")
    output_format: str = Field("text", description="输出格式: text/json/markdown")
    language: str = Field("auto", description="语言设置: auto/zh/en/ja/ko")
    task_type: str = Field("general", description="任务类型: general/document/table/formula")
    confidence_threshold: float = Field(0.0, ge=0.0, le=1.0, description="置信度阈值")
    custom_prompt: Optional[str] = Field(None, description="自定义Prompt")

    class Config:
        json_schema_extra = {
            "example": {
                "image_url": "https://example.com/image.jpg",
                "output_format": "json",
                "language": "zh",
                "task_type": "document"
            }
        }
