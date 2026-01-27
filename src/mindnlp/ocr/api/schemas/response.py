"""
API响应模型
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """边界框坐标"""
    x: int = Field(..., description="左上角X坐标")
    y: int = Field(..., description="左上角Y坐标")
    width: int = Field(..., description="宽度")
    height: int = Field(..., description="高度")


class TextBlock(BaseModel):
    """文本块"""
    text: str = Field(..., description="识别的文本内容")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    bounding_box: Optional[BoundingBox] = Field(None, description="边界框坐标")


class OCRResponse(BaseModel):
    """OCR响应模型"""

    success: bool = Field(..., description="是否成功")
    texts: List[str] = Field(default_factory=list, description="识别的文本列表")
    boxes: List[List[float]] = Field(default_factory=list, description="边界框坐标列表 [[x,y,w,h], ...]")
    confidences: List[float] = Field(default_factory=list, description="置信度列表")
    raw_output: str = Field(..., description="原始输出文本")
    inference_time: float = Field(..., description="推理时间 (秒)")
    model_name: str = Field(..., description="使用的模型名称")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")
    error: Optional[str] = Field(None, description="错误信息")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "texts": ["第一行文本", "第二行文本"],
                "boxes": [[10, 20, 200, 30], [10, 60, 180, 28]],
                "confidences": [0.95, 0.92],
                "raw_output": "第一行文本\\n第二行文本",
                "inference_time": 0.5,
                "model_name": "Qwen/Qwen2-VL-2B-Instruct",
                "metadata": {
                    "language": "zh",
                    "format": "json"
                }
            }
        }


class BatchOCRResponse(BaseModel):
    """批量OCR响应模型"""

    success: bool = Field(..., description="是否成功")
    results: List[OCRResponse] = Field(..., description="OCR结果列表")
    total_images: int = Field(..., description="总图像数量")
    total_time: float = Field(..., description="总处理时间 (秒)")
    model_name: str = Field(..., description="使用的模型名称")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "results": [
                    {
                        "success": True,
                        "texts": ["文本1"],
                        "boxes": [[10, 20, 200, 30]],
                        "confidences": [0.95],
                        "raw_output": "文本1",
                        "inference_time": 0.5,
                        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
                        "metadata": {}
                    }
                ],
                "total_images": 1,
                "total_time": 0.5,
                "model_name": "Qwen/Qwen2-VL-2B-Instruct"
            }
        }
