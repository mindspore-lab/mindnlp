"""
Mock OCR引擎 - 用于API测试
"""

from typing import Dict, List, Any, Optional, Union
try:
    from PIL import Image
    import numpy as np
except ImportError:
    # 如果没有这些库，使用Any作为类型
    Image = Any
    np = Any


class MockVLMOCREngine:
    """模拟的VLM OCR引擎（用于测试API层）"""
    
    def __init__(self, model_name: str = "qwen2-vl", device: str = "cpu"):
        """初始化mock引擎"""
        self.model_name = model_name
        self.device = device
        self.initialized = True
    
    def predict(
        self,
        image: Union[Image.Image, np.ndarray, str],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        模拟OCR识别
        
        Args:
            image: 输入图像
            options: 识别选项
            
        Returns:
            识别结果
        """
        # 返回模拟数据（符合OCRResponse格式）
        return {
            "success": True,
            "texts": [
                "这是第一行文本",
                "这是第二行文本",
                "Mock OCR Result"
            ],
            "boxes": [
                [10, 10, 190, 20],
                [10, 40, 190, 20],
                [10, 70, 190, 20]
            ],
            "confidences": [0.95, 0.92, 0.98],
            "raw_output": "这是第一行文本\n这是第二行文本\nMock OCR Result",
            "inference_time": 0.1,
            "model_name": self.model_name,
            "metadata": {
                "language": options.get("lang", "ch") if options else "ch",
                "image_size": "100x100",
                "format": options.get("format", "text") if options else "text"
            },
            "error": None
        }
    
    def predict_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, str]],
        options: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        批量模拟OCR识别
        
        Args:
            images: 输入图像列表
            options: 识别选项
            
        Returns:
            识别结果列表
        """
        return [self.predict(img, options) for img in images]
    
    def is_initialized(self) -> bool:
        """检查引擎是否初始化"""
        return self.initialized
