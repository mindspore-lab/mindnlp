"""
批处理整理器
负责批量数据的整理和padding
"""

from typing import List
import numpy as np
from utils.logger import get_logger


logger = get_logger(__name__)


class BatchCollator:
    """批处理整理器"""
    
    def __init__(self):
        """初始化批处理整理器"""
        logger.info("BatchCollator initialized")
    
    def collate(self, images: List[np.ndarray], prompts: List[str]) -> dict:
        """
        整理批量数据
        
        Args:
            images: 图像列表
            prompts: Prompt列表
            
        Returns:
            dict: 整理后的批量数据
        """
        # 将图像堆叠成批量 [B, C, H, W]
        batch_images = np.stack(images, axis=0)
        
        logger.debug(f"Collated batch: {len(images)} images, shape={batch_images.shape}")
        
        return {
            'images': batch_images,
            'prompts': prompts
        }
