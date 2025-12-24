"""
图像预处理器
负责图像加载、增强、归一化等预处理操作
"""

import io
import numpy as np
from PIL import Image
from typing import Union
from utils.logger import get_logger


logger = get_logger(__name__)


class ImageProcessor:
    """图像预处理器"""
    
    def __init__(self, target_size: tuple = (448, 448)):
        """
        初始化图像处理器
        
        Args:
            target_size: 目标图像尺寸 (height, width)
        """
        self.target_size = target_size
        logger.info(f"ImageProcessor initialized with target size: {target_size}")
    
    def process(self, image_data: Union[bytes, str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        处理图像
        
        Args:
            image_data: 图像数据 (bytes/路径/数组/PIL Image)
            
        Returns:
            np.ndarray: 处理后的图像数组
        """
        # 1. 加载图像
        image = self._load_image(image_data)
        
        # 2. 图像增强 (可选)
        image = self._enhance_image(image)
        
        # 3. 尺寸归一化
        image = self._resize_with_padding(image)
        
        # 4. 数值归一化
        image = self._normalize(image)
        
        return image
    
    def _load_image(self, image_data: Union[bytes, str, np.ndarray, Image.Image]) -> Image.Image:
        """加载图像"""
        if isinstance(image_data, bytes):
            # 从bytes加载
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, str):
            # 从文件路径加载
            image = Image.open(image_data)
        elif isinstance(image_data, np.ndarray):
            # 从numpy数组加载
            image = Image.fromarray(image_data)
        elif isinstance(image_data, Image.Image):
            # 已经是PIL Image
            image = image_data
        else:
            raise ValueError(f"Unsupported image data type: {type(image_data)}")
        
        # 转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.debug(f"Image loaded: size={image.size}, mode={image.mode}")
        return image
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """
        图像增强 (可选)
        包括去噪、对比度调整等
        """
        # TODO: 实现图像增强功能
        # - 去噪处理
        # - 对比度调整
        # - 倾斜校正
        return image
    
    def _resize_with_padding(self, image: Image.Image) -> Image.Image:
        """
        智能缩放并padding到目标尺寸
        保持宽高比
        """
        original_width, original_height = image.size
        target_height, target_width = self.target_size
        
        # 计算缩放比例 (保持宽高比)
        scale = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # 缩放图像
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 创建目标尺寸的画布 (黑色背景)
        canvas = Image.new('RGB', (target_width, target_height), (0, 0, 0))
        
        # 计算居中位置
        offset_x = (target_width - new_width) // 2
        offset_y = (target_height - new_height) // 2
        
        # 粘贴缩放后的图像到画布中心
        canvas.paste(image, (offset_x, offset_y))
        
        logger.debug(f"Image resized: {original_width}x{original_height} -> {target_width}x{target_height}")
        return canvas
    
    def _normalize(self, image: Image.Image) -> np.ndarray:
        """
        数值归一化
        [0, 255] -> [0, 1] -> 标准化
        """
        # 转换为numpy数组
        image_array = np.array(image).astype(np.float32)
        
        # 归一化到 [0, 1]
        image_array = image_array / 255.0
        
        # 标准化 (使用ImageNet统计值)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        # 转换维度 [H, W, C] -> [C, H, W]
        image_array = np.transpose(image_array, (2, 0, 1))
        
        logger.debug(f"Image normalized: shape={image_array.shape}, dtype={image_array.dtype}")
        return image_array
