"""
图像预处理器
负责图像加载、增强、归一化等预处理操作
"""

import io
from typing import Union, Dict, Tuple, Any
import numpy as np
import torch
from PIL import Image
from utils.logger import get_logger


logger = get_logger(__name__)


class ImageProcessor:
    """
    图像预处理器

    功能:
    1. 支持多种图像输入格式 (bytes/PIL/numpy/路径)
    2. 智能缩放 (保持宽高比)
    3. 自适应 Padding (填充至目标尺寸)
    4. 数值归一化和标准化
    5. Tensor 转换
    6. 记录变换信息 (用于坐标还原)
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (448, 448),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        初始化图像处理器

        Args:
            target_size: 目标图像尺寸 (width, height)
            mean: 归一化均值 (R, G, B)
            std: 归一化标准差 (R, G, B)
        """
        self.target_size = target_size
        # pylint: disable=too-many-function-args
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        logger.info(f"ImageProcessor initialized: target_size={target_size}, mean={mean}, std={std}")

    def process(
        self,
        image_data: Union[bytes, str, np.ndarray, Image.Image]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        处理图像并返回 Tensor 和变换信息

        Args:
            image_data: 图像数据 (bytes/路径/数组/PIL Image)

        Returns:
            pixel_values: 处理后的 Tensor [1, 3, H, W]
            transform_info: 变换信息字典，包含:
                - original_size: 原始尺寸 (width, height)
                - resized_size: 缩放后尺寸 (width, height)
                - target_size: 目标尺寸 (width, height)
                - scale: 缩放比例
                - padding: Padding 信息 {left, top, right, bottom}
                - offset: 图像在画布上的偏移 (x, y)
        """
        try:
            # 1. 加载图像
            image = self._load_image(image_data)
            original_size = image.size  # (width, height)

            # 2. 智能缩放和 Padding
            resized_image, padding_info = self._resize_with_padding(image)

            # 3. 转换为 NumPy 数组
            image_array = np.array(resized_image, dtype=np.float32)

            # 4. 归一化 [0, 255] -> [0, 1]
            image_array = image_array / 255.0

            # 5. 转换维度 HWC -> CHW
            image_array = np.transpose(image_array, (2, 0, 1))

            # 6. 标准化
            image_array = (image_array - self.mean) / self.std

            # 7. 转换为 Tensor 并添加 batch 维度
            pixel_values = torch.from_numpy(image_array).unsqueeze(0)

            # 8. 构建变换信息
            transform_info = {
                'original_size': original_size,
                'resized_size': padding_info['resized_size'],
                'target_size': self.target_size,
                'scale': padding_info['scale'],
                'padding': padding_info['padding'],
                'offset': padding_info['offset']
            }

            logger.debug(f"Image processed: {original_size} -> {self.target_size}, scale={transform_info['scale']:.4f}")

            return pixel_values, transform_info

        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            raise

    def _load_image(self, image_data: Union[bytes, str, np.ndarray, Image.Image]) -> Image.Image:
        """
        加载图像并转换为 RGB 格式

        Args:
            image_data: 图像数据

        Returns:
            Image.Image: RGB 格式的 PIL Image

        Raises:
            ValueError: 不支持的图像类型
            IOError: 图像加载失败
        """
        try:
            if isinstance(image_data, bytes):
                # 从 bytes 加载
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, str):
                # 从文件路径加载
                import os
                if not os.path.exists(image_data):
                    raise ValueError(f"Invalid image data: '{image_data}' is not a valid file path")
                image = Image.open(image_data)
            elif isinstance(image_data, np.ndarray):
                # 检查数组是否为空
                if image_data.size == 0 or image_data.shape[0] == 0 or image_data.shape[1] == 0:
                    raise ValueError("Empty image: array has zero dimensions")

                # 从 numpy 数组加载
                # 处理不同的数组形状
                if len(image_data.shape) == 2:
                    # 灰度图
                    image = Image.fromarray(image_data, mode='L')
                elif len(image_data.shape) == 3:
                    if image_data.shape[2] == 3:
                        # RGB
                        image = Image.fromarray(image_data.astype(np.uint8), mode='RGB')
                    elif image_data.shape[2] == 4:
                        # RGBA
                        image = Image.fromarray(image_data.astype(np.uint8), mode='RGBA')
                    else:
                        raise ValueError(f"Unsupported array shape: {image_data.shape}")
                else:
                    raise ValueError(f"Unsupported array shape: {image_data.shape}")
            elif isinstance(image_data, Image.Image):
                # 已经是 PIL Image
                # 检查图像是否为空
                if image_data.size[0] == 0 or image_data.size[1] == 0:
                    raise ValueError("Empty image: width or height is zero")
                image = image_data
            else:
                raise ValueError(f"Unsupported image data type: {type(image_data)}")

            # 统一转换为 RGB
            if image.mode != 'RGB':
                if image.mode == 'RGBA':
                    # RGBA -> RGB (处理透明通道)
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])  # 使用 alpha 通道作为 mask
                    image = background
                else:
                    # 其他模式直接转换
                    image = image.convert('RGB')

            logger.debug(f"Image loaded: size={image.size}, mode={image.mode}")
            return image

        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            if isinstance(e, ValueError):
                raise
            raise IOError(f"Cannot load image: {str(e)}") from e

    def _resize_with_padding(self, image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        智能缩放并 padding 到目标尺寸（保持宽高比）

        Args:
            image: PIL Image

        Returns:
            resized_image: 处理后的图像
            padding_info: Padding 信息字典
        """
        original_width, original_height = image.size
        target_width, target_height = self.target_size

        # 处理空图像
        if original_width == 0 or original_height == 0:
            raise ValueError(f"Invalid image dimensions: {original_width}x{original_height}")

        # 计算缩放比例 (保持宽高比)
        scale = min(target_width / original_width, target_height / original_height)

        # 避免放大小图
        if scale > 1.0:
            scale = 1.0

        # 计算缩放后的尺寸
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # 缩放图像 (使用 LANCZOS 高质量插值)
        if scale != 1.0:
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            resized = image.copy()

        # 创建目标尺寸的黑色画布
        canvas = Image.new('RGB', self.target_size, (0, 0, 0))

        # 计算居中粘贴的位置
        offset_x = (target_width - new_width) // 2
        offset_y = (target_height - new_height) // 2

        # 粘贴缩放后的图像到画布中心
        canvas.paste(resized, (offset_x, offset_y))

        # 构建 padding 信息
        padding_info = {
            'resized_size': (new_width, new_height),
            'scale': scale,
            'padding': {
                'left': offset_x,
                'top': offset_y,
                'right': target_width - new_width - offset_x,
                'bottom': target_height - new_height - offset_y
            },
            'offset': (offset_x, offset_y)
        }

        logger.debug(f"Resize: {original_width}x{original_height} -> {new_width}x{new_height}, "
                    f"padding=({offset_x},{offset_y}), scale={scale:.4f}")

        return canvas, padding_info

    def restore_coordinates(
        self,
        boxes: np.ndarray,
        transform_info: Dict[str, Any]
    ) -> np.ndarray:
        """
        将模型输出的坐标还原到原始图像坐标系

        Args:
            boxes: 边界框坐标 [[x1, y1, x2, y2], ...] 或 [[x, y, w, h], ...]
            transform_info: 变换信息

        Returns:
            restored_boxes: 还原后的坐标
        """
        boxes = np.array(boxes, dtype=np.float32)
        if boxes.size == 0:
            return boxes

        scale = transform_info['scale']
        offset_x, offset_y = transform_info['offset']

        # 减去 padding 偏移
        boxes[..., 0::2] -= offset_x  # x 坐标
        boxes[..., 1::2] -= offset_y  # y 坐标

        # 缩放回原始尺寸
        boxes = boxes / scale

        return boxes
