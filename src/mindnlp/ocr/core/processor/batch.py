"""
批处理整理器
负责批量数据的整理和padding
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from mindnlp.ocr.utils.logger import get_logger


logger = get_logger(__name__)


class BatchCollator:
    """
    批处理整理器

    功能:
    1. 动态分组（按图像尺寸相似度）
    2. 智能 Padding（最小化计算浪费）
    3. 批次构建
    """

    def __init__(self, max_group_diff: float = 0.2):
        """
        初始化批处理整理器

        Args:
            max_group_diff: 最大尺寸差异比例（用于分组）
        """
        self.max_group_diff = max_group_diff
        logger.info(f"BatchCollator initialized with max_group_diff={max_group_diff}")

    def collate(
        self,
        images: List[torch.Tensor],
        transform_infos: List[Dict[str, Any]] = None
    ):
        """
        整理批量数据

        Args:
            images: 图像 Tensor 列表 [每个为 [1, 3, H, W] 或 [3, H, W]]
            transform_infos: 变换信息列表（可选）

        Returns:
            如果 transform_infos 提供:
                (batch_images, transform_infos): 批量图像 Tensor [B, 3, H, W] 和变换信息
            否则:
                batch_images: 批量图像 Tensor [B, 3, H, W]
        """
        if not images:
            raise ValueError("Empty image list")

        # 检查通道数是否一致
        channels = [img.shape[0] if img.dim() == 3 else img.shape[1] for img in images]
        if len(set(channels)) > 1:
            raise ValueError(f"Inconsistent channel counts: {channels}")

        # 移除单个 batch 维度并堆叠
        images = [img.squeeze(0) if img.dim() == 4 else img for img in images]

        # 堆叠成批量
        batch_images = torch.stack(images, dim=0)

        logger.debug(f"Collated batch: {len(images)} images, shape={batch_images.shape}")

        # 根据是否提供 transform_infos 返回不同格式
        if transform_infos is not None:
            return batch_images, transform_infos
        else:
            return batch_images

    def group_by_size(
        self,
        images,
        transform_infos: List[Dict[str, Any]] = None,
        max_group_diff: float = None
    ) -> List:
        """
        按尺寸相似度分组

        Args:
            images: 图像数组列表或尺寸元组列表 [(width, height), ...]
            transform_infos: 变换信息列表（可选）
            max_group_diff: 最大分组差异（可选，默认使用初始化值）

        Returns:
            groups: 分组后的列表
        """
        if not images:
            return []

        # 使用参数中的 max_group_diff 或默认值
        group_diff = max_group_diff if max_group_diff is not None else self.max_group_diff

        # 计算每个图像的尺寸（宽高比）
        sizes = []

        # 支持两种输入格式：带 transform_infos 或直接是尺寸列表
        if transform_infos is not None:
            for info in transform_infos:
                orig_w, orig_h = info['original_size']
                aspect_ratio = orig_w / max(orig_h, 1)
                sizes.append((orig_w, orig_h, aspect_ratio))
        else:
            # images 是尺寸元组列表
            for size in images:
                orig_w, orig_h = size
                aspect_ratio = orig_w / max(orig_h, 1)
                sizes.append((orig_w, orig_h, aspect_ratio))

        # 按宽高比排序
        sorted_indices = sorted(range(len(sizes)), key=lambda i: sizes[i][2])

        # 分组
        groups = []
        current_group_images = []
        current_group_infos = []
        current_aspect = None

        for idx in sorted_indices:
            aspect_ratio = sizes[idx][2]

            if current_aspect is None:
                # 第一个元素
                current_aspect = aspect_ratio
                current_group_images.append(images[idx])
                if transform_infos is not None:
                    current_group_infos.append(transform_infos[idx])
            else:
                # 检查是否应该新建组
                diff = abs(aspect_ratio - current_aspect) / max(current_aspect, 0.001)

                if diff <= group_diff:
                    # 加入当前组
                    current_group_images.append(images[idx])
                    if transform_infos is not None:
                        current_group_infos.append(transform_infos[idx])
                else:
                    # 保存当前组，开始新组
                    if transform_infos is not None:
                        groups.append((current_group_images, current_group_infos))
                    else:
                        groups.append(current_group_images)
                    current_group_images = [images[idx]]
                    if transform_infos is not None:
                        current_group_infos = [transform_infos[idx]]
                    else:
                        current_group_infos = []
                    current_aspect = aspect_ratio

        # 添加最后一组
        if current_group_images:
            if transform_infos is not None:
                groups.append((current_group_images, current_group_infos))
            else:
                groups.append(current_group_images)

        logger.debug(f"Grouped {len(images)} images into {len(groups)} groups")
        return groups

    def smart_padding(
        self,
        sizes: List[Tuple[int, int]],
        target_size: Tuple[int, int] = None,
        alignment: int = 32
    ) -> Tuple[int, int]:
        """
        计算智能 Padding 尺寸

        Args:
            sizes: 图像尺寸列表 [(width, height), ...]
            target_size: 目标尺寸（可选）
            alignment: 对齐单位（默认32）

        Returns:
            padded_size: 对齐后的尺寸 (width, height)
        """
        if not sizes:
            return target_size if target_size else (448, 448)

        # 找到最大宽高
        max_width = max(s[0] for s in sizes)
        max_height = max(s[1] for s in sizes)

        # 对齐到指定单位
        padded_width = ((max_width + alignment - 1) // alignment) * alignment
        padded_height = ((max_height + alignment - 1) // alignment) * alignment

        return (padded_width, padded_height)

    def _smart_padding_batch(
        self,
        images: List[np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        智能 Padding（最小化计算浪费）- 用于批量图像数组

        Args:
            images: 图像数组列表 [每个为 [C, H, W]]

        Returns:
            padded_images: Padding 后的批量图像 [B, C, H, W]
            padding_info: Padding 信息
        """
        if not images:
            raise ValueError("Empty image list")

        # 找到批次中的最大尺寸
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)

        # 对齐到 32 的倍数（优化GPU计算）
        max_h = ((max_h + 31) // 32) * 32
        max_w = ((max_w + 31) // 32) * 32

        # 创建 Padding 后的批量数组
        batch_size = len(images)
        channels = images[0].shape[0]
        padded_images = np.zeros((batch_size, channels, max_h, max_w), dtype=np.float32)

        # Padding 每个图像
        for i, img in enumerate(images):
            _, h, w = img.shape
            padded_images[i, :, :h, :w] = img

        padding_info = {
            'max_height': max_h,
            'max_width': max_w,
            'original_shapes': [img.shape for img in images]
        }

        logger.debug(f"Smart padding: {batch_size} images to {max_h}x{max_w}")

        return padded_images, padding_info
