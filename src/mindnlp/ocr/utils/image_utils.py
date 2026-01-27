# -*- coding: utf-8 -*-
"""
图像工具函数
"""

import io
import requests
from PIL import Image
from .logger import get_logger


logger = get_logger(__name__)


def download_image_from_url(url: str, timeout: int = 10) -> bytes:
    """
    从URL下载图像

    Args:
        url: 图像URL
        timeout: 超时时间(

    Returns:
        bytes: 图像数据

    Raises:
        Exception: 下载失败
    """
    try:
        logger.info("Downloading image from: %s", url)
        response = requests.get(url, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()

        # 检查响应内容类
        content_type = response.headers.get('Content-Type', '')
        logger.info("Response Content-Type: %s, Size: %d bytes", content_type, len(response.content))

        # 验证是否为有效图
        # 注意: verify() 会使 Image 对象失效，所以需要重新打开
        try:
            image = Image.open(io.BytesIO(response.content))
            # 获取图像信息（这会验证图像格式）
            image_format = image.format
            image_size = image.size
            logger.info("Image downloaded successfully: %s, %s", image_size, image_format)
        except Exception as e:
            logger.error("Failed to parse image - Content-Type: %s, Size: %d, First 100 bytes: %s",
                        content_type, len(response.content), response.content[:100])
            raise ValueError(f"Invalid image format (Content-Type: {content_type}): {str(e)}") from e

        return response.content

    except requests.RequestException as e:
        logger.error("Failed to download image: %s", e)
        raise IOError(f"Failed to download image from URL: {str(e)}") from e
    except ValueError:
        # 重新抛出图像验证错误
        raise
    except Exception as e:
        logger.error("Invalid image data: %s", e)
        raise ValueError(f"Invalid image data from URL: {str(e)}") from e


def validate_image_format(image_data: bytes) -> bool:
    """
    验证图像格式

    Args:
        image_data: 图像数据

    Returns:
        bool: 是否为有效图
    """
    try:
        image = Image.open(io.BytesIO(image_data))
        image.verify()
        return True
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def resize_image(image: Image.Image, max_size: tuple = (1024, 1024)) -> Image.Image:
    """
    缩放图像

    Args:
        image: PIL Image对象
        max_size: 最大尺(width, height)

    Returns:
        Image.Image: 缩放后的图像
    """
    # 计算缩放比例
    width, height = image.size
    max_width, max_height = max_size

    if width <= max_width and height <= max_height:
        return image

    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
