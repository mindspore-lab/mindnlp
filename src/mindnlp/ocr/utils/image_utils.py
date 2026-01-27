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
        timeout: 超时时间(秒)

    Returns:
        bytes: 图像数据

    Raises:
        Exception: 下载失败
    """
    try:
        logger.info("Downloading image from: %s", url)
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        # 验证是否为有效图像
        image = Image.open(io.BytesIO(response.content))
        image.verify()

        logger.info("Image downloaded successfully: %s, %s", image.size, image.format)
        return response.content

    except requests.RequestException as e:
        logger.error("Failed to download image: %s", e)
        raise IOError(f"Failed to download image from URL: {str(e)}") from e
    except Exception as e:
        logger.error("Invalid image data: %s", e)
        raise ValueError(f"Invalid image data from URL: {str(e)}") from e


def validate_image_format(image_data: bytes) -> bool:
    """
    验证图像格式

    Args:
        image_data: 图像数据

    Returns:
        bool: 是否为有效图像
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
        max_size: 最大尺寸 (width, height)

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
