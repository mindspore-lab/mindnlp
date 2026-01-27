"""
输入验证器
验证输入数据的有效性
"""

import io
from PIL import Image
from mindnlp.ocr.utils.logger import get_logger


logger = get_logger(__name__)


class InputValidator:
    """输入验证器"""

    def __init__(
        self,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        max_image_size: tuple = (4096, 4096),
        allowed_formats: tuple = ('JPEG', 'PNG', 'BMP', 'TIFF')
    ):
        """
        初始化输入验证器

        Args:
            max_file_size: 最大文件大小 (bytes)
            max_image_size: 最大图像尺寸 (width, height)
            allowed_formats: 允许的图像格式
        """
        self.max_file_size = max_file_size
        self.max_image_size = max_image_size
        self.allowed_formats = allowed_formats
        logger.info("InputValidator initialized")

    def validate_image(self, image_data: bytes) -> bool:
        """
        验证图像数据

        Args:
            image_data: 图像数据

        Returns:
            bool: 是否有效

        Raises:
            ValueError: 验证失败
        """
        # 检查文件大小
        if len(image_data) > self.max_file_size:
            raise ValueError(
                f"Image size {len(image_data)} exceeds maximum allowed size {self.max_file_size}"
            )

        # 尝试打开图像
        try:
            image = Image.open(io.BytesIO(image_data))

            # 检查图像格式
            if image.format not in self.allowed_formats:
                raise ValueError(
                    f"Image format {image.format} not in allowed formats: {self.allowed_formats}"
                )

            # 检查图像尺寸
            width, height = image.size
            if width > self.max_image_size[0] or height > self.max_image_size[1]:
                raise ValueError(
                    f"Image size {width}x{height} exceeds maximum allowed size {self.max_image_size}"
                )

            logger.debug(f"Image validation passed: format={image.format}, size={width}x{height}")
            return True

        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}") from e

    def validate_params(
        self,
        output_format: str,
        language: str,
        task_type: str
    ) -> bool:
        """
        验证参数

        Args:
            output_format: 输出格式
            language: 语言
            task_type: 任务类型

        Returns:
            bool: 是否有效

        Raises:
            ValueError: 验证失败
        """
        # 验证输出格式
        allowed_formats = ('text', 'json', 'markdown')
        if output_format not in allowed_formats:
            raise ValueError(f"output_format must be one of {allowed_formats}")

        # 验证语言
        allowed_languages = ('auto', 'zh', 'en', 'ja', 'ko')
        if language not in allowed_languages:
            raise ValueError(f"language must be one of {allowed_languages}")

        # 验证任务类型
        allowed_tasks = ('general', 'document', 'table', 'formula')
        if task_type not in allowed_tasks:
            raise ValueError(f"task_type must be one of {allowed_tasks}")

        logger.debug(f"Params validation passed: format={output_format}, lang={language}, task={task_type}")
        return True
