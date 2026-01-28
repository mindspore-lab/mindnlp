"""
OCR 自定义异常类
提供统一的异常处理体系
"""

from typing import Optional, Dict, Any


class OCRException(Exception):
    """
    OCR基础异常类

    所有OCR相关异常的基类，提供统一的错误信息格式
    """

    def __init__(
        self,
        message: str,
        error_code: str = "OCR_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        初始化异常

        Args:
            message: 错误消息
            error_code: 错误代码，用于标识错误类型
            details: 错误详情，包含额外的上下文信息
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        将异常转换为字典格式

        Returns:
            包含错误信息的字典
        """
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }

    def __str__(self) -> str:
        """返回异常的字符串表示"""
        if self.details:
            return f"[{self.error_code}] {self.message} - Details: {self.details}"
        return f"[{self.error_code}] {self.message}"


class ImageProcessingError(OCRException):
    """
    图像处理异常

    在图像预处理、格式转换、尺寸调整等操作失败时抛出
    """

    def __init__(
        self,
        message: str,
        image_info: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        初始化图像处理异常

        Args:
            message: 错误消息
            image_info: 图像信息（尺寸、格式等）
            details: 额外的错误详情
        """
        error_details = details or {}
        if image_info:
            error_details["image_info"] = image_info

        super().__init__(
            message=message,
            error_code="IMAGE_PROCESSING_ERROR",
            details=error_details
        )


class ModelInferenceError(OCRException):
    """
    模型推理异常

    在模型加载、推理、解码等操作失败时抛出
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        stage: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        初始化模型推理异常

        Args:
            message: 错误消息
            model_name: 模型名称
            stage: 失败的阶段（loading, inference, decoding等）
            details: 额外的错误详情
        """
        error_details = details or {}
        if model_name:
            error_details["model_name"] = model_name
        if stage:
            error_details["stage"] = stage

        super().__init__(
            message=message,
            error_code="MODEL_INFERENCE_ERROR",
            details=error_details
        )


class ValidationError(OCRException):
    """
    验证异常

    在输入参数、文件格式、配置验证失败时抛出
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        初始化验证异常

        Args:
            message: 错误消息
            field: 验证失败的字段名
            value: 导致失败的值
            details: 额外的错误详情
        """
        error_details = details or {}
        if field:
            error_details["field"] = field
        if value is not None:
            error_details["value"] = str(value)

        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=error_details
        )


class ModelLoadingError(OCRException):
    """
    模型加载异常

    在模型、处理器、分词器加载失败时抛出
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        component: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        初始化模型加载异常

        Args:
            message: 错误消息
            model_name: 模型名称
            component: 失败的组件（model, processor, tokenizer等）
            details: 额外的错误详情
        """
        error_details = details or {}
        if model_name:
            error_details["model_name"] = model_name
        if component:
            error_details["component"] = component

        super().__init__(
            message=message,
            error_code="MODEL_LOADING_ERROR",
            details=error_details
        )


class ConfigurationError(OCRException):
    """
    配置异常

    在配置文件读取、解析、验证失败时抛出
    """

    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        初始化配置异常

        Args:
            message: 错误消息
            config_file: 配置文件路径
            config_key: 配置项键名
            details: 额外的错误详情
        """
        error_details = details or {}
        if config_file:
            error_details["config_file"] = config_file
        if config_key:
            error_details["config_key"] = config_key

        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=error_details
        )


class ResourceNotFoundError(OCRException):
    """
    资源未找到异常

    在找不到必需的文件、模型、配置等资源时抛出
    """

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        初始化资源未找到异常

        Args:
            message: 错误消息
            resource_type: 资源类型（file, model, config等）
            resource_path: 资源路径
            details: 额外的错误详情
        """
        error_details = details or {}
        if resource_type:
            error_details["resource_type"] = resource_type
        if resource_path:
            error_details["resource_path"] = resource_path

        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            details=error_details
        )


class TimeoutError(OCRException):
    """
    超时异常

    在操作超时时抛出
    """

    def __init__(
        self,
        message: str,
        timeout: Optional[float] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        初始化超时异常

        Args:
            message: 错误消息
            timeout: 超时时间（秒）
            operation: 超时的操作
            details: 额外的错误详情
        """
        error_details = details or {}
        if timeout is not None:
            error_details["timeout"] = timeout
        if operation:
            error_details["operation"] = operation

        super().__init__(
            message=message,
            error_code="TIMEOUT_ERROR",
            details=error_details
        )


class BatchProcessingError(OCRException):
    """
    批处理异常

    在批量处理操作失败时抛出
    """

    def __init__(
        self,
        message: str,
        batch_size: Optional[int] = None,
        failed_indices: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        初始化批处理异常

        Args:
            message: 错误消息
            batch_size: 批次大小
            failed_indices: 失败的样本索引列表
            details: 额外的错误详情
        """
        error_details = details or {}
        if batch_size is not None:
            error_details["batch_size"] = batch_size
        if failed_indices:
            error_details["failed_indices"] = failed_indices

        super().__init__(
            message=message,
            error_code="BATCH_PROCESSING_ERROR",
            details=error_details
        )
