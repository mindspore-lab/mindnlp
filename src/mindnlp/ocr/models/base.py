"""
VLM模型基类
定义统一的模型接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from mindnlp.ocr.utils.logger import get_logger


logger = get_logger(__name__)


class VLMModelBase(ABC):
    """VLM模型抽象基类"""

    def __init__(self, model_name: str, device: str = "cuda"):
        """
        初始化VLM模型

        Args:
            model_name: 模型名称
            device: 运行设备
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        logger.info(f"Initializing {self.__class__.__name__} with model: {model_name}")

    @abstractmethod
    def load_model(self):
        """加载模型"""
        pass  # pylint: disable=unnecessary-pass

    @abstractmethod
    def load_tokenizer(self):
        """加载tokenizer"""
        pass  # pylint: disable=unnecessary-pass

    @abstractmethod
    def generate(self, inputs: Dict[str, Any], **kwargs):
        """
        生成输出

        Args:
            inputs: 模型输入
            **kwargs: 生成参数

        Returns:
            模型输出
        """
        pass  # pylint: disable=unnecessary-pass

    def to(self, device: str):
        """
        移动模型到指定设备

        Args:
            device: 目标设备
        """
        if self.model is not None:
            self.model = self.model.to(device)
            self.device = device
            logger.info(f"Model moved to {device}")
