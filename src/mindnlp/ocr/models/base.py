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

    def prepare_inputs(self, messages: list, **kwargs):
        """
        准备模型输入 (可选实现)

        Args:
            messages: 消息列表
            **kwargs: 额外参数

        Returns:
            处理后的输入
        """
        raise NotImplementedError("Subclass should implement prepare_inputs if needed")

    def decode_output(self, generated_ids, input_ids):
        """
        解码生成的输出 (可选实现)

        Args:
            generated_ids: 生成的 token IDs
            input_ids: 输入的 token IDs

        Returns:
            解码后的文本
        """
        raise NotImplementedError("Subclass should implement decode_output if needed")

    def batch_generate(self, batch_messages: list, **kwargs):
        """
        批量生成推理 (可选实现)

        Args:
            batch_messages: 批量消息列表
            **kwargs: 生成参数

        Returns:
            批量输出文本列表
        """
        raise NotImplementedError("Subclass should implement batch_generate if needed")

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
