"""
Qwen2-VL模型封装
"""

from typing import Any, Dict
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from mindnlp.ocr.utils.logger import get_logger
from .base import VLMModelBase


logger = get_logger(__name__)


class Qwen2VLModel(VLMModelBase):
    """Qwen2-VL模型封装"""

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct", device: str = "cuda"):
        """
        初始化Qwen2-VL模型

        Args:
            model_name: 模型名称
            device: 运行设备
        """
        super().__init__(model_name, device)
        self.processor = None
        self.load_model()
        self.load_processor()
        self.load_tokenizer()

    def load_model(self):
        """加载Qwen2-VL模型"""
        try:
            logger.info(f"Loading Qwen2-VL model: {self.model_name}")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=self.device if self.device != "cpu" else None
            )
            if self.device == "cpu":
                self.model = self.model.to("cpu")
            self.model.eval()
            logger.info("Qwen2-VL model loaded successfully")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def load_processor(self):
        """加载Qwen2-VL processor"""
        try:
            logger.info(f"Loading Qwen2-VL processor: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            logger.info("Qwen2-VL processor loaded successfully")
            return self.processor
        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            raise

    def load_tokenizer(self):
        """加载Qwen2-VL tokenizer"""
        try:
            logger.info(f"Loading Qwen2-VL tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            logger.info("Qwen2-VL tokenizer loaded successfully")
            return self.tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def generate(self, inputs: Dict[str, Any], **kwargs):
        """
        生成输出

        Args:
            inputs: 模型输入字典
            **kwargs: 生成参数

        Returns:
            生成的token ids
        """
        # 设置默认生成参数
        generation_config = {
            'max_new_tokens': kwargs.get('max_new_tokens', 512),
            'do_sample': kwargs.get('do_sample', False),
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.9),
        }

        logger.info("Generating output with Qwen2-VL...")

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )

            logger.info("Generation completed")
            return outputs

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
