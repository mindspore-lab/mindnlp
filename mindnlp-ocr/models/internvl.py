"""
InternVL模型封装
"""

from transformers import AutoModel, AutoTokenizer
from .base import VLMModelBase
from typing import Any, Dict
from utils.logger import get_logger


logger = get_logger(__name__)


class InternVLModel(VLMModelBase):
    """InternVL模型封装"""
    
    def __init__(self, model_name: str = "OpenGVLab/InternVL-Chat-V1-5", device: str = "cuda"):
        """
        初始化InternVL模型
        
        Args:
            model_name: 模型名称
            device: 运行设备
        """
        super().__init__(model_name, device)
        self.load_model()
        self.load_tokenizer()
    
    def load_model(self):
        """加载InternVL模型"""
        try:
            logger.info(f"Loading InternVL model: {self.model_name}")
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map=self.device
            )
            logger.info("InternVL model loaded successfully")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_tokenizer(self):
        """加载InternVL tokenizer"""
        try:
            logger.info(f"Loading InternVL tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            logger.info("InternVL tokenizer loaded successfully")
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
        # TODO: 实现InternVL特定的生成逻辑
        logger.info("Generating output with InternVL...")
        
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_new_tokens', 512),
            )
            
            logger.info("Generation completed")
            return outputs
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
