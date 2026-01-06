"""
模型加载器
根据模型名称加载对应的VLM模型
"""

from mindnlp.ocr.utils.logger import get_logger
from .base import VLMModelBase
from .qwen2vl import Qwen2VLModel
from .internvl import InternVLModel
# from .got_ocr import GOTOCRModel  # TODO: 待实现


logger = get_logger(__name__)


class ModelLoader:
    """模型加载器"""

    # 支持的模型映射
    MODEL_MAPPING = {
        'qwen2-vl': Qwen2VLModel,
        'qwen2vl': Qwen2VLModel,
        'qwen': Qwen2VLModel,
        'internvl': InternVLModel,
        # 'got-ocr': GOTOCRModel,  # TODO: 待实现
        # 'got_ocr': GOTOCRModel,
        # 'got': GOTOCRModel,
    }

    def __init__(self, model_name: str, device: str = "cuda"):
        """
        初始化模型加载器

        Args:
            model_name: 模型名称或HuggingFace model ID
            device: 运行设备
        """
        self.model_name = model_name
        self.device = device
        self.model_instance = None
        logger.info(f"ModelLoader initialized with model: {model_name}")

    def load_model(self) -> VLMModelBase:
        """
        加载模型

        Returns:
            VLMModelBase: 加载的模型实例
        """
        # 检测模型类型
        model_type = self._detect_model_type(self.model_name)

        # 获取对应的模型类
        model_class = self.MODEL_MAPPING.get(model_type)

        if model_class is None:
            logger.warning(f"Unknown model type: {model_type}, using Qwen2VL as default")
            model_class = Qwen2VLModel

        # 实例化并加载模型
        logger.info(f"Loading model with {model_class.__name__}")
        self.model_instance = model_class(self.model_name, self.device)

        return self.model_instance.model

    def load_tokenizer(self):
        """
        加载tokenizer

        Returns:
            Tokenizer实例
        """
        if self.model_instance is None:
            self.load_model()

        return self.model_instance.tokenizer

    def _detect_model_type(self, model_name: str) -> str:
        """
        检测模型类型

        Args:
            model_name: 模型名称

        Returns:
            str: 模型类型
        """
        model_name_lower = model_name.lower()

        if 'got' in model_name_lower and 'ocr' in model_name_lower:
            return 'got-ocr'
        elif 'qwen' in model_name_lower:
            return 'qwen2-vl'
        elif 'internvl' in model_name_lower:
            return 'internvl'
        else:
            logger.warning(f"Cannot detect model type from name: {model_name}")
            return 'qwen2-vl'  # 默认使用Qwen2-VL
