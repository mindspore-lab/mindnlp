"""
Qwen2-VL模型封装
"""

import base64
from io import BytesIO
from typing import Any, Dict, List, Union
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from mindnlp.ocr.utils.logger import get_logger
from mindnlp.ocr.core.exceptions import ModelLoadingError, ModelInferenceError
from .base import VLMModelBase


logger = get_logger(__name__)

# 尝试导入 qwen_vl_utils，如果不存在则使用内置实现
try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
    logger.info("qwen_vl_utils is available")
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False
    logger.warning("qwen_vl_utils not found, using built-in implementation")


class Qwen2VLModel(VLMModelBase):
    """Qwen2-VL模型封装"""

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct", device: str = "cuda",
                 min_pixels: int = 128*28*28, max_pixels: int = 512*28*28):
        """
        初始化Qwen2-VL模型

        Args:
            model_name: 模型名称
            device: 运行设备
            min_pixels: 最小像素数 (用于动态分辨率)
            max_pixels: 最大像素数 (用于动态分辨率)
        """
        super().__init__(model_name, device)
        self.processor = None
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.load_model()
        self.load_processor()
        self.load_tokenizer()

    def load_model(self):
        """
        加载Qwen2-VL模型
        
        Raises:
            ModelLoadingError: 模型加载失败
        """
        try:
            logger.info(f"Loading Qwen2-VL model: {self.model_name}")
            # Qwen2-VL 官方推荐的加载方式：
            # 使用 Qwen2VLForConditionalGeneration 而非 AutoModel
            try:
                # 方法1：尝试直接导入 Qwen2VLForConditionalGeneration
                from transformers import Qwen2VLForConditionalGeneration
                
                # 使用float16节省显存
                if self.device != "cpu":
                    device_type = "NPU" if "npu" in self.device else "GPU"
                    logger.info(f"Using float16 precision to optimize {device_type} memory...")
                    # NPU设备不支持device_map="auto"，需要手动指定设备
                    if "npu" in self.device:
                        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                            self.model_name,
                            torch_dtype=torch.float16,
                            device_map=None,
                            low_cpu_mem_usage=True
                        ).to(self.device)
                    else:
                        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                            self.model_name,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            low_cpu_mem_usage=True
                        )
                else:
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        device_map=None
                    )
                logger.info(f"Loaded with Qwen2VLForConditionalGeneration on device: {self.model.device if hasattr(self.model, 'device') else 'auto'}")
            except ImportError:
                # 方法2：使用 trust_remote_code 动态导入
                logger.info("Qwen2VLForConditionalGeneration not in transformers, trying trust_remote_code...")
                # 导入配置
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                # 获取模型类名
                if config.architectures:
                    model_class_name = config.architectures[0]
                    logger.info(f"Model architecture: {model_class_name}")
                
                # 使用 trust_remote_code 加载完整模型
                from transformers import AutoModelForVision2Seq
                
                # 使用float16优化
                if self.device != "cpu":
                    device_type = "NPU" if "npu" in self.device else "GPU"
                    logger.info(f"Using float16 precision ({device_type}, trust_remote_code path)...")
                    # NPU设备不支持device_map="auto"，需要手动指定设备
                    if "npu" in self.device:
                        self.model = AutoModelForVision2Seq.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            torch_dtype=torch.float16,
                            device_map=None,
                            low_cpu_mem_usage=True
                        ).to(self.device)
                    else:
                        self.model = AutoModelForVision2Seq.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            low_cpu_mem_usage=True
                        )
                else:
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                        device_map=None
                    )
                logger.info(f"Loaded with AutoModelForVision2Seq on device: {self.model.device if hasattr(self.model, 'device') else 'auto'}")
            
            # 如果使用 CPU，需要显式移动模型
            if self.device == "cpu":
                self.model = self.model.to("cpu").to(torch.float32)
            
            self.model.eval()
            
            logger.info(f"Qwen2-VL model loaded successfully (type: {type(self.model).__name__}, has_generate: {hasattr(self.model, 'generate')})")
            return self.model
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}", exc_info=True)
            raise ModelLoadingError(
                message=f"Missing required dependencies for {self.model_name}: {str(e)}",
                model_name=self.model_name,
                details={"error_type": "ImportError", "missing_module": str(e)}
            ) from e
        except OSError as e:
            logger.error(f"Model files not found: {e}", exc_info=True)
            raise ModelLoadingError(
                message=f"Model files not found for {self.model_name}: {str(e)}",
                model_name=self.model_name,
                details={"error_type": "OSError", "suggestion": "Check model path or download model files"}
            ) from e
        except RuntimeError as e:
            logger.error(f"Runtime error during model loading: {e}", exc_info=True)
            raise ModelLoadingError(
                message=f"Runtime error loading {self.model_name}: {str(e)}",
                model_name=self.model_name,
                details={"error_type": "RuntimeError", "device": self.device}
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error loading model: {e}", exc_info=True)
            raise ModelLoadingError(
                message=f"Unexpected error loading {self.model_name}: {str(e)}",
                model_name=self.model_name,
                details={"error_type": type(e).__name__}
            ) from e

    def load_processor(self):
        """
        加载Qwen2-VL processor
        
        Raises:
            ModelLoadingError: Processor加载失败
        """
        try:
            logger.info(f"Loading Qwen2-VL processor: {self.model_name}")
            
            # 尝试从本地加载，如果失败则尝试在线加载
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels
                )
                logger.info("Processor loaded from local cache")
            except Exception as local_err:
                logger.warning(f"Failed to load from local cache: {local_err}")
                logger.info("Attempting to load processor from online...")
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        min_pixels=self.min_pixels,
                        max_pixels=self.max_pixels
                    )
                    logger.info("Processor loaded from online")
                except Exception as online_err:
                    raise ModelLoadingError(
                        message=f"Failed to load processor from both local and online: {str(online_err)}",
                        model_name=self.model_name,
                        details={"local_error": str(local_err), "online_error": str(online_err)}
                    ) from online_err
                
            logger.info(f"Qwen2-VL processor loaded successfully (min_pixels={self.min_pixels}, max_pixels={self.max_pixels})")
            return self.processor
        except ModelLoadingError:
            # 重新抛出我们自己的异常
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading processor: {e}", exc_info=True)
            raise ModelLoadingError(
                message=f"Unexpected error loading processor for {self.model_name}: {str(e)}",
                model_name=self.model_name,
                details={"error_type": type(e).__name__}
            ) from e

    def load_tokenizer(self):
        """
        加载Qwen2-VL tokenizer
        
        Raises:
            ModelLoadingError: Tokenizer加载失败
        """
        try:
            logger.info(f"Loading Qwen2-VL tokenizer: {self.model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                logger.info("Tokenizer loaded from local cache")
            except Exception as local_err:
                logger.warning(f"Failed to load tokenizer from local cache: {local_err}")
                logger.info("Attempting to load tokenizer from online...")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )
                    logger.info("Tokenizer loaded from online")
                except Exception as online_err:
                    raise ModelLoadingError(
                        message=f"Failed to load tokenizer from both local and online: {str(online_err)}",
                        model_name=self.model_name,
                        details={"local_error": str(local_err), "online_error": str(online_err)}
                    ) from online_err
            
            logger.info("Qwen2-VL tokenizer loaded successfully")
            return self.tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def _process_image_input(self, image_input: Union[str, bytes, Image.Image]) -> str:
        """
        处理不同格式的图像输入

        Args:
            image_input: 图像输入 (本地路径、URL、base64、PIL Image)

        Returns:
            处理后的图像路径或数据URI
        """
        # PIL Image
        if isinstance(image_input, Image.Image):
            buffered = BytesIO()
            image_input.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"

        # bytes (图像数据)
        if isinstance(image_input, bytes):
            # 检查是否是 base64
            try:
                # 尝试解码 base64
                if image_input.startswith(b'data:image'):
                    return image_input.decode()
                else:
                    # 原始图像字节
                    img_str = base64.b64encode(image_input).decode()
                    return f"data:image/jpeg;base64,{img_str}"
            except Exception:
                # 如果不是 base64，作为原始图像处理
                img_str = base64.b64encode(image_input).decode()
                return f"data:image/jpeg;base64,{img_str}"

        # 字符串 (路径或URL或base64)
        if isinstance(image_input, str):
            # 检查是否是 data URI
            if image_input.startswith('data:image'):
                return image_input
            # 检查是否是 URL
            elif image_input.startswith(('http://', 'https://')):
                return image_input
            # 检查是否是本地文件路径
            elif image_input.startswith('file://'):
                return image_input
            else:
                # 假定是本地文件路径
                return f"file://{image_input}"

        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def _builtin_process_vision_info(self, messages: List[Dict]) -> tuple:
        """
        内置的视觉信息处理 (当 qwen_vl_utils 不可用时使用)

        Args:
            messages: 消息列表

        Returns:
            (image_inputs, video_inputs) 元组
        """
        image_inputs = []
        video_inputs = []

        for message in messages:
            if "content" in message:
                for content in message["content"]:
                    if isinstance(content, dict) and content.get("type") == "image":
                        image_input = content.get("image")
                        if image_input:
                            # 处理图像输入
                            processed_image = self._process_image_input(image_input)
                            image_inputs.append(processed_image)
                    elif isinstance(content, dict) and content.get("type") == "video":
                        video_input = content.get("video")
                        if video_input:
                            video_inputs.append(video_input)

        return image_inputs, video_inputs

    def prepare_inputs(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        准备模型输入

        Args:
            messages: 消息列表，格式参考 Qwen2-VL
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "path/to/image.jpg"},
                            {"type": "text", "text": "描述这张图片"}
                        ]
                    }
                ]
            **kwargs: 额外参数 (min_pixels, max_pixels, etc.)

        Returns:
            处理后的输入张量字典
        """
        try:
            # 1. 应用对话模板
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.debug(f"Applied chat template: {text[:100]}...")

            # 2. 处理视觉信息
            if QWEN_VL_UTILS_AVAILABLE:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs, video_inputs = self._builtin_process_vision_info(messages)

            logger.debug(f"Processed {len(image_inputs)} images and {len(video_inputs)} videos")

            # 3. 准备输入
            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt"
            )

            # 4. 移动到目标设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            logger.info("Model inputs prepared successfully")
            return inputs

        except Exception as e:
            logger.error(f"Failed to prepare inputs: {e}")
            raise

    def decode_output(self, generated_ids: torch.Tensor, input_ids: torch.Tensor) -> List[str]:
        """
        解码生成的输出

        Args:
            generated_ids: 生成的 token IDs [batch_size, seq_len]
            input_ids: 输入的 token IDs [batch_size, seq_len]

        Returns:
            解码后的文本列表
        """
        try:
            # Token trimming: 只保留新生成的 tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(input_ids, generated_ids)
            ]

            # 解码
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            logger.info(f"Decoded {len(output_text)} outputs")
            return output_text

        except Exception as e:
            logger.error(f"Failed to decode output: {e}")
            raise

    def generate(self, inputs: Dict[str, Any], **kwargs) -> List[str]:
        """
        生成输出并解码

        Args:
            inputs: 模型输入字典 (来自 prepare_inputs)
            **kwargs: 生成参数
                - max_new_tokens: 最大生成token数
                - do_sample: 是否采样
                - temperature: 温度参数
                - top_p: top-p 采样

        Returns:
            解码后的文本列表
        """
        # 设置默认生成参数
        do_sample = kwargs.get('do_sample', False)
        generation_config = {
            'max_new_tokens': kwargs.get('max_new_tokens', 2048),  # 增加到2048以支持更长的文本
            'do_sample': do_sample,
        }
        
        # 只在采样时添加 temperature 和 top_p
        if do_sample:
            generation_config['temperature'] = kwargs.get('temperature', 0.7)
            generation_config['top_p'] = kwargs.get('top_p', 0.9)

        logger.info(f"Generating output with Qwen2-VL (max_new_tokens={generation_config['max_new_tokens']}, do_sample={do_sample})...")

        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    **generation_config
                )

            logger.info("Generation completed, decoding...")

            # 解码输出
            output_text = self.decode_output(generated_ids, inputs['input_ids'])

            return output_text

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def batch_generate(self, batch_messages: List[List[Dict]], **kwargs) -> List[str]:
        """
        批量生成推理

        Args:
            batch_messages: 批量消息列表
                [
                    [  # 第一个样本的消息
                        {"role": "user", "content": [...]},
                    ],
                    [  # 第二个样本的消息
                        {"role": "user", "content": [...]},
                    ],
                    ...
                ]
            **kwargs: 生成参数

        Returns:
            批量输出文本列表
        """
        logger.info(f"Batch generating for {len(batch_messages)} samples...")

        try:
            # 1. 批量处理输入
            batch_texts = []
            all_image_inputs = []
            all_video_inputs = []

            for messages in batch_messages:
                # 应用对话模板
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                batch_texts.append(text)

                # 处理视觉信息
                if QWEN_VL_UTILS_AVAILABLE:
                    result = process_vision_info(messages)
                    if result is not None:
                        image_inputs, video_inputs = result
                    else:
                        image_inputs, video_inputs = [], []
                else:
                    image_inputs, video_inputs = self._builtin_process_vision_info(messages)

                # 确保返回值不为 None
                if image_inputs is not None:
                    all_image_inputs.extend(image_inputs)
                if video_inputs is not None:
                    all_video_inputs.extend(video_inputs)

            # 2. 批量推理
            inputs = self.processor(
                text=batch_texts,
                images=all_image_inputs if all_image_inputs else None,
                videos=all_video_inputs if all_video_inputs else None,
                padding=True,
                return_tensors="pt"
            )

            # 移动到目标设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 3. 生成
            do_sample = kwargs.get('do_sample', False)
            generation_config = {
                'max_new_tokens': kwargs.get('max_new_tokens', 512),
                'do_sample': do_sample,
            }
            
            # 只在采样时添加 temperature 和 top_p
            if do_sample:
                generation_config['temperature'] = kwargs.get('temperature', 0.7)
                generation_config['top_p'] = kwargs.get('top_p', 0.9)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    **generation_config
                )

            # 4. 解码输出
            output_text = self.decode_output(generated_ids, inputs['input_ids'])

            logger.info(f"Batch generation completed for {len(output_text)} samples")
            return output_text

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            raise
