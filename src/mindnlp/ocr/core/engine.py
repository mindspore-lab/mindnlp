"""
VLM-OCR主引擎
协调各个组件完成端到端的OCR流程
"""

import time
from typing import List, Optional
from mindnlp.ocr.api.schemas.request import OCRRequest, OCRBatchRequest, OCRURLRequest
from mindnlp.ocr.api.schemas.response import OCRResponse
from mindnlp.ocr.models.loader import ModelLoader
from mindnlp.ocr.utils.logger import get_logger
from mindnlp.ocr.utils.image_utils import download_image_from_url
from mindnlp.ocr.core.exceptions import (
    ImageProcessingError,
    ModelInferenceError,
    ValidationError,
    BatchProcessingError,
    ResourceNotFoundError
)
from mindnlp.ocr.core.monitor import get_performance_monitor
from .processor.image import ImageProcessor
from .processor.prompt import PromptBuilder
from .parser.decoder import TokenDecoder
from .parser.result import ResultParser
from .parser.formatter import OutputFormatter
from .validator.input import InputValidator


logger = get_logger(__name__)


class VLMOCREngine:
    """VLM-OCR主引擎"""

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct", device: str = "cuda",
                 enable_monitoring: bool = True, quantization_mode: str = "none",
                 quantization_config: dict = None, lora_weights_path: str = None):
        """
        初始化OCR引擎

        Args:
            model_name: 模型名称
            device: 运行设备
            enable_monitoring: 是否启用性能监控
            quantization_mode: 量化模式 ("none", "fp16", "int8", "int4")
            quantization_config: 量化配置字典
            lora_weights_path: LoRA权重路径 (可选，用于加载微调模型)
        """
        logger.info(f"Initializing VLM-OCR Engine with model: {model_name}, quantization: {quantization_mode}, lora: {lora_weights_path}")

        # 保存模型名称
        self.model_name = model_name

        # 性能监控
        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            self.monitor = get_performance_monitor()
            logger.info("Performance monitoring enabled")
        else:
            self.monitor = None

        # 加载模型（传递量化参数和LoRA路径）
        self.model_loader = ModelLoader(
            model_name,
            device,
            quantization_mode=quantization_mode,
            quantization_config=quantization_config,
            lora_weights_path=lora_weights_path
        )
        self.model_loader.load_model()  # 加载模型到 model_instance
        self.model_instance = self.model_loader.model_instance  # 保存 model_instance
        self.tokenizer = self.model_loader.load_tokenizer()

        # 初始化组件
        self.image_processor = ImageProcessor()
        self.prompt_builder = PromptBuilder()
        self.token_decoder = TokenDecoder(self.tokenizer)
        self.result_parser = ResultParser()
        self.output_formatter = OutputFormatter()
        self.input_validator = InputValidator()

        logger.info("VLM-OCR Engine initialized successfully")

    def predict(self, request: OCRRequest) -> OCRResponse:
        """
        单张图像OCR预测

        Args:
            request: OCR请求

        Returns:
            OCRResponse: OCR识别结果

        Raises:
            ValidationError: 输入验证失败
            ImageProcessingError: 图像处理失败
            ModelInferenceError: 模型推理失败
        """
        start_time = time.time()

        try:
            # 1. 输入验证
            try:
                self.input_validator.validate_image(request.image)
                self.input_validator.validate_params(
                    output_format=request.output_format,
                    language=request.language,
                    task_type=request.task_type
                )
            except ValueError as e:
                raise ValidationError(
                    message=f"Input validation failed: {str(e)}",
                    field="request",
                    details={"output_format": request.output_format, "language": request.language}
                ) from e

            # 2. 图像预处理
            logger.info("Processing image...")

            # 加载为 PIL Image (Qwen2-VL processor 需要)
            try:
                from PIL import Image
                import io
                if isinstance(request.image, bytes):
                    pil_image = Image.open(io.BytesIO(request.image)).convert("RGB")
                elif isinstance(request.image, str):
                    pil_image = Image.open(request.image).convert("RGB")
                else:
                    pil_image = request.image
            except Exception as e:
                raise ImageProcessingError(
                    message=f"Failed to load image: {str(e)}",
                    details={"image_type": type(request.image).__name__}
                ) from e

            # 3. 构建Prompt
            logger.info(f"Building prompt for task: {request.task_type}")
            # pylint: disable=no-member
            try:
                prompt = self.prompt_builder.build(
                    task_type=request.task_type,
                    output_format=request.output_format,
                    language=request.language,
                    custom_prompt=request.custom_prompt
                )
            except Exception as e:
                raise ValidationError(
                    message=f"Failed to build prompt: {str(e)}",
                    field="prompt_config",
                    details={"task_type": request.task_type, "language": request.language}
                ) from e

            # 4. 将 PIL Image 转换为 data URI（qwen_vl_utils 需要）
            try:
                import base64
                from io import BytesIO
                buffered = BytesIO()
                pil_image.save(buffered, format="JPEG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode()
                image_uri = f"data:image/jpeg;base64,{img_b64}"
            except Exception as e:
                raise ImageProcessingError(
                    message=f"Failed to encode image: {str(e)}",
                    details={"image_size": pil_image.size, "mode": pil_image.mode}
                ) from e

            # 5. 构建消息格式并进行推理
            logger.info("Running model inference...")
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_uri},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]

                # 使用 model_instance 的 batch_generate 方法（已包含输入准备和解码）
                # 推理加速优化：根据任务类型动态调整 max_new_tokens
                task_type = request.task_type.lower() if request.task_type else "general"
                if task_type in ["general", "simple"]:
                    max_tokens = 128  # 简单OCR使用128（4x加速）
                elif task_type in ["document", "table"]:
                    max_tokens = 256  # 复杂文档使用256（2x加速）
                else:
                    max_tokens = 512  # 其他任务保持512

                outputs = self.model_instance.batch_generate(
                    batch_messages=[messages],
                    max_new_tokens=max_tokens,
                    do_sample=False,     # 禁用采样，使用贪婪解码（更快）
                )
            except Exception as e:
                raise ModelInferenceError(
                    message=f"Model inference failed: {str(e)}",
                    model_name=self.model_name,
                    details={"error_type": type(e).__name__}
                ) from e

            # 6. 获取解码后的文本（batch_generate 已返回解码文本）
            logger.info(f"Batch generate returned: {type(outputs)}, value: {outputs}")

            try:
                if outputs is None:
                    raise ModelInferenceError(
                        message="Model returned None",
                        model_name=self.model_name
                    )
                if not isinstance(outputs, (list, tuple)):
                    raise ModelInferenceError(
                        message=f"Model returned unexpected type: {type(outputs)}",
                        model_name=self.model_name,
                        details={"output_type": type(outputs).__name__}
                    )
                if len(outputs) == 0:
                    raise ModelInferenceError(
                        message="Model returned empty output",
                        model_name=self.model_name
                    )

                decoded_text = outputs[0]
                logger.info(f"Decoded text: {decoded_text[:100] if decoded_text else 'None'}...")
            except ModelInferenceError:
                raise
            except Exception as e:
                raise ModelInferenceError(
                    message=f"Failed to process model output: {str(e)}",
                    model_name=self.model_name
                ) from e

            # 7. 结果解析
            try:
                parsed_result = self.result_parser.parse(
                    decoded_text,
                    output_format=request.output_format,
                    confidence_threshold=request.confidence_threshold
                )
            except Exception as e:
                logger.error(f"Result parsing failed: {e}", exc_info=True)
                raise ModelInferenceError(
                    message=f"Failed to parse model output: {str(e)}",
                    model_name=self.model_name,
                    details={"raw_output": decoded_text[:200]}
                ) from e

            # 8. 格式化输出
            try:
                formatted_result = self.output_formatter.format(
                    parsed_result,
                    output_format=request.output_format
                )
            except Exception as e:
                logger.error(f"Output formatting failed: {e}", exc_info=True)
                raise ModelInferenceError(
                    message=f"Failed to format output: {str(e)}",
                    model_name=self.model_name
                ) from e

            # 构建响应
            processing_time = time.time() - start_time

            # 记录性能指标
            if self.monitor:
                self.monitor.record_inference(
                    inference_time=processing_time,
                    image_count=1,
                    success=True
                )

            # 提取文本列表和置信度
            texts = []
            confidences = []
            if isinstance(formatted_result, dict):
                blocks = formatted_result.get('blocks')
                if blocks is not None and isinstance(blocks, list):
                    for block in blocks:
                        if isinstance(block, dict):
                            texts.append(block.get('text', ''))
                            confidences.append(block.get('confidence', 1.0))
                elif 'text' in formatted_result:
                    texts = [formatted_result['text']]
                    confidences = [1.0]
            elif isinstance(formatted_result, str):
                texts = [formatted_result]
                confidences = [1.0]

            return OCRResponse(
                success=True,
                texts=texts if texts else [decoded_text],
                boxes=[],  # 目前不支持边界框
                confidences=confidences if confidences else [1.0],
                raw_output=decoded_text,
                inference_time=processing_time,
                model_name=self.model_name,
                metadata={
                    'format': request.output_format,
                    'language': request.language,
                    'task_type': request.task_type
                }
            )

        except (ValidationError, ImageProcessingError, ModelInferenceError) as e:
            # 记录失败的性能指标
            processing_time = time.time() - start_time
            if self.monitor:
                self.monitor.record_inference(
                    inference_time=processing_time,
                    image_count=1,
                    success=False,
                    error_message=str(e)
                )
            # 重新抛出我们自己的异常
            logger.error(f"OCR prediction failed: {e}", exc_info=True)
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            # 记录失败的性能指标
            processing_time = time.time() - start_time
            if self.monitor:
                self.monitor.record_inference(
                    inference_time=processing_time,
                    image_count=1,
                    success=False,
                    error_message=str(e)
                )
            # 捕获任何未预期的异常
            logger.error(f"Unexpected error during OCR prediction: {e}", exc_info=True)
            raise ModelInferenceError(
                message=f"Unexpected error: {str(e)}",
                model_name=self.model_name,
                details={"error_type": type(e).__name__}
            ) from e
        finally:
            # 显式清理中间变量释放内存（NPU 内存管理）
            try:
                # 清理局部变量
                if 'pil_image' in locals():
                    del pil_image
                if 'image_uri' in locals():
                    del image_uri
                if 'messages' in locals():
                    del messages
                if 'outputs' in locals():
                    del outputs

                # NPU 内存清理
                if "npu" in self.device:
                    import gc
                    gc.collect()
                    if hasattr(__import__('torch'), 'npu'):
                        torch_module = __import__('torch')
                        if hasattr(torch_module.npu, 'empty_cache'):
                            torch_module.npu.empty_cache()
            except Exception as cleanup_error:
                logger.debug(f"Memory cleanup warning: {cleanup_error}")

    def predict_batch(self, request: OCRBatchRequest) -> List[OCRResponse]:
        """
        批量图像OCR预测

        Args:
            request: 批量OCR请求

        Returns:
            List[OCRResponse]: OCR识别结果列表

        Raises:
            ValidationError: 批量输入验证失败
            BatchProcessingError: 批量处理失败（当所有图像都失败时）
        """
        logger.info(f"Processing batch of {len(request.images)} images...")
        batch_start_time = time.time()

        if not request.images:
            raise ValidationError(
                message="Empty batch: no images provided",
                field="images"
            )

        results = []
        errors = []

        for idx, image in enumerate(request.images):
            logger.info(f"Processing image {idx + 1}/{len(request.images)}")
            try:
                single_request = OCRRequest(
                    image=image,
                    output_format=request.output_format,
                    language=request.language,
                    task_type=request.task_type,
                    confidence_threshold=request.confidence_threshold,
                    custom_prompt=request.custom_prompt
                )
                result = self.predict(single_request)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image {idx + 1}: {e}", exc_info=True)
                errors.append({"index": idx, "error": str(e), "error_type": type(e).__name__})
                # 创建失败的响应
                results.append(OCRResponse(
                    success=False,
                    texts=[],
                    boxes=[],
                    confidences=[],
                    raw_output="",
                    inference_time=0.0,
                    model_name=self.model_name,
                    metadata={"image_index": idx},
                    error=str(e)
                ))

        # 记录批处理性能指标
        batch_time = time.time() - batch_start_time
        successful_count = len(request.images) - len(errors)
        if self.monitor:
            self.monitor.record_inference(
                inference_time=batch_time,
                image_count=len(request.images),
                success=(len(errors) == 0),
                error_message=f"Batch: {len(errors)} failures" if errors else None
            )

        # 如果所有图像都失败，抛出批处理异常
        if len(errors) == len(request.images):
            raise BatchProcessingError(
                message="All images in batch failed to process",
                total=len(request.images),
                failed=len(errors),
                errors=errors
            )

        # 如果部分失败，记录警告但返回结果
        if errors:
            logger.warning(f"Batch processing completed with {len(errors)} failures out of {len(request.images)} images")

        return results

    def predict_from_url(self, request: OCRURLRequest) -> OCRResponse:
        """
        从URL预测OCR

        Args:
            request: URL OCR请求

        Returns:
            OCRResponse: OCR识别结果

        Raises:
            ValidationError: URL验证失败
            ResourceNotFoundError: 图像下载失败
            ImageProcessingError: 图像处理失败
            ModelInferenceError: 模型推理失败
        """
        logger.info(f"Downloading image from URL: {request.image_url}")

        # 验证URL格式
        if not request.image_url:
            raise ValidationError(
                message="Empty image URL",
                field="image_url"
            )

        if not request.image_url.startswith(('http://', 'https://')):
            raise ValidationError(
                message=f"Invalid URL scheme: {request.image_url}",
                field="image_url",
                value=request.image_url
            )

        # 下载图像
        try:
            image_bytes = download_image_from_url(str(request.image_url))
        except Exception as e:
            raise ResourceNotFoundError(
                message=f"Failed to download image from URL: {str(e)}",
                resource_type="image",
                resource_path=str(request.image_url),
                details={"error_type": type(e).__name__}
            ) from e

        # 创建单个请求并处理
        try:
            single_request = OCRRequest(
                image=image_bytes,
                output_format=request.output_format,
                language=request.language,
                task_type=request.task_type,
                confidence_threshold=request.confidence_threshold,
                custom_prompt=request.custom_prompt
            )

            return self.predict(single_request)
        except (ValidationError, ImageProcessingError, ModelInferenceError):
            # 重新抛出我们的异常
            raise
        except Exception as e:
            # 捕获任何其他异常
            logger.error(f"Unexpected error processing URL image: {e}", exc_info=True)
            raise ImageProcessingError(
                message=f"Unexpected error processing image from URL: {str(e)}",
                details={"url": str(request.image_url), "error_type": type(e).__name__}
            ) from e

    def _prepare_inputs(self, image, prompt):
        """
        准备模型输入

        Args:
            image: 处理后的图像 (PIL Image)
            prompt: 构建的Prompt

        Returns:
            dict: 模型输入字典
        """
        # 检查模型是否有 processor (Qwen2-VL 需要)
        if hasattr(self.model_loader.model_instance, 'processor') and \
           self.model_loader.model_instance.processor is not None:
            # 使用 processor 处理图像和文本
            processor = self.model_loader.model_instance.processor

            # 构建消息格式 (Qwen2-VL 格式)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # 应用聊天模板
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 处理图像和文本
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            )
        else:
            # 回退到 tokenizer (用于其他模型)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            )
            inputs['pixel_values'] = image

        # 将输入移动到正确的设备
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        return inputs

    def process_batch(self, batch_items: List[OCRRequest]) -> List[OCRResponse]:
        """
        批处理接口（适配DynamicBatcher）

        Args:
            batch_items: OCR请求列表

        Returns:
            List[OCRResponse]: OCR响应列表
        """
        logger.info(f"Processing batch of {len(batch_items)} requests via process_batch")

        # 转换为OCRBatchRequest并调用predict_batch
        batch_request = OCRBatchRequest(
            images=[item.image for item in batch_items],
            output_format=batch_items[0].output_format if batch_items else "json",
            language=batch_items[0].language if batch_items else "en",
            task_type=batch_items[0].task_type if batch_items else "ocr",
            confidence_threshold=batch_items[0].confidence_threshold if batch_items else 0.5,
            custom_prompt=batch_items[0].custom_prompt if batch_items else None
        )

        return self.predict_batch(batch_request)
