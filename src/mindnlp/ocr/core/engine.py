"""
VLM-OCR主引擎
协调各个组件完成端到端的OCR流程
"""

import time
from typing import List
from mindnlp.ocr.api.schemas.request import OCRRequest, OCRBatchRequest, OCRURLRequest
from mindnlp.ocr.api.schemas.response import OCRResponse
from mindnlp.ocr.models.loader import ModelLoader
from mindnlp.ocr.utils.logger import get_logger
from mindnlp.ocr.utils.image_utils import download_image_from_url
from .processor.image import ImageProcessor
from .processor.prompt import PromptBuilder
from .parser.decoder import TokenDecoder
from .parser.result import ResultParser
from .parser.formatter import OutputFormatter
from .validator.input import InputValidator


logger = get_logger(__name__)


class VLMOCREngine:
    """VLM-OCR主引擎"""

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct", device: str = "cuda"):
        """
        初始化OCR引擎

        Args:
            model_name: 模型名称
            device: 运行设备
        """
        logger.info(f"Initializing VLM-OCR Engine with model: {model_name}")

        # 保存模型名称
        self.model_name = model_name
        
        # 加载模型
        self.model_loader = ModelLoader(model_name, device)
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
        """
        start_time = time.time()

        try:
            # 1. 输入验证
            self.input_validator.validate_image(request.image)
            self.input_validator.validate_params(
                output_format=request.output_format,
                language=request.language,
                task_type=request.task_type
            )

            # 2. 图像预处理
            logger.info("Processing image...")

            # 加载为 PIL Image (Qwen2-VL processor 需要)
            from PIL import Image
            import io
            if isinstance(request.image, bytes):
                pil_image = Image.open(io.BytesIO(request.image)).convert("RGB")
            elif isinstance(request.image, str):
                pil_image = Image.open(request.image).convert("RGB")
            else:
                pil_image = request.image

            # 3. 构建Prompt
            logger.info(f"Building prompt for task: {request.task_type}")
            # pylint: disable=no-member
            prompt = self.prompt_builder.build(
                task_type=request.task_type,
                output_format=request.output_format,
                language=request.language,
                custom_prompt=request.custom_prompt
            )

            # 4. 将 PIL Image 转换为 data URI（qwen_vl_utils 需要）
            import base64
            from io import BytesIO
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            image_uri = f"data:image/jpeg;base64,{img_b64}"
            
            # 5. 构建消息格式并进行推理
            logger.info("Running model inference...")
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
            outputs = self.model_instance.batch_generate(
                batch_messages=[messages],
                max_new_tokens=512
            )

            # 6. 获取解码后的文本（batch_generate 已返回解码文本）
            logger.info(f"Batch generate returned: {type(outputs)}, value: {outputs}")
            
            if outputs is None:
                raise ValueError("Model batch_generate returned None")
            if not isinstance(outputs, (list, tuple)):
                raise ValueError(f"Model batch_generate returned unexpected type: {type(outputs)}")
            if len(outputs) == 0:
                raise ValueError("Model batch_generate returned empty list")
                
            decoded_text = outputs[0]
            logger.info(f"Decoded text: {decoded_text[:100] if decoded_text else 'None'}...")

            # 7. 结果解析
            parsed_result = self.result_parser.parse(
                decoded_text,
                output_format=request.output_format,
                confidence_threshold=request.confidence_threshold
            )

            # 8. 格式化输出
            formatted_result = self.output_formatter.format(
                parsed_result,
                output_format=request.output_format
            )

            # 构建响应
            processing_time = time.time() - start_time
            
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

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"OCR prediction failed: {str(e)}")
            processing_time = time.time() - start_time
            return OCRResponse(
                success=False,
                texts=[],
                boxes=[],
                confidences=[],
                raw_output="",
                inference_time=processing_time,
                model_name=self.model_name,
                metadata={
                    'format': request.output_format,
                    'language': request.language
                },
                error=str(e)
            )

    def predict_batch(self, request: OCRBatchRequest) -> List[OCRResponse]:
        """
        批量图像OCR预测

        Args:
            request: 批量OCR请求

        Returns:
            List[OCRResponse]: OCR识别结果列表
        """
        logger.info(f"Processing batch of {len(request.images)} images...")
        results = []

        for idx, image in enumerate(request.images):
            logger.info(f"Processing image {idx + 1}/{len(request.images)}")
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

        return results

    def predict_from_url(self, request: OCRURLRequest) -> OCRResponse:
        """
        从URL预测OCR

        Args:
            request: URL OCR请求

        Returns:
            OCRResponse: OCR识别结果
        """
        logger.info(f"Downloading image from URL: {request.image_url}")
        image_bytes = download_image_from_url(str(request.image_url))

        single_request = OCRRequest(
            image=image_bytes,
            output_format=request.output_format,
            language=request.language,
            task_type=request.task_type,
            confidence_threshold=request.confidence_threshold,
            custom_prompt=request.custom_prompt
        )

        return self.predict(single_request)

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
