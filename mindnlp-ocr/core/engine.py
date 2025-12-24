"""
VLM-OCR主引擎
协调各个组件完成端到端的OCR流程
"""

import time
from typing import List
from api.schemas.request import OCRRequest, OCRBatchRequest, OCRURLRequest
from api.schemas.response import OCRResponse
from .processor.image import ImageProcessor
from .processor.prompt import PromptBuilder
from .parser.decoder import TokenDecoder
from .parser.result import ResultParser
from .parser.formatter import OutputFormatter
from .validator.input import InputValidator
from models.loader import ModelLoader
from utils.logger import get_logger
from utils.image_utils import download_image_from_url


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
        
        # 加载模型
        self.model_loader = ModelLoader(model_name, device)
        self.model = self.model_loader.load_model()
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
            processed_image = self.image_processor.process(request.image)
            
            # 3. 构建Prompt
            logger.info(f"Building prompt for task: {request.task_type}")
            prompt = self.prompt_builder.build_prompt(
                task_type=request.task_type,
                output_format=request.output_format,
                language=request.language,
                custom_prompt=request.custom_prompt
            )
            
            # 4. 准备模型输入
            inputs = self._prepare_inputs(processed_image, prompt)
            
            # 5. 模型推理
            logger.info("Running model inference...")
            outputs = self.model.generate(**inputs)
            
            # 6. Token解码
            decoded_text = self.token_decoder.decode(outputs[0])
            
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
            return OCRResponse(
                success=True,
                text=formatted_result.get('text', ''),
                blocks=formatted_result.get('blocks'),
                format=request.output_format,
                language=request.language,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"OCR prediction failed: {str(e)}")
            processing_time = time.time() - start_time
            return OCRResponse(
                success=False,
                text="",
                format=request.output_format,
                language=request.language,
                processing_time=processing_time,
                error_message=str(e)
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
            image: 处理后的图像
            prompt: 构建的Prompt
            
        Returns:
            dict: 模型输入字典
        """
        # 这里使用transformers的标准接口
        # 具体实现依赖于所使用的VLM模型
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )
        
        # 添加图像输入 (具体格式取决于模型)
        inputs['pixel_values'] = image
        
        return inputs
