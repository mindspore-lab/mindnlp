# MindNLP OCR模块设计文档

## 1. 设计概述

### 1.1 设计目标

基于VLM-OCR引擎的设计理念,为MindNLP构建独立的OCR模块,具备以下特点:
- **轻耦合**: 与mindnlp其他模块解耦,独立运行
- **标准化**: 完整的RESTful API,易于集成
- **模块化**: 清晰的分层架构,易于维护
- **可扩展**: 支持多种VLM模型,灵活配置

### 1.2 核心设计原则

**1. 模型层依赖Transformers**
- 直接调用HuggingFace Transformers库的VLM模型
- 支持Qwen2-VL、InternVL、LLaVA等主流模型
- 模型加载和推理使用transformers标准接口

**2. 业务层自研实现**
- 图像预处理、Prompt构建、结果后处理完全自己实现
- API服务层使用FastAPI从零搭建
- 不依赖mindnlp现有的任何模块

**3. 先完成流程,后优化性能**
- Phase 1: 实现完整的端到端流程(本文档重点)
- Phase 2: 引入性能优化(KV Cache、Flash Attention等,后续设计)

## 2. 模块架构设计

### 2.1 总体架构

```
mindnlp-ocr/
├── api/                    # API服务层(自研)
│   ├── app.py             # FastAPI应用入口
│   ├── routes/            # 路由定义
│   │   ├── __init__.py
│   │   ├── ocr.py        # OCR预测接口
│   │   └── health.py     # 健康检查接口
│   ├── schemas/           # 请求/响应模型
│   │   ├── __init__.py
│   │   ├── request.py    # OCRRequest
│   │   └── response.py   # OCRResponse
│   └── middleware/        # 中间件
│       ├── __init__.py
│       ├── error.py      # 错误处理
│       └── logging.py    # 日志记录
│
├── core/                  # 核心业务层(自研)
│   ├── __init__.py
│   ├── engine.py         # VLMOCREngine主引擎
│   ├── processor/        # 预处理器
│   │   ├── __init__.py
│   │   ├── image.py     # ImageProcessor
│   │   ├── prompt.py    # PromptBuilder
│   │   └── batch.py     # BatchCollator
│   ├── parser/           # 后处理器
│   │   ├── __init__.py
│   │   ├── decoder.py   # TokenDecoder
│   │   ├── result.py    # ResultParser
│   │   └── formatter.py # OutputFormatter
│   └── validator/        # 验证器
│       ├── __init__.py
│       └── input.py     # 输入验证
│
├── models/               # 模型层(调用transformers)
│   ├── __init__.py
│   ├── base.py          # VLMModelBase抽象类
│   ├── qwen2vl.py       # Qwen2VL模型封装
│   ├── internvl.py      # InternVL模型封装
│   └── loader.py        # 模型加载器
│
├── utils/               # 工具库(自研)
│   ├── __init__.py
│   ├── image_utils.py   # 图像工具
│   ├── text_utils.py    # 文本工具
│   └── logger.py        # 日志工具
│
├── config/              # 配置管理(自研)
│   ├── __init__.py
│   ├── settings.py      # 配置类
│   └── prompts.yaml     # Prompt模板
│
├── tests/               # 测试
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_processor.py
│   └── test_models.py
│
├── requirements.txt     # 依赖列表
├── setup.py            # 安装脚本
├── README.md           # 使用文档
└── main.py             # 启动入口
```

### 2.2 层级职责划分

| 层级 | 职责 | 实现方式 | 依赖 |
|-----|------|---------|------|
| **API层** | 接收HTTP请求,返回结果 | FastAPI自研 | FastAPI, Pydantic |
| **核心层** | 预处理、后处理、流程编排 | 完全自研 | PIL, NumPy, OpenCV |
| **模型层** | VLM模型推理 | **调用transformers** | transformers, torch |
| **工具层** | 通用工具函数 | 完全自研 | 标准库 |

### 2.3 数据流转

```
HTTP请求 (POST /api/v1/ocr/predict)
    ↓
[API层] 请求验证 & 参数解析 (FastAPI)
    ↓
[核心层] ImageProcessor.process()
    ├─ 图像加载 (PIL)
    ├─ 尺寸归一化 (自研)
    ├─ 数值标准化 (NumPy)
    └─ Tensor转换 → [B,3,H,W]
    ↓
[核心层] PromptBuilder.build()
    ├─ 任务类型识别 (自研)
    ├─ Prompt模板填充 (自研)
    └─ 返回Prompt字符串
    ↓
[模型层] VLMModel.generate()
    ├─ transformers.AutoModel.from_pretrained()
    ├─ model.generate(pixel_values, prompt)  ← **使用transformers**
    └─ 返回Token IDs
    ↓
[模型层] VLMModel.decode()
    └─ tokenizer.decode(token_ids)  ← **使用transformers**
    ↓
[核心层] ResultParser.parse()
    ├─ 格式识别 (JSON/Text/Markdown)
    ├─ 结构化提取 (自研正则/JSON解析)
    └─ 返回OCRResult对象
    ↓
[核心层] OutputFormatter.format()
    ├─ 坐标映射 (自研)
    ├─ 置信度过滤 (自研)
    └─ 结果排序 (自研)
    ↓
[API层] 响应封装 & 返回 (Pydantic)
    ↓
HTTP响应 (JSON)
```

## 3. 核心模块详细设计

### 3.1 API服务层

#### 3.1.1 FastAPI应用入口 (`api/app.py`)

```python
"""
FastAPI应用主入口
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

from .routes import ocr, health
from .middleware.error import setup_exception_handlers
from .middleware.logging import setup_logging
from core.engine import VLMOCREngine
from config.settings import Settings

# 全局引擎实例
engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global engine
    
    # 启动时初始化
    settings = Settings()
    engine = VLMOCREngine(
        model_name=settings.MODEL_NAME,
        device=settings.DEVICE,
        model_path=settings.MODEL_PATH
    )
    print(f"OCR Engine initialized: {settings.MODEL_NAME}")
    
    yield
    
    # 关闭时清理
    if engine:
        engine.cleanup()
        print("OCR Engine cleaned up")


def create_app() -> FastAPI:
    """创建FastAPI应用"""
    
    app = FastAPI(
        title="MindNLP OCR API",
        description="基于VLM的OCR服务",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    app.include_router(ocr.router, prefix="/api/v1/ocr", tags=["OCR"])
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    
    # 异常处理
    setup_exception_handlers(app)
    
    # 日志配置
    setup_logging()
    
    return app


def get_engine() -> VLMOCREngine:
    """获取全局引擎实例"""
    return engine


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
```

#### 3.1.2 OCR预测路由 (`api/routes/ocr.py`)

```python
"""
OCR预测API路由
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import time
import base64

from ..schemas.request import OCRRequest, BatchOCRRequest
from ..schemas.response import OCRResponse, BatchOCRResponse
from ..app import get_engine
from core.engine import VLMOCREngine

router = APIRouter()


@router.post("/predict", response_model=OCRResponse)
async def predict_ocr(
    file: UploadFile = File(...),
    output_format: str = Form("json"),
    language: str = Form("auto"),
    confidence_threshold: float = Form(0.0),
    custom_prompt: Optional[str] = Form(None),
    engine: VLMOCREngine = Depends(get_engine)
):
    """
    单张图像OCR预测
    
    Args:
        file: 上传的图像文件
        output_format: 输出格式 (json/text/markdown)
        language: 语言设置 (auto/zh/en/ja/ko)
        confidence_threshold: 置信度阈值
        custom_prompt: 自定义Prompt
        
    Returns:
        OCRResponse: OCR识别结果
    """
    start_time = time.time()
    
    try:
        # 读取图像
        image_bytes = await file.read()
        
        # 构建请求
        request = OCRRequest(
            image=image_bytes,
            output_format=output_format,
            language=language,
            confidence_threshold=confidence_threshold,
            custom_prompt=custom_prompt
        )
        
        # 执行预测
        result = engine.predict(request)
        
        # 计算耗时
        inference_time = time.time() - start_time
        
        # 构建响应
        response = OCRResponse(
            success=True,
            texts=result.texts,
            boxes=result.boxes,
            confidences=result.confidences,
            raw_output=result.raw_output,
            inference_time=inference_time,
            model_name=engine.model_name,
            metadata=result.metadata
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict_batch", response_model=BatchOCRResponse)
async def predict_batch(
    files: list[UploadFile] = File(...),
    output_format: str = Form("json"),
    language: str = Form("auto"),
    confidence_threshold: float = Form(0.0),
    engine: VLMOCREngine = Depends(get_engine)
):
    """
    批量图像OCR预测
    
    Args:
        files: 上传的图像文件列表
        output_format: 输出格式
        language: 语言设置
        confidence_threshold: 置信度阈值
        
    Returns:
        BatchOCRResponse: 批量OCR结果
    """
    start_time = time.time()
    
    try:
        # 读取所有图像
        images = []
        for file in files:
            image_bytes = await file.read()
            images.append(image_bytes)
        
        # 构建批量请求
        batch_request = BatchOCRRequest(
            images=images,
            output_format=output_format,
            language=language,
            confidence_threshold=confidence_threshold
        )
        
        # 执行批量预测
        results = engine.predict_batch(batch_request)
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        # 构建批量响应
        response = BatchOCRResponse(
            success=True,
            results=results,
            total_images=len(results),
            total_time=total_time,
            model_name=engine.model_name
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict_url")
async def predict_from_url(
    image_url: str = Form(...),
    output_format: str = Form("json"),
    language: str = Form("auto"),
    confidence_threshold: float = Form(0.0),
    engine: VLMOCREngine = Depends(get_engine)
):
    """
    从URL预测OCR
    
    Args:
        image_url: 图像URL
        output_format: 输出格式
        language: 语言设置
        confidence_threshold: 置信度阈值
        
    Returns:
        OCRResponse: OCR识别结果
    """
    import httpx
    
    start_time = time.time()
    
    try:
        # 下载图像
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, timeout=10.0)
            response.raise_for_status()
            image_bytes = response.content
        
        # 构建请求
        request = OCRRequest(
            image=image_bytes,
            output_format=output_format,
            language=language,
            confidence_threshold=confidence_threshold
        )
        
        # 执行预测
        result = engine.predict(request)
        
        inference_time = time.time() - start_time
        
        return OCRResponse(
            success=True,
            texts=result.texts,
            boxes=result.boxes,
            confidences=result.confidences,
            raw_output=result.raw_output,
            inference_time=inference_time,
            model_name=engine.model_name,
            metadata=result.metadata
        )
        
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### 3.1.3 请求/响应Schema (`api/schemas/`)

```python
# api/schemas/request.py
"""
API请求模型
"""
from pydantic import BaseModel, Field, validator
from typing import Optional


class OCRRequest(BaseModel):
    """OCR预测请求"""
    image: bytes = Field(..., description="图像字节流")
    output_format: str = Field("json", description="输出格式: json/text/markdown")
    language: str = Field("auto", description="语言: auto/zh/en/ja/ko/multi")
    confidence_threshold: float = Field(0.0, ge=0.0, le=1.0, description="置信度阈值")
    custom_prompt: Optional[str] = Field(None, description="自定义Prompt")
    
    @validator('output_format')
    def validate_format(cls, v):
        if v not in ['json', 'text', 'markdown']:
            raise ValueError('output_format must be json/text/markdown')
        return v
    
    @validator('language')
    def validate_language(cls, v):
        if v not in ['auto', 'zh', 'en', 'ja', 'ko', 'multi']:
            raise ValueError('language must be auto/zh/en/ja/ko/multi')
        return v


class BatchOCRRequest(BaseModel):
    """批量OCR请求"""
    images: list[bytes] = Field(..., description="图像字节流列表")
    output_format: str = Field("json", description="输出格式")
    language: str = Field("auto", description="语言设置")
    confidence_threshold: float = Field(0.0, ge=0.0, le=1.0, description="置信度阈值")
    
    @validator('images')
    def validate_images(cls, v):
        if len(v) == 0:
            raise ValueError('images cannot be empty')
        if len(v) > 100:
            raise ValueError('images count cannot exceed 100')
        return v
```

```python
# api/schemas/response.py
"""
API响应模型
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class OCRResponse(BaseModel):
    """OCR预测响应"""
    success: bool = Field(..., description="是否成功")
    texts: List[str] = Field(..., description="识别的文本列表")
    boxes: List[List[float]] = Field(..., description="文本框坐标 [[x1,y1,x2,y2],...]")
    confidences: List[float] = Field(..., description="置信度列表")
    raw_output: str = Field(..., description="原始模型输出")
    inference_time: float = Field(..., description="推理耗时(秒)")
    model_name: str = Field(..., description="模型名称")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    error: Optional[str] = Field(None, description="错误信息")


class BatchOCRResponse(BaseModel):
    """批量OCR响应"""
    success: bool = Field(..., description="是否成功")
    results: List[OCRResponse] = Field(..., description="每张图的结果")
    total_images: int = Field(..., description="图像总数")
    total_time: float = Field(..., description="总耗时(秒)")
    model_name: str = Field(..., description="模型名称")
    error: Optional[str] = Field(None, description="错误信息")
```

### 3.2 核心业务层

#### 3.2.1 主引擎 (`core/engine.py`)

```python
"""
VLM-OCR主引擎
协调各个组件完成端到端OCR流程
"""
from typing import List, Optional
import torch

from .processor.image import ImageProcessor
from .processor.prompt import PromptBuilder
from .processor.batch import BatchCollator
from .parser.decoder import TokenDecoder
from .parser.result import ResultParser
from .parser.formatter import OutputFormatter
from .validator.input import InputValidator
from models.loader import ModelLoader
from api.schemas.request import OCRRequest, BatchOCRRequest
from api.schemas.response import OCRResponse


class VLMOCREngine:
    """
    VLM-OCR引擎主类
    
    负责:
    1. 组件初始化和管理
    2. 流程编排
    3. 异常处理
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        model_path: Optional[str] = None
    ):
        """
        初始化引擎
        
        Args:
            model_name: 模型名称
            device: 设备 (cuda/cpu)
            model_path: 本地模型路径
        """
        self.model_name = model_name
        self.device = device
        
        # 初始化组件
        self.validator = InputValidator()
        self.image_processor = ImageProcessor()
        self.prompt_builder = PromptBuilder()
        self.batch_collator = BatchCollator()
        self.token_decoder = TokenDecoder()
        self.result_parser = ResultParser()
        self.output_formatter = OutputFormatter()
        
        # 加载模型 (调用transformers)
        self.model_loader = ModelLoader()
        self.model = self.model_loader.load_model(
            model_name=model_name,
            device=device,
            model_path=model_path
        )
        
        print(f"[Engine] Initialized with model: {model_name}")
    
    def predict(self, request: OCRRequest) -> OCRResponse:
        """
        单张图像预测
        
        Args:
            request: OCR请求
            
        Returns:
            OCRResponse: OCR结果
        """
        try:
            # 1. 输入验证
            self.validator.validate_image(request.image)
            self.validator.validate_params(
                output_format=request.output_format,
                language=request.language,
                confidence_threshold=request.confidence_threshold
            )
            
            # 2. 图像预处理
            pixel_values, transform_info = self.image_processor.process(
                request.image
            )
            
            # 3. Prompt构建
            prompt = self.prompt_builder.build(
                output_format=request.output_format,
                language=request.language,
                custom_prompt=request.custom_prompt
            )
            
            # 4. 模型推理 (使用transformers)
            with torch.no_grad():
                output_ids = self.model.generate(
                    pixel_values=pixel_values.to(self.device),
                    prompt=prompt,
                    max_new_tokens=2048,
                    do_sample=False
                )
            
            # 5. Token解码 (使用transformers tokenizer)
            decoded_text = self.token_decoder.decode(
                output_ids,
                tokenizer=self.model.tokenizer
            )
            
            # 6. 结果解析
            parsed_result = self.result_parser.parse(
                decoded_text,
                output_format=request.output_format
            )
            
            # 7. 输出格式化
            formatted_result = self.output_formatter.format(
                parsed_result,
                transform_info=transform_info,
                confidence_threshold=request.confidence_threshold
            )
            
            # 8. 构建响应
            response = OCRResponse(
                success=True,
                texts=formatted_result['texts'],
                boxes=formatted_result['boxes'],
                confidences=formatted_result['confidences'],
                raw_output=decoded_text,
                inference_time=0.0,  # 由上层计算
                model_name=self.model_name,
                metadata=formatted_result['metadata']
            )
            
            return response
            
        except Exception as e:
            print(f"[Engine] Prediction error: {str(e)}")
            raise
    
    def predict_batch(self, request: BatchOCRRequest) -> List[OCRResponse]:
        """
        批量图像预测
        
        Args:
            request: 批量OCR请求
            
        Returns:
            List[OCRResponse]: 批量OCR结果
        """
        results = []
        
        for image in request.images:
            # 为每张图创建单独请求
            single_request = OCRRequest(
                image=image,
                output_format=request.output_format,
                language=request.language,
                confidence_threshold=request.confidence_threshold
            )
            
            try:
                result = self.predict(single_request)
                results.append(result)
            except Exception as e:
                # 单张图失败不影响其他
                error_result = OCRResponse(
                    success=False,
                    texts=[],
                    boxes=[],
                    confidences=[],
                    raw_output="",
                    inference_time=0.0,
                    model_name=self.model_name,
                    metadata={},
                    error=str(e)
                )
                results.append(error_result)
        
        return results
    
    def cleanup(self):
        """清理资源"""
        if self.model:
            del self.model
            torch.cuda.empty_cache()
        print("[Engine] Cleaned up")
```

#### 3.2.2 图像处理器 (`core/processor/image.py`)

```python
"""
图像预处理器
"""
from PIL import Image
import numpy as np
import torch
from typing import Tuple, Dict
import io


class ImageProcessor:
    """
    图像预处理器
    
    功能:
    1. 图像加载和格式转换
    2. 尺寸归一化
    3. 数值标准化
    4. Tensor转换
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (448, 448),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        初始化处理器
        
        Args:
            target_size: 目标尺寸
            mean: 归一化均值
            std: 归一化标准差
        """
        self.target_size = target_size
        self.mean = np.array(mean).reshape(3, 1, 1)
        self.std = np.array(std).reshape(3, 1, 1)
    
    def process(self, image_bytes: bytes) -> Tuple[torch.Tensor, Dict]:
        """
        处理图像
        
        Args:
            image_bytes: 图像字节流
            
        Returns:
            pixel_values: 处理后的Tensor [1,3,H,W]
            transform_info: 变换信息(用于坐标还原)
        """
        # 1. 加载图像
        image = Image.open(io.BytesIO(image_bytes))
        
        # 2. 转RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 3. 记录原始尺寸
        original_size = image.size  # (width, height)
        
        # 4. 智能缩放 (保持宽高比)
        resized_image, padding = self._smart_resize(image, self.target_size)
        
        # 5. 转NumPy数组
        image_array = np.array(resized_image).astype(np.float32)
        
        # 6. 归一化 [0,255] -> [0,1]
        image_array = image_array / 255.0
        
        # 7. 转换维度 HWC -> CHW
        image_array = np.transpose(image_array, (2, 0, 1))
        
        # 8. 标准化
        image_array = (image_array - self.mean) / self.std
        
        # 9. 转Tensor并增加batch维度
        pixel_values = torch.from_numpy(image_array).unsqueeze(0)
        
        # 10. 记录变换信息
        transform_info = {
            'original_size': original_size,
            'resized_size': resized_image.size,
            'padding': padding,
            'scale': self._calculate_scale(original_size, resized_image.size)
        }
        
        return pixel_values, transform_info
    
    def _smart_resize(
        self,
        image: Image.Image,
        target_size: Tuple[int, int]
    ) -> Tuple[Image.Image, Dict]:
        """
        智能缩放(保持宽高比)
        
        Args:
            image: PIL图像
            target_size: 目标尺寸 (width, height)
            
        Returns:
            resized_image: 缩放后的图像
            padding: Padding信息
        """
        original_width, original_height = image.size
        target_width, target_height = target_size
        
        # 计算缩放比例
        scale = min(
            target_width / original_width,
            target_height / original_height
        )
        
        # 计算新尺寸
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # 缩放图像
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        
        # 创建目标尺寸的画布(黑色背景)
        canvas = Image.new('RGB', target_size, (0, 0, 0))
        
        # 计算居中paste的位置
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        
        # Paste到画布
        canvas.paste(resized, (paste_x, paste_y))
        
        padding = {
            'left': paste_x,
            'top': paste_y,
            'right': target_width - new_width - paste_x,
            'bottom': target_height - new_height - paste_y
        }
        
        return canvas, padding
    
    def _calculate_scale(
        self,
        original_size: Tuple[int, int],
        resized_size: Tuple[int, int]
    ) -> float:
        """计算实际缩放比例"""
        original_width, original_height = original_size
        resized_width, resized_height = resized_size
        
        scale = min(
            resized_width / original_width,
            resized_height / original_height
        )
        
        return scale
```

### 3.3 模型层

#### 3.3.1 模型加载器 (`models/loader.py`)

```python
"""
模型加载器
调用transformers加载VLM模型
"""
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import torch
from typing import Optional

from .base import VLMModelBase
from .qwen2vl import Qwen2VLModel


class ModelLoader:
    """
    模型加载器
    
    支持的模型:
    - Qwen2-VL系列
    - InternVL系列 (待实现)
    - LLaVA系列 (待实现)
    """
    
    SUPPORTED_MODELS = {
        'qwen2-vl': Qwen2VLModel,
        # 'internvl': InternVLModel,
        # 'llava': LLaVAModel,
    }
    
    def load_model(
        self,
        model_name: str,
        device: str = "cuda",
        model_path: Optional[str] = None,
        **kwargs
    ) -> VLMModelBase:
        """
        加载VLM模型
        
        Args:
            model_name: 模型名称或HuggingFace模型ID
            device: 设备
            model_path: 本地模型路径(可选)
            
        Returns:
            VLMModelBase: 模型实例
        """
        # 识别模型类型
        model_type = self._identify_model_type(model_name)
        
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # 获取模型类
        model_class = self.SUPPORTED_MODELS[model_type]
        
        # 实例化模型
        model = model_class(
            model_name=model_name,
            device=device,
            model_path=model_path,
            **kwargs
        )
        
        print(f"[ModelLoader] Loaded {model_type} model: {model_name}")
        
        return model
    
    def _identify_model_type(self, model_name: str) -> str:
        """识别模型类型"""
        model_name_lower = model_name.lower()
        
        if 'qwen2-vl' in model_name_lower or 'qwen/qwen2-vl' in model_name_lower:
            return 'qwen2-vl'
        elif 'internvl' in model_name_lower:
            return 'internvl'
        elif 'llava' in model_name_lower:
            return 'llava'
        else:
            # 默认尝试Qwen2-VL
            return 'qwen2-vl'
```

#### 3.3.2 Qwen2-VL模型封装 (`models/qwen2vl.py`)

```python
"""
Qwen2-VL模型封装
基于transformers实现
"""
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from typing import Optional, List

from .base import VLMModelBase


class Qwen2VLModel(VLMModelBase):
    """
    Qwen2-VL模型封装
    
    使用transformers库加载和推理
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        model_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16
    ):
        """
        初始化Qwen2-VL模型
        
        Args:
            model_name: 模型名称
            device: 设备
            model_path: 本地模型路径
            torch_dtype: 数据类型
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        
        # 加载模型路径
        load_path = model_path if model_path else model_name
        
        # 使用transformers加载模型
        print(f"[Qwen2VL] Loading model from {load_path}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            load_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True
        )
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            trust_remote_code=True
        )
        
        # 加载processor (用于图像处理)
        self.processor = AutoProcessor.from_pretrained(
            load_path,
            trust_remote_code=True
        )
        
        self.model.eval()
        
        print(f"[Qwen2VL] Model loaded successfully")
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 2048,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        生成OCR结果
        
        Args:
            pixel_values: 图像Tensor [B,3,H,W]
            prompt: 文本提示
            max_new_tokens: 最大生成长度
            do_sample: 是否采样
            temperature: 温度参数
            top_p: Top-p采样
            
        Returns:
            output_ids: 生成的Token IDs
        """
        # 准备输入
        # Qwen2-VL需要特定的输入格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # 使用processor准备输入
        text_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            text_prompt,
            return_tensors="pt"
        ).to(self.device)
        
        # 生成
        output_ids = self.model.generate(
            **inputs,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        return output_ids
    
    def decode(self, output_ids: torch.Tensor) -> str:
        """
        解码Token IDs
        
        Args:
            output_ids: Token IDs
            
        Returns:
            decoded_text: 解码后的文本
        """
        # 使用tokenizer解码
        decoded_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        
        return decoded_text
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'model_type': 'Qwen2-VL',
            'device': self.device,
            'dtype': str(self.torch_dtype),
            'vocab_size': self.tokenizer.vocab_size
        }
```

### 3.4 配置管理

#### 3.4.1 配置类 (`config/settings.py`)

```python
"""
配置管理
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用配置"""
    
    # API配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    
    # 模型配置
    MODEL_NAME: str = "Qwen/Qwen2-VL-7B-Instruct"
    MODEL_PATH: Optional[str] = None
    DEVICE: str = "cuda"
    TORCH_DTYPE: str = "float16"
    
    # 图像处理配置
    IMAGE_SIZE: int = 448
    MAX_IMAGE_SIZE: int = 4096
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # 推理配置
    MAX_NEW_TOKENS: int = 2048
    TEMPERATURE: float = 1.0
    TOP_P: float = 1.0
    DO_SAMPLE: bool = False
    
    # 批处理配置
    MAX_BATCH_SIZE: int = 100
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = "logs/ocr.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

## 4. 部署与使用

### 4.1 安装依赖 (`requirements.txt`)

```txt
# API框架
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
httpx==0.26.0

# 模型框架 (核心依赖)
transformers==4.37.0
torch==2.1.0
torchvision==0.16.0

# 图像处理
Pillow==10.2.0
opencv-python==4.9.0.80
numpy==1.24.3

# 数据验证
pydantic==2.5.0
pydantic-settings==2.1.0

# 工具
PyYAML==6.0.1
loguru==0.7.2

# 开发工具
pytest==7.4.3
black==24.1.0
flake8==7.0.0
```

### 4.2 启动命令

```bash
# 方式1: 直接运行
python -m api.app

# 方式2: 使用uvicorn
uvicorn api.app:create_app --factory --host 0.0.0.0 --port 8000

# 方式3: 多worker
uvicorn api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 4
```

### 4.3 API调用示例

```python
import requests

# 单张图像预测
url = "http://localhost:8000/api/v1/ocr/predict"

with open("test.jpg", "rb") as f:
    files = {"file": f}
    data = {
        "output_format": "json",
        "language": "zh",
        "confidence_threshold": 0.5
    }
    
    response = requests.post(url, files=files, data=data)
    result = response.json()
    
    print(f"识别文本: {result['texts']}")
    print(f"推理耗时: {result['inference_time']}秒")
```

```bash
# 使用curl
curl -X POST "http://localhost:8000/api/v1/ocr/predict" \
  -F "file=@test.jpg" \
  -F "output_format=json" \
  -F "language=zh"
```

## 5. 开发计划

### Phase 1: 基础流程搭建 (当前阶段)

**目标**: 实现完整的端到端OCR流程

- [x] API服务层设计
- [x] 核心业务层设计
- [x] 模型层设计(transformers集成)
- [ ] ImageProcessor实现
- [ ] PromptBuilder实现
- [ ] ResultParser实现
- [ ] Qwen2VL模型封装
- [ ] API接口实现
- [ ] 基础测试

**验收标准**:
- 能够成功加载Qwen2-VL模型
- 能够接收图像并返回OCR结果
- API接口可正常调用
- 基本功能测试通过

### Phase 2: 功能完善 (后续)

- [ ] 批处理优化(BatchCollator)
- [ ] 多模型支持(InternVL、LLaVA)
- [ ] 错误处理和异常恢复
- [ ] 日志和监控
- [ ] 单元测试和集成测试
- [ ] API文档完善

### Phase 3: 性能优化 (独立阶段)

- [ ] KV Cache实现
- [ ] Flash Attention集成
- [ ] 模型量化(FP16/INT8)
- [ ] 并发处理优化
- [ ] 显存优化
- [ ] 推理加速

## 6. 设计优势总结

**1. 解耦设计**
- 模型层仅依赖transformers,独立可测
- 业务层完全自研,不依赖mindnlp其他模块
- API层标准化,易于集成

**2. 渐进式开发**
- Phase 1专注功能实现,快速搭建原型
- Phase 2完善功能和稳定性
- Phase 3独立优化性能,不影响核心流程

**3. 易于扩展**
- 新增模型只需实现VLMModelBase接口
- 新增输出格式只需扩展ResultParser
- 新增API只需添加路由

**4. 工程化实践**
- 完整的类型注解和文档
- 清晰的目录结构
- 标准的配置管理
- 完善的错误处理

## 7. 与VLM-OCR设计的对应关系

| VLM-OCR设计 | MindNLP OCR实现 | 实现方式 |
|------------|----------------|---------|
| API层 | `api/` | FastAPI自研 |
| ImageProcessor | `core/processor/image.py` | PIL+NumPy自研 |
| PromptBuilder | `core/processor/prompt.py` | 完全自研 |
| VLM模型 | `models/qwen2vl.py` | **transformers** |
| TokenDecoder | `core/parser/decoder.py` | transformers tokenizer |
| ResultParser | `core/parser/result.py` | 正则+JSON自研 |
| OutputFormatter | `core/parser/formatter.py` | NumPy自研 |
| 优化层 | Phase 3实现 | 后续优化 |

---

**设计完成时间**: 2025-12-23  
**设计版本**: v1.0  
**设计者**: MindNLP OCR Team
