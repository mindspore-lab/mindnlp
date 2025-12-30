"""
OCR预测路由
"""

import time
from typing import List
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from ..schemas.request import OCRRequest, OCRURLRequest
from ..schemas.response import OCRResponse, BatchOCRResponse
from utils.logger import get_logger
from config.settings import get_settings


logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()


def get_engine():
    """获取OCR引擎实例（延迟导入避免循环依赖）"""
    from ..app import get_engine as _get_engine
    return _get_engine()


@router.post("/predict", response_model=OCRResponse)
async def predict_image(
    file: UploadFile = File(...),
    output_format: str = Form("text"),
    language: str = Form("auto"),
    task_type: str = Form("general"),
    confidence_threshold: float = Form(0.0)
):
    """
    单张图像OCR预测

    Args:
        file: 上传的图像文件
        output_format: 输出格式 (text/json/markdown)
        language: 语言设置 (auto/zh/en/ja/ko)
        task_type: 任务类型 (general/document/table/formula)
        confidence_threshold: 置信度阈值

    Returns:
        OCRResponse: OCR识别结果
    """
    start_time = time.time()

    try:
        # 验证文件类型
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Only image files are allowed."
            )

        # 获取引擎
        _engine = get_engine()

        # 读取图像数据
        image_bytes = await file.read()

        # 构建请求
        _request = OCRRequest(
            image=image_bytes,
            output_format=output_format,
            language=language,
            task_type=task_type,
            confidence_threshold=confidence_threshold
        )

        # 执行OCR (这里暂时返回模拟数据，等待engine.predict实现)
        # result = engine.predict(request)

        # 模拟响应
        inference_time = time.time() - start_time
        result = OCRResponse(
            success=True,
            texts=["识别的文本内容"],
            boxes=[[10, 20, 200, 30]],
            confidences=[0.95],
            raw_output="识别的文本内容",
            inference_time=inference_time,
            model_name=settings.default_model,
            metadata={
                "language": language,
                "format": output_format,
                "task_type": task_type
            }
        )

        return result

    except HTTPException:
        # 重新抛出HTTPException，不要捕获
        raise
    except RuntimeError as e:
        logger.error(f"Engine not ready: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"OCR prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR prediction failed: {str(e)}")


@router.post("/predict_batch", response_model=BatchOCRResponse)
async def predict_batch(
    files: List[UploadFile] = File(...),
    output_format: str = Form("text"),
    language: str = Form("auto"),
    task_type: str = Form("general"),
    confidence_threshold: float = Form(0.0)
):
    """
    批量图像OCR预测

    Args:
        files: 上传的图像文件列表
        output_format: 输出格式 (text/json/markdown)
        language: 语言设置 (auto/zh/en/ja/ko)
        task_type: 任务类型 (general/document/table/formula)
        confidence_threshold: 置信度阈值

    Returns:
        BatchOCRResponse: OCR识别结果列表
    """
    start_time = time.time()

    try:
        # 获取引擎
        _engine = get_engine()

        # 处理每个图像
        results = []
        for file in files:
            image_bytes = await file.read()

            # 执行单张OCR
            _request = OCRRequest(
                image=image_bytes,
                output_format=output_format,
                language=language,
                task_type=task_type,
                confidence_threshold=confidence_threshold
            )

            # 模拟单张处理
            single_result = OCRResponse(
                success=True,
                texts=["文本内容"],
                boxes=[[10, 20, 200, 30]],
                confidences=[0.95],
                raw_output="文本内容",
                inference_time=0.5,
                model_name=settings.default_model,
                metadata={"language": language}
            )
            results.append(single_result)

        total_time = time.time() - start_time

        return BatchOCRResponse(
            success=True,
            results=results,
            total_images=len(files),
            total_time=total_time,
            model_name=settings.default_model
        )

    except HTTPException:
        # 重新抛出HTTPException，不要捕获
        raise
    except RuntimeError as e:
        logger.error(f"Engine not ready: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Batch OCR prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch OCR prediction failed: {str(e)}")


@router.post("/predict_url", response_model=OCRResponse)
async def predict_from_url(request: OCRURLRequest):
    """
    从URL预测OCR

    Args:
        request: 包含图像URL的请求

    Returns:
        OCRResponse: OCR识别结果
    """
    start_time = time.time()

    try:
        # 获取引擎
        _engine = get_engine()

        # 下载图像
        from utils.image_utils import download_image_from_url
        image_bytes = download_image_from_url(str(request.image_url))

        # 执行OCR
        _ocr_request = OCRRequest(
            image=image_bytes,
            output_format=request.output_format,
            language=request.language,
            task_type=request.task_type,
            confidence_threshold=request.confidence_threshold
        )

        # 模拟响应
        inference_time = time.time() - start_time
        result = OCRResponse(
            success=True,
            texts=["URL图像识别的文本"],
            boxes=[[10, 20, 200, 30]],
            confidences=[0.95],
            raw_output="URL图像识别的文本",
            inference_time=inference_time,
            model_name=settings.default_model,
            metadata={"source": "url"}
        )

        return result

    except HTTPException:
        # 重新抛出HTTPException，不要捕获
        raise
    except RuntimeError as e:
        logger.error(f"Engine not ready: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"URL OCR prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"URL OCR prediction failed: {str(e)}")
