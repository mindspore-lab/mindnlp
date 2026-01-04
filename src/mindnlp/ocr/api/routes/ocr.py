"""
OCR预测路由
"""

import time
from typing import List
from fastapi import APIRouter, File, UploadFile, Form, HTTPException  # pylint: disable=import-error
from mindnlp.ocr.utils.logger import get_logger
from mindnlp.ocr.config.settings import get_settings
from mindnlp.ocr.core.exceptions import (
    ValidationError,
    ImageProcessingError,
    ModelInferenceError,
    OCRException
)
from ..schemas.request import OCRRequest, OCRURLRequest
from ..schemas.response import OCRResponse, BatchOCRResponse


logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()


def get_engine():
    """获取OCR引擎实例（延迟导入避免循环依赖）"""
    from ..app import get_engine as _get_engine
    return _get_engine()


@router.get("/predict")
async def predict_info():
    """
    OCR预测端点使用说明
    
    此端点需要使用 POST 方法并上传图像文件。
    请访问 /api/docs 查看交互式文档进行测试。
    """
    return {
        "message": "此端点需要使用 POST 方法上传图像",
        "method": "POST",
        "content_type": "multipart/form-data",
        "parameters": {
            "file": "图像文件 (必需)",
            "output_format": "输出格式 (可选, 默认: text)",
            "language": "语言 (可选, 默认: auto)",
            "task_type": "任务类型 (可选, 默认: general)"
        },
        "documentation": "/api/docs"
    }


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
        
        # 验证图像大小
        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )

        # 构建请求
        _request = OCRRequest(
            image=image_bytes,
            output_format=output_format,
            language=language,
            task_type=task_type,
            confidence_threshold=confidence_threshold
        )

        # 执行真实 OCR 推理
        logger.info(f"Processing image with real model: {settings.default_model}")
        try:
            result = _engine.predict(_request)
        except ValidationError as e:
            logger.error(f"Validation error: {e.to_dict()}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": e.message,
                    "field": e.details.get("field") if e.details else None,
                    "details": e.details
                }
            ) from e
        except ImageProcessingError as e:
            logger.error(f"Image processing error: {e.to_dict()}", exc_info=True)
            raise HTTPException(
                status_code=422,
                detail={
                    "error": e.message,
                    "details": e.details
                }
            ) from e
        except ModelInferenceError as e:
            logger.error(f"Model inference error: {e.to_dict()}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": e.message,
                    "model": e.details.get("model_name") if e.details else None,
                    "details": e.details
                }
            ) from e
        except OCRException as e:
            logger.error(f"OCR error: {e.to_dict()}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=e.to_dict()
            ) from e
        
        # 如果推理成功，添加额外的元数据
        if result.success:
            if not result.metadata:
                result.metadata = {}
            result.metadata.update({
                "language": language,
                "format": output_format,
                "task_type": task_type,
                "filename": file.filename
            })
        
        logger.info(f"OCR completed in {result.inference_time:.2f}s")
        return result

    except HTTPException:
        # 重新抛出HTTPException，不要捕获
        raise
    except RuntimeError as e:
        logger.error(f"Engine not ready: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Unexpected error in OCR prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": str(e),
                "type": type(e).__name__
            }
        ) from e


@router.get("/predict_batch")
async def predict_batch_info():
    """
    批量OCR预测端点使用说明
    
    此端点需要使用 POST 方法并上传多个图像文件。
    请访问 /api/docs 查看交互式文档进行测试。
    """
    return {
        "message": "此端点需要使用 POST 方法上传多个图像",
        "method": "POST",
        "content_type": "multipart/form-data",
        "parameters": {
            "files": "多个图像文件 (必需)",
            "output_format": "输出格式 (可选, 默认: text)",
            "language": "语言 (可选, 默认: auto)"
        },
        "documentation": "/api/docs"
    }


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
        # 验证文件列表
        if not files:
            raise HTTPException(
                status_code=400,
                detail="No files uploaded"
            )

        # 获取引擎
        _engine = get_engine()

        # 处理每个图像
        results = []
        logger.info(f"Processing batch of {len(files)} images with real model")
        
        for idx, file in enumerate(files):
            try:
                # 验证文件类型
                if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"]:
                    logger.warning(f"Skipping file {file.filename} with invalid type: {file.content_type}")
                    results.append(OCRResponse(
                        success=False,
                        texts=[],
                        boxes=[],
                        confidences=[],
                        raw_output="",
                        inference_time=0.0,
                        model_name=settings.default_model,
                        metadata={"filename": file.filename},
                        error=f"Invalid file type: {file.content_type}"
                    ))
                    continue
                
                image_bytes = await file.read()
                
                if not image_bytes:
                    logger.warning(f"Skipping empty file {file.filename}")
                    results.append(OCRResponse(
                        success=False,
                        texts=[],
                        boxes=[],
                        confidences=[],
                        raw_output="",
                        inference_time=0.0,
                        model_name=settings.default_model,
                        metadata={"filename": file.filename},
                        error="Empty file"
                    ))
                    continue

                # 执行真实 OCR 推理
                _request = OCRRequest(
                    image=image_bytes,
                    output_format=output_format,
                    language=language,
                    task_type=task_type,
                    confidence_threshold=confidence_threshold
                )
                
                logger.info(f"Processing image {idx+1}/{len(files)}: {file.filename}")
                try:
                    single_result = _engine.predict(_request)
                except OCRException as e:
                    logger.error(f"OCR error for {file.filename}: {e.to_dict()}")
                    results.append(OCRResponse(
                        success=False,
                        texts=[],
                        boxes=[],
                        confidences=[],
                        raw_output="",
                        inference_time=0.0,
                        model_name=settings.default_model,
                        metadata={"filename": file.filename},
                        error=e.message
                    ))
                    continue
                
                # 添加文件名到元数据
                if single_result.success:
                    if not single_result.metadata:
                        single_result.metadata = {}
                    single_result.metadata["filename"] = file.filename
                
                results.append(single_result)
                
            except Exception as e:
                logger.error(f"Failed to process image {file.filename}: {e}", exc_info=True)
                # 为失败的图像添加错误结果
                error_result = OCRResponse(
                    success=False,
                    texts=[],
                    boxes=[],
                    confidences=[],
                    raw_output="",
                    inference_time=0.0,
                    model_name=settings.default_model,
                    metadata={"filename": file.filename, "error": str(e)}
                )
                results.append(error_result)

        total_time = time.time() - start_time
        successful_count = sum(1 for r in results if r.success)
        logger.info(f"Batch processing completed: {successful_count}/{len(files)} successful in {total_time:.2f}s")

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
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Batch OCR prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch OCR prediction failed: {str(e)}") from e


@router.get("/predict_url")
async def predict_url_info():
    """
    URL OCR预测端点使用说明
    
    此端点需要使用 POST 方法并提供图像 URL。
    请访问 /api/docs 查看交互式文档进行测试。
    """
    return {
        "message": "此端点需要使用 POST 方法提供图像 URL",
        "method": "POST",
        "content_type": "application/json",
        "parameters": {
            "image_url": "图像URL (必需)",
            "output_format": "输出格式 (可选, 默认: text)",
            "language": "语言 (可选, 默认: auto)",
            "task_type": "任务类型 (可选, 默认: general)"
        },
        "documentation": "/api/docs"
    }


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
        from mindnlp.ocr.utils.image_utils import download_image_from_url
        logger.info(f"Downloading image from URL: {request.image_url}")
        image_bytes = download_image_from_url(str(request.image_url))

        # 构建请求并执行真实 OCR 推理
        _ocr_request = OCRRequest(
            image=image_bytes,
            output_format=request.output_format,
            language=request.language,
            task_type=request.task_type,
            confidence_threshold=request.confidence_threshold
        )

        # 执行真实 OCR 推理
        logger.info(f"Processing URL image with real model: {settings.default_model}")
        result = _engine.predict(_ocr_request)
        
        # 添加 URL 到元数据
        if result.success:
            if not result.metadata:
                result.metadata = {}
            result.metadata["source_url"] = str(request.image_url)
        
        logger.info(f"URL OCR completed in {result.inference_time:.2f}s")
        return result

    except HTTPException:
        # 重新抛出HTTPException，不要捕获
        raise
    except RuntimeError as e:
        logger.error(f"Engine not ready: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.error(f"URL OCR prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"URL OCR prediction failed: {str(e)}") from e
