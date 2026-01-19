"""
健康检查路由
"""

from fastapi import APIRouter  # pylint: disable=import-error
from pydantic import BaseModel
from mindnlp.ocr.utils.logger import get_logger


logger = get_logger(__name__)
router = APIRouter()


def get_engine():
    """获取OCR引擎实例（延迟导入避免循环依赖）"""
    from ..app import get_engine as _get_engine
    return _get_engine()


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    message: str


class ReadyResponse(BaseModel):
    """就绪检查响应"""
    ready: bool
    engine_status: str
    message: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查接口
    检查服务是否在运行

    Returns:
        HealthResponse: 服务健康状态
    """
    return HealthResponse(
        status="healthy",
        message="VLM-OCR service is running"
    )


@router.get("/health/ready", response_model=ReadyResponse)
async def ready_check():
    """
    就绪检查接口
    检查服务是否准备好处理请求（引擎已初始化）

    Returns:
        ReadyResponse: 服务就绪状态
    """
    try:
        _engine = get_engine()
        return ReadyResponse(
            ready=True,
            engine_status="initialized",
            message="OCR engine is ready to process requests"
        )
    except RuntimeError:
        return ReadyResponse(
            ready=False,
            engine_status="not_initialized",
            message="OCR engine is not initialized"
        )
