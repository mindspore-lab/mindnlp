"""
FastAPI应用入口
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from utils.logger import get_logger
from config.settings import get_settings
from .routes import ocr, health
from .middleware.error import setup_exception_handlers
from .middleware.logging import setup_logging, add_logging_middleware


logger = get_logger(__name__)
settings = get_settings()

# 全局引擎实例
_engine = None


@asynccontextmanager
async def lifespan(_app: FastAPI):  # pylint: disable=redefined-outer-name
    """
    应用生命周期管理
    启动时初始化引擎，关闭时清理资源
    """
    global _engine  # pylint: disable=global-statement

    # 启动时初始化
    logger.info("Initializing OCR engine...")
    try:
        if settings.use_mock_engine:
            # 使用 Mock 引擎（快速启动）
            logger.info("Using Mock OCR engine for testing...")
            from core.mock_engine import MockVLMOCREngine
            _engine = MockVLMOCREngine()
            logger.info("Mock OCR engine initialized successfully")
        else:
            # 使用真实引擎
            from core.engine import VLMOCREngine
            _engine = VLMOCREngine(
                model_name=settings.default_model,
                device=settings.device
            )
            logger.info("OCR engine initialized successfully")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Failed to initialize OCR engine: {e}")
        _engine = None

    yield

    # 关闭时清理
    logger.info("Shutting down OCR engine...")
    _engine = None


def get_engine():
    """
    依赖注入: 获取全局引擎实例

    Returns:
        VLMOCREngine: OCR引擎实例

    Raises:
        RuntimeError: 引擎未初始化
    """
    global _engine  # pylint: disable=global-statement

    # 如果引擎未初始化，尝试创建默认引擎（用于测试环境）
    if _engine is None:
        logger.warning("OCR engine not initialized via lifespan, creating mock engine for testing")
        try:
            from core.mock_engine import MockVLMOCREngine
            _engine = MockVLMOCREngine()
            logger.info("Mock OCR engine created successfully")
        except Exception as e:
            logger.error(f"Failed to create mock OCR engine: {e}")
            raise RuntimeError("OCR Engine not initialized") from e

    return _engine


def create_app() -> FastAPI:
    """
    创建FastAPI应用实例

    Returns:
        FastAPI: FastAPI应用实例
    """

    app = FastAPI(
        title="MindNLP VLM-OCR API",
        description="基于Vision-Language Model的OCR服务",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan
    )

    # 配置CORS (开发阶段允许所有来源)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
    )

    # 设置日志系统
    setup_logging()

    # 添加日志中间件
    add_logging_middleware(app)

    # 设置异常处理
    setup_exception_handlers(app)

    # 注册路由
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(ocr.router, prefix="/api/v1/ocr", tags=["ocr"])

    logger.info("FastAPI application created")

    return app
