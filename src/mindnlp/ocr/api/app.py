"""
FastAPI应用入口
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI  # pylint: disable=import-error
from fastapi.middleware.cors import CORSMiddleware  # pylint: disable=import-error
from mindnlp.ocr.utils.logger import get_logger
from mindnlp.ocr.config.settings import get_settings
from .routes import ocr, health, monitor, metrics
from .middleware.error import setup_exception_handlers
from .middleware.logging import setup_logging, add_logging_middleware


logger = get_logger(__name__)
settings = get_settings()

# 全局引擎实例
_engine = None
# 全局服务管理器
_service_manager = None


@asynccontextmanager
async def lifespan(_app: FastAPI):  # pylint: disable=redefined-outer-name
    """
    应用生命周期管理
    启动时初始化引擎和服务管理器，关闭时清理资源
    """
    global _engine, _service_manager  # pylint: disable=global-statement

    # 清除settings缓存以重新读取环境变量
    get_settings.cache_clear()
    # 重新获取settings
    fresh_settings = get_settings()

    # 启动时初始化引擎
    logger.info("Initializing OCR engine...")
    try:
        if fresh_settings.use_mock_engine:
            # 使用 Mock 引擎（快速启动）
            logger.info("Using Mock OCR engine for testing...")
            from mindnlp.ocr.core.mock_engine import MockVLMOCREngine
            _engine = MockVLMOCREngine()
            logger.info("Mock OCR engine initialized successfully")
        else:
            # 使用真实引擎
            from mindnlp.ocr.core.engine import VLMOCREngine
            from mindnlp.ocr.core.exceptions import ModelLoadingError
            try:
                _engine = VLMOCREngine(
                    model_name=fresh_settings.default_model,
                    device=fresh_settings.device,
                    lora_weights_path=fresh_settings.lora_weights_path
                )
                logger.info("OCR engine initialized successfully")
            except ModelLoadingError as e:
                logger.error(f"Model loading failed: {e.to_dict()}", exc_info=True)
                _engine = None
                raise RuntimeError(f"Failed to load OCR model: {e.message}") from e
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Failed to initialize OCR engine: {e}", exc_info=True)
        _engine = None
        raise

    # 初始化并发管理器（包含动态批处理、限流、熔断等功能）
    logger.info("Initializing concurrency manager...")
    try:
        from mindnlp.ocr.core.concurrency.manager import ConcurrencyManager

        # 创建并发管理器（内部会自动配置批处理器、限流器等）
        _service_manager = ConcurrencyManager(engine=_engine)

        # 启动并发管理器
        await _service_manager.start()

        logger.info("Concurrency manager initialized and started successfully")
        logger.info(f"  - Batch size: {getattr(fresh_settings, 'max_batch_size', 8)}")
        logger.info(f"  - QPS limit: {getattr(fresh_settings, 'qps_limit', 10)}")

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Failed to initialize concurrency manager: {e}", exc_info=True)
        _service_manager = None
        # 不抛出异常，降级到直接处理模式
        logger.warning("Falling back to direct processing mode (no batching/rate limiting)")

    yield

    # 关闭时清理
    logger.info("Shutting down services...")
    if _service_manager:
        # await _service_manager.stop()
        _service_manager = None
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
            from mindnlp.ocr.core.mock_engine import MockVLMOCREngine
            _engine = MockVLMOCREngine()
            logger.info("Mock OCR engine created successfully")
        except Exception as e:
            logger.error(f"Failed to create mock OCR engine: {e}")
            raise RuntimeError("OCR Engine not initialized") from e

    return _engine


def get_service_manager():
    """
    依赖注入: 获取全局服务管理器实例

    Returns:
        OCRServiceManager: 服务管理器实例

    Raises:
        RuntimeError: 服务管理器未初始化
    """
    global _service_manager  # pylint: disable=global-statement

    if _service_manager is None:
        raise RuntimeError("Service Manager not initialized")

    return _service_manager


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
    app.include_router(monitor.router, prefix="/api/v1/monitor", tags=["monitor"])
    app.include_router(metrics.router, prefix="/api/v1/metrics", tags=["metrics"])

    # 添加根路径重定向
    @app.get("/", include_in_schema=False)
    async def root():
        """根路径重定向到文档"""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/api/docs")

    logger.info("FastAPI application created")

    return app
