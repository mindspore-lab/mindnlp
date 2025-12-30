"""
日志记录中间件
"""

import time
import logging
from fastapi import FastAPI, Request
from utils.logger import get_logger


logger = get_logger(__name__)


def setup_logging():
    """
    配置日志系统
    """
    # 配置uvicorn日志
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.INFO)

    # 配置uvicorn.access日志
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.setLevel(logging.INFO)

    logger.info("Logging system configured")


def add_logging_middleware(app: FastAPI):
    """
    添加日志记录中间件

    Args:
        app: FastAPI应用实例
    """

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """
        记录请求和响应日志

        Args:
            request: 请求对象
            call_next: 下一个中间件

        Returns:
            Response: 响应对象
        """
        start_time = time.time()

        # 记录请求信息
        logger.info(
            f"→ {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )

        # 执行请求
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"Request processing failed: {str(e)}", exc_info=True)
            raise

        # 计算处理时间
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"

        # 记录响应信息
        logger.info(
            f"← {request.method} {request.url.path} "
            f"Status: {response.status_code} Time: {process_time:.4f}s"
        )

        return response

    logger.info("Logging middleware configured")
