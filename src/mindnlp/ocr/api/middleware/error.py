"""
错误处理中间件
"""

from fastapi import FastAPI, Request, status  # pylint: disable=import-error
from fastapi.responses import JSONResponse  # pylint: disable=import-error
from fastapi.exceptions import RequestValidationError  # pylint: disable=import-error
from starlette.exceptions import HTTPException as StarletteHTTPException  # pylint: disable=import-error
from mindnlp.ocr.utils.logger import get_logger


logger = get_logger(__name__)


def setup_exception_handlers(app: FastAPI):
    """
    配置全局异常处理

    Args:
        app: FastAPI应用实例
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """
        HTTP异常处理

        Args:
            request: 请求对象
            exc: HTTP异常

        Returns:
            JSONResponse: 错误响应
        """
        logger.warning(f"HTTP {exc.status_code}: {exc.detail} - Path: {request.url.path}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "type": "HTTPException"
                }
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """
        请求验证异常处理

        Args:
            request: 请求对象
            exc: 验证异常

        Returns:
            JSONResponse: 错误响应
        """
        logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": {
                    "code": 422,
                    "message": "Validation error",
                    "type": "ValidationError",
                    "details": exc.errors()
                }
            }
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError):
        """
        运行时错误处理

        Args:
            request: 请求对象
            exc: 运行时异常

        Returns:
            JSONResponse: 错误响应
        """
        logger.error(f"Runtime error on {request.url.path}: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "success": False,
                "error": {
                    "code": 503,
                    "message": str(exc),
                    "type": "RuntimeError"
                }
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """
        通用异常处理

        Args:
            request: 请求对象
            exc: 异常

        Returns:
            JSONResponse: 错误响应
        """
        logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": 500,
                    "message": "Internal server error",
                    "type": "Exception",
                    "details": str(exc)
                }
            }
        )

    logger.info("Exception handlers configured")
