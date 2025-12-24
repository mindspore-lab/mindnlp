"""
MindNLP VLM-OCR 服务启动入口
"""

import uvicorn
from config.settings import get_settings
from utils.logger import get_logger


logger = get_logger(__name__)
settings = get_settings()


def main():
    """主函数"""
    logger.info("Starting MindNLP VLM-OCR Service...")
    logger.info(f"Model: {settings.default_model}")
    logger.info(f"Device: {settings.device}")
    logger.info(f"Host: {settings.api_host}:{settings.api_port}")
    
    # 启动服务 (使用factory模式)
    uvicorn.run(
        "api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
