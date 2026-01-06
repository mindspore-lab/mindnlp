"""
配置设置
"""

import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings

# 加载 .env 文件
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # 如果没有安装 python-dotenv，继续使用环境变量


class Settings(BaseSettings):
    """应用配置"""

    # API配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # 模型配置
    default_model: str = "Qwen/Qwen2-VL-7B-Instruct"  # 使用7B模型提高精度
    device: str = "npu:0"

    @property
    def use_mock_engine(self) -> bool:
        """是否使用 Mock 引擎（用于测试）"""
        return os.getenv("OCR_USE_MOCK_ENGINE", "false").lower() in ("true", "1", "yes")

    # 图像处理配置
    max_image_size: int = 10 * 1024 * 1024  # 10MB
    target_image_size: tuple = (448, 448)

    # 推理配置（针对 NPU 优化）
    max_new_tokens: int = 512  # 减少生成长度提升速度
    temperature: float = 0.1  # 降低温度提高准确性
    top_p: float = 0.9
    batch_size: int = 1  # NPU 批处理大小
    use_cache: bool = True  # 启用 KV cache

    # 日志配置
    log_level: str = "INFO"

    class Config:
        env_prefix = "OCR_"
        # 允许额外的环境变量（不会报错）
        extra = "ignore"
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置单例

    Returns:
        Settings: 配置对象
    """
    return Settings()
