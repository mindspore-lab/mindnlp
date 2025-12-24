"""
配置设置
"""

from typing import Optional
from pydantic import BaseModel
from functools import lru_cache


class Settings(BaseModel):
    """应用配置"""
    
    # API配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    
    # 模型配置
    default_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    device: str = "cuda"
    
    # 图像处理配置
    max_image_size: int = 10 * 1024 * 1024  # 10MB
    target_image_size: tuple = (448, 448)
    
    # 推理配置
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # 日志配置
    log_level: str = "INFO"
    
    class Config:
        env_prefix = "OCR_"


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置单例
    
    Returns:
        Settings: 配置对象
    """
    return Settings()
