"""
结构化日志系统 - JSON格式日志记录

使用 structlog 实现结构化日志,支持:
- JSON 格式输出
- 请求追踪 (request_id)
- 性能指标记录
- 自动字段绑定
"""

import sys
import logging
import structlog
from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime


def setup_structured_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = True,
    enable_console: bool = True
) -> None:
    """
    配置结构化日志系统
    
    Args:
        log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        log_file: 日志文件路径 (可选)
        json_format: 是否使用JSON格式
        enable_console: 是否输出到控制台
    """
    # 配置 structlog
    timestamper = structlog.processors.TimeStamper(fmt="iso")
    
    # 处理器链
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if json_format:
        # JSON 格式
        structlog.configure(
            processors=shared_processors + [
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # 人类可读格式
        structlog.configure(
            processors=shared_processors + [
                structlog.dev.ConsoleRenderer()
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    # 配置标准 logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout if enable_console else None,
        level=getattr(logging, log_level.upper()),
    )
    
    # 配置文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logging.root.addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    获取结构化日志器
    
    Args:
        name: 日志器名称 (通常是模块名)
    
    Returns:
        structlog 日志器
    """
    return structlog.get_logger(name)


class LogContext:
    """
    日志上下文管理器
    
    用于在特定代码块中绑定额外的日志字段
    
    Example:
        with LogContext(request_id="123", user_id="456"):
            logger.info("processing request")
            # 自动包含 request_id 和 user_id
    """
    
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: 要绑定的日志字段
        """
        self.context = kwargs
        self.tokens = []
    
    def __enter__(self):
        # 绑定上下文变量
        for key, value in self.context.items():
            token = structlog.contextvars.bind_contextvars(**{key: value})
            self.tokens.append(token)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 清除上下文变量
        for token in reversed(self.tokens):
            structlog.contextvars.unbind_contextvars(token)


class RequestLogger:
    """
    请求日志记录器
    
    自动记录HTTP请求的关键信息
    """
    
    def __init__(self, logger_name: str = "ocr.api"):
        """
        Args:
            logger_name: 日志器名称
        """
        self.logger = get_logger(logger_name)
    
    def log_request(
        self,
        method: str,
        endpoint: str,
        request_id: str,
        status_code: int,
        latency_ms: float,
        error: Optional[str] = None,
        **extra_fields
    ):
        """
        记录HTTP请求
        
        Args:
            method: HTTP方法
            endpoint: 请求端点
            request_id: 请求ID
            status_code: 响应状态码
            latency_ms: 延迟(毫秒)
            error: 错误信息(可选)
            **extra_fields: 额外字段
        """
        log_data = {
            "event": "http_request",
            "method": method,
            "endpoint": endpoint,
            "request_id": request_id,
            "status_code": status_code,
            "latency_ms": latency_ms,
            **extra_fields
        }
        
        if error:
            log_data["error"] = error
            self.logger.error("request_failed", **log_data)
        elif status_code >= 500:
            self.logger.error("request_server_error", **log_data)
        elif status_code >= 400:
            self.logger.warning("request_client_error", **log_data)
        else:
            self.logger.info("request_completed", **log_data)
    
    def log_inference(
        self,
        request_id: str,
        model_name: str,
        inference_time_ms: float,
        batch_size: int = 1,
        success: bool = True,
        error: Optional[str] = None,
        **extra_fields
    ):
        """
        记录模型推理
        
        Args:
            request_id: 请求ID
            model_name: 模型名称
            inference_time_ms: 推理时间(毫秒)
            batch_size: 批次大小
            success: 是否成功
            error: 错误信息(可选)
            **extra_fields: 额外字段
        """
        log_data = {
            "event": "model_inference",
            "request_id": request_id,
            "model_name": model_name,
            "inference_time_ms": inference_time_ms,
            "batch_size": batch_size,
            "success": success,
            **extra_fields
        }
        
        if error:
            log_data["error"] = error
            self.logger.error("inference_failed", **log_data)
        else:
            self.logger.info("inference_completed", **log_data)


class PerformanceLogger:
    """
    性能日志记录器
    
    记录系统资源使用和性能指标
    """
    
    def __init__(self, logger_name: str = "ocr.performance"):
        """
        Args:
            logger_name: 日志器名称
        """
        self.logger = get_logger(logger_name)
    
    def log_resource_usage(
        self,
        cpu_percent: float,
        memory_mb: float,
        gpu_utilization: Optional[float] = None,
        gpu_memory_mb: Optional[float] = None,
        **extra_fields
    ):
        """
        记录资源使用情况
        
        Args:
            cpu_percent: CPU使用率(百分比)
            memory_mb: 内存使用(MB)
            gpu_utilization: GPU利用率(百分比,可选)
            gpu_memory_mb: GPU显存使用(MB,可选)
            **extra_fields: 额外字段
        """
        log_data = {
            "event": "resource_usage",
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            **extra_fields
        }
        
        if gpu_utilization is not None:
            log_data["gpu_utilization"] = gpu_utilization
        if gpu_memory_mb is not None:
            log_data["gpu_memory_mb"] = gpu_memory_mb
        
        self.logger.info("resource_snapshot", **log_data)
    
    def log_queue_metrics(
        self,
        queue_size: int,
        queue_capacity: int,
        avg_wait_time_ms: float,
        **extra_fields
    ):
        """
        记录队列指标
        
        Args:
            queue_size: 当前队列大小
            queue_capacity: 队列容量
            avg_wait_time_ms: 平均等待时间(毫秒)
            **extra_fields: 额外字段
        """
        log_data = {
            "event": "queue_metrics",
            "queue_size": queue_size,
            "queue_capacity": queue_capacity,
            "queue_utilization": queue_size / queue_capacity if queue_capacity > 0 else 0,
            "avg_wait_time_ms": avg_wait_time_ms,
            **extra_fields
        }
        
        self.logger.info("queue_snapshot", **log_data)


# 全局日志器实例
_request_logger: Optional[RequestLogger] = None
_performance_logger: Optional[PerformanceLogger] = None


def get_request_logger() -> RequestLogger:
    """获取请求日志器"""
    global _request_logger
    if _request_logger is None:
        _request_logger = RequestLogger()
    return _request_logger


def get_performance_logger() -> PerformanceLogger:
    """获取性能日志器"""
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger()
    return _performance_logger
