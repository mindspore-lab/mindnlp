"""
OCR 服务管理器 - 集成批处理、队列和限流

提供完整的并发处理能力。
"""

import asyncio
from typing import Any, Optional, List
from dataclasses import dataclass
from mindnlp.ocr.core.batching import DynamicBatcher, PriorityBatcher, BatchConfig
from mindnlp.ocr.core.queue_limiter import RateLimiter, RateLimitConfig, RequestQueue
from mindnlp.ocr.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ServiceConfig:
    """服务配置"""
    # 批处理配置
    batch_config: BatchConfig = None
    enable_priority_batching: bool = False
    
    # 限流配置
    rate_limit_config: RateLimitConfig = None
    enable_rate_limit: bool = True
    
    # 队列配置
    queue_maxsize: int = 1000
    enable_priority_queue: bool = False
    request_timeout: float = 30.0  # 请求超时（秒）
    
    def __post_init__(self):
        if self.batch_config is None:
            self.batch_config = BatchConfig()
        if self.rate_limit_config is None:
            self.rate_limit_config = RateLimitConfig()


class OCRServiceManager:
    """
    OCR 服务管理器
    
    功能:
    - 动态批处理
    - 请求队列
    - 限流保护
    - 统一监控
    """
    
    def __init__(
        self,
        engine: Any,  # VLMOCREngine 实例
        config: Optional[ServiceConfig] = None
    ):
        """
        Args:
            engine: OCR引擎实例
            config: 服务配置
        """
        self.engine = engine
        self.config = config or ServiceConfig()
        
        # 初始化组件
        self.batcher: Optional[DynamicBatcher] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self.request_queue: Optional[RequestQueue] = None
        
        self._setup_components()
    
    def _setup_components(self):
        """设置组件"""
        # 批处理器
        if self.config.enable_priority_batching:
            self.batcher = PriorityBatcher(
                process_func=self._process_batch,
                config=self.config.batch_config
            )
        else:
            self.batcher = DynamicBatcher(
                process_func=self._process_batch,
                config=self.config.batch_config
            )
        
        # 限流器
        if self.config.enable_rate_limit:
            self.rate_limiter = RateLimiter(
                config=self.config.rate_limit_config
            )
        
        # 请求队列
        self.request_queue = RequestQueue(
            maxsize=self.config.queue_maxsize,
            enable_priority=self.config.enable_priority_queue
        )
        
        logger.info(
            f"OCRServiceManager initialized: "
            f"batch_size={self.config.batch_config.max_batch_size}, "
            f"qps_limit={self.config.rate_limit_config.qps_limit}, "
            f"queue_size={self.config.queue_maxsize}"
        )
    
    async def start(self):
        """启动服务"""
        if self.batcher:
            await self.batcher.start()
        
        logger.info("OCRServiceManager started")
    
    async def stop(self):
        """停止服务"""
        if self.batcher:
            await self.batcher.stop()
        
        logger.info("OCRServiceManager stopped")
    
    async def process_request(
        self,
        request_id: str,
        image_data: Any,
        priority: int = 0
    ) -> Any:
        """
        处理OCR请求
        
        Args:
            request_id: 请求ID
            image_data: 图像数据
            priority: 优先级
        
        Returns:
            OCR结果
        """
        # 限流检查
        if self.rate_limiter:
            if not await self.rate_limiter.acquire():
                raise Exception("Rate limit exceeded. Please try again later.")
        
        try:
            # 提交到批处理器
            if self.batcher:
                result = await self.batcher.submit(
                    request_id=request_id,
                    data=image_data,
                    priority=priority
                )
            else:
                # 直接处理（单个）
                result = await self._process_single(image_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            raise
    
    async def _process_batch(self, batch_images: List[Any]) -> List[Any]:
        """
        批处理图像
        
        Args:
            batch_images: 图像列表
        
        Returns:
            结果列表
        """
        try:
            # 调用引擎的批处理方法
            if hasattr(self.engine, 'process_batch'):
                results = await asyncio.to_thread(
                    self.engine.process_batch,
                    batch_images
                )
            else:
                # 逐个处理
                results = []
                for img in batch_images:
                    result = await asyncio.to_thread(
                        self.engine.process,
                        img
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            # 返回错误结果
            return [{"error": str(e)} for _ in batch_images]
    
    async def _process_single(self, image_data: Any) -> Any:
        """
        处理单个图像
        
        Args:
            image_data: 图像数据
        
        Returns:
            OCR结果
        """
        return await asyncio.to_thread(
            self.engine.process,
            image_data
        )
    
    def get_stats(self) -> dict:
        """获取服务统计信息"""
        stats = {
            'service': {
                'is_running': self.batcher.is_running if self.batcher else False,
            }
        }
        
        if self.batcher:
            stats['batching'] = self.batcher.get_stats()
        
        if self.rate_limiter:
            stats['rate_limiting'] = self.rate_limiter.get_stats()
        
        if self.request_queue:
            stats['queue'] = self.request_queue.get_stats()
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        if self.batcher:
            self.batcher.reset_stats()
        
        if self.rate_limiter:
            self.rate_limiter.reset_stats()
