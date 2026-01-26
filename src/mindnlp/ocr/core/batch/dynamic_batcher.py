"""
动态批处理器 - 实现请求聚合和批量推理
"""

import asyncio
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from mindnlp.ocr.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BatchRequest:
    """批处理请求"""
    request_id: str
    data: Any  # OCRRequest
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # 优先级 (0=普通, 1=高, 2=紧急)
    future: Optional[asyncio.Future] = None
    
    def __lt__(self, other):
        """优先级比较 (用于优先级队列)"""
        if self.priority != other.priority:
            return self.priority > other.priority  # 高优先级优先
        return self.timestamp < other.timestamp  # 相同优先级按时间顺序


@dataclass
class BatchConfig:
    """批处理配置"""
    max_batch_size: int = 4  # 最大批大小
    wait_timeout_ms: int = 100  # 等待超时(毫秒)
    min_batch_size: int = 1  # 最小批大小
    enable_dynamic_batching: bool = True  # 启用动态批处理
    max_queue_size: int = 100  # 最大队列大小
    

class DynamicBatcher:
    """
    动态批处理器
    
    功能:
    - 请求聚合: 等待一定时间或达到批大小后批量处理
    - 优先级队列: 支持高优先级请求插队
    - 超时机制: 避免请求无限等待
    - 统计监控: 记录批处理命中率、队列长度等
    """
    
    def __init__(self, config: BatchConfig, batch_processor):
        """
        初始化批处理器
        
        Args:
            config: 批处理配置
            batch_processor: 批处理函数 (接收 List[Any], 返回 List[Any])
        """
        self.config = config
        self.batch_processor = batch_processor
        self.queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.is_running = False
        self.worker_task = None
        
        # 统计指标
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "batched_requests": 0,  # 被批处理的请求数
            "single_requests": 0,  # 单个处理的请求数
            "queue_full_rejects": 0,
            "timeout_count": 0,
            "avg_batch_size": 0.0,
            "avg_wait_time_ms": 0.0,
        }
        
        logger.info(
            f"DynamicBatcher initialized: "
            f"max_batch_size={config.max_batch_size}, "
            f"wait_timeout_ms={config.wait_timeout_ms}, "
            f"max_queue_size={config.max_queue_size}"
        )
    
    async def start(self):
        """启动批处理工作线程"""
        if self.is_running:
            logger.warning("DynamicBatcher already running")
            return
        
        self.is_running = True
        self.worker_task = asyncio.create_task(self._batch_worker())
        logger.info("DynamicBatcher started")
    
    async def stop(self):
        """停止批处理工作线程"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        
        logger.info("DynamicBatcher stopped")
    
    async def submit(
        self, 
        request_id: str, 
        data: Any, 
        priority: int = 0,
        timeout: float = 30.0
    ) -> Any:
        """
        提交请求到批处理队列
        
        Args:
            request_id: 请求ID
            data: 请求数据
            priority: 优先级 (0=普通, 1=高, 2=紧急)
            timeout: 超时时间(秒)
        
        Returns:
            处理结果
        
        Raises:
            asyncio.QueueFull: 队列已满
            asyncio.TimeoutError: 处理超时
        """
        if not self.is_running:
            raise RuntimeError("DynamicBatcher not started")
        
        # 创建 Future 用于等待结果
        future = asyncio.Future()
        batch_req = BatchRequest(
            request_id=request_id,
            data=data,
            priority=priority,
            future=future
        )
        
        # 提交到队列
        try:
            self.queue.put_nowait(batch_req)
            self.stats["total_requests"] += 1
        except asyncio.QueueFull:
            self.stats["queue_full_rejects"] += 1
            logger.warning(f"Queue full, rejecting request {request_id}")
            raise
        
        # 等待结果
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self.stats["timeout_count"] += 1
            logger.error(f"Request {request_id} timeout after {timeout}s")
            raise
    
    async def _batch_worker(self):
        """批处理工作线程"""
        logger.info("Batch worker started")
        
        while self.is_running:
            try:
                # 收集一批请求
                batch = await self._collect_batch()
                
                if not batch:
                    continue
                
                # 执行批处理
                await self._process_batch(batch)
                
            except asyncio.CancelledError:
                logger.info("Batch worker cancelled")
                break
            except Exception as e:
                logger.error(f"Batch worker error: {e}", exc_info=True)
                await asyncio.sleep(0.1)  # 避免错误循环
        
        logger.info("Batch worker stopped")
    
    async def _collect_batch(self) -> List[BatchRequest]:
        """
        收集一批请求
        
        策略:
        1. 等待第一个请求
        2. 在超时时间内尽可能收集更多请求
        3. 达到 max_batch_size 立即返回
        """
        batch = []
        timeout_s = self.config.wait_timeout_ms / 1000.0
        
        # 等待第一个请求
        try:
            first_req = await asyncio.wait_for(
                self.queue.get(), 
                timeout=1.0  # 1秒超时,避免无限等待
            )
            batch.append(first_req)
        except asyncio.TimeoutError:
            return batch
        
        # 在超时时间内收集更多请求
        start_time = time.time()
        while len(batch) < self.config.max_batch_size:
            remaining_time = timeout_s - (time.time() - start_time)
            
            if remaining_time <= 0:
                break
            
            try:
                req = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=remaining_time
                )
                batch.append(req)
            except asyncio.TimeoutError:
                break
        
        return batch
    
    async def _process_batch(self, batch: List[BatchRequest]):
        """
        处理一批请求
        
        Args:
            batch: 请求列表
        """
        batch_size = len(batch)
        start_time = time.time()
        
        try:
            # 提取数据
            data_list = [req.data for req in batch]
            
            # 执行批处理
            logger.debug(f"Processing batch of {batch_size} requests")
            results = await self.batch_processor(data_list)
            
            # 验证结果数量
            if len(results) != batch_size:
                raise ValueError(
                    f"Batch processor returned {len(results)} results "
                    f"for {batch_size} requests"
                )
            
            # 分发结果
            for req, result in zip(batch, results):
                if not req.future.cancelled():
                    req.future.set_result(result)
            
            # 更新统计
            process_time = time.time() - start_time
            wait_time_ms = (start_time - batch[0].timestamp) * 1000
            
            self.stats["total_batches"] += 1
            if batch_size > 1:
                self.stats["batched_requests"] += batch_size
            else:
                self.stats["single_requests"] += 1
            
            # 计算平均值
            total_batches = self.stats["total_batches"]
            self.stats["avg_batch_size"] = (
                (self.stats["avg_batch_size"] * (total_batches - 1) + batch_size) 
                / total_batches
            )
            self.stats["avg_wait_time_ms"] = (
                (self.stats["avg_wait_time_ms"] * (total_batches - 1) + wait_time_ms)
                / total_batches
            )
            
            logger.debug(
                f"Batch processed: size={batch_size}, "
                f"time={process_time:.3f}s, "
                f"wait={wait_time_ms:.1f}ms"
            )
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}", exc_info=True)
            
            # 通知所有请求处理失败
            for req in batch:
                if not req.future.cancelled():
                    req.future.set_exception(e)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        stats["queue_size"] = self.queue.qsize()
        stats["is_running"] = self.is_running
        
        # 计算批处理命中率
        total_requests = self.stats["total_requests"]
        if total_requests > 0:
            stats["batch_hit_rate"] = (
                self.stats["batched_requests"] / total_requests
            )
        else:
            stats["batch_hit_rate"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "batched_requests": 0,
            "single_requests": 0,
            "queue_full_rejects": 0,
            "timeout_count": 0,
            "avg_batch_size": 0.0,
            "avg_wait_time_ms": 0.0,
        }
        logger.info("Statistics reset")
