"""
动态批处理模块 - Dynamic Batching

实现请求聚合和批量处理，提升推理吞吐量。
"""

import asyncio
import time
from typing import List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
from mindnlp.ocr.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BatchRequest:
    """批处理请求"""
    request_id: str
    data: Any
    future: asyncio.Future
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # 优先级，数字越小优先级越高


@dataclass
class BatchConfig:
    """批处理配置"""
    max_batch_size: int = 4
    wait_timeout_ms: int = 100
    enable_dynamic_batching: bool = True
    min_batch_size: int = 1


class DynamicBatcher:
    """
    动态批处理器

    功能:
    - 聚合多个请求到一个batch
    - 支持超时机制（等待窗口）
    - 支持最大batch size限制
    - 支持优先级调度
    """

    def __init__(
        self,
        process_func: Callable,
        config: Optional[BatchConfig] = None
    ):
        """
        Args:
            process_func: 批处理函数，接收List[Any]，返回List[Any]
            config: 批处理配置
        """
        self.process_func = process_func
        self.config = config or BatchConfig()

        self.queue: deque[BatchRequest] = deque()
        self.lock = asyncio.Lock()
        self.condition = asyncio.Condition(self.lock)
        self.is_running = False
        self.worker_task: Optional[asyncio.Task] = None

        # 统计信息
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'total_batch_size': 0,
            'avg_batch_size': 0.0,
            'batch_hit_rate': 0.0,
        }

    async def start(self):
        """启动批处理worker"""
        if self.is_running:
            logger.warning("DynamicBatcher already running")
            return

        self.is_running = True
        self.worker_task = asyncio.create_task(self._worker_loop())
        logger.info(
            f"DynamicBatcher started: "
            f"max_batch_size={self.config.max_batch_size}, "
            f"wait_timeout={self.config.wait_timeout_ms}ms"
        )

    async def stop(self):
        """停止批处理worker"""
        if not self.is_running:
            return

        self.is_running = False

        # 唤醒worker以便退出
        async with self.condition:
            self.condition.notify()

        # 等待worker完成
        if self.worker_task:
            await self.worker_task
            self.worker_task = None

        logger.info("DynamicBatcher stopped")

    async def submit(
        self,
        request_id: str,
        data: Any,
        priority: int = 0
    ) -> Any:
        """
        提交请求到批处理队列

        Args:
            request_id: 请求ID
            data: 请求数据
            priority: 优先级（数字越小优先级越高）

        Returns:
            处理结果
        """
        if not self.is_running:
            raise RuntimeError("DynamicBatcher not running")

        # 创建future等待结果
        future = asyncio.get_event_loop().create_future()

        request = BatchRequest(
            request_id=request_id,
            data=data,
            future=future,
            priority=priority
        )

        async with self.condition:
            self.queue.append(request)
            self.stats['total_requests'] += 1

            # 如果达到最大batch size，立即唤醒worker
            if len(self.queue) >= self.config.max_batch_size:
                self.condition.notify()

        # 等待结果
        return await future

    async def _worker_loop(self):
        """Worker循环，持续处理批次"""
        logger.info("DynamicBatcher worker started")

        while self.is_running:
            try:
                batch = await self._collect_batch()

                if not batch:
                    # 没有请求，继续等待
                    await asyncio.sleep(0.01)
                    continue

                # 处理批次
                await self._process_batch(batch)

            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

        logger.info("DynamicBatcher worker stopped")

    async def _collect_batch(self) -> List[BatchRequest]:
        """
        收集一个批次的请求

        策略:
        1. 如果队列为空，等待新请求或超时
        2. 如果队列有请求，等待更多请求或超时
        3. 如果达到max_batch_size，立即返回
        """
        async with self.condition:
            # 等待第一个请求
            if not self.queue:
                try:
                    await asyncio.wait_for(
                        self.condition.wait(),
                        timeout=self.config.wait_timeout_ms / 1000
                    )
                except asyncio.TimeoutError:
                    return []

            if not self.queue:
                return []

            # 如果未达到max_batch_size，等待更多请求
            if len(self.queue) < self.config.max_batch_size:
                try:
                    await asyncio.wait_for(
                        self.condition.wait(),
                        timeout=self.config.wait_timeout_ms / 1000
                    )
                except asyncio.TimeoutError:
                    pass

            # 收集batch
            batch_size = min(len(self.queue), self.config.max_batch_size)
            batch = []

            # 按优先级排序（如果需要）
            if any(req.priority != 0 for req in self.queue):
                sorted_queue = sorted(self.queue, key=lambda x: x.priority)
                self.queue.clear()
                self.queue.extend(sorted_queue)

            for _ in range(batch_size):
                if self.queue:
                    batch.append(self.queue.popleft())

            return batch

    async def _process_batch(self, batch: List[BatchRequest]):
        """处理一个批次"""
        if not batch:
            return

        batch_size = len(batch)
        logger.debug(f"Processing batch of size {batch_size}")

        try:
            # 提取数据
            batch_data = [req.data for req in batch]

            # 调用处理函数
            results = await self._call_process_func(batch_data)

            # 确保结果数量匹配
            if len(results) != len(batch):
                raise ValueError(
                    f"Process function returned {len(results)} results "
                    f"for {len(batch)} requests"
                )

            # 设置每个请求的结果
            for request, result in zip(batch, results):
                if not request.future.done():
                    request.future.set_result(result)

            # 更新统计
            self.stats['total_batches'] += 1
            self.stats['total_batch_size'] += batch_size
            self.stats['avg_batch_size'] = (
                self.stats['total_batch_size'] / self.stats['total_batches']
            )
            self.stats['batch_hit_rate'] = (
                (self.stats['total_batch_size'] - self.stats['total_batches'])
                / self.stats['total_requests']
                if self.stats['total_requests'] > 0 else 0.0
            )

            logger.debug(
                f"Batch processed successfully: size={batch_size}, "
                f"avg_batch_size={self.stats['avg_batch_size']:.2f}"
            )

        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)

            # 设置所有请求的异常
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)

    async def _call_process_func(self, batch_data: List[Any]) -> List[Any]:
        """调用处理函数（支持同步和异步）"""
        if asyncio.iscoroutinefunction(self.process_func):
            return await self.process_func(batch_data)
        else:
            # 在executor中运行同步函数
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.process_func,
                batch_data
            )

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            **self.stats,
            'queue_size': len(self.queue),
            'is_running': self.is_running,
        }

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'total_batch_size': 0,
            'avg_batch_size': 0.0,
            'batch_hit_rate': 0.0,
        }


class PriorityBatcher(DynamicBatcher):
    """
    带优先级的批处理器

    扩展功能:
    - 优先处理高优先级请求
    - 避免低优先级请求饥饿
    """

    def __init__(
        self,
        process_func: Callable,
        config: Optional[BatchConfig] = None,
        max_wait_time: float = 5.0  # 最大等待时间，避免饥饿
    ):
        super().__init__(process_func, config)
        self.max_wait_time = max_wait_time

    async def _collect_batch(self) -> List[BatchRequest]:
        """
        收集批次，考虑优先级和等待时间
        """
        async with self.condition:
            # 等待第一个请求
            if not self.queue:
                try:
                    await asyncio.wait_for(
                        self.condition.wait(),
                        timeout=self.config.wait_timeout_ms / 1000
                    )
                except asyncio.TimeoutError:
                    return []

            if not self.queue:
                return []

            # 检查是否有请求等待过久
            current_time = time.time()
            has_starving = any(
                (current_time - req.timestamp) > self.max_wait_time
                for req in self.queue
            )

            # 如果未达到max_batch_size且没有饥饿请求，等待更多
            if len(self.queue) < self.config.max_batch_size and not has_starving:
                try:
                    await asyncio.wait_for(
                        self.condition.wait(),
                        timeout=self.config.wait_timeout_ms / 1000
                    )
                except asyncio.TimeoutError:
                    pass

            # 按优先级和等待时间排序
            def sort_key(req: BatchRequest) -> Tuple[int, float]:
                wait_time = current_time - req.timestamp
                # 等待过久的请求提升优先级
                if wait_time > self.max_wait_time:
                    return (0, wait_time)
                return (req.priority, wait_time)

            sorted_queue = sorted(self.queue, key=sort_key)
            self.queue.clear()
            self.queue.extend(sorted_queue)

            # 收集batch
            batch_size = min(len(self.queue), self.config.max_batch_size)
            batch = []
            for _ in range(batch_size):
                if self.queue:
                    batch.append(self.queue.popleft())

            return batch
