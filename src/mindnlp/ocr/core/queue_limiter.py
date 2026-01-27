"""
请求队列和限流模块

实现:
- 异步请求队列
- Token Bucket 限流算法
- Circuit Breaker 熔断器
- 背压机制
"""

import asyncio
import time
from typing import Optional, Callable
from dataclasses import dataclass
from mindnlp.ocr.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """限流配置"""
    qps_limit: int = 100  # 每秒请求数限制
    bucket_capacity: int = 100  # 桶容量
    refill_rate: float = 100.0  # 令牌补充速率（tokens/s）
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 50  # 熔断阈值（连续失败次数）
    circuit_breaker_timeout: float = 30.0  # 熔断恢复时间（秒）


class TokenBucket:
    """
    令牌桶限流器

    原理:
    - 固定容量的桶
    - 以固定速率补充令牌
    - 请求消耗令牌，无令牌则拒绝
    """

    def __init__(
        self,
        capacity: int,
        refill_rate: float
    ):
        """
        Args:
            capacity: 桶容量（最大令牌数）
            refill_rate: 补充速率（tokens/s）
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        获取令牌

        Args:
            tokens: 需要的令牌数
            timeout: 超时时间（秒），None表示不等待

        Returns:
            是否成功获取令牌
        """
        start_time = time.time()

        while True:
            async with self.lock:
                # 补充令牌
                await self._refill()

                # 检查是否有足够令牌
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

            # 如果不等待或超时，返回失败
            if timeout is None:
                return False

            if timeout is not None and (time.time() - start_time) >= timeout:
                return False

            # 等待一小段时间再重试
            await asyncio.sleep(0.01)

    async def _refill(self):
        """补充令牌"""
        now = time.time()
        elapsed = now - self.last_refill

        # 计算应补充的令牌数
        new_tokens = elapsed * self.refill_rate

        if new_tokens > 0:
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now

    async def get_available_tokens(self) -> float:
        """获取当前可用令牌数"""
        async with self.lock:
            await self._refill()
            return self.tokens


class CircuitBreaker:
    """
    熔断器

    状态:
    - CLOSED: 正常状态，请求正常通过
    - OPEN: 熔断状态，拒绝所有请求
    - HALF_OPEN: 半开状态，允许少量请求测试服务是否恢复
    """

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(
        self,
        failure_threshold: int = 50,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3
    ):
        """
        Args:
            failure_threshold: 连续失败次数阈值
            recovery_timeout: 熔断后恢复时间（秒）
            half_open_max_calls: 半开状态最大请求数
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = self.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0

        self.lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs):
        """
        通过熔断器调用函数

        Args:
            func: 要调用的函数
            *args, **kwargs: 函数参数

        Returns:
            函数返回值

        Raises:
            Exception: 熔断器OPEN状态或函数执行失败
        """
        async with self.lock:
            # 检查状态
            if self.state == self.OPEN:
                # 检查是否可以进入半开状态
                if self.last_failure_time and \
                   (time.time() - self.last_failure_time) >= self.recovery_timeout:
                    logger.info("Circuit breaker entering HALF_OPEN state")
                    self.state = self.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise Exception("Circuit breaker is OPEN")

            if self.state == self.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise Exception("Circuit breaker HALF_OPEN limit reached")
                self.half_open_calls += 1

        # 执行函数
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # 成功
            await self._on_success()
            return result

        except Exception as e:
            # 失败
            await self._on_failure()
            raise e

    async def _on_success(self):
        """处理成功调用"""
        async with self.lock:
            self.success_count += 1

            if self.state == self.HALF_OPEN:
                # 半开状态成功，尝试关闭
                if self.success_count >= self.half_open_max_calls:
                    logger.info("Circuit breaker entering CLOSED state")
                    self.state = self.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == self.CLOSED:
                # 关闭状态成功，重置失败计数
                self.failure_count = 0

    async def _on_failure(self):
        """处理失败调用"""
        async with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == self.HALF_OPEN:
                # 半开状态失败，重新打开
                logger.warning("Circuit breaker re-entering OPEN state")
                self.state = self.OPEN
                self.success_count = 0
            elif self.state == self.CLOSED:
                # 关闭状态失败次数过多，打开熔断器
                if self.failure_count >= self.failure_threshold:
                    logger.error(
                        f"Circuit breaker entering OPEN state "
                        f"(failures: {self.failure_count})"
                    )
                    self.state = self.OPEN
                    self.success_count = 0

    def get_state(self) -> str:
        """获取当前状态"""
        return self.state

    def reset(self):
        """重置熔断器"""
        self.state = self.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0


class RateLimiter:
    """
    综合限流器

    结合:
    - Token Bucket 限流
    - Circuit Breaker 熔断
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()

        # Token Bucket
        self.token_bucket = TokenBucket(
            capacity=self.config.bucket_capacity,
            refill_rate=self.config.refill_rate
        )

        # Circuit Breaker
        self.circuit_breaker = None
        if self.config.enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=self.config.circuit_breaker_threshold,
                recovery_timeout=self.config.circuit_breaker_timeout
            )

        # 统计
        self.stats = {
            'total_requests': 0,
            'accepted_requests': 0,
            'rejected_requests': 0,
            'circuit_breaker_rejects': 0,
        }

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        获取访问许可

        Args:
            timeout: 超时时间（秒）

        Returns:
            是否获得许可
        """
        self.stats['total_requests'] += 1

        # 检查熔断器
        if self.circuit_breaker and self.circuit_breaker.get_state() == CircuitBreaker.OPEN:
            self.stats['rejected_requests'] += 1
            self.stats['circuit_breaker_rejects'] += 1
            return False

        # 获取令牌
        acquired = await self.token_bucket.acquire(tokens=1, timeout=timeout)

        if acquired:
            self.stats['accepted_requests'] += 1
        else:
            self.stats['rejected_requests'] += 1

        return acquired

    async def call(self, func: Callable, *args, **kwargs):
        """
        通过限流器调用函数

        包含:
        - 令牌桶限流
        - 熔断器保护
        """
        # 获取令牌
        if not await self.acquire():
            raise Exception("Rate limit exceeded")

        # 通过熔断器调用
        if self.circuit_breaker:
            return await self.circuit_breaker.call(func, *args, **kwargs)
        else:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

    def get_stats(self) -> dict:
        """获取统计信息"""
        result = {**self.stats}

        if self.circuit_breaker:
            result['circuit_breaker_state'] = self.circuit_breaker.get_state()

        return result

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_requests': 0,
            'accepted_requests': 0,
            'rejected_requests': 0,
            'circuit_breaker_rejects': 0,
        }


class RequestQueue:
    """
    请求队列

    功能:
    - 异步队列
    - 优先级支持
    - 队列长度限制（背压）
    - 超时控制
    """

    def __init__(
        self,
        maxsize: int = 1000,
        enable_priority: bool = False
    ):
        """
        Args:
            maxsize: 最大队列长度
            enable_priority: 是否启用优先级队列
        """
        self.maxsize = maxsize
        self.enable_priority = enable_priority

        if enable_priority:
            self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=maxsize)
        else:
            self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)

        self.stats = {
            'total_enqueued': 0,
            'total_dequeued': 0,
            'total_timeout': 0,
            'total_rejected': 0,
        }

    async def put(
        self,
        item: any,
        priority: int = 0,
        timeout: Optional[float] = None
    ) -> bool:
        """
        放入队列

        Args:
            item: 队列项
            priority: 优先级（仅priority queue）
            timeout: 超时时间（秒）

        Returns:
            是否成功放入
        """
        try:
            if self.enable_priority:
                queue_item = (priority, time.time(), item)
            else:
                queue_item = item

            if timeout is not None:
                await asyncio.wait_for(
                    self.queue.put(queue_item),
                    timeout=timeout
                )
            else:
                # 非阻塞put
                self.queue.put_nowait(queue_item)

            self.stats['total_enqueued'] += 1
            return True

        except asyncio.TimeoutError:
            self.stats['total_timeout'] += 1
            return False
        except asyncio.QueueFull:
            self.stats['total_rejected'] += 1
            return False

    async def get(self, timeout: Optional[float] = None):
        """
        从队列获取

        Args:
            timeout: 超时时间（秒）

        Returns:
            队列项
        """
        try:
            if timeout is not None:
                queue_item = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=timeout
                )
            else:
                queue_item = await self.queue.get()

            self.stats['total_dequeued'] += 1

            # 解包priority queue item
            if self.enable_priority:
                _, _, item = queue_item
                return item
            else:
                return queue_item

        except asyncio.TimeoutError:
            raise

    def qsize(self) -> int:
        """获取队列大小"""
        return self.queue.qsize()

    def empty(self) -> bool:
        """队列是否为空"""
        return self.queue.empty()

    def full(self) -> bool:
        """队列是否已满"""
        return self.queue.full()

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            **self.stats,
            'current_size': self.qsize(),
            'is_full': self.full(),
        }
