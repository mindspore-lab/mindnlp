"""
限流器 - Token Bucket算法实现
"""

import time
import asyncio
from dataclasses import dataclass
from typing import Optional
from mindnlp.ocr.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """限流配置"""
    qps: int = 100  # 每秒查询数
    burst: int = 20  # 突发容量
    enable_circuit_breaker: bool = True  # 启用熔断器
    circuit_breaker_threshold: int = 10  # 熔断阈值(连续失败次数)
    circuit_breaker_timeout_s: int = 30  # 熔断恢复时间(秒)


class TokenBucketRateLimiter:
    """
    Token Bucket 限流器
    
    原理:
    - 桶中存储 token,每个请求消耗1个token
    - token 以固定速率生成 (rate = qps)
    - 桶容量 = qps + burst (支持突发流量)
    - 无token时拒绝请求
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        初始化限流器
        
        Args:
            config: 限流配置
        """
        self.config = config
        self.capacity = config.qps + config.burst  # 桶容量
        self.tokens = float(self.capacity)  # 当前token数
        self.rate = config.qps  # token生成速率(个/秒)
        self.last_update = time.time()
        self.lock = asyncio.Lock()
        
        # 统计
        self.stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "rejected_requests": 0,
        }
        
        logger.info(
            f"TokenBucketRateLimiter initialized: "
            f"qps={config.qps}, burst={config.burst}, capacity={self.capacity}"
        )
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        获取token (非阻塞)
        
        Args:
            tokens: 需要的token数
        
        Returns:
            True if acquired, False if rejected
        """
        async with self.lock:
            # 更新token数
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            # 统计
            self.stats["total_requests"] += 1
            
            # 检查是否有足够token
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.stats["allowed_requests"] += 1
                return True
            else:
                self.stats["rejected_requests"] += 1
                logger.debug(
                    f"Rate limit exceeded: {self.tokens:.2f}/{self.capacity} tokens"
                )
                return False
    
    async def wait_for_token(self, tokens: int = 1, timeout: Optional[float] = None):
        """
        等待token (阻塞,直到获取成功或超时)
        
        Args:
            tokens: 需要的token数
            timeout: 超时时间(秒), None表示无限等待
        
        Raises:
            asyncio.TimeoutError: 超时
        """
        start_time = time.time()
        
        while True:
            if await self.acquire(tokens):
                return
            
            # 检查超时
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise asyncio.TimeoutError(
                        f"Failed to acquire token within {timeout}s"
                    )
            
            # 计算等待时间
            async with self.lock:
                if self.tokens < tokens:
                    wait_time = (tokens - self.tokens) / self.rate
                else:
                    wait_time = 0.01
            
            await asyncio.sleep(min(wait_time, 0.1))
    
    def get_stats(self):
        """获取统计信息"""
        stats = self.stats.copy()
        stats["current_tokens"] = self.tokens
        stats["capacity"] = self.capacity
        stats["rate"] = self.rate
        
        if stats["total_requests"] > 0:
            stats["reject_rate"] = (
                stats["rejected_requests"] / stats["total_requests"]
            )
        else:
            stats["reject_rate"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """重置统计"""
        self.stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "rejected_requests": 0,
        }


class CircuitBreaker:
    """
    熔断器 - 防止服务雪崩
    
    状态:
    - CLOSED: 正常工作
    - OPEN: 熔断打开,拒绝所有请求
    - HALF_OPEN: 半开状态,允许部分请求测试服务是否恢复
    """
    
    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half_open"
    
    def __init__(self, config: RateLimitConfig):
        """
        初始化熔断器
        
        Args:
            config: 限流配置
        """
        self.config = config
        self.state = self.STATE_CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.lock = asyncio.Lock()
        
        # 统计
        self.stats = {
            "total_calls": 0,
            "success_calls": 0,
            "failure_calls": 0,
            "rejected_calls": 0,  # 熔断拒绝的请求
            "state_changes": 0,
        }
        
        logger.info(
            f"CircuitBreaker initialized: "
            f"threshold={config.circuit_breaker_threshold}, "
            f"timeout={config.circuit_breaker_timeout_s}s"
        )
    
    async def call(self, func, *args, **kwargs):
        """
        通过熔断器调用函数
        
        Args:
            func: 要调用的函数
            *args, **kwargs: 函数参数
        
        Returns:
            函数返回值
        
        Raises:
            RuntimeError: 熔断器打开,拒绝请求
            Exception: 函数执行异常
        """
        async with self.lock:
            self.stats["total_calls"] += 1
            
            # 检查熔断器状态
            if self.state == self.STATE_OPEN:
                # 检查是否可以转为半开状态
                if (
                    self.last_failure_time and
                    time.time() - self.last_failure_time >= 
                    self.config.circuit_breaker_timeout_s
                ):
                    logger.info("CircuitBreaker: OPEN -> HALF_OPEN")
                    self.state = self.STATE_HALF_OPEN
                    self.stats["state_changes"] += 1
                else:
                    self.stats["rejected_calls"] += 1
                    raise RuntimeError("Circuit breaker is OPEN")
        
        # 执行函数
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # 成功回调
            await self._on_success()
            return result
            
        except Exception as e:
            # 失败回调
            await self._on_failure()
            raise
    
    async def _on_success(self):
        """成功回调"""
        async with self.lock:
            self.stats["success_calls"] += 1
            
            if self.state == self.STATE_HALF_OPEN:
                # 半开状态成功,转为关闭
                logger.info("CircuitBreaker: HALF_OPEN -> CLOSED")
                self.state = self.STATE_CLOSED
                self.failure_count = 0
                self.stats["state_changes"] += 1
    
    async def _on_failure(self):
        """失败回调"""
        async with self.lock:
            self.stats["failure_calls"] += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == self.STATE_HALF_OPEN:
                # 半开状态失败,重新打开
                logger.warning("CircuitBreaker: HALF_OPEN -> OPEN")
                self.state = self.STATE_OPEN
                self.stats["state_changes"] += 1
            
            elif (
                self.state == self.STATE_CLOSED and
                self.failure_count >= self.config.circuit_breaker_threshold
            ):
                # 连续失败达到阈值,打开熔断器
                logger.error(
                    f"CircuitBreaker: CLOSED -> OPEN "
                    f"(failures={self.failure_count})"
                )
                self.state = self.STATE_OPEN
                self.stats["state_changes"] += 1
    
    def get_stats(self):
        """获取统计信息"""
        stats = self.stats.copy()
        stats["state"] = self.state
        stats["failure_count"] = self.failure_count
        
        if stats["total_calls"] > 0:
            stats["failure_rate"] = (
                stats["failure_calls"] / stats["total_calls"]
            )
            stats["reject_rate"] = (
                stats["rejected_calls"] / stats["total_calls"]
            )
        else:
            stats["failure_rate"] = 0.0
            stats["reject_rate"] = 0.0
        
        return stats
    
    def reset(self):
        """重置熔断器"""
        self.state = self.STATE_CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.stats = {
            "total_calls": 0,
            "success_calls": 0,
            "failure_calls": 0,
            "rejected_calls": 0,
            "state_changes": 0,
        }
        logger.info("CircuitBreaker reset")
