"""
性能 Profiling 工具集

提供多种性能分析工具:
- CPU Profiling (cProfile, py-spy)
- GPU Profiling (torch.profiler)
- 内存 Profiling (memory_profiler)
- 自定义性能计时器
"""

import time
import cProfile
import pstats
import io
import logging
from typing import Optional, Any, Dict
from contextlib import contextmanager
from pathlib import Path

# 使用标准logger避免循环导入
logger = logging.getLogger(__name__)


class CPUProfiler:
    """
    CPU Profiling 工具

    使用 cProfile 分析函数执行时间
    """

    def __init__(self, output_dir: str = "profiling_results"):
        """
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profiler: Optional[cProfile.Profile] = None

    def start(self):
        """启动 profiler"""
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        logger.info("CPU profiling started")

    def stop(self, output_file: Optional[str] = None):
        """
        停止 profiler 并保存结果

        Args:
            output_file: 输出文件名 (可选)
        """
        if self.profiler is None:
            logger.warning("Profiler not started")
            return

        self.profiler.disable()

        # 保存结果
        if output_file:
            output_path = self.output_dir / output_file
            self.profiler.dump_stats(str(output_path))
            logger.info(f"Profiling results saved to {output_path}")

        # 打印统计
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.strip_dirs().sort_stats('cumulative')
        ps.print_stats(20)  # 打印前20个函数

        logger.info(f"CPU Profiling Results:\n{s.getvalue()}")

        self.profiler = None

    @contextmanager
    def profile(self, name: str = "profile"):
        """
        上下文管理器方式使用

        Args:
            name: Profile名称

        Example:
            with cpu_profiler.profile("inference"):
                # 执行推理
                pass
        """
        output_file = f"{name}_{int(time.time())}.prof"

        self.start()
        try:
            yield
        finally:
            self.stop(output_file)


class GPUProfiler:
    """
    GPU Profiling 工具

    使用 torch.profiler 分析 GPU 性能
    """

    def __init__(self, output_dir: str = "profiling_results"):
        """
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profiler: Optional[Any] = None

    def create_profiler(
        self,
        activities: list = None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True
    ):
        """
        创建 profiler

        Args:
            activities: 要profile的活动 (CPU/CUDA)
            record_shapes: 是否记录tensor shape
            profile_memory: 是否profile内存
            with_stack: 是否记录调用栈
        """
        try:
            import torch
            from torch.profiler import profile, ProfilerActivity

            if activities is None:
                activities = [ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(ProfilerActivity.CUDA)

            self.profiler = profile(
                activities=activities,
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=with_stack
            )

            logger.info("GPU profiler created")

        except ImportError:
            logger.warning("torch.profiler not available")
            self.profiler = None

    @contextmanager
    def profile(self, name: str = "gpu_profile"):
        """
        上下文管理器方式使用

        Args:
            name: Profile名称

        Example:
            with gpu_profiler.profile("forward_pass"):
                # 执行前向传播
                pass
        """
        if self.profiler is None:
            self.create_profiler()

        if self.profiler is None:
            # Profiler 不可用，直接yield
            yield
            return

        self.profiler.__enter__()

        try:
            yield
        finally:
            self.profiler.__exit__(None, None, None)

            # 保存结果
            output_file = self.output_dir / f"{name}_{int(time.time())}.json"
            self.profiler.export_chrome_trace(str(output_file))
            logger.info(f"GPU profiling results saved to {output_file}")

            # 打印统计
            logger.info(f"GPU Profiling Key Stats:\n{self.profiler.key_averages().table()}")


class MemoryProfiler:
    """
    内存 Profiling 工具

    监控内存使用情况
    """

    def __init__(self):
        """初始化内存profiler"""
        self.start_memory: Optional[float] = None
        self.peak_memory: Optional[float] = None

    def get_memory_usage(self) -> float:
        """
        获取当前内存使用

        Returns:
            内存使用量(MB)
        """
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            logger.warning("psutil not available")
            return 0.0

    def get_gpu_memory_usage(self) -> Optional[float]:
        """
        获取GPU显存使用

        Returns:
            显存使用量(MB)
        """
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024  # MB
            return None
        except ImportError:
            return None

    def start(self):
        """开始内存监控"""
        self.start_memory = self.get_memory_usage()
        self.peak_memory = self.start_memory
        logger.info(f"Memory profiling started: {self.start_memory:.2f} MB")

    def snapshot(self) -> Dict[str, float]:
        """
        获取内存快照

        Returns:
            内存使用情况
        """
        current_memory = self.get_memory_usage()
        gpu_memory = self.get_gpu_memory_usage()

        if self.peak_memory is not None:
            self.peak_memory = max(self.peak_memory, current_memory)

        snapshot = {
            "current_mb": current_memory,
            "peak_mb": self.peak_memory or current_memory,
        }

        if gpu_memory is not None:
            snapshot["gpu_mb"] = gpu_memory

        if self.start_memory is not None:
            snapshot["delta_mb"] = current_memory - self.start_memory

        return snapshot

    @contextmanager
    def profile(self, name: str = "memory_profile"):
        """
        上下文管理器方式使用

        Args:
            name: Profile名称

        Example:
            with memory_profiler.profile("data_loading"):
                # 加载数据
                pass
        """
        self.start()

        try:
            yield
        finally:
            snapshot = self.snapshot()
            logger.info(
                f"Memory Profile [{name}]: "
                f"Current={snapshot['current_mb']:.2f}MB, "
                f"Peak={snapshot['peak_mb']:.2f}MB, "
                f"Delta={snapshot.get('delta_mb', 0):.2f}MB"
            )

            if 'gpu_mb' in snapshot:
                logger.info(f"GPU Memory: {snapshot['gpu_mb']:.2f}MB")


class PerformanceTimer:
    """
    性能计时器

    测量代码块执行时间
    """

    def __init__(self, name: str = "operation"):
        """
        Args:
            name: 操作名称
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self):
        """开始计时"""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """结束计时并记录"""
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            logger.info(f"[Timer] {self.name}: {self.elapsed*1000:.2f}ms")

    def get_elapsed(self) -> Optional[float]:
        """
        获取已经过的时间

        Returns:
            已经过的时间(秒)
        """
        return self.elapsed


class ProfilingManager:
    """
    Profiling 管理器

    统一管理所有 profiler
    """

    def __init__(self, output_dir: str = "profiling_results"):
        """
        Args:
            output_dir: 输出目录
        """
        self.cpu_profiler = CPUProfiler(output_dir)
        self.gpu_profiler = GPUProfiler(output_dir)
        self.memory_profiler = MemoryProfiler()

    @contextmanager
    def profile_all(self, name: str = "full_profile"):
        """
        同时启用所有 profiler

        Args:
            name: Profile名称

        Example:
            with profiling_manager.profile_all("inference"):
                # 执行推理
                pass
        """
        # 启动所有 profiler
        with self.cpu_profiler.profile(f"{name}_cpu"), \
             self.gpu_profiler.profile(f"{name}_gpu"), \
             self.memory_profiler.profile(f"{name}_mem"):
            yield

    @contextmanager
    def profile_cpu(self, name: str = "cpu_profile"):
        """CPU profiling"""
        with self.cpu_profiler.profile(name):
            yield

    @contextmanager
    def profile_gpu(self, name: str = "gpu_profile"):
        """GPU profiling"""
        with self.gpu_profiler.profile(name):
            yield

    @contextmanager
    def profile_memory(self, name: str = "memory_profile"):
        """Memory profiling"""
        with self.memory_profiler.profile(name):
            yield


# 全局 profiling manager
_profiling_manager: Optional[ProfilingManager] = None


def get_profiling_manager() -> ProfilingManager:
    """获取全局 profiling manager"""
    global _profiling_manager
    if _profiling_manager is None:
        _profiling_manager = ProfilingManager()
    return _profiling_manager
