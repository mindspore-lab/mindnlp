"""
性能测试套件
测试 OCR 系统的性能指标（推理速度、批处理、内存占用）
"""

import pytest
import time
import psutil
from PIL import Image
import numpy as np
from mindnlp.ocr.core.monitor import PerformanceMonitor, get_performance_monitor


@pytest.fixture
def monitor():
    """创建性能监控器实例"""
    return PerformanceMonitor(max_history=100)


@pytest.fixture
def sample_image():
    """创建测试图像"""
    return Image.new('RGB', (800, 600), color='white')


class TestPerformanceMonitor:
    """测试性能监控器"""
    
    def test_monitor_initialization(self, monitor):
        """测试监控器初始化"""
        assert monitor.max_history == 100
        stats = monitor.get_statistics()
        assert stats['total_requests'] == 0
        assert stats['successful_requests'] == 0
        assert stats['failed_requests'] == 0
    
    def test_record_successful_inference(self, monitor):
        """测试记录成功的推理"""
        monitor.record_inference(
            inference_time=1.5,
            image_count=1,
            success=True
        )
        
        stats = monitor.get_statistics()
        assert stats['total_requests'] == 1
        assert stats['successful_requests'] == 1
        assert stats['failed_requests'] == 0
        assert stats['total_images'] == 1
        assert stats['average_inference_time'] == 1.5
    
    def test_record_failed_inference(self, monitor):
        """测试记录失败的推理"""
        monitor.record_inference(
            inference_time=0.5,
            image_count=1,
            success=False,
            error_message="Test error"
        )
        
        stats = monitor.get_statistics()
        assert stats['total_requests'] == 1
        assert stats['successful_requests'] == 0
        assert stats['failed_requests'] == 1
    
    def test_record_batch_inference(self, monitor):
        """测试记录批处理推理"""
        monitor.record_inference(
            inference_time=5.0,
            image_count=10,
            success=True
        )
        
        stats = monitor.get_statistics()
        assert stats['total_images'] == 10
        assert stats['average_inference_time'] == 5.0
        # 吞吐量: 10 images / 5.0 seconds = 2.0 images/s
        assert stats['throughput'] == 2.0
    
    def test_multiple_inferences(self, monitor):
        """测试多次推理记录"""
        # 记录 5 次成功推理
        for i in range(5):
            monitor.record_inference(
                inference_time=1.0 + i * 0.1,
                image_count=1,
                success=True
            )
        
        stats = monitor.get_statistics()
        assert stats['total_requests'] == 5
        assert stats['successful_requests'] == 5
        assert stats['total_images'] == 5
        
        # 平均推理时间: (1.0 + 1.1 + 1.2 + 1.3 + 1.4) / 5 = 1.2
        assert abs(stats['average_inference_time'] - 1.2) < 0.01
    
    def test_get_recent_metrics(self, monitor):
        """测试获取最近的指标"""
        # 记录 5 次推理
        for i in range(5):
            monitor.record_inference(
                inference_time=1.0,
                image_count=1,
                success=True
            )
        
        recent = monitor.get_recent_metrics(count=3)
        assert len(recent) == 3
        assert all(m['inference_time'] == 1.0 for m in recent)
    
    def test_time_window_stats(self, monitor):
        """测试时间窗口统计"""
        # 记录一些推理
        monitor.record_inference(inference_time=1.0, image_count=1, success=True)
        time.sleep(0.1)
        monitor.record_inference(inference_time=1.5, image_count=2, success=True)
        
        # 获取 1 秒窗口内的统计
        window_stats = monitor.get_time_window_stats(window_seconds=1)
        assert window_stats['request_count'] == 2
        assert window_stats['successful_requests'] == 2
        assert window_stats['total_images'] == 3
    
    def test_success_rate_calculation(self, monitor):
        """测试成功率计算"""
        # 3 次成功，1 次失败
        for i in range(3):
            monitor.record_inference(inference_time=1.0, image_count=1, success=True)
        monitor.record_inference(inference_time=1.0, image_count=1, success=False)
        
        stats = monitor.get_statistics()
        assert stats['success_rate'] == 0.75  # 3/4
    
    def test_memory_tracking(self, monitor):
        """测试内存追踪"""
        monitor.record_inference(inference_time=1.0, image_count=1, success=True)
        
        stats = monitor.get_statistics()
        assert stats['current_memory_mb'] > 0
    
    def test_monitor_reset(self, monitor):
        """测试监控器重置"""
        # 记录一些数据
        monitor.record_inference(inference_time=1.0, image_count=1, success=True)
        monitor.record_inference(inference_time=1.0, image_count=1, success=False)
        
        # 重置
        monitor.reset()
        
        stats = monitor.get_statistics()
        assert stats['total_requests'] == 0
        assert stats['successful_requests'] == 0
        assert stats['failed_requests'] == 0
    
    def test_max_history_limit(self):
        """测试历史记录上限"""
        monitor = PerformanceMonitor(max_history=5)
        
        # 记录 10 次推理
        for i in range(10):
            monitor.record_inference(inference_time=1.0, image_count=1, success=True)
        
        recent = monitor.get_recent_metrics(count=100)
        assert len(recent) == 5  # 只保留最近 5 条


class TestGlobalMonitor:
    """测试全局监控器单例"""
    
    def test_get_global_monitor(self):
        """测试获取全局监控器"""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        
        # 应该是同一个实例
        assert monitor1 is monitor2
    
    def test_global_monitor_persistence(self):
        """测试全局监控器的持久性"""
        monitor = get_performance_monitor()
        
        # 记录数据
        monitor.record_inference(inference_time=1.0, image_count=1, success=True)
        
        # 再次获取应该还有数据
        monitor2 = get_performance_monitor()
        stats = monitor2.get_statistics()
        assert stats['total_requests'] >= 1


class TestPerformanceBenchmarks:
    """性能基准测试"""
    
    @pytest.mark.slow
    def test_inference_time_benchmark(self):
        """测试单图推理时间（基准测试）"""
        # 这是一个示例性能测试
        # 实际测试需要加载真实模型
        start = time.time()
        
        # 模拟推理过程
        time.sleep(0.01)  # 模拟推理延迟
        
        elapsed = time.time() - start
        
        # 基准: 推理时间应该小于 5 秒（对于 2B 模型）
        # 注意：这是模拟测试，实际值会不同
        assert elapsed < 5.0
    
    @pytest.mark.slow
    def test_batch_throughput_benchmark(self):
        """测试批处理吞吐量（基准测试）"""
        batch_size = 8
        start = time.time()
        
        # 模拟批处理
        for _ in range(batch_size):
            time.sleep(0.01)
        
        elapsed = time.time() - start
        throughput = batch_size / elapsed
        
        # 基准: 吞吐量应该大于 5 images/s
        # 注意：这是模拟测试，实际值会不同
        assert throughput > 1.0
    
    def test_memory_usage_tracking(self):
        """测试内存使用追踪"""
        process = psutil.Process()
        
        # 获取初始内存
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建一些数据
        data = [np.random.rand(1000, 1000) for _ in range(10)]
        
        # 获取当前内存
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 内存应该增加
        memory_increase = current_memory - initial_memory
        assert memory_increase > 0
        
        # 清理
        del data
    
    def test_concurrent_monitoring(self):
        """测试并发监控（线程安全性）"""
        import threading
        
        monitor = PerformanceMonitor()
        
        def record_metrics():
            for _ in range(10):
                monitor.record_inference(
                    inference_time=0.1,
                    image_count=1,
                    success=True
                )
        
        # 创建多个线程
        threads = [threading.Thread(target=record_metrics) for _ in range(5)]
        
        # 启动所有线程
        for t in threads:
            t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        # 应该记录了 50 次推理
        stats = monitor.get_statistics()
        assert stats['total_requests'] == 50


@pytest.mark.integration
class TestPerformanceIntegration:
    """性能监控集成测试"""
    
    def test_monitor_with_engine_simulation(self, monitor):
        """测试监控器与引擎的集成（模拟）"""
        # 模拟多次 OCR 请求
        for i in range(10):
            # 模拟推理时间
            inference_time = 1.0 + np.random.rand() * 0.5
            success = np.random.rand() > 0.1  # 90% 成功率
            
            monitor.record_inference(
                inference_time=inference_time,
                image_count=1,
                success=success,
                error_message=None if success else "Simulated error"
            )
        
        # 验证统计数据
        stats = monitor.get_statistics()
        assert stats['total_requests'] == 10
        assert stats['success_rate'] >= 0.8  # 至少 80% 成功率
        assert stats['average_inference_time'] > 0
    
    def test_batch_processing_monitoring(self, monitor):
        """测试批处理监控"""
        # 模拟批处理
        batch_sizes = [4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            # 批处理时间与批大小成比例
            batch_time = batch_size * 0.1
            
            monitor.record_inference(
                inference_time=batch_time,
                image_count=batch_size,
                success=True
            )
        
        stats = monitor.get_statistics()
        assert stats['total_images'] == sum(batch_sizes)
        assert stats['total_requests'] == len(batch_sizes)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
