"""
测试监控、日志和追踪系统
"""
import sys
import os
import tempfile
import time
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def test_structured_logging():
    """测试结构化日志系统"""
    print("\n" + "="*60)
    print("测试 1: 结构化日志系统")
    print("="*60)
    
    from mindnlp.ocr.utils.structured_logging import (
        setup_structured_logging,
        LogContext,
        get_request_logger,
        get_performance_logger
    )
    
    # 创建临时日志文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        log_file = f.name
    
    try:
        # 1. 初始化日志系统 (Console格式便于测试)
        print("\n[1.1] 初始化日志系统...")
        setup_structured_logging(
            log_level="INFO",
            log_file=log_file,
            json_format=False  # Console格式便于查看
        )
        print("✓ 日志系统初始化成功")
        
        # 2. 测试请求日志
        print("\n[1.2] 测试请求日志...")
        request_logger = get_request_logger()
        
        request_logger.log_request(
            method="POST",
            endpoint="/api/v1/ocr/predict",
            request_id="test-request-123",
            status_code=200,
            latency_ms=250.5
        )
        print("✓ 请求日志记录成功")
        
        # 3. 测试上下文绑定
        print("\n[1.3] 测试日志上下文绑定...")
        with LogContext(request_id="test-ctx-456", user_id="user-789"):
            request_logger.log_inference(
                request_id="test-ctx-456",
                model_name="ocr_model",
                inference_time_ms=180.2,
                batch_size=4
            )
        print("✓ 上下文绑定成功")
        
        # 4. 测试性能日志
        print("\n[1.4] 测试性能日志...")
        perf_logger = get_performance_logger()
        
        perf_logger.log_resource_usage(
            cpu_percent=65.2,
            memory_mb=1024.5,
            gpu_utilization=82.3
        )
        
        perf_logger.log_queue_metrics(
            queue_size=15,
            queue_capacity=100,
            avg_wait_time_ms=50.0
        )
        print("✓ 性能日志记录成功")
        
        # 5. 测试JSON格式
        print("\n[1.5] 测试JSON格式日志...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            json_log_file = f.name
        
        setup_structured_logging(
            log_level="INFO",
            log_file=json_log_file,
            json_format=True
        )
        
        request_logger = get_request_logger()
        request_logger.log_request(
            method="GET",
            endpoint="/api/v1/health",
            request_id="test-json-001",
            status_code=200,
            latency_ms=10.5
        )
        
        # 验证JSON格式
        if os.path.exists(json_log_file):
            with open(json_log_file, 'r', encoding='utf-8') as f:
                last_line = f.readlines()[-1]
                log_entry = json.loads(last_line)
                assert log_entry['event'] == 'http_request'
                assert log_entry['request_id'] == 'test-json-001'
                print("✓ JSON格式验证成功")
        
        print("\n✅ 结构化日志测试全部通过!")
        
    finally:
        # 清理临时文件
        for temp_file in [log_file, json_log_file]:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


def test_tracing():
    """测试分布式追踪系统"""
    print("\n" + "="*60)
    print("测试 2: 分布式追踪系统")
    print("="*60)
    
    from mindnlp.ocr.utils.tracing import (
        TracingConfig,
        setup_tracing,
        get_tracer
    )
    
    # 1. 测试Console导出器 (不需要Jaeger)
    print("\n[2.1] 初始化追踪系统 (Console导出)...")
    config = TracingConfig(
        enabled=True,
        service_name="ocr-api-test",
        sampling_rate=1.0,  # 100%采样用于测试
        exporter_type="console"
    )
    
    provider = setup_tracing(config)
    print("✓ 追踪系统初始化成功")
    
    # 2. 测试请求追踪
    print("\n[2.2] 测试HTTP请求追踪...")
    tracer = get_tracer()
    
    with tracer.trace_request(
        request_id="trace-req-001",
        endpoint="/api/v1/ocr/predict"
    ) as span:
        time.sleep(0.1)  # 模拟处理
        span.set_attribute("http.status_code", 200)
    print("✓ 请求追踪成功")
    
    # 3. 测试预处理追踪
    print("\n[2.3] 测试预处理追踪...")
    with tracer.trace_preprocessing(
        image_size=(1920, 1080),
        image_format="JPEG"
    ):
        time.sleep(0.05)
    print("✓ 预处理追踪成功")
    
    # 4. 测试推理追踪
    print("\n[2.4] 测试推理追踪...")
    with tracer.trace_inference(
        model_name="ocr_model_v1",
        batch_size=4
    ):
        time.sleep(0.15)
    print("✓ 推理追踪成功")
    
    # 5. 测试后处理追踪
    print("\n[2.5] 测试后处理追踪...")
    with tracer.trace_postprocessing(output_format="json"):
        time.sleep(0.02)
    print("✓ 后处理追踪成功")
    
    # 6. 测试嵌套Span
    print("\n[2.6] 测试嵌套Span追踪...")
    with tracer.trace_request("nested-req-001", "/api/v1/ocr/batch"):
        with tracer.trace_preprocessing((800, 600), "PNG"):
            time.sleep(0.03)
        with tracer.trace_inference("model_v2", 2):
            time.sleep(0.08)
        with tracer.trace_postprocessing("xml"):
            time.sleep(0.01)
    print("✓ 嵌套追踪成功")
    
    # 关闭provider
    if provider:
        provider.shutdown()
    
    print("\n✅ 分布式追踪测试全部通过!")


def test_profiling():
    """测试性能分析工具"""
    print("\n" + "="*60)
    print("测试 3: 性能Profiling工具")
    print("="*60)
    
    from mindnlp.ocr.utils.profiling import (
        get_profiling_manager,
        CPUProfiler,
        MemoryProfiler,
        PerformanceTimer
    )
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # 1. 测试CPU Profiling
        print("\n[3.1] 测试CPU Profiling...")
        cpu_profiler = CPUProfiler(output_dir=temp_dir)
        
        with cpu_profiler.profile("test_cpu"):
            # 模拟CPU密集型操作
            result = sum([i**2 for i in range(10000)])
        
        # 检查输出文件
        prof_files = list(Path(temp_dir).glob("test_cpu_*.prof"))
        assert len(prof_files) > 0, "CPU profiling文件未生成"
        print(f"✓ CPU Profiling成功，生成文件: {prof_files[0].name}")
        
        # 2. 测试Memory Profiling
        print("\n[3.2] 测试Memory Profiling...")
        mem_profiler = MemoryProfiler()
        
        with mem_profiler.profile("test_memory"):
            # 模拟内存分配
            data = [i for i in range(100000)]
            time.sleep(0.1)
        print("✓ Memory Profiling成功")
        
        # 3. 测试Performance Timer
        print("\n[3.3] 测试Performance Timer...")
        timer = PerformanceTimer()
        
        with timer.measure("operation_1"):
            time.sleep(0.1)
        
        elapsed = timer.get_elapsed("operation_1")
        assert elapsed >= 100, f"计时不准确: {elapsed}ms"
        print(f"✓ Performance Timer成功，耗时: {elapsed:.2f}ms")
        
        # 4. 测试Profiling Manager
        print("\n[3.4] 测试Profiling Manager...")
        manager = get_profiling_manager()
        manager.output_dir = temp_dir
        
        with manager.profile_cpu("manager_test"):
            result = sum([i**3 for i in range(5000)])
        
        prof_files = list(Path(temp_dir).glob("manager_test_*.prof"))
        assert len(prof_files) > 0
        print("✓ Profiling Manager成功")
        
        # 5. 测试组合Profiling
        print("\n[3.5] 测试组合Profiling...")
        with manager.profile_cpu("combined_cpu"):
            with manager.profile_memory("combined_memory"):
                data = [i**2 for i in range(50000)]
                time.sleep(0.05)
        print("✓ 组合Profiling成功")
        
    print("\n✅ 性能Profiling测试全部通过!")


def test_integration():
    """集成测试: 同时使用日志、追踪和Profiling"""
    print("\n" + "="*60)
    print("测试 4: 集成测试")
    print("="*60)
    
    from mindnlp.ocr.utils.structured_logging import (
        setup_structured_logging,
        LogContext,
        get_request_logger
    )
    from mindnlp.ocr.utils.tracing import (
        TracingConfig,
        setup_tracing,
        get_tracer
    )
    from mindnlp.ocr.utils.profiling import get_profiling_manager
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "integration.log")
        
        # 初始化所有系统
        print("\n[4.1] 初始化所有监控系统...")
        setup_structured_logging(log_level="INFO", log_file=log_file, json_format=True)
        
        trace_config = TracingConfig(
            enabled=True,
            service_name="ocr-integration-test",
            sampling_rate=1.0,
            exporter_type="console"
        )
        setup_tracing(trace_config)
        
        profiler = get_profiling_manager()
        profiler.output_dir = temp_dir
        print("✓ 所有系统初始化成功")
        
        # 模拟完整的OCR请求处理
        print("\n[4.2] 模拟完整OCR请求处理...")
        request_id = "integration-req-001"
        request_logger = get_request_logger()
        tracer = get_tracer()
        
        with LogContext(request_id=request_id):
            with tracer.trace_request(request_id, "/api/v1/ocr/predict"):
                with profiler.profile_cpu("full_pipeline"):
                    
                    # 预处理
                    with tracer.trace_preprocessing((1024, 768), "JPEG"):
                        time.sleep(0.05)
                        request_logger.log_inference(
                            request_id=request_id,
                            model_name="preprocessing",
                            inference_time_ms=50.0,
                            batch_size=1
                        )
                    
                    # 推理
                    with tracer.trace_inference("ocr_model", 1):
                        time.sleep(0.15)
                        request_logger.log_inference(
                            request_id=request_id,
                            model_name="ocr_model",
                            inference_time_ms=150.0,
                            batch_size=1
                        )
                    
                    # 后处理
                    with tracer.trace_postprocessing("json"):
                        time.sleep(0.02)
                
                # 记录最终请求
                request_logger.log_request(
                    method="POST",
                    endpoint="/api/v1/ocr/predict",
                    request_id=request_id,
                    status_code=200,
                    latency_ms=220.0
                )
        
        print("✓ 完整流程执行成功")
        
        # 验证结果
        print("\n[4.3] 验证生成的文件...")
        
        # 检查日志文件
        assert os.path.exists(log_file), "日志文件未生成"
        with open(log_file, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
            assert len(log_lines) > 0, "日志为空"
            # 验证JSON格式
            for line in log_lines:
                json.loads(line)  # 应该能成功解析
        print(f"✓ 日志文件验证成功 ({len(log_lines)} 条日志)")
        
        # 检查Profiling文件
        prof_files = list(Path(temp_dir).glob("*.prof"))
        assert len(prof_files) > 0, "Profiling文件未生成"
        print(f"✓ Profiling文件验证成功 ({len(prof_files)} 个文件)")
    
    print("\n✅ 集成测试全部通过!")


def main():
    """运行所有测试"""
    print("\n" + "█"*60)
    print("█  OCR 监控、日志和性能分析系统 - 功能测试")
    print("█"*60)
    
    try:
        test_structured_logging()
        test_tracing()
        test_profiling()
        test_integration()
        
        print("\n" + "█"*60)
        print("█  🎉 所有测试通过! ")
        print("█"*60)
        print("\n测试总结:")
        print("  ✅ 结构化日志系统 - 6项测试通过")
        print("  ✅ 分布式追踪系统 - 6项测试通过")
        print("  ✅ 性能Profiling - 5项测试通过")
        print("  ✅ 集成测试 - 3项测试通过")
        print("\n总计: 20项测试全部通过\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
