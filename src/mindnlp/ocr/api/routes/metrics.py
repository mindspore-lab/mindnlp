"""
监控和指标路由
"""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from mindnlp.ocr.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


def get_engine():
    """获取OCR引擎实例"""
    from ..app import get_engine as _get_engine
    return _get_engine()


@router.get("/stats")
async def get_stats():
    """
    获取服务统计信息
    
    Returns:
        JSON格式的统计信息
    """
    engine = get_engine()
    if not engine:
        return {
            "success": False,
            "error": "Engine not initialized"
        }
    
    try:
        # 从engine的monitor获取统计
        if hasattr(engine, 'monitor') and engine.monitor:
            stats_data = engine.monitor.get_statistics()
            recent_metrics = engine.monitor.get_recent_metrics(count=10)
            
            # 计算延迟百分位数
            if recent_metrics:
                latencies = [m['inference_time'] for m in recent_metrics if m.get('success', True)]
                if latencies:
                    latencies.sort()
                    n = len(latencies)
                    p50 = latencies[int(n * 0.5)] if n > 0 else 0
                    p95 = latencies[int(n * 0.95)] if n > 0 else 0
                    p99 = latencies[int(n * 0.99)] if n > 0 else 0
                    avg = sum(latencies) / n if n > 0 else 0
                else:
                    p50 = p95 = p99 = avg = 0
            else:
                p50 = p95 = p99 = avg = 0
            
            stats = {
                "monitor": {
                    "qps": stats_data.get('throughput', 0),
                    "success_rate": stats_data.get('success_rate', 1.0) * 100,
                    "total_requests": stats_data.get('total_requests', 0),
                    "total_successes": stats_data.get('successful_requests', 0),
                    "total_failures": stats_data.get('failed_requests', 0),
                    "avg_inference_time": stats_data.get('average_inference_time', 0),
                    "memory_mb": stats_data.get('current_memory_mb', 0),
                    "latency": {
                        "p50": p50,
                        "p95": p95,
                        "p99": p99,
                        "avg": avg,
                    }
                }
            }
            
            return {
                "success": True,
                "data": stats
            }
        else:
            return {
                "success": True,
                "data": {
                    "monitor": {
                        "status": "Monitor not available (simplified mode)"
                    }
                }
            }
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """
    获取 Prometheus 格式的指标
    
    Returns:
        Prometheus metrics 文本
    """
    engine = get_engine()
    if not engine:
        return "# Engine not initialized\n"
    
    try:
        # 从engine获取性能监控器
        if hasattr(engine, 'monitor') and engine.monitor:
            stats = engine.monitor.get_statistics()
            metrics_text = f"""# HELP ocr_requests_total Total OCR requests
# TYPE ocr_requests_total counter
ocr_requests_total {stats.get('total_requests', 0)}

# HELP ocr_requests_success Successful OCR requests
# TYPE ocr_requests_success counter
ocr_requests_success {stats.get('successful_requests', 0)}

# HELP ocr_requests_failed Failed OCR requests
# TYPE ocr_requests_failed counter
ocr_requests_failed {stats.get('failed_requests', 0)}

# HELP ocr_success_rate OCR success rate
# TYPE ocr_success_rate gauge
ocr_success_rate {stats.get('success_rate', 0)}

# HELP ocr_avg_inference_time Average inference time in seconds
# TYPE ocr_avg_inference_time gauge
ocr_avg_inference_time {stats.get('average_inference_time', 0)}

# HELP ocr_memory_mb Memory usage in MB
# TYPE ocr_memory_mb gauge
ocr_memory_mb {stats.get('current_memory_mb', 0)}
"""
            return metrics_text
        else:
            return "# Monitor not available (simplified mode)\n"
    except Exception as e:
        logger.error(f"Error exporting metrics: {e}", exc_info=True)
        return f"# Error: {e}\n"


@router.post("/stats/reset")
async def reset_stats():
    """
    重置统计信息
    
    Returns:
        操作结果
    """
    engine = get_engine()
    if not engine:
        return {
            "success": False,
            "error": "Engine not initialized"
        }
    
    try:
        if hasattr(engine, 'monitor') and engine.monitor:
            # PerformanceMonitor没有reset方法,返回提示
            return {
                "success": False,
                "error": "Reset not supported by current monitor implementation"
            }
        else:
            return {
                "success": False,
                "error": "Monitor not available"
            }
    except Exception as e:
        logger.error(f"Error resetting stats: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/health/detailed")
async def detailed_health():
    """
    详细健康检查(包含统计信息)
    
    Returns:
        详细的健康状态
    """
    engine = get_engine()
    
    if not engine:
        return {
            "status": "unhealthy",
            "reason": "Engine not initialized",
            "components": {}
        }
    
    try:
        components = {}
        
        # 检查engine状态
        components['engine'] = {
            "status": "healthy" if engine else "unhealthy",
            "model": getattr(engine, 'model_name', 'unknown') if engine else None
        }
        
        # 检查monitor状态
        if hasattr(engine, 'monitor') and engine.monitor:
            stats_data = engine.monitor.get_statistics()
            recent_metrics = engine.monitor.get_recent_metrics(count=10)
            
            # 计算延迟百分位数
            if recent_metrics:
                latencies = [m['inference_time'] for m in recent_metrics if m.get('success', True)]
                if latencies:
                    latencies.sort()
                    n = len(latencies)
                    p50 = latencies[int(n * 0.5)] if n > 0 else 0
                    p95 = latencies[int(n * 0.95)] if n > 0 else 0
                    p99 = latencies[int(n * 0.99)] if n > 0 else 0
                    avg = sum(latencies) / n if n > 0 else 0
                else:
                    p50 = p95 = p99 = avg = 0
            else:
                p50 = p95 = p99 = avg = 0
            
            components['monitor'] = {
                "status": "healthy",
                "qps": stats_data.get('throughput', 0),
                "success_rate": stats_data.get('success_rate', 1.0) * 100,
                "p95_latency": p95,
                "total_requests": stats_data.get('total_requests', 0)
            }
            
            stats = {
                "monitor": stats_data,
                "latency": {
                    "p50": p50,
                    "p95": p95,
                    "p99": p99,
                    "avg": avg,
                }
            }
        else:
            components['monitor'] = {
                "status": "not_available",
                "message": "Running in simplified mode"
            }
            stats = {}
        
        # 判断整体状态
        overall_status = "healthy"
        
        return {
            "status": overall_status,
            "components": components,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error in detailed health check: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "reason": str(e),
            "components": {}
        }
