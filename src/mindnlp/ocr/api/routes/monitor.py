"""
监控路由
提供性能监控和系统状态查询接口
"""

from fastapi import APIRouter, Depends  # pylint: disable=import-error
from mindnlp.ocr.utils.logger import get_logger
from mindnlp.ocr.core.monitor import get_performance_monitor
from mindnlp.ocr.api.app import get_service_manager


logger = get_logger(__name__)
router = APIRouter()


@router.get("/statistics")
async def get_statistics():
    """
    获取总体性能统计数据

    Returns:
        Dict: 包含总请求数、成功率、平均推理时间、吞吐量等指标
    """
    monitor = get_performance_monitor()
    stats = monitor.get_statistics()

    return {
        "status": "success",
        "data": stats
    }


@router.get("/recent")
async def get_recent_metrics(count: int = 10):
    """
    获取最近的性能指标记录

    Args:
        count: 返回的记录数量（默认 10）

    Returns:
        Dict: 最近的性能指标列表
    """
    monitor = get_performance_monitor()

    # 验证 count 参数
    if count < 1:
        return {
            "status": "error",
            "message": "count must be >= 1"
        }
    if count > 1000:
        return {
            "status": "error",
            "message": "count must be <= 1000"
        }

    recent = monitor.get_recent_metrics(count=count)

    return {
        "status": "success",
        "data": {
            "count": len(recent),
            "metrics": recent
        }
    }


@router.get("/window")
async def get_time_window_stats(window_seconds: int = 60):
    """
    获取时间窗口内的统计数据

    Args:
        window_seconds: 时间窗口大小（秒，默认 60）

    Returns:
        Dict: 时间窗口内的统计数据
    """
    monitor = get_performance_monitor()

    # 验证 window_seconds 参数
    if window_seconds < 1:
        return {
            "status": "error",
            "message": "window_seconds must be >= 1"
        }
    if window_seconds > 3600:
        return {
            "status": "error",
            "message": "window_seconds must be <= 3600 (1 hour)"
        }

    window_stats = monitor.get_time_window_stats(window_seconds=window_seconds)

    return {
        "status": "success",
        "data": window_stats
    }


@router.post("/reset")
async def reset_statistics():
    """
    重置所有性能统计数据

    Returns:
        Dict: 操作结果
    """
    monitor = get_performance_monitor()
    monitor.reset()

    logger.info("Performance statistics reset via API")

    return {
        "status": "success",
        "message": "Performance statistics reset successfully"
    }


@router.get("/service-stats")
async def get_service_stats():
    """
    获取服务统计信息（批处理、限流、队列等）

    Returns:
        Dict: 服务统计数据
    """
    try:
        service_mgr = get_service_manager()
        stats = service_mgr.get_stats()

        return {
            "status": "success",
            "data": stats
        }
    except RuntimeError as e:
        return {
            "status": "error",
            "message": str(e),
            "data": None
        }


@router.post("/service-stats/reset")
async def reset_service_stats():
    """
    重置服务统计信息

    Returns:
        Dict: 操作结果
    """
    try:
        service_mgr = get_service_manager()
        service_mgr.reset_stats()

        logger.info("Service statistics reset via API")

        return {
            "status": "success",
            "message": "Service statistics reset successfully"
        }
    except RuntimeError as e:
        return {
            "status": "error",
            "message": str(e)
        }
