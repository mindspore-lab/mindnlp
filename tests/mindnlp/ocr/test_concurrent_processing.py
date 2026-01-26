"""
并发处理和批处理测试脚本

测试:
1. 动态批处理功能
2. 请求队列和限流
3. 并发性能
4. QPS 和延迟指标
"""

import asyncio
import time
import aiohttp
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    image_path: str,
    request_id: int
) -> Dict[str, Any]:
    """
    发送单个OCR请求
    
    Args:
        session: aiohttp会话
        url: 请求URL
        image_path: 图像路径
        request_id: 请求ID
    
    Returns:
        请求结果和耗时
    """
    start_time = time.time()
    
    try:
        # 构造请求数据
        data = aiohttp.FormData()
        data.add_field('file',
                      open(image_path, 'rb'),
                      filename='test.jpg',
                      content_type='image/jpeg')
        
        # 发送请求
        async with session.post(url, data=data) as response:
            result = await response.json()
            elapsed = time.time() - start_time
            
            return {
                'request_id': request_id,
                'status_code': response.status,
                'success': response.status == 200,
                'elapsed': elapsed,
                'result': result
            }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'request_id': request_id,
            'status_code': 0,
            'success': False,
            'elapsed': elapsed,
            'error': str(e)
        }


async def concurrent_test(
    base_url: str,
    image_path: str,
    num_requests: int,
    concurrency: int
) -> List[Dict[str, Any]]:
    """
    并发测试
    
    Args:
        base_url: API基础URL
        image_path: 测试图像路径
        num_requests: 总请求数
        concurrency: 并发数
    
    Returns:
        请求结果列表
    """
    url = f"{base_url}/api/v1/ocr/process"
    
    # 创建客户端会话
    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # 创建任务
        tasks = [
            send_request(session, url, image_path, i)
            for i in range(num_requests)
        ]
        
        # 执行任务
        results = await asyncio.gather(*tasks)
        
    return results


def analyze_results(results: List[Dict[str, Any]]):
    """
    分析测试结果
    
    Args:
        results: 请求结果列表
    """
    # 成功请求
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    # 延迟统计
    latencies = [r['elapsed'] * 1000 for r in successful]  # 转换为毫秒
    
    print("\n" + "=" * 60)
    print("测试结果分析")
    print("=" * 60)
    
    print(f"\n总请求数: {len(results)}")
    print(f"成功请求: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"失败请求: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    
    if latencies:
        print(f"\n延迟统计 (ms):")
        print(f"  最小: {min(latencies):.2f}")
        print(f"  最大: {max(latencies):.2f}")
        print(f"  平均: {statistics.mean(latencies):.2f}")
        print(f"  中位数: {statistics.median(latencies):.2f}")
        
        # P95, P99
        sorted_latencies = sorted(latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)
        print(f"  P95: {sorted_latencies[p95_idx]:.2f}")
        print(f"  P99: {sorted_latencies[p99_idx]:.2f}")
    
    # 计算总耗时和QPS
    total_elapsed = max(r['elapsed'] for r in results)
    qps = len(results) / total_elapsed
    print(f"\n总耗时: {total_elapsed:.2f}s")
    print(f"实际 QPS: {qps:.2f}")
    
    # 错误统计
    if failed:
        print(f"\n失败请求详情:")
        error_types = {}
        for r in failed:
            error = r.get('error', 'Unknown')
            error_types[error] = error_types.get(error, 0) + 1
        
        for error, count in error_types.items():
            print(f"  {error}: {count}")


async def get_service_stats(base_url: str):
    """
    获取服务统计信息
    
    Args:
        base_url: API基础URL
    """
    url = f"{base_url}/api/v1/monitor/service-stats"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            result = await response.json()
            
            if result['status'] == 'success':
                stats = result['data']
                
                print("\n" + "=" * 60)
                print("服务统计信息")
                print("=" * 60)
                
                # 批处理统计
                if 'batching' in stats:
                    batch_stats = stats['batching']
                    print(f"\n批处理统计:")
                    print(f"  总请求数: {batch_stats.get('total_requests', 0)}")
                    print(f"  总批次数: {batch_stats.get('total_batches', 0)}")
                    print(f"  平均批大小: {batch_stats.get('avg_batch_size', 0):.2f}")
                    print(f"  批处理命中率: {batch_stats.get('batch_hit_rate', 0)*100:.2f}%")
                    print(f"  队列大小: {batch_stats.get('queue_size', 0)}")
                
                # 限流统计
                if 'rate_limiting' in stats:
                    rate_stats = stats['rate_limiting']
                    print(f"\n限流统计:")
                    print(f"  总请求数: {rate_stats.get('total_requests', 0)}")
                    print(f"  接受请求: {rate_stats.get('accepted_requests', 0)}")
                    print(f"  拒绝请求: {rate_stats.get('rejected_requests', 0)}")
                    print(f"  熔断拒绝: {rate_stats.get('circuit_breaker_rejects', 0)}")
                    
                    if 'circuit_breaker_state' in rate_stats:
                        print(f"  熔断器状态: {rate_stats['circuit_breaker_state']}")
                
                # 队列统计
                if 'queue' in stats:
                    queue_stats = stats['queue']
                    print(f"\n队列统计:")
                    print(f"  当前大小: {queue_stats.get('current_size', 0)}")
                    print(f"  是否已满: {queue_stats.get('is_full', False)}")
                    print(f"  总入队: {queue_stats.get('total_enqueued', 0)}")
                    print(f"  总出队: {queue_stats.get('total_dequeued', 0)}")
                    print(f"  超时数: {queue_stats.get('total_timeout', 0)}")
                    print(f"  拒绝数: {queue_stats.get('total_rejected', 0)}")


async def batch_test_scenario(
    base_url: str,
    image_path: str,
    batch_sizes: List[int] = [1, 2, 4, 8],
    requests_per_batch: int = 10
):
    """
    批处理场景测试
    
    测试不同并发度下的批处理效果
    
    Args:
        base_url: API基础URL
        image_path: 测试图像路径
        batch_sizes: 批大小列表
        requests_per_batch: 每个批次的请求数
    """
    print("\n" + "=" * 60)
    print("批处理场景测试")
    print("=" * 60)
    
    for batch_size in batch_sizes:
        print(f"\n测试批大小: {batch_size}")
        
        # 重置统计信息
        async with aiohttp.ClientSession() as session:
            await session.post(f"{base_url}/api/v1/monitor/service-stats/reset")
        
        # 等待一下
        await asyncio.sleep(0.5)
        
        # 执行测试
        results = await concurrent_test(
            base_url=base_url,
            image_path=image_path,
            num_requests=requests_per_batch,
            concurrency=batch_size
        )
        
        # 分析结果
        analyze_results(results)
        
        # 获取服务统计
        await get_service_stats(base_url)
        
        # 等待下一轮测试
        await asyncio.sleep(1)


async def stress_test(
    base_url: str,
    image_path: str,
    target_qps: int = 100,
    duration_seconds: int = 10
):
    """
    压力测试
    
    持续发送请求，测试系统在目标QPS下的表现
    
    Args:
        base_url: API基础URL
        image_path: 测试图像路径
        target_qps: 目标QPS
        duration_seconds: 测试持续时间（秒）
    """
    print("\n" + "=" * 60)
    print(f"压力测试 - 目标QPS: {target_qps}, 持续时间: {duration_seconds}s")
    print("=" * 60)
    
    # 重置统计
    async with aiohttp.ClientSession() as session:
        await session.post(f"{base_url}/api/v1/monitor/service-stats/reset")
    
    results = []
    start_time = time.time()
    request_id = 0
    
    # 创建客户端会话
    connector = aiohttp.TCPConnector(limit=target_qps)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        while time.time() - start_time < duration_seconds:
            # 发送请求
            task = send_request(
                session,
                f"{base_url}/api/v1/ocr/process",
                image_path,
                request_id
            )
            
            # 不等待完成，立即创建下一个请求
            asyncio.create_task(task).add_done_callback(
                lambda t: results.append(t.result())
            )
            
            request_id += 1
            
            # 控制发送速率
            await asyncio.sleep(1.0 / target_qps)
    
    # 等待所有请求完成
    await asyncio.sleep(5)
    
    # 分析结果
    analyze_results(results)
    
    # 获取服务统计
    await get_service_stats(base_url)


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OCR并发处理和批处理测试')
    parser.add_argument('--base-url', default='http://localhost:8000',
                      help='API基础URL')
    parser.add_argument('--image', required=True,
                      help='测试图像路径')
    parser.add_argument('--test-type', choices=['concurrent', 'batch', 'stress'],
                      default='concurrent',
                      help='测试类型')
    parser.add_argument('--num-requests', type=int, default=100,
                      help='总请求数（concurrent测试）')
    parser.add_argument('--concurrency', type=int, default=10,
                      help='并发数（concurrent测试）')
    parser.add_argument('--target-qps', type=int, default=100,
                      help='目标QPS（stress测试）')
    parser.add_argument('--duration', type=int, default=10,
                      help='测试持续时间（秒，stress测试）')
    
    args = parser.parse_args()
    
    if args.test_type == 'concurrent':
        # 并发测试
        results = await concurrent_test(
            base_url=args.base_url,
            image_path=args.image,
            num_requests=args.num_requests,
            concurrency=args.concurrency
        )
        analyze_results(results)
        await get_service_stats(args.base_url)
        
    elif args.test_type == 'batch':
        # 批处理场景测试
        await batch_test_scenario(
            base_url=args.base_url,
            image_path=args.image
        )
        
    elif args.test_type == 'stress':
        # 压力测试
        await stress_test(
            base_url=args.base_url,
            image_path=args.image,
            target_qps=args.target_qps,
            duration_seconds=args.duration
        )


if __name__ == '__main__':
    asyncio.run(main())
