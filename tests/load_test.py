"""
OCR 服务性能测试脚本

测试并发处理能力:
- QPS测试
- 延迟测试
- 稳定性测试
"""

import asyncio
import aiohttp
import time
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import statistics


class OCRLoadTester:
    """OCR负载测试器"""
    
    def __init__(self, base_url: str, image_path: str):
        """
        初始化测试器
        
        Args:
            base_url: API基础URL (例如: http://localhost:8000)
            image_path: 测试图片路径
        """
        self.base_url = base_url.rstrip('/')
        self.image_path = Path(image_path)
        self.results = []
        
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
    
    async def send_request(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """
        发送单个OCR请求
        
        Returns:
            结果字典 (包含延迟、状态等)
        """
        start_time = time.time()
        
        try:
            with open(self.image_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file',
                              f,
                              filename=self.image_path.name,
                              content_type='image/jpeg')
                data.add_field('task_type', 'document')
                
                async with session.post(
                    f'{self.base_url}/api/v1/ocr/predict',
                    data=data
                ) as resp:
                    latency = time.time() - start_time
                    status = resp.status
                    
                    if status == 200:
                        result = await resp.json()
                        success = result.get('success', False)
                    else:
                        success = False
                        result = await resp.text()
                    
                    return {
                        'latency': latency,
                        'status': status,
                        'success': success,
                        'error': None if success else result
                    }
        
        except Exception as e:
            latency = time.time() - start_time
            return {
                'latency': latency,
                'status': 0,
                'success': False,
                'error': str(e)
            }
    
    async def run_concurrent_test(
        self,
        total_requests: int,
        concurrency: int
    ) -> List[Dict[str, Any]]:
        """
        运行并发测试
        
        Args:
            total_requests: 总请求数
            concurrency: 并发数
        
        Returns:
            结果列表
        """
        results = []
        
        async with aiohttp.ClientSession() as session:
            # 创建任务
            tasks = []
            for _ in range(total_requests):
                task = self.send_request(session)
                tasks.append(task)
                
                # 控制并发数
                if len(tasks) >= concurrency:
                    batch_results = await asyncio.gather(*tasks)
                    results.extend(batch_results)
                    tasks = []
            
            # 处理剩余任务
            if tasks:
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
        
        return results
    
    async def run_qps_test(
        self,
        duration_seconds: int,
        target_qps: int
    ) -> List[Dict[str, Any]]:
        """
        运行QPS测试
        
        Args:
            duration_seconds: 测试持续时间(秒)
            target_qps: 目标QPS
        
        Returns:
            结果列表
        """
        results = []
        start_time = time.time()
        request_interval = 1.0 / target_qps
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < duration_seconds:
                loop_start = time.time()
                
                # 发送请求
                result = await self.send_request(session)
                results.append(result)
                
                # 等待以维持目标QPS
                elapsed = time.time() - loop_start
                if elapsed < request_interval:
                    await asyncio.sleep(request_interval - elapsed)
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析测试结果
        
        Args:
            results: 结果列表
        
        Returns:
            分析报告
        """
        if not results:
            return {"error": "No results to analyze"}
        
        latencies = [r['latency'] for r in results]
        successes = [r for r in results if r['success']]
        failures = [r for r in results if not r['success']]
        
        # 按状态码统计
        status_codes = {}
        for r in results:
            status = r['status']
            status_codes[status] = status_codes.get(status, 0) + 1
        
        # 计算百分位数
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        return {
            "total_requests": len(results),
            "successful_requests": len(successes),
            "failed_requests": len(failures),
            "success_rate": len(successes) / len(results) * 100,
            "status_codes": status_codes,
            "latency": {
                "min_ms": min(latencies) * 1000,
                "max_ms": max(latencies) * 1000,
                "avg_ms": statistics.mean(latencies) * 1000,
                "median_ms": statistics.median(latencies) * 1000,
                "p50_ms": sorted_latencies[int(n * 0.50)] * 1000,
                "p95_ms": sorted_latencies[int(n * 0.95)] * 1000,
                "p99_ms": sorted_latencies[int(n * 0.99)] * 1000,
            }
        }
    
    def print_report(self, report: Dict[str, Any], test_type: str):
        """打印测试报告"""
        print(f"\n{'='*60}")
        print(f"  {test_type} Test Report")
        print(f"{'='*60}")
        
        print(f"\nTotal Requests:      {report['total_requests']}")
        print(f"Successful:          {report['successful_requests']}")
        print(f"Failed:              {report['failed_requests']}")
        print(f"Success Rate:        {report['success_rate']:.2f}%")
        
        print(f"\nStatus Codes:")
        for code, count in sorted(report['status_codes'].items()):
            print(f"  {code}: {count}")
        
        print(f"\nLatency (ms):")
        lat = report['latency']
        print(f"  Min:     {lat['min_ms']:.2f}")
        print(f"  Avg:     {lat['avg_ms']:.2f}")
        print(f"  Median:  {lat['median_ms']:.2f}")
        print(f"  P50:     {lat['p50_ms']:.2f}")
        print(f"  P95:     {lat['p95_ms']:.2f}")
        print(f"  P99:     {lat['p99_ms']:.2f}")
        print(f"  Max:     {lat['max_ms']:.2f}")
        
        print(f"\n{'='*60}\n")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='OCR Service Load Testing')
    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='API base URL'
    )
    parser.add_argument(
        '--image',
        required=True,
        help='Path to test image'
    )
    parser.add_argument(
        '--test-type',
        choices=['concurrent', 'qps', 'stress'],
        default='concurrent',
        help='Test type'
    )
    parser.add_argument(
        '--requests',
        type=int,
        default=100,
        help='Total number of requests (for concurrent test)'
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=10,
        help='Concurrent requests (for concurrent test)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Test duration in seconds (for QPS test)'
    )
    parser.add_argument(
        '--qps',
        type=int,
        default=10,
        help='Target QPS (for QPS test)'
    )
    
    args = parser.parse_args()
    
    try:
        tester = OCRLoadTester(args.url, args.image)
        
        if args.test_type == 'concurrent':
            print(f"\nRunning concurrent test:")
            print(f"  Total Requests: {args.requests}")
            print(f"  Concurrency:    {args.concurrency}")
            
            start_time = time.time()
            results = await tester.run_concurrent_test(
                total_requests=args.requests,
                concurrency=args.concurrency
            )
            duration = time.time() - start_time
            
            report = tester.analyze_results(results)
            report['duration_seconds'] = duration
            report['actual_qps'] = len(results) / duration
            
            tester.print_report(report, "Concurrent")
            print(f"Test Duration:       {duration:.2f}s")
            print(f"Actual QPS:          {report['actual_qps']:.2f}")
        
        elif args.test_type == 'qps':
            print(f"\nRunning QPS test:")
            print(f"  Duration:    {args.duration}s")
            print(f"  Target QPS:  {args.qps}")
            
            results = await tester.run_qps_test(
                duration_seconds=args.duration,
                target_qps=args.qps
            )
            
            report = tester.analyze_results(results)
            report['duration_seconds'] = args.duration
            report['target_qps'] = args.qps
            report['actual_qps'] = len(results) / args.duration
            
            tester.print_report(report, "QPS")
            print(f"Target QPS:          {args.qps}")
            print(f"Actual QPS:          {report['actual_qps']:.2f}")
        
        elif args.test_type == 'stress':
            # 压力测试:逐步增加QPS
            print(f"\nRunning stress test (gradually increasing load)...")
            
            qps_levels = [1, 10, 50, 100, 200]
            duration_per_level = 30
            
            for qps in qps_levels:
                print(f"\n--- Testing at {qps} QPS for {duration_per_level}s ---")
                
                results = await tester.run_qps_test(
                    duration_seconds=duration_per_level,
                    target_qps=qps
                )
                
                report = tester.analyze_results(results)
                report['target_qps'] = qps
                report['actual_qps'] = len(results) / duration_per_level
                
                print(f"Actual QPS: {report['actual_qps']:.2f}")
                print(f"Success Rate: {report['success_rate']:.2f}%")
                print(f"P95 Latency: {report['latency']['p95_ms']:.2f}ms")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
