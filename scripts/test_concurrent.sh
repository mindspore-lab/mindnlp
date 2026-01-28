#!/bin/bash

# OCR 并发处理快速测试脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# 默认参数
BASE_URL="http://localhost:8000"
IMAGE_PATH="test_images/sample.jpg"

echo -e "${BLUE}=========================================="
echo "OCR 并发处理功能测试"
echo "==========================================${NC}"

# 检查服务是否运行
echo -e "\n${YELLOW}检查服务状态...${NC}"
if curl -s "$BASE_URL/api/v1/health" > /dev/null; then
    echo -e "${GREEN}✓ 服务正在运行${NC}"
else
    echo -e "${RED}✗ 服务未运行，请先启动服务:${NC}"
    echo -e "  cd src/mindnlp/ocr"
    echo -e "  python -m uvicorn api.app:create_app --factory --host 0.0.0.0 --port 8000"
    exit 1
fi

# 显示菜单
echo -e "\n${BLUE}选择测试类型:${NC}"
echo "1. 基础功能测试（单个请求）"
echo "2. 并发测试（10 并发，100 请求）"
echo "3. 批处理场景测试（测试不同 batch size）"
echo "4. 压力测试（100 QPS，10 秒）"
echo "5. 查看服务统计"
echo "6. 重置服务统计"
echo "7. 持续监控（每 5 秒刷新）"
echo "0. 退出"

read -p $'\n选择 (0-7): ' choice

case $choice in
    1)
        # 基础功能测试
        echo -e "\n${YELLOW}[测试 1] 基础功能测试${NC}"
        
        # 检查图像文件
        if [ ! -f "$IMAGE_PATH" ]; then
            echo -e "${RED}错误: 图像文件不存在: $IMAGE_PATH${NC}"
            exit 1
        fi
        
        echo "发送单个请求..."
        response=$(curl -s -X POST "$BASE_URL/api/v1/ocr/process" \
            -F "file=@$IMAGE_PATH")
        
        echo -e "${GREEN}响应:${NC}"
        echo "$response" | jq .
        
        echo -e "\n${GREEN}✓ 基础功能测试完成${NC}"
        ;;
        
    2)
        # 并发测试
        echo -e "\n${YELLOW}[测试 2] 并发测试${NC}"
        echo "参数: 10 并发，100 请求"
        
        cd tests
        python test_concurrent_processing.py \
            --base-url "$BASE_URL" \
            --image "$IMAGE_PATH" \
            --test-type concurrent \
            --num-requests 100 \
            --concurrency 10
        ;;
        
    3)
        # 批处理场景测试
        echo -e "\n${YELLOW}[测试 3] 批处理场景测试${NC}"
        echo "测试 batch size: 1, 2, 4, 8"
        
        cd tests
        python test_concurrent_processing.py \
            --base-url "$BASE_URL" \
            --image "$IMAGE_PATH" \
            --test-type batch
        ;;
        
    4)
        # 压力测试
        echo -e "\n${YELLOW}[测试 4] 压力测试${NC}"
        echo "参数: 100 QPS，持续 10 秒"
        echo -e "${RED}警告: 此测试会产生大量请求${NC}"
        read -p "确认继续? (y/n) " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cd tests
            python test_concurrent_processing.py \
                --base-url "$BASE_URL" \
                --image "$IMAGE_PATH" \
                --test-type stress \
                --target-qps 100 \
                --duration 10
        else
            echo "已取消"
        fi
        ;;
        
    5)
        # 查看统计
        echo -e "\n${YELLOW}[查询] 服务统计${NC}"
        
        stats=$(curl -s "$BASE_URL/api/v1/monitor/service-stats")
        
        echo -e "\n${BLUE}批处理统计:${NC}"
        echo "$stats" | jq '.data.batching'
        
        echo -e "\n${BLUE}限流统计:${NC}"
        echo "$stats" | jq '.data.rate_limiting'
        
        echo -e "\n${BLUE}队列统计:${NC}"
        echo "$stats" | jq '.data.queue'
        ;;
        
    6)
        # 重置统计
        echo -e "\n${YELLOW}[操作] 重置统计${NC}"
        read -p "确认重置所有统计信息? (y/n) " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            response=$(curl -s -X POST "$BASE_URL/api/v1/monitor/service-stats/reset")
            echo "$response" | jq .
            echo -e "${GREEN}✓ 统计信息已重置${NC}"
        else
            echo "已取消"
        fi
        ;;
        
    7)
        # 持续监控
        echo -e "\n${YELLOW}[监控] 持续监控 (Ctrl+C 停止)${NC}"
        echo ""
        
        while true; do
            clear
            echo -e "${BLUE}========== OCR 服务监控 $(date) ==========${NC}"
            
            stats=$(curl -s "$BASE_URL/api/v1/monitor/service-stats")
            
            # 批处理统计
            echo -e "\n${GREEN}批处理统计:${NC}"
            total_requests=$(echo "$stats" | jq -r '.data.batching.total_requests // 0')
            total_batches=$(echo "$stats" | jq -r '.data.batching.total_batches // 0')
            avg_batch_size=$(echo "$stats" | jq -r '.data.batching.avg_batch_size // 0')
            batch_hit_rate=$(echo "$stats" | jq -r '.data.batching.batch_hit_rate // 0')
            queue_size=$(echo "$stats" | jq -r '.data.batching.queue_size // 0')
            
            echo "  总请求数:       $total_requests"
            echo "  总批次数:       $total_batches"
            echo "  平均批大小:     $avg_batch_size"
            echo "  批处理命中率:   $(echo "$batch_hit_rate * 100" | bc -l | cut -c1-5)%"
            echo "  队列大小:       $queue_size"
            
            # 限流统计
            echo -e "\n${GREEN}限流统计:${NC}"
            total_req=$(echo "$stats" | jq -r '.data.rate_limiting.total_requests // 0')
            accepted=$(echo "$stats" | jq -r '.data.rate_limiting.accepted_requests // 0')
            rejected=$(echo "$stats" | jq -r '.data.rate_limiting.rejected_requests // 0')
            cb_rejects=$(echo "$stats" | jq -r '.data.rate_limiting.circuit_breaker_rejects // 0')
            cb_state=$(echo "$stats" | jq -r '.data.rate_limiting.circuit_breaker_state // "N/A"')
            
            echo "  总请求数:       $total_req"
            echo "  接受请求:       $accepted"
            echo "  拒绝请求:       $rejected"
            echo "  熔断拒绝:       $cb_rejects"
            echo "  熔断器状态:     $cb_state"
            
            # 队列统计
            echo -e "\n${GREEN}队列统计:${NC}"
            current_size=$(echo "$stats" | jq -r '.data.queue.current_size // 0')
            is_full=$(echo "$stats" | jq -r '.data.queue.is_full // false')
            enqueued=$(echo "$stats" | jq -r '.data.queue.total_enqueued // 0')
            dequeued=$(echo "$stats" | jq -r '.data.queue.total_dequeued // 0')
            
            echo "  当前大小:       $current_size"
            echo "  是否已满:       $is_full"
            echo "  总入队:         $enqueued"
            echo "  总出队:         $dequeued"
            
            echo -e "\n${BLUE}=================================================${NC}"
            
            sleep 5
        done
        ;;
        
    0)
        echo "退出"
        exit 0
        ;;
        
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}测试完成！${NC}"
