#!/bin/bash
# 快速性能测试脚本

echo "=========================================="
echo "OCR 服务性能测试"
echo "=========================================="

# 检查服务是否运行
echo ""
echo "检查服务状态..."
curl -s http://localhost:8000/api/v1/health | python -m json.tool

# 获取初始统计
echo ""
echo "=========================================="
echo "初始统计信息"
echo "=========================================="
curl -s http://localhost:8000/api/v1/metrics/stats | python -m json.tool

# 准备测试图片
TEST_IMAGE="$HOME/mindnlp/datasets/funsd_converted/images/test_00046.png"

if [ ! -f "$TEST_IMAGE" ]; then
    echo "警告: 测试图片不存在: $TEST_IMAGE"
    echo "请指定正确的图片路径"
    exit 1
fi

echo ""
echo "=========================================="
echo "测试 1: 单次请求 (预热)"
echo "=========================================="
curl -X POST http://localhost:8000/api/v1/ocr/predict \
  -F "file=@$TEST_IMAGE" \
  -F "task_type=document" \
  -s | python -c "import sys, json; d=json.load(sys.stdin); print(f\"Success: {d.get('success')}, Time: {d.get('inference_time', 0):.2f}s\")"

echo ""
echo "=========================================="
echo "测试 2: 连续10次请求 (测试批处理)"
echo "=========================================="

for i in {1..10}; do
    echo -n "请求 $i: "
    curl -X POST http://localhost:8000/api/v1/ocr/predict \
      -F "file=@$TEST_IMAGE" \
      -F "task_type=document" \
      -s | python -c "import sys, json; d=json.load(sys.stdin); print(f\"Success: {d.get('success')}, Time: {d.get('inference_time', 0):.2f}s\")" &
    sleep 0.1  # 100ms间隔,触发批处理
done

wait

echo ""
echo "=========================================="
echo "测试 3: 查看批处理统计"
echo "=========================================="
sleep 2  # 等待批处理完成
curl -s http://localhost:8000/api/v1/metrics/stats | python -m json.tool

echo ""
echo "=========================================="
echo "测试 4: 详细健康检查"
echo "=========================================="
curl -s http://localhost:8000/api/v1/metrics/health/detailed | python -m json.tool

echo ""
echo "=========================================="
echo "测试完成!"
echo "=========================================="
