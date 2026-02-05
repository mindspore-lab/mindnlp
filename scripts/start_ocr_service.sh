#!/bin/bash
# OCR服务启动脚本

echo "=========================================="
echo "启动OCR服务"
echo "=========================================="

# 进入mindnlp目录
cd ~/mindnlp || exit 1

# 激活conda环境
echo "激活conda环境: mindspore"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mindspore

# 设置环境变量
echo "设置环境变量..."

# 关键: 使用ASCEND_RT_VISIBLE_DEVICES屏蔽被占用的NPU 0
# 这样物理NPU 1-7 映射为逻辑设备0-6
export ASCEND_RT_VISIBLE_DEVICES=1,2,3,4,5,6,7
echo "可见NPU设备: 1,2,3,4,5,6,7 (屏蔽了被占用的NPU 0)"

export OCR_DEVICE="npu:0"  # 逻辑设备0 = 物理NPU 1
export OCR_LORA_WEIGHTS_PATH="/data1/mindnlp_output/lora_final_20260108_222408/checkpoint-39"
export OCR_API_PORT=8000
export OCR_LOG_LEVEL="INFO"

# 检查端口是否被占用
PORT_IN_USE=$(lsof -ti:8000)
if [ -n "$PORT_IN_USE" ]; then
    echo "警告: 端口8000已被占用 (PID: $PORT_IN_USE)"
    echo "是否停止现有进程? (y/n)"
    read -r STOP_PROCESS
    if [ "$STOP_PROCESS" = "y" ]; then
        echo "停止进程 $PORT_IN_USE..."
        kill -9 "$PORT_IN_USE"
        sleep 2
    else
        echo "请手动停止进程或使用其他端口"
        exit 1
    fi
fi

# 清除Python缓存
echo "清除Python缓存..."
find ~/mindnlp/src/mindnlp/ocr -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# 启动服务
echo ""
echo "=========================================="
echo "启动OCR API服务..."
echo "  - 端口: 8000"
echo "  - 可见NPU: 1,2,3,4,5,6,7 (屏蔽NPU 0)"
echo "  - 使用设备: npu:0 (物理NPU 1)"
echo "  - LoRA权重: $OCR_LORA_WEIGHTS_PATH"
echo "=========================================="
echo ""

# 以后台方式启动
python -m mindnlp.ocr.main > ~/ocr_service.log 2>&1 &
SERVICE_PID=$!

echo "服务已启动 (PID: $SERVICE_PID)"
echo "日志文件: ~/ocr_service.log"
echo ""

# 等待服务启动
echo "等待服务启动..."
for i in {1..30}; do
    if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        echo ""
        echo "✅ 服务启动成功!"
        echo ""
        echo "=========================================="
        echo "服务信息:"
        echo "  - PID: $SERVICE_PID"
        echo "  - 日志: ~/ocr_service.log"
        echo "  - 健康检查: http://localhost:8000/api/v1/health"
        echo "  - API文档: http://localhost:8000/api/docs"
        echo ""
        echo "按 Ctrl+C 终止服务并退出"
        echo "=========================================="
        echo ""
        
        # 设置trap捕获Ctrl+C信号
        trap "echo ''; echo '正在停止服务...'; kill $SERVICE_PID 2>/dev/null; echo '服务已停止'; exit 0" INT
        
        # 显示实时日志
        tail -f ~/ocr_service.log
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "❌ 服务启动超时,显示最近的日志:"
echo "=========================================="
tail -50 ~/ocr_service.log
echo "=========================================="
echo ""
echo "服务PID: $SERVICE_PID"
echo "完整日志: tail -f ~/ocr_service.log"
echo "停止服务: kill $SERVICE_PID"
exit 1
