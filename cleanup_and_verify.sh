#!/bin/bash
# 清理和验证服务器代码
# 在服务器上执行: bash cleanup_and_verify.sh

set -e

echo "🧹 清理临时文件..."
cd /data1/mindnlp

# 删除 __pycache__
if [ -d "scripts/ocr/__pycache__" ]; then
    rm -rf scripts/ocr/__pycache__
    echo "  ✓ 删除 scripts/ocr/__pycache__"
fi

# 检查是否可以删除 scripts/ocr 目录
if [ -d "scripts/ocr" ]; then
    remaining=$(ls -A scripts/ocr 2>/dev/null | wc -l)
    if [ $remaining -eq 0 ]; then
        rmdir scripts/ocr
        echo "  ✓ 删除空目录 scripts/ocr"
    else
        echo "  ⚠️  scripts/ocr 还有内容，保留"
    fi
fi

echo ""
echo "✅ 清理完成！"
echo ""

# 验证关键文件
echo "🔍 验证关键代码..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. 检查 qwen2vl.py 的 BF16 修复
echo ""
echo "1️⃣  检查 qwen2vl.py BF16 修复代码:"
if grep -q "Converting BF16 parameters to FP16 for NPU compatibility" src/mindnlp/ocr/models/qwen2vl.py; then
    echo "  ✓ 包含 BF16 转换日志"
    
    if grep -q "param.data.to('cpu', dtype=torch.float32)" src/mindnlp/ocr/models/qwen2vl.py; then
        echo "  ✓ 包含 CPU 中转转换代码"
        
        # 显示关键代码行
        echo ""
        echo "  关键代码片段:"
        grep -n -A 2 "Converting BF16 parameters" src/mindnlp/ocr/models/qwen2vl.py | head -10
    else
        echo "  ✗ 缺少 CPU 中转转换代码"
    fi
else
    echo "  ✗ 缺少 BF16 转换代码"
    echo "  ⚠️  需要手动更新 qwen2vl.py"
fi

echo ""
echo "2️⃣  检查 cache_manager.py:"
if [ -f "src/mindnlp/ocr/utils/cache_manager.py" ]; then
    size=$(du -h src/mindnlp/ocr/utils/cache_manager.py | cut -f1)
    echo "  ✓ cache_manager.py 存在 ($size)"
    
    if grep -q "class KVCacheManager" src/mindnlp/ocr/utils/cache_manager.py; then
        echo "  ✓ 包含 KVCacheManager 类"
    fi
else
    echo "  ✗ cache_manager.py 缺失"
fi

echo ""
echo "3️⃣  检查 benchmark 脚本:"
for file in benchmark_kv_cache.py benchmark_comparison.py; do
    if [ -f "src/mindnlp/ocr/benchmarks/$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file 缺失"
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "📊 完整目录结构:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tree -L 3 src/mindnlp/ocr/ 2>/dev/null || find src/mindnlp/ocr/ -type d | sed 's|[^/]*/| |g' | sed 's|^ ||'
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "✅ 验证完成！"
echo ""
echo "📝 下一步操作:"
echo ""
echo "1️⃣  快速功能测试 (验证导入和基本功能):"
echo "   python -c \"from mindnlp.ocr.utils.cache_manager import KVCacheManager; print('✓ Import successful')\""
echo ""
echo "2️⃣  运行 KV Cache 性能测试 (约 30 分钟):"
echo "   python src/mindnlp/ocr/benchmarks/benchmark_comparison.py \\"
echo "       --model_path /data1/mindnlp_output/lora_final_20260108_222408/checkpoint-39/adapter_model.npz \\"
echo "       --device npu:0 \\"
echo "       --output /data1/benchmark_results/kv_cache_final_\$(date +%Y%m%d_%H%M%S).json"
echo ""
echo "3️⃣  查看之前的测试结果 (如果存在):"
echo "   ls -lh /data1/benchmark_results/"
echo ""
