#!/bin/bash
# 服务器初始化和更新脚本
# 在服务器上执行: bash init_and_update.sh

set -e

echo "🔍 检查当前目录..."
pwd

# 检查是否是 Git 仓库
if [ -d ".git" ]; then
    echo "✅ 已是 Git 仓库，直接更新..."
    git fetch origin
    git checkout feature/issue-2377-quantization-phase3
    git pull origin feature/issue-2377-quantization-phase3
else
    echo "⚠️  当前目录不是 Git 仓库"
    
    # 检查是否有 mindnlp 代码
    if [ -f "src/mindnlp/__init__.py" ]; then
        echo "📁 发现已有代码，初始化 Git 仓库..."
        
        # 备份当前代码
        backup_dir="/data1/mindnlp_backup_$(date +%Y%m%d_%H%M%S)"
        echo "💾 备份当前代码到: $backup_dir"
        cp -r . "$backup_dir"
        
        # 初始化 Git 并关联远程仓库
        git init
        git remote add origin https://github.com/mindspore-lab/mindnlp.git
        git fetch origin
        git checkout -b feature/issue-2377-quantization-phase3 origin/feature/issue-2377-quantization-phase3
        
        echo "✅ Git 仓库初始化完成"
    else
        echo "📦 克隆新仓库..."
        cd /data1
        
        # 备份旧目录
        if [ -d "mindnlp" ]; then
            backup_dir="mindnlp_backup_$(date +%Y%m%d_%H%M%S)"
            echo "💾 备份旧目录为: $backup_dir"
            mv mindnlp "$backup_dir"
        fi
        
        # 克隆仓库
        git clone -b feature/issue-2377-quantization-phase3 https://github.com/mindspore-lab/mindnlp.git
        cd mindnlp
        
        echo "✅ 仓库克隆完成"
    fi
fi

echo ""
echo "✅ 代码更新完成！"
echo ""

# 验证新的目录结构
echo "📁 验证 OCR 目录结构:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

dirs=(
    "src/mindnlp/ocr/benchmarks"
    "src/mindnlp/ocr/tests"
    "src/mindnlp/ocr/tools"
    "src/mindnlp/ocr/scripts"
    "src/mindnlp/ocr/models"
    "src/mindnlp/ocr/utils"
)

all_exist=true
for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        count=$(ls -1 "$dir" 2>/dev/null | wc -l)
        echo "  ✓ $dir/ ($count 文件)"
    else
        echo "  ✗ $dir/ (缺失)"
        all_exist=false
    fi
done
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 验证关键文件
echo "🔍 验证关键文件:"
files=(
    "src/mindnlp/ocr/models/qwen2vl.py"
    "src/mindnlp/ocr/utils/cache_manager.py"
    "src/mindnlp/ocr/benchmarks/benchmark_kv_cache.py"
    "src/mindnlp/ocr/benchmarks/benchmark_comparison.py"
    "src/mindnlp/ocr/tests/test_kv_cache.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "  ✓ $file ($size)"
    else
        echo "  ✗ $file (缺失)"
        all_exist=false
    fi
done
echo ""

if [ "$all_exist" = false ]; then
    echo "⚠️  部分文件缺失，请检查分支是否正确！"
    echo ""
    echo "当前分支:"
    git branch -vv
    exit 1
fi

echo "✅ 所有文件验证通过！"
echo ""
echo "📝 最新提交:"
git log --oneline -3
echo ""

# 检查 qwen2vl.py 中的 BF16 修复
echo "🔍 验证 BF16 修复代码..."
if grep -q "Converting BF16 parameters" src/mindnlp/ocr/models/qwen2vl.py; then
    echo "  ✓ BF16 转换逻辑已包含"
else
    echo "  ✗ BF16 转换逻辑未找到"
fi

if grep -q "param.data.to('cpu', dtype=torch.float32)" src/mindnlp/ocr/models/qwen2vl.py; then
    echo "  ✓ CPU 中转转换代码已包含"
else
    echo "  ✗ CPU 中转转换代码未找到"
fi
echo ""

echo "✅ 服务器代码更新完成！"
echo ""
echo "📋 后续操作:"
echo "  1. 快速测试: python src/mindnlp/ocr/tests/test_kv_cache.py"
echo "  2. 性能测试: python src/mindnlp/ocr/benchmarks/benchmark_comparison.py --help"
echo "  3. 查看文档: cat src/mindnlp/ocr/docs/directory_structure.md"
