#!/bin/bash
# 从备份恢复并初始化 Git (适用于网络问题)
# 在服务器上执行: bash restore_and_init.sh

set -e

echo "🔍 检查备份目录..."
cd /data1

# 查找最新备份
latest_backup=$(ls -td mindnlp_backup_* 2>/dev/null | head -1)

if [ -z "$latest_backup" ]; then
    echo "❌ 未找到备份目录"
    echo "请检查 /data1/ 下是否有 mindnlp_backup_* 目录"
    exit 1
fi

echo "📁 找到备份: $latest_backup"

# 删除失败的克隆目录
if [ -d "mindnlp" ]; then
    echo "🗑️  删除失败的克隆目录..."
    rm -rf mindnlp
fi

# 从备份恢复
echo "📦 从备份恢复代码..."
cp -r "$latest_backup" mindnlp
cd mindnlp

echo "✅ 代码已恢复"
echo ""

# 初始化 Git
echo "🔧 初始化 Git 仓库..."
git init

# 添加远程仓库
echo "🔗 添加远程仓库..."
git remote add origin https://github.com/mindspore-lab/mindnlp.git

# 尝试 fetch（如果网络仍有问题会失败，但不影响后续手动操作）
echo "📥 尝试获取远程分支信息..."
if git fetch origin feature/issue-2377-quantization-phase3 2>/dev/null; then
    echo "✅ 远程分支信息获取成功"
    git checkout -b feature/issue-2377-quantization-phase3 FETCH_HEAD
else
    echo "⚠️  网络问题，无法获取远程分支"
    echo "📝 手动下载更新文件..."
    
    # 手动下载关键文件
    echo ""
    echo "下载 qwen2vl.py..."
    wget -q -O src/mindnlp/ocr/models/qwen2vl.py \
        https://raw.githubusercontent.com/messere1/mindnlp/feature/issue-2377-quantization-phase3/src/mindnlp/ocr/models/qwen2vl.py \
        || echo "  ⚠️  下载失败，使用现有文件"
    
    echo "下载 cache_manager.py..."
    wget -q -O src/mindnlp/ocr/utils/cache_manager.py \
        https://raw.githubusercontent.com/messere1/mindnlp/feature/issue-2377-quantization-phase3/src/mindnlp/ocr/utils/cache_manager.py \
        || echo "  ⚠️  下载失败，使用现有文件"
    
    # 创建本地分支
    git checkout -b feature/issue-2377-quantization-phase3
fi

echo ""
echo "✅ Git 仓库初始化完成"
echo ""

# 验证文件结构
echo "📁 验证目录结构:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 检查旧位置的文件（应该已移动）
if [ -d "scripts/ocr" ]; then
    count=$(ls -1 scripts/ocr/*.py 2>/dev/null | wc -l)
    if [ $count -gt 0 ]; then
        echo "  ⚠️  scripts/ocr/ 仍有 $count 个 Python 文件（应该已移动）"
        echo "     需要手动整理或重新克隆"
    else
        echo "  ✓ scripts/ocr/ 已清空"
    fi
fi

# 检查新位置
dirs=(
    "src/mindnlp/ocr/benchmarks"
    "src/mindnlp/ocr/tests"
    "src/mindnlp/ocr/tools"
    "src/mindnlp/ocr/scripts"
)

need_reorganize=false
for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        count=$(ls -1 "$dir" 2>/dev/null | wc -l)
        echo "  ✓ $dir/ ($count 文件)"
    else
        echo "  ✗ $dir/ (缺失 - 需要整理文件)"
        need_reorganize=true
    fi
done
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$need_reorganize" = true ]; then
    echo "⚠️  新目录结构缺失，需要整理文件"
    echo ""
    echo "📝 手动整理步骤:"
    echo "  1. 创建目录:"
    echo "     mkdir -p src/mindnlp/ocr/{benchmarks,tests,tools,scripts}"
    echo ""
    echo "  2. 移动文件 (如果 scripts/ocr 存在):"
    echo "     mv scripts/ocr/benchmark_*.py src/mindnlp/ocr/benchmarks/"
    echo "     mv scripts/ocr/test_*.py src/mindnlp/ocr/tests/"
    echo "     mv scripts/ocr/{check_config,convert_*,ocr_toolkit}.py src/mindnlp/ocr/tools/"
    echo "     mv scripts/ocr/*.{sh,ps1} src/mindnlp/ocr/scripts/"
    echo ""
else
    echo "✅ 目录结构正确"
fi

# 验证关键文件
echo "🔍 验证关键文件:"
files=(
    "src/mindnlp/ocr/models/qwen2vl.py"
    "src/mindnlp/ocr/utils/cache_manager.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "  ✓ $file ($size)"
        
        # 检查 BF16 修复
        if [ "$file" = "src/mindnlp/ocr/models/qwen2vl.py" ]; then
            if grep -q "Converting BF16 parameters" "$file"; then
                echo "    ✓ 包含 BF16 转换代码"
            else
                echo "    ⚠️  缺少 BF16 转换代码 - 需要更新"
            fi
        fi
    else
        echo "  ✗ $file (缺失)"
    fi
done
echo ""

echo "✅ 初始化完成！"
echo ""
echo "📝 后续操作:"
echo "  1. 如果文件需要整理，执行上面的手动整理步骤"
echo "  2. 验证 BF16 修复: grep -n 'Converting BF16' src/mindnlp/ocr/models/qwen2vl.py"
echo "  3. 运行测试验证功能正常"
