#!/bin/bash
# OCR数据集下载脚本
# 存储位置: /data1/datasets/

set -e

echo "========================================"
echo "OCR数据集下载脚本"
echo "========================================"
echo ""

cd /data1/datasets

# 函数：下载FUNSD数据集
download_funsd() {
    echo " 下载FUNSD数据集..."
    
    if [ -d "funsd" ]; then
        echo "  FUNSD目录已存在，跳过下载"
        return 0
    fi
    
    # FUNSD可以直接下载
    echo "正在下载FUNSD数据集..."
    wget -O funsd.zip https://guillaumejaume.github.io/FUNSD/dataset.zip
    
    echo "解压FUNSD数据集..."
    unzip -q funsd.zip
    mv dataset funsd
    rm funsd.zip
    
    echo " FUNSD数据集下载完成"
    echo "位置: /data1/datasets/funsd/"
    ls -lh funsd/
}

# 函数：ICDAR 2015说明
show_icdar_instructions() {
    echo ""
    echo " ICDAR 2015数据集下载说明"
    echo "----------------------------------------"
    echo "ICDAR 2015需要手动下载（需要注册账号）"
    echo ""
    echo "步骤："
    echo "1. 访问: https://rrc.cvc.uab.es/?ch=4&com=downloads"
    echo "2. 注册并登录账号"
    echo "3. 下载以下文件:"
    echo "   - ch4_training_images.zip"
    echo "   - ch4_training_localization_transcription_gt.zip"
    echo "   - ch4_test_images.zip"
    echo "   - Task4.1_and_Task4.2_Test_Ground_Truth.zip"
    echo ""
    echo "4. 将下载的文件上传到服务器:"
    echo "   scp ch4_*.zip mseco@192.168.88.19:/data1/datasets/"
    echo ""
    echo "5. 在服务器上解压:"
    echo "   cd /data1/datasets"
    echo "   mkdir -p icdar2015/train/images icdar2015/train/gt"
    echo "   mkdir -p icdar2015/test/images icdar2015/test/gt"
    echo "   unzip ch4_training_images.zip -d icdar2015/train/images/"
    echo "   unzip ch4_training_localization_transcription_gt.zip -d icdar2015/train/gt/"
    echo "   unzip ch4_test_images.zip -d icdar2015/test/images/"
    echo "   unzip Task4.1_and_Task4.2_Test_Ground_Truth.zip -d icdar2015/test/gt/"
    echo ""
}

# 函数：SROIE说明
show_sroie_instructions() {
    echo ""
    echo " SROIE数据集下载说明"
    echo "----------------------------------------"
    echo "SROIE需要手动下载（需要注册账号）"
    echo ""
    echo "步骤："
    echo "1. 访问: https://rrc.cvc.uab.es/?ch=13&com=downloads"
    echo "2. 注册并登录账号"
    echo "3. 下载Task 1数据集"
    echo ""
    echo "4. 将下载的文件上传到服务器:"
    echo "   scp sroie_*.zip mseco@192.168.88.19:/data1/datasets/"
    echo ""
    echo "5. 在服务器上解压:"
    echo "   cd /data1/datasets"
    echo "   mkdir -p sroie"
    echo "   unzip sroie_*.zip -d sroie/"
    echo ""
}

# 主流程
echo "当前目录: $(pwd)"
echo "可用空间:"
df -h /data1 | tail -1
echo ""

# 选择下载方式
case "${1:-all}" in
    funsd)
        download_funsd
        ;;
    icdar)
        show_icdar_instructions
        ;;
    sroie)
        show_sroie_instructions
        ;;
    all)
        download_funsd
        show_icdar_instructions
        show_sroie_instructions
        ;;
    help)
        echo "用法: bash download_datasets.sh [funsd|icdar|sroie|all]"
        echo ""
        echo "  funsd  - 自动下载FUNSD数据集"
        echo "  icdar  - 显示ICDAR 2015下载说明"
        echo "  sroie  - 显示SROIE下载说明"
        echo "  all    - 执行所有操作（默认）"
        ;;
    *)
        echo " 未知选项: $1"
        echo "使用 'bash download_datasets.sh help' 查看帮助"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo " 完成！"
echo "========================================"
echo ""
echo "下一步："
echo "1. 下载ICDAR 2015和SROIE数据集（按照上述说明）"
echo "2. 运行数据集转换:"
echo "   cd ~/mindnlp"
echo "   bash scripts/ocr/prepare_test_dataset.sh all"
echo ""
echo "数据将转换到: /data1/ocr_test/"