# OCR测试数据集准备脚本 (PowerShell)
# 用于下载和转换公开数据集

param(
    [string]$DatasetType = "all",
    [string]$DatasetRoot = "./datasets",
    [string]$OutputDir = "./data/ocr_test",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "OCR Test Dataset Preparation" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Dataset root: $DatasetRoot"
Write-Host "Output directory: $OutputDir"
Write-Host ""

# 创建目录
New-Item -ItemType Directory -Force -Path $DatasetRoot | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# 函数：转换ICDAR 2015数据集
function Prepare-ICDAR2015 {
    Write-Host "📦 Preparing ICDAR 2015 dataset..." -ForegroundColor Yellow
    
    $IcdarDir = Join-Path $DatasetRoot "icdar2015"
    
    if (-not (Test-Path $IcdarDir)) {
        Write-Host "⚠️  ICDAR 2015 数据集未找到" -ForegroundColor Red
        Write-Host "请从以下地址下载数据集："
        Write-Host "  https://rrc.cvc.uab.es/?ch=4&com=downloads"
        Write-Host "并解压到: $IcdarDir"
        return $false
    }
    
    # 转换训练集
    if (Test-Path (Join-Path $IcdarDir "train")) {
        & $Python -m mindnlp.ocr.finetune.prepare_dataset `
            --format icdar2015 `
            --data_dir $IcdarDir `
            --output_dir "$OutputDir/icdar2015" `
            --split train `
            --validate
        Write-Host "✅ ICDAR 2015 训练集转换完成" -ForegroundColor Green
    }
    
    # 转换测试集
    if (Test-Path (Join-Path $IcdarDir "test")) {
        & $Python -m mindnlp.ocr.finetune.prepare_dataset `
            --format icdar2015 `
            --data_dir $IcdarDir `
            --output_dir "$OutputDir/icdar2015" `
            --split test `
            --validate
        Write-Host "✅ ICDAR 2015 测试集转换完成" -ForegroundColor Green
    }
    
    return $true
}

# 函数：转换FUNSD数据集
function Prepare-FUNSD {
    Write-Host "📦 Preparing FUNSD dataset..." -ForegroundColor Yellow
    
    $FunsdDir = Join-Path $DatasetRoot "funsd"
    
    if (-not (Test-Path $FunsdDir)) {
        Write-Host "⚠️  FUNSD 数据集未找到" -ForegroundColor Red
        Write-Host "请从以下地址下载数据集:"
        Write-Host "  https://guillaumejaume.github.io/FUNSD/"
        Write-Host "并解压到: $FunsdDir"
        return $false
    }
    
    # 转换训练集
    if (Test-Path (Join-Path $FunsdDir "train")) {
        & $Python -m mindnlp.ocr.finetune.prepare_dataset `
            --format funsd `
            --data_dir $FunsdDir `
            --output_dir "$OutputDir/funsd" `
            --split train `
            --validate
        Write-Host "✅ FUNSD 训练集转换完成" -ForegroundColor Green
    }
    
    # 转换测试集
    if (Test-Path (Join-Path $FunsdDir "test")) {
        & $Python -m mindnlp.ocr.finetune.prepare_dataset `
            --format funsd `
            --data_dir $FunsdDir `
            --output_dir "$OutputDir/funsd" `
            --split test `
            --validate
        Write-Host "✅ FUNSD 测试集转换完成" -ForegroundColor Green
    }
    
    return $true
}

# 函数：转换SROIE数据集
function Prepare-SROIE {
    Write-Host "📦 Preparing SROIE dataset..." -ForegroundColor Yellow
    
    $SroieDir = Join-Path $DatasetRoot "sroie"
    
    if (-not (Test-Path $SroieDir)) {
        Write-Host "⚠️  SROIE 数据集未找到" -ForegroundColor Red
        Write-Host "请从以下地址下载数据集:"
        Write-Host "  https://rrc.cvc.uab.es/?ch=13&com=downloads"
        Write-Host "并解压到: $SroieDir"
        return $false
    }
    
    # 转换训练集
    if (Test-Path (Join-Path $SroieDir "train")) {
        & $Python -m mindnlp.ocr.finetune.prepare_dataset `
            --format sroie `
            --data_dir $SroieDir `
            --output_dir "$OutputDir/sroie" `
            --split train `
            --validate
        Write-Host "✅ SROIE 训练集转换完成" -ForegroundColor Green
    }
    
    # 转换测试集
    if (Test-Path (Join-Path $SroieDir "test")) {
        & $Python -m mindnlp.ocr.finetune.prepare_dataset `
            --format sroie `
            --data_dir $SroieDir `
            --output_dir "$OutputDir/sroie" `
            --split test `
            --validate
        Write-Host "✅ SROIE 测试集转换完成" -ForegroundColor Green
    }
    
    return $true
}

# 主流程
switch ($DatasetType.ToLower()) {
    "icdar" { 
        Prepare-ICDAR2015
    }
    "icdar2015" { 
        Prepare-ICDAR2015
    }
    "funsd" { 
        Prepare-FUNSD
    }
    "sroie" { 
        Prepare-SROIE
    }
    "all" {
        $results = @()
        $results += Prepare-ICDAR2015
        $results += Prepare-FUNSD
        $results += Prepare-SROIE
        
        if ($results -contains $false) {
            Write-Host "⚠️  部分数据集准备失败" -ForegroundColor Yellow
        }
    }
    default {
        Write-Host "Usage: .\prepare_test_dataset.ps1 -DatasetType [icdar|funsd|sroie|all]" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✅ Dataset preparation completed!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "输出目录: $OutputDir"
Write-Host ""
Write-Host "下一步："
Write-Host "1. 检查转换后的数据集"
Write-Host "2. 运行训练脚本进行微调"
Write-Host "3. 使用评估脚本验证模型性能"
