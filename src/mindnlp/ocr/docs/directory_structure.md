# MindNLP OCR 目录结构

本文档说明 `src/mindnlp/ocr` 的完整目录结构和各模块功能。

## 📁 目录结构概览

```
src/mindnlp/ocr/
├── api/                    # API 服务层
│   ├── routes/            # API 路由定义
│   └── server.py          # FastAPI 服务器
│
├── benchmarks/            # 性能测试脚本 ⭐ (新整理)
│   ├── benchmark_kv_cache.py           # KV Cache 性能测试
│   ├── benchmark_comparison.py         # KV Cache 对比测试
│   └── validate_acceptance_criteria.py # 验收标准验证
│
├── config/                # 配置管理
│   ├── model_config.yaml  # 模型配置
│   └── training_config.yaml
│
├── core/                  # 核心业务逻辑
│   ├── inference.py       # 推理引擎
│   ├── preprocessing.py   # 预处理
│   └── postprocessing.py  # 后处理
│
├── docs/                  # 文档 ⭐ (更新)
│   ├── directory_structure.md  # 目录结构说明 (本文档)
│   ├── kv_cache_guide.md      # KV Cache 使用指南
│   └── scripts_guide.md       # 脚本使用指南
│
├── finetune/              # 模型微调
│   ├── lora_trainer.py    # LoRA 训练器
│   └── data_loader.py     # 数据加载器
│
├── models/                # 模型实现
│   ├── base.py           # 模型基类
│   ├── qwen2vl.py        # Qwen2-VL 模型 (含 KV Cache)
│   └── __init__.py
│
├── scripts/               # Shell/PowerShell 脚本 ⭐ (新整理)
│   ├── download_datasets.sh          # 下载数据集
│   ├── prepare_test_dataset.sh/.ps1  # 准备测试数据
│   ├── run_full_evaluation.sh        # 完整评估流程
│   ├── setup_environment.sh          # 环境设置
│   └── start_lora_api.sh             # 启动 LoRA API 服务
│
├── tests/                 # 测试套件 ⭐ (新整理)
│   ├── test_kv_cache.py          # KV Cache 功能测试
│   ├── test_lora_loading.py      # LoRA 加载测试
│   └── test_server_kv_cache.py   # 服务器端测试
│
├── tools/                 # 开发工具 ⭐ (新整理)
│   ├── check_config.py               # 配置检查
│   ├── convert_datasets.py           # 数据集转换
│   ├── convert_features_to_numpy.py  # 特征转换
│   └── ocr_toolkit.py                # OCR 工具集
│
├── utils/                 # 工具函数
│   ├── cache_manager.py   # KV Cache 管理器
│   ├── logger.py          # 日志工具
│   └── exceptions.py      # 自定义异常
│
├── main.py               # 主入口
├── README.md             # 项目说明
└── __init__.py           # 包初始化
```

---

## 📦 模块功能说明

### 🎯 核心模块 (Core Modules)

#### `models/` - 模型实现
- **base.py**: 所有模型的基类，定义通用接口
- **qwen2vl.py**: Qwen2-VL 模型实现
  - 集成 KV Cache 优化
  - NPU 完全兼容 (Ascend 910)
  - LoRA 微调支持

#### `utils/` - 工具库
- **cache_manager.py**: KV Cache 管理器
  - LRU 缓存策略
  - TTL 过期控制
  - 内存限制管理
- **logger.py**: 统一日志管理
- **exceptions.py**: 自定义异常类

#### `core/` - 业务逻辑
- **inference.py**: 推理引擎
- **preprocessing.py**: 图像预处理
- **postprocessing.py**: 结果后处理

---

### 🧪 测试与评估 (Testing & Benchmarking)

#### `benchmarks/` - 性能测试
从 `scripts/ocr/` 迁移而来，包含专业的性能测试脚本：

**benchmark_kv_cache.py**
- 单图推理延迟测试
- 批量推理吞吐量测试
- 长序列生成测试
- 内存使用分析

**benchmark_comparison.py**
- KV Cache enabled vs disabled 对比
- 生成完整性能报告 (JSON)
- 支持多种测试场景

**validate_acceptance_criteria.py**
- 自动验证性能指标
- 检查是否满足验收标准
- 生成验收报告

#### `tests/` - 功能测试
从 `scripts/ocr/` 迁移而来，包含集成测试：

- **test_kv_cache.py**: KV Cache 功能正确性测试
- **test_lora_loading.py**: LoRA 模型加载测试
- **test_server_kv_cache.py**: 服务器端 KV Cache 测试

---

### 🔧 开发工具 (Development Tools)

#### `tools/` - 工具脚本
从 `scripts/ocr/` 迁移而来，包含开发辅助工具：

- **check_config.py**: 验证配置文件完整性
- **convert_datasets.py**: 数据集格式转换
- **convert_features_to_numpy.py**: 特征提取和转换
- **ocr_toolkit.py**: OCR 通用工具集

#### `scripts/` - 自动化脚本
从 `scripts/ocr/` 迁移而来，包含 Shell/PowerShell 脚本：

- **download_datasets.sh**: 自动下载数据集
- **prepare_test_dataset.sh/.ps1**: 准备测试数据
- **setup_environment.sh**: 环境初始化
- **run_full_evaluation.sh**: 完整评估流程
- **start_lora_api.sh**: 启动 API 服务

---

### 🌐 服务层 (Service Layer)

#### `api/` - API 服务
- **server.py**: FastAPI 服务器
- **routes/**: API 路由定义

#### `finetune/` - 微调模块
- **lora_trainer.py**: LoRA 训练器
- **data_loader.py**: 数据加载

---

## 🚀 使用指南

### 运行性能测试

```bash
# 单项测试 (KV Cache)
python src/mindnlp/ocr/benchmarks/benchmark_kv_cache.py \
    --model_path /path/to/model.npz \
    --device npu:0

# 对比测试 (KV Cache ON vs OFF)
python src/mindnlp/ocr/benchmarks/benchmark_comparison.py \
    --model_path /path/to/model.npz \
    --device npu:0 \
    --output results.json

# 验收标准检查
python src/mindnlp/ocr/benchmarks/validate_acceptance_criteria.py \
    --results results.json
```

### 运行功能测试

```bash
# KV Cache 功能测试
python src/mindnlp/ocr/tests/test_kv_cache.py

# LoRA 加载测试
python src/mindnlp/ocr/tests/test_lora_loading.py
```

### 使用开发工具

```bash
# 检查配置
python src/mindnlp/ocr/tools/check_config.py --config config.yaml

# 转换数据集
python src/mindnlp/ocr/tools/convert_datasets.py \
    --input data.json \
    --output data.npz
```

### 使用自动化脚本

```bash
# Linux/Mac
bash src/mindnlp/ocr/scripts/setup_environment.sh
bash src/mindnlp/ocr/scripts/download_datasets.sh

# Windows
powershell src/mindnlp/ocr/scripts/prepare_test_dataset.ps1
```

---

## 📝 迁移说明

### 从 `scripts/ocr/` 迁移

以下文件已从 `scripts/ocr/` 迁移到新位置：

| 原位置 | 新位置 | 说明 |
|--------|--------|------|
| `scripts/ocr/benchmark_*.py` | `src/mindnlp/ocr/benchmarks/` | 性能测试脚本 |
| `scripts/ocr/test_*.py` | `src/mindnlp/ocr/tests/` | 功能测试脚本 |
| `scripts/ocr/check_config.py` | `src/mindnlp/ocr/tools/` | 配置检查工具 |
| `scripts/ocr/convert_*.py` | `src/mindnlp/ocr/tools/` | 转换工具 |
| `scripts/ocr/*.sh` | `src/mindnlp/ocr/scripts/` | Shell 脚本 |
| `scripts/ocr/*.ps1` | `src/mindnlp/ocr/scripts/` | PowerShell 脚本 |
| `scripts/ocr/README.md` | `src/mindnlp/ocr/docs/scripts_guide.md` | 脚本使用指南 |

### 更新导入路径

如果你的代码引用了这些文件，请更新导入路径：

```python
# 旧路径 (已弃用)
from scripts.ocr.benchmark_kv_cache import run_benchmark

# 新路径
from mindnlp.ocr.benchmarks.benchmark_kv_cache import run_benchmark
```

---

## 🔄 持续更新

本目录结构遵循模块化和职责分离原则，便于：
- ✅ 代码维护和扩展
- ✅ 测试和调试
- ✅ 文档生成
- ✅ CI/CD 集成

如有新增模块或调整，请更新本文档。

---

**最后更新**: 2026-01-24  
**维护者**: MindNLP OCR Team
