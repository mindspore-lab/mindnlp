# MindNLP OCR 测试套件

OCR模块的核心测试文件。

## 📁 测试文件

```
tests/mindnlp/ocr/
├── test_qwen2vl.py                 # Qwen2-VL模型测试（Mock + 真实模型）
├── test_preprocessing.py           # 预处理组件单元测试
├── test_quantization.py            # 量化性能测试
├── test_monitoring_standalone.py   # 监控系统独立测试
└── README.md                       # 本文件
```

## 🧪 测试说明

### 1. test_qwen2vl.py - 核心模型测试
**用途**: 测试Qwen2-VL模型推理功能（Issue #2366）  
**测试内容**:
- Mock测试：验证API实现正确性（无需下载模型，快速）
- 真实模型测试：验证与transformers的完整兼容性

**运行方式**:
```bash
# Mock测试（默认，快速）
pytest tests/mindnlp/ocr/test_qwen2vl.py -v

# 真实模型测试（需要下载约4GB模型）
pytest tests/mindnlp/ocr/test_qwen2vl.py -v --run-real-model
```

### 2. test_preprocessing.py - 预处理组件测试
**用途**: 测试图像处理、Prompt构建、批处理等核心组件  
**测试类**:
- `TestImageProcessor` - 图像预处理
- `TestPromptBuilder` - Prompt构建
- `TestBatchCollator` - 批量数据整理
- `TestInputValidator` - 输入验证
- `TestIntegration` - 集成测试

**运行方式**:
```bash
pytest tests/mindnlp/ocr/test_preprocessing.py -v
```

### 3. test_quantization.py - 量化性能测试
**用途**: 测试不同量化模式的推理速度和精度（Issue #2377）  
**测试内容**:
- 8位量化性能
- 4位量化性能
- 内存占用对比
- 精度损失评估

**运行方式**:
```bash
pytest tests/mindnlp/ocr/test_quantization.py -v
```

### 4. test_monitoring_standalone.py - 监控系统测试
**用途**: 测试监控、日志和性能分析系统（Issue #2381）  
**测试内容**:
- 结构化日志系统（structlog）
- 分布式追踪（OpenTelemetry）
- 性能Profiling（CPU/Memory）
- 系统集成测试

**运行方式**:
```bash
python tests/mindnlp/ocr/test_monitoring_standalone.py
```

## 🚀 快速开始

### 安装依赖

```bash
# 基础依赖
pip install -r requirements/requirements.txt

# OCR模块依赖
pip install -r requirements/ocr-requirements.txt
```

### 运行所有测试

### 运行所有测试

```bash
# 仅运行Mock测试（快速）
pytest tests/mindnlp/ocr/ -v

# 包含真实模型测试
pytest tests/mindnlp/ocr/ -v --run-real-model
```

### 运行单个测试文件

```bash
# Qwen2-VL测试
pytest tests/mindnlp/ocr/test_qwen2vl.py -v

# 预处理测试
pytest tests/mindnlp/ocr/test_preprocessing.py -v

# 量化测试
pytest tests/mindnlp/ocr/test_quantization.py -v

# 监控系统测试（使用python直接运行）
python tests/mindnlp/ocr/test_monitoring_standalone.py
```

## 📊 测试覆盖范围

| 测试文件 | 测试内容 | Issue | 行数 |
|---------|---------|-------|------|
| test_qwen2vl.py | Qwen2-VL模型推理 | #2366 | 427 |
| test_preprocessing.py | 预处理组件 | #2350 | 621 |
| test_quantization.py | 模型量化 | #2377 | 319 |
| test_monitoring_standalone.py | 监控日志 | #2381 | 396 |

## 🔍 已清理的测试文件

以下测试文件已被删除（功能已被上述核心测试覆盖）：
- ~~test_api_complete.py~~ - 功能已整合到 test_qwen2vl.py
- ~~test_api_real_model.py~~ - 功能已整合到 test_qwen2vl.py
- ~~test_concurrent_processing.py~~ - 非核心功能，暂不测试
- ~~test_evaluate_model.py~~ - 评估功能移至 src/mindnlp/ocr/finetune/evaluate.py
- ~~test_kv_cache.py~~ - KV Cache测试已整合
- ~~test_lora_loading.py~~ - LoRA测试已整合
- ~~test_monitoring.py~~ - 替换为 test_monitoring_standalone.py
- ~~test_multi_scenario.py~~ - 多场景测试移至专项工具
- ~~test_performance.py~~ - 性能测试移至 benchmarks/
- ~~test_server_kv_cache.py~~ - 服务器测试已整合

## 📝 注意事项

1. **Mock测试优先**: 默认运行Mock测试，速度快，适合CI/CD
2. **真实模型测试**: 使用 `--run-real-model` 标志，首次会下载约4GB模型
3. **独立测试**: test_monitoring_standalone.py 需要单独运行，避免循环导入
4. **环境隔离**: 真实模型测试建议使用独立conda环境

## 🔗 相关链接

- [OCR模块文档](../../../src/mindnlp/ocr/README.md)
- [Issue #2348 - VLM-OCR模块](https://github.com/mindspore-lab/mindnlp/issues/2348)
  - 图像识别能力
  - 完整推理流程

## 环境要求

### 最低配置（Mock 测试）
- Python 3.10+
- PyTorch 2.1.2+
- transformers 4.37.0+
- 2 GB 内存

### 推荐配置（真实模型测试）
- Python 3.10+
- PyTorch 2.4.0+
- transformers 4.37.0+
- 8 GB 内存
- 10 GB 磁盘空间（存储模型）

## 故障排除

### 问题 1: mindnlp patch 冲突
**症状**: `TypeError: typing.Optional type checking conflict`

**解决方案**: 使用独立环境运行真实模型测试
```bash
conda create -n qwen2vl_test python=3.10 -y
conda activate qwen2vl_test
pip install -r requirements/ocr-requirements.txt
pytest tests/mindnlp/ocr/test_qwen2vl.py -v --run-real-model
```

### 问题 2: NumPy 版本冲突
**症状**: `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

**解决方案**:
```bash
pip install "numpy<2.0" --force-reinstall
```

### 问题 3: 网络连接问题
**症状**: 无法下载模型

**解决方案**: 设置镜像源（已在测试代码中配置）
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题 4: torch.compiler 属性错误
**症状**: `module 'torch.compiler' has no attribute 'is_compiling'`

**解决方案**: 升级 PyTorch
```bash
pip install torch>=2.4.0 torchvision>=0.19.0 --force-reinstall
```

## 验证结果

### ✅ Mock 测试（23/23 通过）
```bash
$ pytest tests/mindnlp/ocr/test_qwen2vl.py -v
========================= 23 passed in 0.5s =========================
```

### ✅ 真实模型测试（1/1 通过）
```bash
$ pytest tests/mindnlp/ocr/test_qwen2vl.py -v --run-real-model
生成的文本: The image shows a blue square...
========================= 24 passed in 15.2s ========================
```

## 贡献指南

添加新测试时：
1. Mock 测试放在 `TestQwen2VLInferenceMock` 类中
2. 真实模型测试放在 `TestQwen2VLInferenceRealModel` 类中，并添加 `@pytest.mark.real_model` 装饰器
3. 确保测试名称清晰描述测试内容
4. 添加适当的文档字符串

## 参考

- Issue: #2366
- 模型: [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- 文档: transformers >= 4.37.0
- OpenAPI Schema

### 7. 配置管理
- Settings 验证
- 环境变量支持

### 8. 代码结构
- 目录结构验证
- 模块导入测试

## 预期结果

```
============================================================
Issue #2349 API 服务层功能验证
============================================================

[✓] 应用创建
[✓] Lifespan 函数
[✓] 引擎依赖注入
[✓] 健康检查端点
[✓] 就绪检查端点
[✓] 单图 OCR 端点
[✓] 批量 OCR 端点
[✓] URL OCR 端点
[✓] 请求 Schema
[✓] 响应 Schema
[✓] 响应字段验证
[✓] 文件类型验证
[✓] 异常处理器
[✓] 日志中间件
[✓] 处理时间记录
[✓] CORS 跨域支持
[✓] Swagger UI 文档
[✓] OpenAPI Schema
[✓] 引擎依赖注入
[✓] 应用配置
[✓] 所有目录结构

总结: 所有核心功能已实现！✓
```

## 依赖项

```bash
pip install fastapi uvicorn pydantic pillow httpx
```

## 注意事项

- 测试使用 Mock 引擎，不需要实际的 VLM 模型
- 测试会自动创建测试图像
- 输出编码设置为 UTF-8
