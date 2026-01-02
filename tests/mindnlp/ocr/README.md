# Qwen2-VL OCR 测试套件

测试 Issue #2366 的 Qwen2-VL OCR 功能实现。

## 文件结构

```
tests/mindnlp/ocr/
├── test_qwen2vl.py          # 统一的测试文件（Mock + 真实模型测试）
├── test_api_real_model.py   # API 真实模型集成测试
└── README.md                # 本文件

requirements/
└── ocr-requirements.txt     # OCR 功能依赖配置

src/mindnlp/ocr/
├── main.py                  # API 服务入口
├── .env                     # 环境配置文件
├── api/                     # FastAPI 路由和接口
├── core/                    # OCR 引擎核心
├── models/                  # 模型封装（Qwen2VL）
└── ...
```

## 快速开始

### 1. 安装依赖

```bash
# 在现有环境中安装
pip install -r requirements/ocr-requirements.txt

# 或创建独立环境（推荐用于真实模型测试）
conda create -n qwen2vl_test python=3.10 -y
conda activate qwen2vl_test
pip install -r requirements/ocr-requirements.txt
```

### 2. 运行单元测试

```bash
# 运行 Mock 测试（快速，不需要下载模型）
pytest tests/mindnlp/ocr/test_qwen2vl.py -v

# 运行所有测试（包括真实模型，首次运行会下载约 4GB）
pytest tests/mindnlp/ocr/test_qwen2vl.py -v --run-real-model

# 只运行特定测试
pytest tests/mindnlp/ocr/test_qwen2vl.py::TestQwen2VLInferenceMock::test_complete_inference_pipeline -v
```

### 3. 启动 API 服务（真实模型）

```bash
# 配置环境变量（使用真实模型）
cd src/mindnlp/ocr
# 编辑 .env 文件，设置 OCR_USE_MOCK_ENGINE=False

# 启动服务（首次运行会下载 Qwen2-VL 模型约 4GB）
python main.py

# 服务启动后访问：
# - API 文档: http://localhost:8000/api/docs
# - 健康检查: http://localhost:8000/api/v1/health
```

### 4. 测试 API（真实模型推理）

```bash
# 启动 API 服务后，在另一个终端运行
python tests/mindnlp/ocr/test_api_real_model.py

# 或使用 curl 测试
curl -X POST "http://localhost:8000/api/v1/ocr/predict" \
  -F "file=@test.png" \
  -F "output_format=text" \
  -F "language=auto" \
  -F "task_type=general"
```

## 测试说明

### Mock 测试（默认运行）
- **测试数量**: 23 个
- **运行时间**: < 1 秒
- **用途**: 验证 API 实现的正确性
- **优点**: 快速、无需下载模型、适合 CI/CD
- **测试内容**:
  - 模型和 Processor 创建
  - 图像处理流程
  - 文本生成逻辑
  - 批量处理
  - 特殊 token 处理
  - 错误处理
  - 设备兼容性

### 真实模型测试（需要 --run-real-model）
- **测试数量**: 1 个
- **运行时间**: 首次约 5-10 分钟（下载模型），后续约 10-30 秒
- **模型大小**: 约 4 GB
- **用途**: 验证与 transformers 库的完整兼容性
- **优点**: 完整的端到端测试
- **测试内容**:
  - 真实模型加载
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
