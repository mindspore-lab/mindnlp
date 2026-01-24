# 更新日志

MindNLP 的所有重要变更都记录在此。

## 版本 0.6.x（当前版本）

**MindSpore**：>=2.7.1 | **Python**：3.10-3.11

### 亮点

- 通过补丁机制完全兼容 HuggingFace Transformers
- 完全兼容 HuggingFace Diffusers
- 支持最新模型架构（Qwen3、Llama3 等）
- 增强的 mindtorch 层，提供 PyTorch API 兼容性
- 改进的设备管理和异构计算支持

### 新功能

- 自动对 `transformers` 和 `diffusers` 库打补丁
- 模型加载支持 `ms_dtype` 参数
- 增强的 `device_map` 支持多设备推理
- 改进的张量序列化和检查点处理

## 版本 0.5.x

**MindSpore**：2.5.0-2.7.0 | **Python**：3.10-3.11

### 亮点

- 重大 API 重构以提供更好的 HuggingFace 兼容性
- 引入 mindtorch 兼容层
- 支持新模型系列（Gemma、Phi-3 等）

### 新功能

- `mindnlp.core` 模块提供 PyTorch 兼容的 API
- 增强的 AutoModel 类支持各种任务
- 改进的分词器支持
- PEFT/LoRA 集成用于参数高效微调

## 版本 0.4.x

**MindSpore**：2.2.x-2.5.0 | **Python**：3.9-3.11

### 亮点

- 扩展的模型支持
- 改进的训练稳定性
- 增强的 Trainer API

### 新功能

- 支持 Qwen2、Mistral、Mixtral 模型
- 增强的梯度检查点
- 改进的分布式训练支持
- 大模型更好的内存管理

## 版本 0.3.x

**MindSpore**：2.1.0-2.3.1 | **Python**：3.8-3.9

### 亮点

- 具有全面模型覆盖的稳定版本
- 改进的文档和示例

### 新功能

- 支持 Llama、Llama2 模型
- ChatGLM 系列支持（ChatGLM、ChatGLM2、ChatGLM3）
- 增强的数据集加载工具
- 改进的模型序列化

## 版本 0.2.x

**MindSpore**：>=2.1.0 | **Python**：3.8-3.9

### 亮点

- 重大架构改进
- 更好地与 HuggingFace API 对齐

### 新功能

- 重构的模型架构
- 改进的分词器实现
- 增强的训练引擎
- 更好的错误消息和调试

## 版本 0.1.x

**MindSpore**：1.8.1-2.0.0 | **Python**：3.7.5-3.9

### 亮点

- MindNLP 初始版本
- 核心 transformer 模型支持

### 新功能

- 基础 transformer 模型（BERT、GPT-2、T5 等）
- 分词器支持
- 数据集加载工具
- 基础训练循环实现

---

详细发布说明请参见 [GitHub Releases](https://github.com/mindspore-lab/mindnlp/releases)。
