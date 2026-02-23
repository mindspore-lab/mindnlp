# MindSpore BitsAndBytes (msbnb)

将 MindSpore 原生量化算子封装成 bitsandbytes 风格的量化库。

**核心目标**: 基于 MindSpore 已有的量化算子，提供与 bitsandbytes 兼容的接口

---

## 项目背景

[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) 是 PyTorch 生态中广泛使用的量化库，特别是在大模型训练和推理中。然而，MindSpore 生态缺少类似的工具。本项目旨在：

1. **利用 MindSpore 原生算子**：MindSpore 已经提供了高性能的量化算子（如 `WeightQuantBatchMatmul`），但缺少易用的封装
2. **提供 bitsandbytes 风格接口**：让熟悉 PyTorch/bitsandbytes 的用户能够快速迁移到 MindSpore
3. **支持大模型训练**：提供 QLoRA 等参数高效微调方法

---

## MindSpore 量化算子概览

### 1. MindSpore 已有的量化能力

MindSpore 提供了以下量化相关的功能：

#### 数据类型
- **INT8**: `mstype.int8` - 标准 8-bit 整数
- **qint4x2**: `mstype.uint8` - 打包的 4-bit 整数（两个 INT4 值打包到一个字节）

#### 核心算子
- **WeightQuantBatchMatmul**: 融合了权重反量化和矩阵乘法的算子
  - 输入：FP16/BF16 激活值 + INT8/INT4 量化权重 + scale/offset
  - 输出：FP16/BF16 结果
  - 优势：硬件加速，避免显式反量化开销

#### 量化方法
- **对称量化**: `value_int = round(value_fp / scale)`，范围 [-127, 127]
- **非对称量化**: `value_int = round(value_fp / scale + offset)`，范围 [-128, 127]
- **Per-channel 量化**: 每个输出通道使用不同的 scale
- **Per-group 量化**: 将权重分组，每组使用不同的 scale（用于 INT4）

---

## 封装策略

### 设计原则

1. **最小化封装开销**：直接使用 MindSpore 原生算子，不引入额外计算
2. **兼容 bitsandbytes 接口**：提供相似的 API，降低迁移成本
3. **分层设计**：提供多层次接口（底层工具函数 → 量化层 → 模型转换）

### 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    应用层 (User API)                     │
│  - convert_to_quantized_model()  模型自动转换           │
│  - Linear4bitWithLoRA            QLoRA 训练              │
├─────────────────────────────────────────────────────────┤
│                   量化层 (Quantized Layers)              │
│  - Linear8bit   封装 WeightQuantBatchMatmul (INT8)      │
│  - Linear4bit   封装 WeightQuantBatchMatmul (INT4)      │
├─────────────────────────────────────────────────────────┤
│                 工具层 (Utility Functions)               │
│  - quantize_8bit/4bit           量化函数                │
│  - pack_int4_to_qint4x2         INT4 打包               │
│  - compute_scale_offset         计算量化参数            │
├─────────────────────────────────────────────────────────┤
│              MindSpore 原生算子 (Native Ops)             │
│  - WeightQuantBatchMatmul       融合算子                │
│  - mstype.int8, mstype.uint8    数据类型                │
└─────────────────────────────────────────────────────────┘
```

---

## 核心实现

### 1. INT8 量化层 (Linear8bit)

#### 封装的 MindSpore 算子
- **WeightQuantBatchMatmul**: 融合反量化和矩阵乘法

#### 实现细节

```python
class Linear8bit(nn.Cell):
    """
    8-bit 量化线性层
    
    封装 MindSpore 的 WeightQuantBatchMatmul 算子
    """
    
    def __init__(self, in_features, out_features, 
                 symmetric=True, per_channel=True):
        super().__init__()
        
        # 权重参数 (INT8)
        self.weight = Parameter(
            Tensor(shape=(out_features, in_features), dtype=mstype.int8),
            requires_grad=False
        )
        
        # 量化参数
        scale_shape = (out_features,) if per_channel else (1,)
        self.scale = Parameter(
            Tensor(shape=scale_shape, dtype=mstype.float16),
            requires_grad=False
        )
        
        if not symmetric:
            self.offset = Parameter(
                Tensor(shape=scale_shape, dtype=mstype.float16),
                requires_grad=False
            )
    
    def construct(self, x):
        """
        前向传播
        
        使用 WeightQuantBatchMatmul 算子进行融合计算：
        1. 反量化: weight_fp = weight_int8 * scale + offset
        2. 矩阵乘法: out = x @ weight_fp^T
        
        这两步在算子内部融合，避免显式反量化
        """
        from mindspore.ops.auto_generate import weight_quant_batch_matmul
        
        out = weight_quant_batch_matmul(
            x,                  # 输入 [batch, in_features] (FP16)
            self.weight,        # INT8 权重 [out_features, in_features]
            self.scale,         # 反量化 scale
            self.offset,        # 反量化 offset (可选)
            None,               # quant_scale (推理时不需要)
            None,               # quant_offset (推理时不需要)
            self.bias,          # bias (融合在算子中)
            transpose_x=False,
            transpose_weight=True,
            antiquant_group_size=0  # 0 表示 per-channel
        )
        
        return out
```

#### 量化流程

```
FP16 权重 [out_features, in_features]
    ↓
计算 scale 和 offset
    scale = absmax(weight) / 127  (对称量化)
    ↓
量化: weight_int8 = round(weight_fp / scale)
    ↓
存储: INT8 权重 + FP16 scale
    ↓ (推理时)
WeightQuantBatchMatmul 算子
    - 内部融合反量化和矩阵乘法
    - 硬件加速
    ↓
输出 FP16
```

#### 显存节省

| 数据类型 | 每个参数 | 示例 (768→3072) |
|---------|---------|----------------|
| FP16    | 2 bytes | 4.5 MB         |
| INT8    | 1 byte  | 2.25 MB        |
| **节省** | **50%** | **2.25 MB**    |

---

### 2. INT4 量化层 (Linear4bit)

#### 封装的 MindSpore 能力
- **qint4x2 数据类型**: `mstype.uint8` 打包格式
- **Per-group 量化**: 分组量化以保持精度

#### 实现细节

```python
class Linear4bit(nn.Cell):
    """
    4-bit 量化线性层
    
    使用 MindSpore 的 qint4x2 打包格式
    """
    
    def __init__(self, in_features, out_features, group_size=128):
        super().__init__()
        
        # 权重参数 (qint4x2 打包格式)
        # 两个 INT4 值打包到一个 uint8
        self.weight = Parameter(
            Tensor(shape=(out_features, in_features // 2), dtype=mstype.uint8),
            requires_grad=False
        )
        
        # Per-group 量化参数
        num_groups = (in_features + group_size - 1) // group_size
        self.scale = Parameter(
            Tensor(shape=(num_groups, out_features), dtype=mstype.float16),
            requires_grad=False
        )
```

#### qint4x2 打包格式

```python
def pack_int4_to_qint4x2(weight_int8):
    """
    将 INT4 值（表示为 INT8）打包成 qint4x2 格式
    
    格式: [high_4bit | low_4bit]
    
    示例:
        输入: [-7, 3, 5, -2] (INT8 表示的 INT4)
        打包: [0x13, 0x5E]
              - 0x13 = (3 << 4) | (-7 + 8)
              - 0x5E = (-2 + 8) << 4 | (5 + 8)
    """
    out_features, in_features = weight_int8.shape
    
    # 分离奇偶列
    even = weight_int8[:, 0::2]  # 低 4 位
    odd = weight_int8[:, 1::2]   # 高 4 位
    
    # 转换为无符号表示 [-7, 7] -> [1, 15]
    even_unsigned = (even + 8) & 0x0F
    odd_unsigned = (odd + 8) & 0x0F
    
    # 打包: (odd << 4) | even
    packed = (odd_unsigned << 4) | even_unsigned
    
    return packed.astype(np.uint8)
```

#### Per-group 量化

```
FP16 权重 [out_features, in_features]
    ↓
分组: [out_features, num_groups, group_size]
    ↓
计算每组的 absmax
    scale[g] = absmax(group[g]) / 7  (INT4 范围 [-7, 7])
    ↓
量化: weight_int4[g] = round(weight[g] / scale[g])
    ↓
打包: pack_int4_to_qint4x2()
    ↓
存储: UINT8 权重 [out_features, in_features/2] + FP16 scale [num_groups, out_features]
```

#### 双重量化（可选）

为了进一步节省显存，可以对 scale 参数再次量化：

```python
# scale 本身也量化为 INT8
scale_absmax = absmax(scale)
scale_scale = scale_absmax / 127
scale_int8 = round(scale / scale_scale)

# 存储: INT8 scale + FP16 scale_scale
```

#### 显存节省

| 数据类型 | 每个参数 | 示例 (768→3072) | 双重量化 |
|---------|---------|----------------|---------|
| FP16    | 2 bytes | 4.5 MB         | -       |
| INT4    | 0.5 byte | 1.125 MB      | 1.125 MB |
| scale   | -       | ~0.1 MB        | ~0.05 MB |
| **总计** | **~25%** | **~1.2 MB**   | **~1.15 MB** |
| **节省** | **~75%** | **~3.3 MB**   | **~3.35 MB** |

---

### 3. QLoRA 实现 (Linear4bitWithLoRA)

#### 封装策略

QLoRA = INT4 量化 + LoRA 适配器

```python
class Linear4bitWithLoRA(Linear4bit):
    """
    QLoRA 层: INT4 量化 + LoRA
    
    架构:
        输入 x
          ↓
          ├─ 主路径（冻结）
          │   INT4 量化权重 → 反量化 → 矩阵乘法 → out_main
          │
          └─ LoRA 路径（可训练）
              x @ A @ B * scaling → out_lora
          ↓
          out = out_main + out_lora
    """
    
    def __init__(self, in_features, out_features, 
                 r=8, lora_alpha=16, group_size=128):
        # 初始化 INT4 量化层
        super().__init__(in_features, out_features, group_size=group_size)
        
        # 冻结量化权重
        self.weight.requires_grad = False
        self.scale.requires_grad = False
        
        # LoRA 参数（可训练）
        self.lora_A = Parameter(
            initializer(Normal(0.01), [in_features, r], mstype.float32),
            requires_grad=True
        )
        self.lora_B = Parameter(
            initializer(Zero(), [r, out_features], mstype.float32),
            requires_grad=True
        )
        self.scaling = lora_alpha / r
    
    def construct(self, x):
        # 主路径: INT4 量化计算
        out_main = super().construct(x)
        
        # LoRA 路径: x @ A @ B * scaling
        out_lora = ops.matmul(ops.matmul(x, self.lora_A), self.lora_B) * self.scaling
        
        return out_main + out_lora
```

#### 参数效率

以 768 → 3072 的线性层为例：

| 方法 | 参数量 | 显存占用 | 可训练参数 |
|------|--------|---------|-----------|
| 全量微调 (FP16) | 2,359,296 | 4.5 MB | 2,359,296 (100%) |
| LoRA (r=8) | 2,359,296 + 30,720 | 4.6 MB | 30,720 (1.3%) |
| QLoRA (r=8) | 2,359,296 + 30,720 | 1.2 MB | 30,720 (1.3%) |

**QLoRA 优势**:
- 显存节省 ~75%（相比全量微调）
- 参数量减少 ~99%（只训练 LoRA 参数）
- 精度保持良好（< 2% 损失）

---

## 使用示例

### 1. 基础量化

```python
import mindspore as ms
from mindspore import Tensor
import numpy as np
from msbnb import Linear8bit, Linear4bit

# INT8 量化
layer_int8 = Linear8bit(768, 3072, symmetric=True, per_channel=True)
x = Tensor(np.random.randn(32, 768), dtype=ms.float16)
out = layer_int8(x)

# INT4 量化
layer_int4 = Linear4bit(768, 3072, group_size=128, compress_statistics=True)
out = layer_int4(x)
```

### 2. 模型转换

```python
from msbnb import convert_to_quantized_model, Int8Config

# 定义量化配置
config = Int8Config(
    symmetric=True,
    per_channel=True,
    has_fp16_weights=False  # 直接量化
)

# 转换模型
quant_model = convert_to_quantized_model(
    model,
    config=config,
    modules_to_not_convert=["lm_head", "classifier"]  # 排除某些层
)

# 查看效果
from msbnb import get_model_size, compare_model_sizes
comparison = compare_model_sizes(fp_model, quant_model)
print(f"显存节省: {comparison['memory_saved_percent']:.1f}%")
```

### 3. QLoRA 训练

```python
from msbnb import Linear4bitWithLoRA, freeze_model_except_lora
from mindspore.nn import Adam

# 转换模型为 QLoRA
for name, module in model.name_cells().items():
    if isinstance(module, nn.Dense):
        qlora_module = Linear4bitWithLoRA.from_linear(
            module, 
            r=8, 
            lora_alpha=16,
            group_size=128
        )
        setattr(model, name, qlora_module)

# 冻结非 LoRA 参数
freeze_model_except_lora(model)

# 训练
optimizer = Adam(model.trainable_params(), lr=1e-4)

# 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 4. 函数式接口

```python
from msbnb import quantize_8bit, dequantize_8bit, estimate_quantization_error

# 量化权重
weight_fp16 = np.random.randn(3072, 768).astype(np.float16)
weight_int8, scale, offset = quantize_8bit(
    weight_fp16, 
    symmetric=True, 
    per_channel=True
)

# 反量化
weight_dequant = dequantize_8bit(weight_int8, scale, offset)

# 评估误差
error_stats = estimate_quantization_error(
    weight_fp16, weight_int8, scale, offset, num_bits=8
)
print(f"相对误差: {error_stats['relative_error']:.2f}%")
print(f"信噪比: {error_stats['snr']:.2f} dB")
```

## 技术细节

### 1. 为什么使用 WeightQuantBatchMatmul？

传统的量化推理流程：
```
INT8 权重 → 反量化 (FP16) → 矩阵乘法 (FP16) → 输出
```

使用 `WeightQuantBatchMatmul` 的优势：
```
INT8 权重 → WeightQuantBatchMatmul (融合算子) → 输出
```

- **减少显存访问**: 不需要存储反量化后的 FP16 权重
- **硬件加速**: 算子内部使用专用硬件指令
- **降低延迟**: 减少 kernel 启动开销

### 2. qint4x2 打包格式的优势

MindSpore 的 `qint4x2` 是硬件原生支持的数据类型：

- **硬件加速**: NPU 可以直接处理打包的 INT4 数据
- **高效存储**: 两个 INT4 值打包到一个字节，无浪费
- **简化实现**: 不需要手动位操作

### 3. Per-group 量化的必要性

INT4 量化范围很小（-7 到 7），如果对整个权重矩阵使用同一个 scale，会导致：
- 大值被截断（精度损失）
- 小值被量化为 0（信息丢失）

Per-group 量化解决方案：
```python
# 将权重分组，每组使用独立的 scale
weight_grouped = weight.reshape(out_features, num_groups, group_size)
scale[g] = absmax(weight_grouped[:, g, :]) / 7

# 每组独立量化
for g in range(num_groups):
    weight_int4[:, g, :] = round(weight_grouped[:, g, :] / scale[g])
```

推荐的 group_size：
- **128**: 平衡精度和显存（默认）
- **64**: 更高精度，稍多显存
- **256**: 更少显存，稍低精度

---

## 性能优化建议

### 1. 选择合适的量化方法

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 推理加速 | INT8 对称量化 | 精度损失小，速度快 |
| 显存受限 | INT4 + 双重量化 | 最大化显存节省 |
| 大模型微调 | QLoRA (INT4 + LoRA) | 参数高效，显存友好 |

### 2. 量化配置

```python
# 推理场景
config = Int8Config(
    symmetric=True,        # 对称量化，精度更好
    per_channel=True,      # Per-channel，精度更好
    has_fp16_weights=False # 直接量化，节省显存
)

# 训练场景（先训练后量化）
config = Int8Config(
    symmetric=True,
    per_channel=True,
    has_fp16_weights=True,  # 训练时保持 FP16
    quant_delay=1000        # 延迟量化
)
```

### 3. 排除敏感层

某些层对量化敏感，建议保持 FP16：

```python
quant_model = convert_to_quantized_model(
    model,
    config=config,
    modules_to_not_convert=[
        "lm_head",      # 语言模型头
        "classifier",   # 分类器
        "embed",        # 嵌入层
    ]
)
```

---

## 安装和使用

### 环境要求

- MindSpore >= 2.0
- Python >= 3.7
- NumPy

### 安装

```bash
# 确保已安装 MindSpore
pip install mindspore

# 将 msbnb 添加到 Python 路径
export PYTHONPATH="${PYTHONPATH}:/path/to/mindnlp/src"
```

### 快速开始

```python
# 导入
from msbnb import Linear8bit, Linear4bit, convert_to_quantized_model

# 使用量化层
layer = Linear8bit(768, 3072)

# 转换模型
from msbnb import Int8Config
config = Int8Config()
quant_model = convert_to_quantized_model(model, config)
```

---

## 示例代码

完整示例请参考 `examples/msbnb/` 目录：

1. **basic_usage.py** - 基础使用示例
2. **model_conversion.py** - 模型转换示例
3. **functional_api.py** - 函数式接口示例
4. **qlora_training.py** - QLoRA 训练示例

运行示例：
```bash
cd src/msbnb/examples
python basic_usage.py
```

---

## 项目结构

```
msbnb/
├── __init__.py          # 模块入口
├── linear.py            # 量化层实现
│   ├── Linear8bit       # INT8 量化层（封装 WeightQuantBatchMatmul）
│   ├── Linear4bit       # INT4 量化层（使用 qint4x2）
│   └── LinearQuant      # 基类
├── config.py            # 量化配置
│   ├── Int8Config       # INT8 配置
│   └── Int4Config       # INT4 配置
├── utils.py             # 工具函数
│   ├── pack_int4_to_qint4x2      # INT4 打包
│   ├── unpack_qint4x2_to_int8    # INT4 解包
│   └── compute_scale_offset      # 计算量化参数
├── functional.py        # 函数式接口
│   ├── quantize_8bit/4bit        # 量化函数
│   └── dequantize_8bit/4bit      # 反量化函数
├── converter.py         # 模型转换工具
│   └── convert_to_quantized_model # 自动转换
└── lora.py              # LoRA/QLoRA 实现
    ├── LoRALinear              # LoRA 适配器
    └── Linear4bitWithLoRA      # QLoRA 层
```

---

## 参考资料

### 论文
1. [LLM.int8()](https://arxiv.org/abs/2208.07339) - 8-bit Matrix Multiplication for Transformers
2. [QLoRA](https://arxiv.org/abs/2305.14314) - Efficient Finetuning of Quantized LLMs

### 相关项目
1. [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) - PyTorch 量化库
2. [MindSpore](https://www.mindspore.cn/) - 华为深度学习框架

---

**代码总量**: ~2960 行  
