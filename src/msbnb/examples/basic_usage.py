"""
MindSpore BitsAndBytes 基础使用示例

演示 INT8 和 INT4 量化的基本用法。
"""

import numpy as np

try:
    import mindspore as ms
    from mindspore import Tensor, context
    import mindspore.nn as nn
    
    # 设置运行模式
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    
    # 导入量化模块
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from msbnb import Linear8bit, Linear4bit, Int8Config, Int4Config
    
    print("=" * 60)
    print("MindSpore BitsAndBytes 基础使用示例")
    print("=" * 60)
    
    # ========== INT8 量化示例 ==========
    print("\n[1] INT8 量化示例")
    print("-" * 60)
    
    # 创建 INT8 量化层
    layer_int8 = Linear8bit(
        in_features=768,
        out_features=3072,
        bias=True,
        has_fp16_weights=True,
        symmetric=True,
        per_channel=True
    )
    
    print(f"创建 Linear8bit 层: {layer_int8.in_features} -> {layer_int8.out_features}")
    
    # 创建输入
    x = Tensor(np.random.randn(32, 768).astype(np.float32))
    print(f"输入形状: {x.shape}")
    
    # 训练模式（FP16 权重）
    print("\n训练模式（FP16 权重）:")
    out_fp16 = layer_int8(x)
    print(f"  输出形状: {out_fp16.shape}")
    print(f"  权重类型: {layer_int8.weight.dtype}")
    
    # 量化权重
    print("\n量化权重...")
    layer_int8.quantize_weights()
    
    # 推理模式（INT8 权重）
    print("\n推理模式（INT8 权重）:")
    out_int8 = layer_int8(x)
    print(f"  输出形状: {out_int8.shape}")
    print(f"  权重类型: {layer_int8.weight.dtype}")
    
    # 计算精度损失
    error = np.abs(out_fp16.asnumpy() - out_int8.asnumpy()).mean()
    print(f"  平均误差: {error:.6f}")
    print(f"  相对误差: {error / np.abs(out_fp16.asnumpy()).mean() * 100:.2f}%")
    
    # ========== INT4 量化示例 ==========
    print("\n[2] INT4 量化示例")
    print("-" * 60)
    
    # 创建标准 Dense 层
    fp16_layer = nn.Dense(768, 3072, has_bias=True)
    print(f"创建标准 Dense 层: {fp16_layer.in_channels} -> {fp16_layer.out_channels}")
    
    # 转换为 INT4 量化层
    print("\n转换为 INT4 量化层...")
    layer_int4 = Linear4bit.from_linear(
        fp16_layer,
        group_size=128,
        compress_statistics=True
    )
    
    # 前向传播
    print("\n前向传播:")
    out_fp16 = fp16_layer(x)
    out_int4 = layer_int4(x)
    print(f"  FP16 输出形状: {out_fp16.shape}")
    print(f"  INT4 输出形状: {out_int4.shape}")
    
    # 计算精度损失
    error = np.abs(out_fp16.asnumpy() - out_int4.asnumpy()).mean()
    print(f"  平均误差: {error:.6f}")
    print(f"  相对误差: {error / np.abs(out_fp16.asnumpy()).mean() * 100:.2f}%")
    
    # ========== 显存占用对比 ==========
    print("\n[3] 显存占用对比")
    print("-" * 60)
    
    def get_param_size(layer):
        """计算参数大小（MB）"""
        total_size = 0
        for param in layer.get_parameters():
            param_size = param.data.nbytes
            total_size += param_size
        return total_size / (1024 * 1024)
    
    fp16_size = get_param_size(fp16_layer)
    int8_size = get_param_size(layer_int8)
    int4_size = get_param_size(layer_int4)
    
    print(f"FP16 层参数大小: {fp16_size:.2f} MB")
    print(f"INT8 层参数大小: {int8_size:.2f} MB (节省 {(1 - int8_size/fp16_size)*100:.1f}%)")
    print(f"INT4 层参数大小: {int4_size:.2f} MB (节省 {(1 - int4_size/fp16_size)*100:.1f}%)")
    
    # ========== 配置管理示例 ==========
    print("\n[4] 配置管理示例")
    print("-" * 60)
    
    # INT8 配置
    int8_config = Int8Config(
        symmetric=True,
        per_channel=True,
        threshold=6.0,
        has_fp16_weights=True
    )
    print(f"INT8 配置: {int8_config}")
    
    # INT4 配置
    int4_config = Int4Config(
        group_size=128,
        compress_statistics=True,
        quant_type='int4'
    )
    print(f"INT4 配置: {int4_config}")
    
    print("示例运行完成！")

except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装 MindSpore 并正确设置路径")
except Exception as e:
    print(f"运行错误: {e}")
    import traceback
    traceback.print_exc()
