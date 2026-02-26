"""
模型转换示例

演示如何将完整模型转换为量化模型。
"""

import numpy as np

try:
    import mindspore as ms
    from mindspore import Tensor, context, nn
    
    # 设置运行模式
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    
    # 导入量化模块
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from msbnb import (
        convert_to_quantized_model,
        replace_linear_layers,
        quantize_model_weights,
        get_model_size,
        compare_model_sizes,
        print_quantization_summary,
        Int8Config,
        Int4Config,
        Linear8bit,
        Linear4bit
    )
    
    print("=" * 60)
    print("模型转换示例")
    print("=" * 60)
    
    # ========== 定义示例模型 ==========
    class SimpleTransformer(nn.Cell):
        """简单的 Transformer 模型"""
        def __init__(self, hidden_size=768, intermediate_size=3072, num_layers=2):
            super().__init__()
            self.layers = nn.CellList([
                nn.SequentialCell([
                    nn.Dense(hidden_size, intermediate_size),
                    nn.GELU(),
                    nn.Dense(intermediate_size, hidden_size),
                ])
                for _ in range(num_layers)
            ])
            self.classifier = nn.Dense(hidden_size, 10)
        
        def construct(self, x):
            for layer in self.layers:
                x = x + layer(x)
            return self.classifier(x)
    
    # ========== 示例 1: 使用配置转换整个模型 ==========
    print("\n[1] 使用配置转换整个模型")
    print("-" * 60)
    
    # 创建原始模型
    model_fp = SimpleTransformer(hidden_size=768, intermediate_size=3072, num_layers=2)
    print(f"原始模型创建完成")
    
    # 获取原始模型大小
    size_fp = get_model_size(model_fp)
    print(f"原始模型大小: {size_fp['total_size_mb']:.2f} MB")
    print(f"总参数量: {size_fp['total_params']:,}")
    
    # 转换为 INT8 量化模型
    print("\n转换为 INT8 量化模型...")
    config = Int8Config(
        symmetric=True,
        per_channel=True,
        has_fp16_weights=False  # 立即量化
    )
    
    model_int8 = SimpleTransformer(hidden_size=768, intermediate_size=3072, num_layers=2)
    model_int8 = convert_to_quantized_model(
        model_int8,
        config=config,
        modules_to_not_convert=["classifier"]  # 不转换分类头
    )
    
    # 比较模型大小
    print("\n模型大小对比:")
    size_int8 = get_model_size(model_int8)
    print(f"INT8 模型大小: {size_int8['total_size_mb']:.2f} MB")
    print(f"显存节省: {(1 - size_int8['total_size_mb']/size_fp['total_size_mb'])*100:.1f}%")
    
    # 测试前向传播
    x = Tensor(np.random.randn(32, 768).astype(np.float32))
    out_fp = model_fp(x)
    out_int8 = model_int8(x)
    
    error = np.abs(out_fp.asnumpy() - out_int8.asnumpy()).mean()
    print(f"\n前向传播误差: {error:.6f}")
    print(f"相对误差: {error / np.abs(out_fp.asnumpy()).mean() * 100:.2f}%")
    
    # ========== 示例 2: 转换为 INT4 量化模型 ==========
    print("\n[2] 转换为 INT4 量化模型")
    print("-" * 60)
    
    model_int4 = SimpleTransformer(hidden_size=768, intermediate_size=3072, num_layers=2)
    config = Int4Config(
        group_size=128,
        compress_statistics=True
    )
    
    model_int4 = convert_to_quantized_model(
        model_int4,
        config=config,
        modules_to_not_convert=["classifier"]
    )
    
    # 比较模型大小
    print("\n模型大小对比:")
    size_int4 = get_model_size(model_int4)
    print(f"FP32 模型: {size_fp['total_size_mb']:.2f} MB")
    print(f"INT8 模型: {size_int8['total_size_mb']:.2f} MB (节省 {(1 - size_int8['total_size_mb']/size_fp['total_size_mb'])*100:.1f}%)")
    print(f"INT4 模型: {size_int4['total_size_mb']:.2f} MB (节省 {(1 - size_int4['total_size_mb']/size_fp['total_size_mb'])*100:.1f}%)")
    
    # 测试前向传播
    out_int4 = model_int4(x)
    error = np.abs(out_fp.asnumpy() - out_int4.asnumpy()).mean()
    print(f"\n前向传播误差: {error:.6f}")
    print(f"相对误差: {error / np.abs(out_fp.asnumpy()).mean() * 100:.2f}%")
    
    # ========== 示例 3: 使用 replace_linear_layers ==========
    print("\n[3] 使用 replace_linear_layers 替换特定层")
    print("-" * 60)
    
    model_custom = SimpleTransformer(hidden_size=768, intermediate_size=3072, num_layers=2)
    
    # 只替换中间层（intermediate_size=3072）
    print("\n只替换 768->3072 的层为 INT8...")
    model_custom = replace_linear_layers(
        model_custom,
        Linear8bit,
        in_features=768,
        out_features=3072,
        has_fp16_weights=False,
        symmetric=True
    )
    
    # 打印量化摘要
    print("\n")
    print_quantization_summary(model_custom)
    
    # ========== 示例 4: 延迟量化 ==========
    print("\n[4] 延迟量化（训练后量化）")
    print("-" * 60)
    
    model_delayed = SimpleTransformer(hidden_size=768, intermediate_size=3072, num_layers=2)
    
    # 转换但保持 FP16 权重
    config = Int8Config(
        symmetric=True,
        per_channel=True,
        has_fp16_weights=True  # 保持 FP16
    )
    
    model_delayed = convert_to_quantized_model(
        model_delayed,
        config=config,
        modules_to_not_convert=["classifier"]
    )
    
    print("模型已转换，权重保持 FP16（训练模式）")
    
    # 模拟训练...
    print("（模拟训练过程...）")
    
    # 训练后量化权重
    print("\n训练完成，量化权重...")
    model_delayed = quantize_model_weights(model_delayed, num_bits=8)
    
    # ========== 示例 5: 模型大小比较 ==========
    print("\n[5] 详细的模型大小比较")
    print("-" * 60)
    
    comparison = compare_model_sizes(model_fp, model_int8)
    print(f"FP32 模型大小: {comparison['fp_size_mb']:.2f} MB")
    print(f"INT8 模型大小: {comparison['quant_size_mb']:.2f} MB")
    print(f"显存节省: {comparison['memory_saved_mb']:.2f} MB ({comparison['memory_saved_percent']:.1f}%)")
    print(f"压缩比: {comparison['compression_ratio']:.2f}x")
    
    print("示例运行完成！")

except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装 MindSpore 并正确设置路径")
except Exception as e:
    print(f"运行错误: {e}")
    import traceback
    traceback.print_exc()
