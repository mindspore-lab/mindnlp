"""
函数式 API 示例

演示如何使用函数式接口进行量化操作。
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from msbnb import (
    quantize_8bit,
    dequantize_8bit,
    quantize_4bit,
    dequantize_4bit,
    quantize_tensor,
    dequantize_tensor,
    estimate_quantization_error,
    get_quantization_info
)

print("=" * 60)
print("函数式 API 示例")
print("=" * 60)

# ========== 示例 1: INT8 量化和反量化 ==========
print("\n[1] INT8 量化和反量化")
print("-" * 60)

# 创建测试权重
weight_fp = np.random.randn(3072, 768).astype(np.float32)
print(f"原始权重形状: {weight_fp.shape}")
print(f"原始权重范围: [{weight_fp.min():.4f}, {weight_fp.max():.4f}]")

# 对称量化
print("\n对称量化:")
weight_int8, scale, offset = quantize_8bit(
    weight_fp,
    symmetric=True,
    per_channel=True
)
print(f"量化权重类型: {weight_int8.dtype}")
print(f"量化权重范围: [{weight_int8.min()}, {weight_int8.max()}]")
print(f"Scale 形状: {scale.shape}")
print(f"Scale 范围: [{scale.min():.6f}, {scale.max():.6f}]")

# 反量化
weight_dequant = dequantize_8bit(weight_int8, scale, offset)
print(f"\n反量化权重形状: {weight_dequant.shape}")

# 计算误差
error = np.abs(weight_fp - weight_dequant).mean()
print(f"平均误差: {error:.6f}")
print(f"相对误差: {error / np.abs(weight_fp).mean() * 100:.2f}%")

# 非对称量化
print("\n非对称量化:")
weight_int8_asym, scale_asym, offset_asym = quantize_8bit(
    weight_fp,
    symmetric=False,
    per_channel=True
)
print(f"Offset 形状: {offset_asym.shape}")
print(f"Offset 范围: [{offset_asym.min():.2f}, {offset_asym.max():.2f}]")

# ========== 示例 2: INT4 量化和反量化 ==========
print("\n[2] INT4 量化和反量化")
print("-" * 60)

# INT4 量化
weight_int4, scale_4bit, offset_4bit = quantize_4bit(
    weight_fp,
    group_size=128,
    compress_statistics=False
)
print(f"量化权重形状: {weight_int4.shape}")
print(f"量化权重类型: {weight_int4.dtype}")
print(f"Scale 形状: {scale_4bit.shape}")

# 反量化
weight_dequant_4bit = dequantize_4bit(
    weight_int4,
    scale_4bit,
    offset_4bit,
    group_size=128
)
print(f"反量化权重形状: {weight_dequant_4bit.shape}")

# 计算误差
error_4bit = np.abs(weight_fp - weight_dequant_4bit).mean()
print(f"平均误差: {error_4bit:.6f}")
print(f"相对误差: {error_4bit / np.abs(weight_fp).mean() * 100:.2f}%")

# 双重量化
print("\n双重量化:")
weight_int4_comp, scale_4bit_comp, offset_4bit_comp = quantize_4bit(
    weight_fp,
    group_size=128,
    compress_statistics=True
)
print(f"Scale 类型: {type(scale_4bit_comp)}")
if isinstance(scale_4bit_comp, tuple):
    print(f"  - Scale INT8 形状: {scale_4bit_comp[0].shape}")
    print(f"  - Scale Scale 形状: {scale_4bit_comp[1].shape}")

# ========== 示例 3: 通用量化接口 ==========
print("\n[3] 通用量化接口")
print("-" * 60)

# INT8 量化
print("INT8 量化:")
tensor_int8, scale_8, offset_8 = quantize_tensor(
    weight_fp,
    num_bits=8,
    symmetric=True,
    per_channel=True
)
print(f"量化张量形状: {tensor_int8.shape}, 类型: {tensor_int8.dtype}")

# INT4 量化
print("\nINT4 量化:")
tensor_int4, scale_4, offset_4 = quantize_tensor(
    weight_fp,
    num_bits=4,
    symmetric=True,
    per_channel=False
)
print(f"量化张量形状: {tensor_int4.shape}, 类型: {tensor_int4.dtype}")

# ========== 示例 4: 量化误差估计 ==========
print("\n[4] 量化误差估计")
print("-" * 60)

# INT8 误差
print("INT8 量化误差:")
error_stats_8 = estimate_quantization_error(
    weight_fp,
    weight_int8,
    scale,
    offset,
    num_bits=8
)
print(f"  平均误差: {error_stats_8['mean_error']:.6f}")
print(f"  最大误差: {error_stats_8['max_error']:.6f}")
print(f"  相对误差: {error_stats_8['relative_error']:.2f}%")
print(f"  信噪比: {error_stats_8['snr']:.2f} dB")

# INT4 误差
print("\nINT4 量化误差:")
error_stats_4 = estimate_quantization_error(
    weight_fp,
    weight_int4,
    scale_4bit,
    offset_4bit,
    num_bits=4
)
print(f"  平均误差: {error_stats_4['mean_error']:.6f}")
print(f"  最大误差: {error_stats_4['max_error']:.6f}")
print(f"  相对误差: {error_stats_4['relative_error']:.2f}%")
print(f"  信噪比: {error_stats_4['snr']:.2f} dB")

# ========== 示例 5: 量化信息查询 ==========
print("\n[5] 量化信息查询")
print("-" * 60)

# INT8 信息
print("INT8 量化信息:")
info_8 = get_quantization_info(
    weight_fp,
    num_bits=8,
    symmetric=True,
    per_channel=True
)
print(f"  Scale 范围: [{info_8['scale_range'][0]:.6f}, {info_8['scale_range'][1]:.6f}]")
print(f"  Scale 平均: {info_8['scale_mean']:.6f}")
print(f"  权重范围: [{info_8['weight_range'][0]:.4f}, {info_8['weight_range'][1]:.4f}]")
print(f"  估计压缩比: {info_8['estimated_compression']:.2f}x")

# INT4 信息
print("\nINT4 量化信息:")
info_4 = get_quantization_info(
    weight_fp,
    num_bits=4,
    symmetric=True,
    per_channel=True
)
print(f"  Scale 范围: [{info_4['scale_range'][0]:.6f}, {info_4['scale_range'][1]:.6f}]")
print(f"  Scale 平均: {info_4['scale_mean']:.6f}")
print(f"  权重范围: [{info_4['weight_range'][0]:.4f}, {info_4['weight_range'][1]:.4f}]")
print(f"  估计压缩比: {info_4['estimated_compression']:.2f}x")

# ========== 示例 6: 不同量化策略对比 ==========
print("\n[6] 不同量化策略对比")
print("-" * 60)

strategies = [
    ("INT8 对称 Per-channel", 8, True, True),
    ("INT8 非对称 Per-channel", 8, False, True),
    ("INT8 对称 Per-layer", 8, True, False),
    ("INT4 Per-group", 4, True, False),
]

print(f"{'策略':<30} {'相对误差':<12} {'SNR (dB)':<10} {'压缩比':<10}")
print("-" * 65)

for name, bits, symmetric, per_channel in strategies:
    # 量化
    if bits == 8:
        q_weight, q_scale, q_offset = quantize_tensor(
            weight_fp,
            num_bits=bits,
            symmetric=symmetric,
            per_channel=per_channel
        )
        # 估计误差
        error_stats = estimate_quantization_error(
            weight_fp, q_weight, q_scale, q_offset, num_bits=bits
        )
    else:
        q_weight, q_scale, q_offset = quantize_4bit(weight_fp, group_size=128)
        error_stats = estimate_quantization_error(
            weight_fp, q_weight, q_scale, q_offset, num_bits=bits
        )
    
    # 获取信息
    info = get_quantization_info(
        weight_fp,
        num_bits=bits,
        symmetric=symmetric,
        per_channel=per_channel
    )
    
    print(f"{name:<30} {error_stats['relative_error']:<12.2f}% {error_stats['snr']:<10.2f} {info['estimated_compression']:<10.2f}x")

print("示例运行完成！")