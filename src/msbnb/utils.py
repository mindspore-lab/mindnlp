"""
Quantization Utility Functions

Provides utility functions for quantization operations.
"""

import numpy as np


def quantize_weight_int4_pergroup(weight, group_size=128, compress_statistics=False):
    """
    Per-group INT4 weight quantization
    
    Args:
        weight (np.ndarray): FP16/FP32 weight with shape [out_features, in_features]
        group_size (int): Size of each group. Default: 128
        compress_statistics (bool): Whether to use double quantization. Default: False
    
    Returns:
        tuple: (weight_int4, scale, offset)
            - weight_int4: Quantized weight in qint4x2 format [out_features, in_features//2]
            - scale: Dequantization scale
                - If compress_statistics=False: [num_groups, out_features] FP16
                - If compress_statistics=True: ([num_groups, out_features] INT8, [out_features] FP16)
            - offset: Dequantization offset [num_groups, out_features] FP16
    
    Examples:
        >>> weight = np.random.randn(3072, 768).astype(np.float16)
        >>> weight_int4, scale, offset = quantize_weight_int4_pergroup(weight, group_size=128)
    """
    out_features, in_features = weight.shape
    num_groups = (in_features + group_size - 1) // group_size
    
    # Pad to multiple of group_size
    pad_size = num_groups * group_size - in_features
    if pad_size > 0:
        weight = np.pad(weight, ((0, 0), (0, pad_size)), mode='constant')
    
    # Reshape to [out_features, num_groups, group_size]
    weight_grouped = weight.reshape(out_features, num_groups, group_size)
    
    # Compute absmax for each group (symmetric quantization)
    absmax = np.abs(weight_grouped).max(axis=2)  # [out_features, num_groups]
    
    # Compute scale: INT4 range [-7, 7] (reserve -8 for special values)
    scale = absmax / 7.0
    scale = np.where(scale == 0, 1.0, scale)  # Avoid division by zero
    
    # Quantize to INT4
    weight_int4_unpacked = np.clip(
        np.round(weight_grouped / scale[:, :, np.newaxis]),
        -7, 7
    ).astype(np.int8)
    
    # Pack to qint4x2 format
    weight_int4 = pack_int4_to_qint4x2(
        weight_int4_unpacked.reshape(out_features, -1)
    )
    
    # Transpose scale: [num_groups, out_features]
    scale = scale.T
    
    # Double quantization (optional)
    if compress_statistics:
        # Quantize scale to INT8
        scale_absmax = np.abs(scale).max(axis=0, keepdims=True)  # [1, out_features]
        scale_scale = scale_absmax / 127.0
        scale_int8 = np.clip(
            np.round(scale / scale_scale),
            -127, 127
        ).astype(np.int8)
        
        scale = (scale_int8, scale_scale.squeeze())
    
    # offset is 0 (symmetric quantization)
    offset = np.zeros((num_groups, out_features), dtype=np.float16)
    
    return weight_int4, scale, offset


def pack_int4_to_qint4x2(weight_int8):
    """
    Pack INT4 values represented as INT8 into qint4x2 format (pack two INT4 values into one byte)
    
    Args:
        weight_int8 (np.ndarray): INT8 array with shape [out_features, in_features], value range [-7, 7]
    
    Returns:
        np.ndarray: qint4x2 array with shape [out_features, in_features // 2]
    
    Description:
        qint4x2 format: Two INT4 values packed into one INT8 byte
        [high_4bit | low_4bit]
    
    Examples:
        >>> weight_int8 = np.random.randint(-7, 7, (1024, 768), dtype=np.int8)
        >>> weight_packed = pack_int4_to_qint4x2(weight_int8)
        >>> print(weight_packed.shape)  # (1024, 384)
    """
    out_features, in_features = weight_int8.shape
    assert in_features % 2 == 0, "in_features must be even"
    
    # Separate even and odd columns
    even = weight_int8[:, 0::2]  # Low 4 bits
    odd = weight_int8[:, 1::2]   # High 4 bits
    
    # Pack: (odd << 4) | (even & 0x0F)
    # Note: INT4 range [-7, 7], need to convert to unsigned representation
    even_unsigned = (even + 8) & 0x0F  # [-7, 7] -> [1, 15]
    odd_unsigned = (odd + 8) & 0x0F
    
    packed = (odd_unsigned << 4) | even_unsigned
    
    return packed.astype(np.uint8)


def unpack_qint4x2_to_int8(weight_qint4x2):
    """
    Unpack qint4x2 format to INT8
    
    Args:
        weight_qint4x2 (np.ndarray): qint4x2 array with shape [out_features, in_features // 2]
    
    Returns:
        np.ndarray: INT8 array with shape [out_features, in_features], value range [-7, 7]
    
    Examples:
        >>> weight_packed = np.random.randint(0, 255, (1024, 384), dtype=np.uint8)
        >>> weight_int8 = unpack_qint4x2_to_int8(weight_packed)
        >>> print(weight_int8.shape)  # (1024, 768)
    """
    out_features, packed_size = weight_qint4x2.shape
    
    # Unpack
    even_unsigned = weight_qint4x2 & 0x0F
    odd_unsigned = (weight_qint4x2 >> 4) & 0x0F
    
    # Convert back to signed
    even = even_unsigned.astype(np.int8) - 8
    odd = odd_unsigned.astype(np.int8) - 8
    
    # Interleave and merge
    weight_int8 = np.empty((out_features, packed_size * 2), dtype=np.int8)
    weight_int8[:, 0::2] = even
    weight_int8[:, 1::2] = odd
    
    return weight_int8


def compute_scale_offset(weight, num_bits=8, symmetric=True, per_channel=True, channel_axis=0):
    """
    Compute quantization scale and offset
    
    Args:
        weight (np.ndarray): Weight tensor
        num_bits (int): Quantization bit width. Default: 8
        symmetric (bool): Whether to use symmetric quantization. Default: True
        per_channel (bool): Whether to use per-channel quantization. Default: True
        channel_axis (int): Channel axis. Default: 0
    
    Returns:
        tuple: (scale, offset)
            - scale: Quantization scale
            - offset: Quantization offset (None for symmetric quantization)
    
    Examples:
        >>> weight = np.random.randn(3072, 768).astype(np.float16)
        >>> scale, offset = compute_scale_offset(weight, num_bits=8, symmetric=True)
    """
    if per_channel:
        # Compute min/max for each channel
        axes = tuple(i for i in range(len(weight.shape)) if i != channel_axis)
        w_min = weight.min(axis=axes, keepdims=True)
        w_max = weight.max(axis=axes, keepdims=True)
    else:
        w_min = weight.min()
        w_max = weight.max()
    
    if symmetric:
        # Symmetric quantization: [-2^(n-1)+1, 2^(n-1)-1]
        quant_max = 2 ** (num_bits - 1) - 1
        absmax = np.maximum(np.abs(w_min), np.abs(w_max))
        scale = absmax / quant_max
        offset = None
    else:
        # Asymmetric quantization: [0, 2^n-1]
        quant_range = 2 ** num_bits - 1
        scale = (w_max - w_min) / quant_range
        offset = -w_min / scale
    
    # Avoid division by zero
    scale = np.where(scale == 0, 1.0, scale)
    
    return scale, offset
