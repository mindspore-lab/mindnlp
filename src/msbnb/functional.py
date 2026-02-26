"""
Functional Quantization Interface

Provides functional quantization and dequantization operations.
"""

import numpy as np
from .utils import compute_scale_offset, quantize_weight_int4_pergroup


def quantize_8bit(weight, symmetric=True, per_channel=True, channel_axis=0):
    """
    8-bit quantization function
    
    Args:
        weight (np.ndarray): Weight tensor
        symmetric (bool): Whether to use symmetric quantization. Default: True
        per_channel (bool): Whether to use per-channel quantization. Default: True
        channel_axis (int): Channel axis. Default: 0
    
    Returns:
        tuple: (weight_int8, scale, offset)
            - weight_int8: INT8 quantized weight
            - scale: Quantization scale
            - offset: Quantization offset (None for symmetric quantization)
    
    Examples:
        >>> weight = np.random.randn(3072, 768).astype(np.float16)
        >>> weight_int8, scale, offset = quantize_8bit(weight, symmetric=True)
    """
    # Compute quantization parameters
    scale, offset = compute_scale_offset(
        weight,
        num_bits=8,
        symmetric=symmetric,
        per_channel=per_channel,
        channel_axis=channel_axis
    )
    
    # Quantize
    if symmetric:
        # Symmetric quantization: [-127, 127]
        weight_int8 = np.clip(
            np.round(weight / scale),
            -127, 127
        ).astype(np.int8)
    else:
        # Asymmetric quantization: [-128, 127]
        weight_int8 = np.clip(
            np.round(weight / scale + offset),
            -128, 127
        ).astype(np.int8)
    
    return weight_int8, scale, offset


def dequantize_8bit(weight_int8, scale, offset=None):
    """
    8-bit dequantization function
    
    Args:
        weight_int8 (np.ndarray): INT8 quantized weight
        scale (np.ndarray): Quantization scale
        offset (np.ndarray, optional): Quantization offset. Default: None
    
    Returns:
        np.ndarray: Dequantized FP16/FP32 weight
    
    Examples:
        >>> weight_fp16 = dequantize_8bit(weight_int8, scale, offset)
    """
    # Dequantize
    weight_fp = weight_int8.astype(np.float32) * scale
    
    if offset is not None:
        weight_fp = weight_fp - offset * scale
    
    return weight_fp


def quantize_4bit(weight, group_size=128, compress_statistics=False):
    """
    4-bit quantization function
    
    Args:
        weight (np.ndarray): Weight tensor [out_features, in_features]
        group_size (int): Group size for per-group quantization. Default: 128
        compress_statistics (bool): Whether to compress statistics (double quantization). Default: False
    
    Returns:
        tuple: (weight_int4, scale, offset)
            - weight_int4: INT4 quantized weight (packed format)
            - scale: Quantization scale
            - offset: Quantization offset
    
    Examples:
        >>> weight = np.random.randn(3072, 768).astype(np.float16)
        >>> weight_int4, scale, offset = quantize_4bit(weight, group_size=128)
    """
    return quantize_weight_int4_pergroup(weight, group_size, compress_statistics)


def dequantize_4bit(weight_int4, scale, offset=None, group_size=128):
    """
    4-bit dequantization function
    
    Args:
        weight_int4 (np.ndarray): INT4 quantized weight (packed format)
        scale (np.ndarray): Quantization scale [num_groups, out_features]
        offset (np.ndarray, optional): Quantization offset. Default: None
        group_size (int): Group size for per-group quantization. Default: 128
    
    Returns:
        np.ndarray: Dequantized FP16/FP32 weight
    
    Examples:
        >>> weight_fp16 = dequantize_4bit(weight_int4, scale, offset, group_size=128)
    """
    from .utils import unpack_qint4x2_to_int8
    
    # Unpack
    weight_int8 = unpack_qint4x2_to_int8(weight_int4)
    
    # Per-group dequantization
    out_features, in_features = weight_int8.shape
    num_groups = scale.shape[0]
    
    # Reshape to [out_features, num_groups, group_size]
    weight_grouped = weight_int8[:, :num_groups * group_size].reshape(
        out_features, num_groups, group_size
    )
    
    # Dequantize: weight_fp16 = weight_int8 * scale
    # scale shape: [num_groups, out_features] -> [out_features, num_groups, 1]
    scale_reshaped = scale.T[:, :, np.newaxis]
    weight_fp = weight_grouped.astype(np.float32) * scale_reshaped
    
    # Reshape back to [out_features, in_features]
    weight_fp = weight_fp.reshape(out_features, -1)[:, :in_features]
    
    return weight_fp


def quantize_tensor(tensor, num_bits=8, symmetric=True, per_channel=False, channel_axis=0):
    """
    Generic tensor quantization function
    
    Args:
        tensor (np.ndarray): Input tensor
        num_bits (int): Quantization bit width. Default: 8
        symmetric (bool): Whether to use symmetric quantization. Default: True
        per_channel (bool): Whether to use per-channel quantization. Default: False
        channel_axis (int): Channel axis. Default: 0
    
    Returns:
        tuple: (tensor_quant, scale, offset)
    
    Examples:
        >>> tensor = np.random.randn(1024, 768).astype(np.float32)
        >>> tensor_quant, scale, offset = quantize_tensor(tensor, num_bits=8)
    """
    if num_bits == 8:
        return quantize_8bit(tensor, symmetric, per_channel, channel_axis)
    elif num_bits == 4:
        # INT4 quantization requires per-group
        group_size = 128
        return quantize_4bit(tensor, group_size=group_size)
    else:
        raise ValueError(f"Unsupported quantization bit width: {num_bits}")


def dequantize_tensor(tensor_quant, scale, offset=None, num_bits=8, group_size=128):
    """
    Generic tensor dequantization function
    
    Args:
        tensor_quant (np.ndarray): Quantized tensor
        scale (np.ndarray): Quantization scale
        offset (np.ndarray, optional): Quantization offset. Default: None
        num_bits (int): Quantization bit width. Default: 8
        group_size (int): Group size for per-group quantization (INT4 only). Default: 128
    
    Returns:
        np.ndarray: Dequantized tensor
    
    Examples:
        >>> tensor_fp = dequantize_tensor(tensor_quant, scale, offset, num_bits=8)
    """
    if num_bits == 8:
        return dequantize_8bit(tensor_quant, scale, offset)
    elif num_bits == 4:
        return dequantize_4bit(tensor_quant, scale, offset, group_size)
    else:
        raise ValueError(f"Unsupported quantization bit width: {num_bits}")


def estimate_quantization_error(weight_fp, weight_quant, scale, offset=None, num_bits=8):
    """
    Estimate quantization error
    
    Args:
        weight_fp (np.ndarray): Original FP weight
        weight_quant (np.ndarray): Quantized weight
        scale (np.ndarray): Quantization scale
        offset (np.ndarray, optional): Quantization offset. Default: None
        num_bits (int): Quantization bit width. Default: 8
    
    Returns:
        dict: Error statistics
            - mean_error: Mean absolute error
            - max_error: Maximum absolute error
            - relative_error: Relative error (percentage)
            - snr: Signal-to-noise ratio (dB)
    
    Examples:
        >>> error_stats = estimate_quantization_error(weight_fp, weight_int8, scale)
        >>> print(f"Mean error: {error_stats['mean_error']:.6f}")
    """
    # Dequantize
    if num_bits == 8:
        weight_dequant = dequantize_8bit(weight_quant, scale, offset)
    elif num_bits == 4:
        weight_dequant = dequantize_4bit(weight_quant, scale, offset)
    else:
        raise ValueError(f"Unsupported quantization bit width: {num_bits}")
    
    # Compute error
    error = np.abs(weight_fp - weight_dequant)
    mean_error = error.mean()
    max_error = error.max()
    
    # Relative error
    relative_error = mean_error / (np.abs(weight_fp).mean() + 1e-8) * 100
    
    # Signal-to-noise ratio (SNR)
    signal_power = np.mean(weight_fp ** 2)
    noise_power = np.mean(error ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
    
    return {
        'mean_error': float(mean_error),
        'max_error': float(max_error),
        'relative_error': float(relative_error),
        'snr': float(snr)
    }


def get_quantization_info(weight, num_bits=8, symmetric=True, per_channel=True):
    """
    Get quantization information (without actually quantizing)
    
    Args:
        weight (np.ndarray): Weight tensor
        num_bits (int): Quantization bit width. Default: 8
        symmetric (bool): Whether to use symmetric quantization. Default: True
        per_channel (bool): Whether to use per-channel quantization. Default: True
    
    Returns:
        dict: Quantization information
            - scale_range: Range of scale [min, max]
            - scale_mean: Mean of scale
            - weight_range: Range of weight [min, max]
            - estimated_compression: Estimated compression ratio
    
    Examples:
        >>> info = get_quantization_info(weight, num_bits=8)
        >>> print(f"Compression ratio: {info['estimated_compression']:.2f}x")
    """
    # Compute scale
    scale, offset = compute_scale_offset(
        weight,
        num_bits=num_bits,
        symmetric=symmetric,
        per_channel=per_channel
    )
    
    # Statistics
    info = {
        'scale_range': [float(scale.min()), float(scale.max())],
        'scale_mean': float(scale.mean()),
        'weight_range': [float(weight.min()), float(weight.max())],
        'estimated_compression': 32.0 / num_bits if weight.dtype == np.float32 else 16.0 / num_bits
    }
    
    return info
