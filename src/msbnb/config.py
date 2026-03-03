"""
Quantization Configuration Classes

Provides configuration management for quantization parameters.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class QuantConfig:
    """
    Base quantization configuration
    
    Args:
        bits (int): Quantization bit width, supports 4, 8. Default: 8 (8:-128~127, 4:-8~7)
        symmetric (bool): Whether to use symmetric quantization. Default: True
        per_channel (bool): Whether to use per-channel quantization. Default: True (each output channel uses different scale and zero_point)
        quant_delay (int): Number of steps to delay quantization. Default: 0
    """
    bits: int = 8
    symmetric: bool = True
    per_channel: bool = True
    quant_delay: int = 0


@dataclass
class Int8Config(QuantConfig):
    """
    INT8 quantization configuration
    
    Args:
        bits (int): Quantization bit width, fixed at 8
        symmetric (bool): Whether to use symmetric quantization. Default: True
        per_channel (bool): Whether to use per-channel quantization. Default: True
        threshold (float): Outlier threshold (for outlier handling). Default: 6.0
        has_fp16_weights (bool): Whether to keep weights in FP16 (training mode). Default: True
        quant_delay (int): Number of steps to delay quantization. Default: 0
    
    Examples:
        >>> config = Int8Config(symmetric=True, threshold=6.0)
        >>> layer = Linear8bit(768, 3072, config=config)
    """
    bits: int = 8
    threshold: float = 6.0
    has_fp16_weights: bool = True


@dataclass
class Int4Config(QuantConfig):
    """
    INT4 quantization configuration
    
    Args:
        bits (int): Quantization bit width, fixed at 4
        group_size (int): Group size for per-group quantization. Default: 128
        compress_statistics (bool): Whether to compress statistics (double quantization). Default: True (scale is also quantized to INT8)
        quant_type (str): Quantization type, supports 'int4'. Default: 'int4'
    
    Examples:
        >>> config = Int4Config(group_size=128, compress_statistics=True)
        >>> layer = Linear4bit(768, 3072, config=config)
    """
    bits: int = 4
    group_size: int = 128
    compress_statistics: bool = True
    quant_type: str = 'int4'
