"""
Quantized Linear Layer Implementation

Provides bitsandbytes-style quantized linear layers.
"""

import numpy as np
try:
    import mindspore as ms
    import mindspore.nn as nn
    import mindspore.ops as ops
    from mindspore import Tensor, Parameter
    from mindspore.common import dtype as mstype
    from mindspore.common.initializer import initializer, Zero, One
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False
    print("Warning: MindSpore not available, quantization layers will not work")

from .utils import quantize_weight_int4_pergroup, compute_scale_offset


class LinearQuant(nn.Cell if MINDSPORE_AVAILABLE else object):
    """
    Base class for quantized linear layers
    
    Provides common interface and functionality for quantized layers.
    """
    
    def __init__(self):
        if MINDSPORE_AVAILABLE:
            super().__init__()
    
    def quantize_weights(self):
        """Quantize weights"""
        raise NotImplementedError
    
    def dequantize_weights(self):
        """Dequantize weights"""
        raise NotImplementedError


class Linear8bit(LinearQuant):
    """
    8-bit quantized linear layer, similar to bitsandbytes.nn.Linear8bitLt
    
    Based on MindSpore quantization operators.
    
    Args:
        in_features (int): Input feature dimension
        out_features (int): Output feature dimension
        bias (bool): Whether to use bias. Default: True
        has_fp16_weights (bool): Whether to keep weights in FP16 (training mode). Default: True
        threshold (float): Outlier threshold. Default: 6.0
        quant_delay (int): Number of steps to delay quantization. Default: 0
        per_channel (bool): Whether to use per-channel quantization. Default: True
        symmetric (bool): Whether to use symmetric quantization. Default: True
    
    Inputs:
        - **x** (Tensor) - Input tensor with shape: [batch, ..., in_features]
    
    Outputs:
        - **output** (Tensor) - Output tensor with shape: [batch, ..., out_features]
    
    Examples:
        >>> # Training mode
        >>> layer = Linear8bit(768, 3072, has_fp16_weights=True)
        >>> x = Tensor(np.random.randn(32, 768), dtype=ms.float16)
        >>> out = layer(x)
        >>> 
        >>> # Inference mode (quantized weights)
        >>> layer.quantize_weights()
        >>> out = layer(x)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        has_fp16_weights: bool = True,
        threshold: float = 6.0,
        quant_delay: int = 0,
        per_channel: bool = True,
        symmetric: bool = True
    ):
        if not MINDSPORE_AVAILABLE:
            raise ImportError("MindSpore is required for Linear8bit")
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_fp16_weights = has_fp16_weights
        self.per_channel = per_channel
        self.threshold = threshold
        self.symmetric = symmetric
        
        # Weight parameters
        if has_fp16_weights:
            # Training mode: keep weights in FP16
            self.weight = Parameter(
                Tensor(np.random.normal(0, 0.02, (out_features, in_features)), 
                       dtype=mstype.float16),
                name='weight'
            )
        else:
            # Inference mode: use INT8 weights
            self.weight = Parameter(
                Tensor(np.zeros((out_features, in_features)), dtype=mstype.int8),
                name='weight',
                requires_grad=False
            )
        
        # Quantization parameters
        scale_shape = (out_features,) if per_channel else (1,)
        self.scale = Parameter(
            Tensor(np.ones(scale_shape), dtype=mstype.float16),
            name='scale',
            requires_grad=False
        )
        
        if not symmetric:
            self.offset = Parameter(
                Tensor(np.zeros(scale_shape), dtype=mstype.float16),
                name='offset',
                requires_grad=False
            )
        else:
            self.offset = None
        
        # Bias
        if bias:
            self.bias = Parameter(
                Tensor(np.zeros(out_features), dtype=mstype.float16),
                name='bias'
            )
        else:
            self.bias = None
        
        # Matrix multiplication operator
        self.matmul = ops.MatMul(transpose_b=True)
        self.cast = ops.Cast()
    
    def construct(self, x):
        """Forward pass"""
        # Ensure consistent input type
        x_dtype = x.dtype
        
        if self.has_fp16_weights:
            # Training mode: use FP16 weights
            compute_dtype = mstype.float32 if x_dtype == mstype.float32 else mstype.float16
            # Convert weight to same type as input
            weight = self.cast(self.weight, compute_dtype)
            out = self.matmul(x, weight)
            
            # Add bias
            if self.bias is not None:
                bias = self.cast(self.bias, compute_dtype)
                out = out + bias
        else:
            # Inference mode: WeightQuantBatchMatmul requires input to be float16 or bfloat16
            # If input is float32, convert to float16 first
            if x_dtype == mstype.float32:
                x = self.cast(x, mstype.float16)
                compute_dtype = mstype.float16
            else:
                compute_dtype = x_dtype
            
            # Use WeightQuantBatchMatmul operator (hardware accelerated)
            try:
                # Try to import auto-generated WeightQuantBatchMatmul operator
                from mindspore.ops.auto_generate import weight_quant_batch_matmul
                
                # Prepare parameters
                scale = self.cast(self.scale, compute_dtype)
                offset = self.cast(self.offset, compute_dtype) if self.offset is not None else None
                bias = self.cast(self.bias, compute_dtype) if self.bias is not None else None
                
                # Use WeightQuantBatchMatmul operator
                # Note: This operator fuses dequantization and matrix multiplication for better performance
                out = weight_quant_batch_matmul(
                    x,                  # Input [batch, in_features] (float16/bfloat16)
                    self.weight,        # INT8 weight [out_features, in_features]
                    scale,              # antiquant_scale [out_features] or [1]
                    offset,             # antiquant_offset (optional)
                    None,               # quant_scale (not needed for inference)
                    None,               # quant_offset (not needed for inference)
                    bias,               # bias (optional, fused in operator)
                    transpose_x=False,
                    transpose_weight=True,
                    antiquant_group_size=0  # 0 means per-channel
                )
                
                # If original input was float32, convert output back to float32
                if x_dtype == mstype.float32:
                    out = self.cast(out, mstype.float32)
            except (ImportError, AttributeError, RuntimeError) as e:
                # If operator is not available, fallback to manual implementation
                print(f"Warning: WeightQuantBatchMatmul not available, using manual dequantization: {e}")
                
                # Manual dequantization + matrix multiplication
                weight_int8_fp = self.cast(self.weight, compute_dtype)
                scale = self.cast(self.scale, compute_dtype)
                
                # Ensure scale shape is correct for broadcasting
                if self.per_channel:
                    scale = scale.reshape(-1, 1)  # [out_features, 1]
                
                weight_fp = weight_int8_fp * scale
                
                if self.offset is not None:
                    offset = self.cast(self.offset, compute_dtype)
                    if self.per_channel:
                        offset = offset.reshape(-1, 1)
                    weight_fp = weight_fp + offset
                
                out = self.matmul(x, weight_fp)
                
                # Add bias
                if self.bias is not None:
                    bias = self.cast(self.bias, compute_dtype)
                    out = out + bias
                
                # If original input was float32, convert output back to float32
                if x_dtype == mstype.float32:
                    out = self.cast(out, mstype.float32)
        
        return out
    
    def quantize_weights(self):
        """
        Quantize FP16 weights to INT8
        
        After calling this method, the model switches to inference mode
        """
        if not self.has_fp16_weights:
            print("Weights are already quantized")
            return
        
        # Compute quantization parameters
        weight_data = self.weight.asnumpy()
        
        scale, offset = compute_scale_offset(
            weight_data,
            num_bits=8,
            symmetric=self.symmetric,
            per_channel=self.per_channel,
            channel_axis=0
        )
        
        if self.symmetric:
            # Symmetric quantization: [-127, 127]
            weight_int8 = np.clip(
                np.round(weight_data / scale),
                -127, 127
            ).astype(np.int8)
        else:
            # Asymmetric quantization
            weight_int8 = np.clip(
                np.round(weight_data / scale + offset),
                -128, 127
            ).astype(np.int8)
            self.offset.set_data(Tensor(offset.squeeze(), dtype=mstype.float16))
        
        # Update parameters
        self.weight = Parameter(
            Tensor(weight_int8, dtype=mstype.int8),
            name='weight',
            requires_grad=False
        )
        # scale remains as 1D array, will be reshaped in construct
        self.scale.set_data(Tensor(scale.squeeze(), dtype=mstype.float16))
        if not self.symmetric and self.offset is not None:
            self.offset.set_data(Tensor(offset.squeeze(), dtype=mstype.float16))
        self.has_fp16_weights = False
        
        print(f"Weights quantized to INT8, scale range: [{scale.min():.6f}, {scale.max():.6f}]")



class Linear4bit(LinearQuant):
    """
    4-bit quantized linear layer, similar to bitsandbytes.nn.Linear4bit (QLoRA)
    
    Based on MindSpore's qint4x2 data type.
    
    Args:
        in_features (int): Input feature dimension
        out_features (int): Output feature dimension
        bias (bool): Whether to use bias. Default: True
        compute_dtype (mstype): Computation data type. Default: mstype.float16
        compress_statistics (bool): Whether to compress statistics (double quantization). Default: True
        quant_type (str): Quantization type ('int4'). Default: 'int4'
        group_size (int): Group size for per-group quantization. Default: 128
    
    Examples:
        >>> # Create INT4 quantized layer
        >>> layer = Linear4bit(768, 3072, group_size=128)
        >>> x = Tensor(np.random.randn(32, 768), dtype=ms.float16)
        >>> out = layer(x)
        >>> 
        >>> # Convert from FP16 layer
        >>> fp16_layer = nn.Dense(768, 3072)
        >>> int4_layer = Linear4bit.from_linear(fp16_layer, group_size=128)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: 'mstype' = None,
        compress_statistics: bool = True,
        quant_type: str = 'int4',
        group_size: int = 128
    ):
        if not MINDSPORE_AVAILABLE:
            raise ImportError("MindSpore is required for Linear4bit")
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype or mstype.float16
        self.group_size = group_size
        self.compress_statistics = compress_statistics
        
        # Weight parameters (use packed uint8 format to store INT4)
        # Two INT4 values packed into one byte
        # Actual storage size: [out_features, in_features // 2]
        self.weight = Parameter(
            initializer(Zero(), [out_features, in_features // 2], mstype.uint8),
            name='weight',
            requires_grad=False
        )
        self.use_qint4x2 = False
        
        # Quantization parameters (per-group)
        num_groups = (in_features + group_size - 1) // group_size
        
        if compress_statistics:
            # Double quantization: scale stored as INT8
            self.scale = Parameter(
                initializer(One(), [num_groups, out_features], mstype.int8),
                name='scale',
                requires_grad=False
            )
            # Scale of scale
            self.scale_scale = Parameter(
                initializer(One(), [out_features], self.compute_dtype),
                name='scale_scale',
                requires_grad=False
            )
        else:
            # Standard quantization: scale stored as FP16/FP32
            self.scale = Parameter(
                initializer(One(), [num_groups, out_features], self.compute_dtype),
                name='scale',
                requires_grad=False
            )
            self.scale_scale = None
        
        # Offset (0 for symmetric quantization)
        self.offset = Parameter(
            initializer(Zero(), [num_groups, out_features], self.compute_dtype),
            name='offset',
            requires_grad=False
        )
        
        # Bias
        if bias:
            self.bias = Parameter(
                initializer(Zero(), [out_features], self.compute_dtype),
                name='bias'
            )
        else:
            self.bias = None
        
        # Operators
        self.matmul = ops.MatMul(transpose_b=True)
        self.cast = ops.Cast()
    
    def construct(self, x):
        """Forward pass"""
        # Save original shape
        original_shape = x.shape
        
        # If input is multi-dimensional, reshape to 2D
        if len(original_shape) > 2:
            x_2d = x.reshape(-1, original_shape[-1])
        else:
            x_2d = x
        
        # Ensure consistent input type
        x_dtype = x_2d.dtype
        compute_dtype = mstype.float32 if x_dtype == mstype.float32 else mstype.float16
        
        # Dequantize scale (if using double quantization)
        if self.scale_scale is not None:
            # scale_fp = scale_int8 * scale_scale
            scale = self.cast(self.scale, compute_dtype) * self.cast(self.scale_scale, compute_dtype)
        else:
            scale = self.cast(self.scale, compute_dtype)
        
        # INT4 quantization uses manual dequantization
        # Note: WeightQuantBatchMatmul has limited support for INT4, manual implementation is more reliable
        from .utils import unpack_qint4x2_to_int8
        
        # Unpack INT4 weights to INT8
        weight_int8 = unpack_qint4x2_to_int8(self.weight.asnumpy())
        
        # Per-group dequantization
        weight_fp = self._dequantize_pergroup(weight_int8, scale.asnumpy())
        weight_fp_tensor = Tensor(weight_fp, dtype=compute_dtype)
        
        # Matrix multiplication
        out = self.matmul(x_2d, weight_fp_tensor)
        
        # Add bias
        if self.bias is not None:
            bias = self.cast(self.bias, compute_dtype)
            out = out + bias
        
        # Restore original shape
        if len(original_shape) > 2:
            out = out.reshape(*original_shape[:-1], self.out_features)
        
        return out
    
    def _dequantize_pergroup(self, weight_int8, scale):
        """Per-group dequantization"""
        out_features, in_features = weight_int8.shape
        num_groups = scale.shape[0]
        group_size = (in_features + num_groups - 1) // num_groups
        
        # Reshape to [out_features, num_groups, group_size]
        weight_grouped = weight_int8[:, :num_groups * group_size].reshape(
            out_features, num_groups, group_size
        )
        
        # Dequantize: weight_fp = weight_int8 * scale
        # scale shape: [num_groups, out_features] -> [out_features, num_groups, 1]
        scale_reshaped = scale.T[:, :, np.newaxis]
        weight_fp = weight_grouped * scale_reshaped
        
        # Reshape back to [out_features, in_features]
        weight_fp = weight_fp.reshape(out_features, -1)[:, :in_features]
        
        return weight_fp.astype(np.float32)
    
    @classmethod
    def from_linear(cls, linear_layer, group_size=128, compress_statistics=True):
        """
        Convert standard Linear layer to Linear4bit
        
        Args:
            linear_layer: nn.Dense or nn.Linear layer
            group_size (int): Quantization group size. Default: 128
            compress_statistics (bool): Whether to use double quantization. Default: True
        
        Returns:
            Linear4bit: Quantized layer
        
        Examples:
            >>> fp16_layer = nn.Dense(768, 3072)
            >>> int4_layer = Linear4bit.from_linear(fp16_layer, group_size=128)
        """
        # Get layer parameters
        if hasattr(linear_layer, 'in_channels'):
            in_features = linear_layer.in_channels
            out_features = linear_layer.out_channels
            has_bias = linear_layer.has_bias
        else:
            in_features = linear_layer.in_features
            out_features = linear_layer.out_features
            has_bias = linear_layer.bias is not None
        
        # Create Linear4bit instance
        quant_layer = cls(
            in_features,
            out_features,
            bias=has_bias,
            group_size=group_size,
            compress_statistics=compress_statistics
        )
        
        # Quantize weights
        weight_fp16 = linear_layer.weight.data.asnumpy()
        weight_int4, scale, offset = quantize_weight_int4_pergroup(
            weight_fp16, group_size, compress_statistics
        )
        
        # Set parameters
        quant_layer.weight.set_data(Tensor(weight_int4, dtype=mstype.uint8))
        
        if compress_statistics:
            # Double quantization
            quant_layer.scale.set_data(Tensor(scale[0], dtype=mstype.int8))
            quant_layer.scale_scale.set_data(Tensor(scale[1], dtype=mstype.float16))
        else:
            quant_layer.scale.set_data(Tensor(scale, dtype=mstype.float16))
        
        quant_layer.offset.set_data(Tensor(offset, dtype=mstype.float16))
        
        if has_bias:
            quant_layer.bias.set_data(linear_layer.bias.data)
        
        print(f"Converted Linear({in_features}, {out_features}) to Linear4bit")
        print(f"  - Group size: {group_size}")
        print(f"  - Double quantization: {compress_statistics}")
        print(f"  - Memory saved: {(1 - 0.25) * 100:.1f}%")
        
        return quant_layer
