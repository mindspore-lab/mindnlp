"""
MindSpore BitsAndBytes 基础测试

测试量化层的基本功能。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pytest

# 尝试导入 MindSpore
try:
    import mindspore as ms
    from mindspore import Tensor, context
    import mindspore.nn as nn
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False
    pytest.skip("MindSpore not available", allow_module_level=True)

from msbnb import Linear8bit, Linear4bit, Int8Config, Int4Config
from msbnb.utils import (
    pack_int4_to_qint4x2,
    unpack_qint4x2_to_int8,
    compute_scale_offset
)


class TestLinear8bit:
    """测试 Linear8bit 层"""
    
    def test_creation(self):
        """测试层创建"""
        layer = Linear8bit(768, 3072, has_fp16_weights=True)
        assert layer.in_features == 768
        assert layer.out_features == 3072
        assert layer.has_fp16_weights == True
    
    def test_forward_fp16(self):
        """测试 FP16 模式前向传播"""
        layer = Linear8bit(768, 3072, has_fp16_weights=True)
        x = Tensor(np.random.randn(32, 768).astype(np.float32))
        out = layer(x)
        assert out.shape == (32, 3072)
    
    def test_quantize_weights(self):
        """测试权重量化"""
        layer = Linear8bit(768, 3072, has_fp16_weights=True)
        x = Tensor(np.random.randn(32, 768).astype(np.float32))
        
        # FP16 输出
        out_fp16 = layer(x)
        
        # 量化权重
        layer.quantize_weights()
        assert layer.has_fp16_weights == False
        assert layer.weight.dtype == ms.int8
        
        # INT8 输出
        out_int8 = layer(x)
        assert out_int8.shape == out_fp16.shape
        
        # 检查精度损失
        error = np.abs(out_fp16.asnumpy() - out_int8.asnumpy()).mean()
        assert error < 0.1  # 平均误差 < 0.1


class TestLinear4bit:
    """测试 Linear4bit 层"""
    
    def test_creation(self):
        """测试层创建"""
        layer = Linear4bit(768, 3072, group_size=128)
        assert layer.in_features == 768
        assert layer.out_features == 3072
        assert layer.group_size == 128
    
    def test_forward(self):
        """测试前向传播"""
        layer = Linear4bit(768, 3072, group_size=128)
        x = Tensor(np.random.randn(32, 768).astype(np.float32))
        out = layer(x)
        assert out.shape == (32, 3072)
    
    def test_from_linear(self):
        """测试从标准层转换"""
        # 创建标准层
        fp16_layer = nn.Dense(768, 3072, has_bias=True)
        x = Tensor(np.random.randn(32, 768).astype(np.float32))
        
        # FP16 输出
        out_fp16 = fp16_layer(x)
        
        # 转换为 INT4
        int4_layer = Linear4bit.from_linear(fp16_layer, group_size=128)
        
        # INT4 输出
        out_int4 = int4_layer(x)
        assert out_int4.shape == out_fp16.shape
        
        # 检查精度损失
        error = np.abs(out_fp16.asnumpy() - out_int4.asnumpy()).mean()
        assert error < 0.2  # 平均误差 < 0.2


class TestUtils:
    """测试工具函数"""
    
    def test_pack_unpack_int4(self):
        """测试 INT4 打包和解包"""
        # 创建测试数据
        weight_int8 = np.random.randint(-7, 7, (1024, 768), dtype=np.int8)
        
        # 打包
        weight_packed = pack_int4_to_qint4x2(weight_int8)
        assert weight_packed.shape == (1024, 384)
        assert weight_packed.dtype == np.uint8
        
        # 解包
        weight_unpacked = unpack_qint4x2_to_int8(weight_packed)
        assert weight_unpacked.shape == weight_int8.shape
        
        # 验证正确性
        assert np.array_equal(weight_int8, weight_unpacked)
    
    def test_compute_scale_offset_symmetric(self):
        """测试对称量化参数计算"""
        weight = np.random.randn(3072, 768).astype(np.float32)
        
        scale, offset = compute_scale_offset(
            weight,
            num_bits=8,
            symmetric=True,
            per_channel=True
        )
        
        assert scale.shape == (3072, 1)
        assert offset is None
    
    def test_compute_scale_offset_asymmetric(self):
        """测试非对称量化参数计算"""
        weight = np.random.randn(3072, 768).astype(np.float32)
        
        scale, offset = compute_scale_offset(
            weight,
            num_bits=8,
            symmetric=False,
            per_channel=True
        )
        
        assert scale.shape == (3072, 1)
        assert offset.shape == (3072, 1)


class TestConfig:
    """测试配置类"""
    
    def test_int8_config(self):
        """测试 INT8 配置"""
        config = Int8Config(
            symmetric=True,
            per_channel=True,
            threshold=6.0
        )
        assert config.bits == 8
        assert config.symmetric == True
        assert config.threshold == 6.0
    
    def test_int4_config(self):
        """测试 INT4 配置"""
        config = Int4Config(
            group_size=128,
            compress_statistics=True
        )
        assert config.bits == 4
        assert config.group_size == 128
        assert config.compress_statistics == True


if __name__ == '__main__':
    # 运行测试
    print("运行 MindSpore BitsAndBytes 测试...")
    print("=" * 60)
    
    if not MINDSPORE_AVAILABLE:
        print("MindSpore 不可用，跳过测试")
        exit(0)
    
    # 测试 Linear8bit
    print("\n[1] 测试 Linear8bit")
    test_linear8bit = TestLinear8bit()
    test_linear8bit.test_creation()
    print("  ✓ 层创建")
    test_linear8bit.test_forward_fp16()
    print("  ✓ FP16 前向传播")
    test_linear8bit.test_quantize_weights()
    print("  ✓ 权重量化")
    
    # 测试 Linear4bit
    print("\n[2] 测试 Linear4bit")
    test_linear4bit = TestLinear4bit()
    test_linear4bit.test_creation()
    print("  ✓ 层创建")
    test_linear4bit.test_forward()
    print("  ✓ 前向传播")
    test_linear4bit.test_from_linear()
    print("  ✓ 从标准层转换")
    
    # 测试工具函数
    print("\n[3] 测试工具函数")
    test_utils = TestUtils()
    test_utils.test_pack_unpack_int4()
    print("  ✓ INT4 打包/解包")
    test_utils.test_compute_scale_offset_symmetric()
    print("  ✓ 对称量化参数计算")
    test_utils.test_compute_scale_offset_asymmetric()
    print("  ✓ 非对称量化参数计算")
    
    # 测试配置
    print("\n[4] 测试配置类")
    test_config = TestConfig()
    test_config.test_int8_config()
    print("  ✓ INT8 配置")
    test_config.test_int4_config()
    print("  ✓ INT4 配置")
    
    print("\n" + "=" * 60)
    print("所有测试通过！✓")
