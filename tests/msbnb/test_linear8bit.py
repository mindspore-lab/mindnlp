"""
Linear8bit 单元测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pytest
import numpy as np

try:
    import mindspore as ms
    from mindspore import Tensor, nn, context
    from mindspore.common import dtype as mstype
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False

if MINDSPORE_AVAILABLE:
    from msbnb import Linear8bit
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")


@pytest.mark.skipif(not MINDSPORE_AVAILABLE, reason="MindSpore not available")
class TestLinear8bit:
    """Linear8bit 单元测试"""
    
    def test_init(self):
        """测试初始化"""
        layer = Linear8bit(768, 3072, has_fp16_weights=True)
        assert layer.in_features == 768
        assert layer.out_features == 3072
    
    def test_forward_fp16_mode(self):
        """测试 FP16 模式前向传播"""
        layer = Linear8bit(768, 3072, has_fp16_weights=True)
        x = Tensor(np.random.randn(32, 768).astype(np.float32))
        out = layer(x)
        assert out.shape == (32, 3072)
    
    def test_quantize_weights(self):
        """测试权重量化"""
        layer = Linear8bit(768, 3072, has_fp16_weights=True)
        layer.quantize_weights()
        assert layer.weight.dtype == mstype.int8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
