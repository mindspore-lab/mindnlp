"""
工具函数单元测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pytest
import numpy as np
from msbnb.utils import pack_int4_to_qint4x2, unpack_qint4x2_to_int8


class TestPackUnpack:
    """测试 INT4 打包/解包"""
    
    def test_pack_unpack_roundtrip(self):
        """测试打包和解包的往返"""
        weight_int8 = np.random.randint(-7, 8, (1024, 768), dtype=np.int8)
        weight_packed = pack_int4_to_qint4x2(weight_int8)
        weight_unpacked = unpack_qint4x2_to_int8(weight_packed)
        assert np.array_equal(weight_int8, weight_unpacked)
    
    def test_pack_shape(self):
        """测试打包后的形状"""
        weight_int8 = np.random.randint(-7, 8, (1024, 768), dtype=np.int8)
        weight_packed = pack_int4_to_qint4x2(weight_int8)
        assert weight_packed.shape == (1024, 768 // 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
