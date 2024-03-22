import pytest
import mindspore
from mindspore import Tensor
from mindspore import nn
import numpy as np


class TestConv2d:
    @pytest.fixture
    def input_tensor(self):
        return Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32), dtype=mindspore.float32)

    @pytest.fixture
    def conv_layer(self):
        # 初始化Conv2d层
        mindspore.set_context(pynative_synchronize=True)
        return nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            has_bias=False,
            pad_mode='pad',
        )

    def test_conv2d_output_shape(self, input_tensor, conv_layer):
        output = conv_layer(input_tensor)
        expected_shape = (1, 64, 112, 112)
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"

    def test_conv2d_parameters(self, conv_layer):
        weight = conv_layer.weight
        expected_shape = (64, 3, 7, 7)
        assert weight.shape == expected_shape, f"Expected weight shape {expected_shape}, but got {weight.shape}"
        assert isinstance(weight, Tensor), "Weight is not an instance of Tensor"
