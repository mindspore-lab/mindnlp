# coding=utf-8
# Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Visual backbone."""
import math
import os
from dataclasses import dataclass
from typing import Optional

import yaml
from addict import Dict

import mindspore as ms
from mindspore import nn, ops


import numpy as np


@dataclass
class ShapeSpec:

    """
    The ShapeSpec class represents a specification for a shape, providing details and parameters
    for creating and manipulating shapes.
    
    This class inherits from [insert name of the parent class here].
    
    Attributes:
        [List any attributes of the class and their descriptions here]
    
    Methods:
        [List any methods of the class and their descriptions here]
    """
    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None


class Conv2d(nn.Conv2d):

    """
    This class represents a custom convolutional layer for 2-dimensional data, inheriting from the nn.Conv2d class.
    
    Attributes:
        norm: A normalization function applied to the output of the convolutional layer.
            If None, no normalization is applied.
        activation: An activation function applied to the output of the normalization step.
            If None, no activation is applied.

    Methods:
        __init__(self, *args, **kwargs): Initializes the Conv2d object with optional normalization and activation parameters.
        construct(self, x): Applies the convolutional operation to the input tensor x,
            followed by optional normalization and activation.

    """
    def __init__(self, *args, **kwargs):
        """
        Initializes an instance of the Conv2d class.

        Args:
            self: The instance of the Conv2d class.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method initializes an instance of the Conv2d class. It takes the following optional keyword arguments:

            - norm: Specifies the normalization method to be applied. Default is None.
            - activation: Specifies the activation function to be applied. Default is None.

        The method first extracts the 'norm' and 'activation' keyword arguments using the pop() method from the kwargs
        dictionary. Next, it calls the __init__() method of the parent class using the super() function, passing all
        the arguments and keyword arguments (*args, **kwargs). After that, it assigns the 'norm' and 'activation' values
        to the instance variables self.norm and self.activation respectively.

        Note:
            - The 'norm' parameter should be of type 'None' or any valid normalization method.
            - The 'activation' parameter should be of type 'None' or any valid activation function.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super(Conv2d, self).__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def construct(self, x):
        """
        Construct method in the Conv2d class.

        Args:
            self (object): Instance of the Conv2d class.
            x (object): Input data to be processed.

        Returns:
            None.

        Raises:
            None
        """
        x = super(Conv2d, self).construct(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class BasicStem(nn.Cell):
    """
    The standard ResNet stem (layers before the first residual block),
    with a conv, relu and max_pool.
    """
    def __init__(self, in_channels=3, out_channels=64, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = 4
        bn1 = None
        if norm == "BN":
            bn1 = nn.BatchNorm2d(out_channels)
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            has_bias=False,
            pad_mode='pad',
            norm=bn1
        )
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, pad_mode="pad")

    def construct(self, x):
        """
        Constructs a basic stem block by applying convolution, ReLU activation, and max pooling operations.

        Args:
            self (object): Instance of the BasicStem class.
            x (tensor): Input tensor to be processed by the basic stem block.

        Returns:
            None: The method modifies the input tensor 'x' in place.

        Raises:
            None.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class BasicBlock(nn.Cell):
    """
    The basic residual block for ResNet-18 and ResNet-34 defined in :paper:`ResNet`,
    with two 3x3 conv layers and a projection shortcut if needed.
    """
    def __init__(self, in_channels, out_channels, *, stride=1, norm="BN"):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                has_bias=False,
                norm=nn.BatchNorm2d(out_channels),
                pad_mode='valid'
            )
        else:
            self.shortcut = None

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            pad_mode='pad',
            has_bias=False,
            norm=nn.BatchNorm2d(out_channels)
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            pad_mode='pad',
            has_bias=False,
            norm=nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def construct(self, x):
        """
        Constructs a basic block by performing convolutional operations and element-wise addition with shortcut connection.

        Args:
            self (object): The instance of the BasicBlock class.
            x (tensor): The input tensor to be processed by the basic block.

        Returns:
            tensor: The output tensor after passing through the basic block operations.

        Raises:
            None.
        """
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = self.relu(out)
        return out


class BottleneckBlock(nn.Cell):
    """
    The standard bottleneck residual block used by ResNet-50, 101 and 152
    defined in :paper:`ResNet`.  It contains 3 conv layers with kernels
    1x1, 3x3, 1x1, and a projection shortcut if needed.
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            *,
            bottleneck_channels,
            stride=1,
            num_groups=1,
            norm="BN",
            stride_in_1x1=False,
            dilation=1,
    ):
        """
        Args:
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            num_groups (int): number of groups for the 3x3 conv layer.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            stride_in_1x1 (bool): when stride>1, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
            dilation (int): the dilation rate of the 3x3 conv layer.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        if norm == "BN" or norm is None:
            norm = nn.BatchNorm2d
        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                has_bias=False,
                norm=norm(out_channels),
                pad_mode='valid'
            )
        else:
            self.shortcut = None
        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            has_bias=False,
            norm=nn.BatchNorm2d(bottleneck_channels),
            pad_mode='valid'
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            has_bias=False,
            group=num_groups,
            dilation=dilation,
            pad_mode='pad',
            norm=norm(bottleneck_channels)
        )
        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            has_bias=False,
            pad_mode='valid',
            norm=norm(out_channels)
        )
        self.relu = nn.ReLU()

    def construct(self, x):
        """
        Constructs a bottleneck block for the BottleneckBlock class.

        Args:
            self (BottleneckBlock): An instance of the BottleneckBlock class.
            x (tensor): The input tensor.

        Returns:
            None

        Raises:
            None
        """
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = self.relu(out)
        return out


class ResNet(nn.Cell):
    """
    Implement :paper:`ResNet`.
    """
    def __init__(self, stem, stages, num_classes=None, out_features=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[CNNBlockBase]]): several (typically 4) stages,
                each contains multiple :class:`CNNBlockBase`.
            num_classes (None or int): if None, will not perform classification.
                Otherwise, will create a linear layer.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
            freeze_at (int): The number of stages at the beginning to freeze.
                see :meth:`freeze` for detailed explanation.
        """
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}
        self.flatten = nn.Flatten(start_dim=1)
        self.stage_names, self.stages = [], []

        if out_features is not None:
            # Avoid keeping unused layers in this module. They consume extra memory
            # and may cause allreduce to fail
            num_stages = max(
                {"res2": 1, "res3": 2, "res4": 3, "res5": 4}.get(f, 0) for f in out_features
            )
            stages = stages[:num_stages]
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, nn.Cell), block

            name = "res" + str(i + 2)
            stage = nn.SequentialCell(*blocks)

            self.insert_child_to_cell(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels
        self.stage_names = tuple(self.stage_names)  # Make it static for scripting

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Dense(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)

    def construct(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = self.flatten(x)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        """
        Method to calculate the output shape of the ResNet model.

        Args:
            self (ResNet): The instance of the ResNet class.

        Returns:
            None: This method does not return any value explicitly,
                but it updates the internal state of the ResNet instance.

        Raises:
            None.
        """
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @staticmethod
    def make_stage(block_class, num_blocks, *, in_channels, out_channels, **kwargs):
        """
        Create a list of blocks of the same type that forms one ResNet stage.

        Args:
            block_class (type): a subclass of CNNBlockBase that's used to create all blocks in this
                stage. A module of this type must not change spatial resolution of inputs unless its
                stride != 1.
            num_blocks (int): number of blocks in this stage
            in_channels (int): input channels of the entire stage.
            out_channels (int): output channels of **every block** in the stage.
            kwargs: other arguments passed to the constructor of
                `block_class`. If the argument name is "xx_per_block", the
                argument is a list of values to be passed to each block in the
                stage. Otherwise, the same argument is passed to every block
                in the stage.

        Returns:
            list[CNNBlockBase]: a list of block module.

        Example:
            ```python
            >>> stage = ResNet.make_stage(
            >>>     BottleneckBlock, 3, in_channels=16, out_channels=64,
            >>>     bottleneck_channels=16, num_groups=1,
            >>>     stride_per_block=[2, 1, 1],
            >>>     dilations_per_block=[1, 1, 2]
            >>> )
            ```

        Usually, layers that produce the same feature map spatial size are defined as one
        "stage" (in :paper:`FPN`). Under such definition, ``stride_per_block[1:]`` should
        all be 1.
        """
        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith("_per_block"):
                    assert len(v) == num_blocks, (
                        f"Argument '{k}' of make_stage should have the "
                        f"same length as num_blocks={num_blocks}."
                    )
                    newk = k[: -len("_per_block")]
                    assert newk not in kwargs, f"Cannot call make_stage with both {k} and {newk}!"
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v

            blocks.append(
                block_class(in_channels=in_channels, out_channels=out_channels, **curr_kwargs)
            )
            in_channels = out_channels
        return blocks

    @staticmethod
    def make_default_stages(depth, block_class=None, **kwargs):
        """
        Created list of ResNet stages from pre-defined depth (one of 18, 34, 50, 101, 152).
        If it doesn't create the ResNet variant you need, please use :meth:`make_stage`
        instead for fine-grained customization.

        Args:
            depth (int): depth of ResNet
            block_class (type): the CNN block class. Has to accept
                `bottleneck_channels` argument for depth > 50.
                By default it is BasicBlock or BottleneckBlock, based on the
                depth.
            kwargs:
                other arguments to pass to `make_stage`. Should not contain
                stride and channels, as they are predefined for each depth.

        Returns:
            list[list[CNNBlockBase]]: modules in all stages; see arguments of
                :class:`ResNet.__init__`.
        """
        num_blocks_per_stage = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[depth]
        if block_class is None:
            block_class = BasicBlock if depth < 50 else BottleneckBlock
        if depth < 50:
            in_channels = [64, 64, 128, 256]
            out_channels = [64, 128, 256, 512]
        else:
            in_channels = [64, 256, 512, 1024]
            out_channels = [256, 512, 1024, 2048]
        ret = []
        for (n, s, i, o) in zip(num_blocks_per_stage, [1, 2, 2, 2], in_channels, out_channels):
            if depth >= 50:
                kwargs["bottleneck_channels"] = o // 4
            ret.append(
                ResNet.make_stage(
                    block_class=block_class,
                    num_blocks=n,
                    stride_per_block=[s] + [1] * (n - 1),
                    in_channels=i,
                    out_channels=o,
                    **kwargs,
                )
            )
        return ret


def build_resnet_backbone(cfg):
    """
    Builds a ResNet backbone network based on the provided configuration.

    Args:
        cfg (object):
            The configuration object containing the following attributes:

            - MODEL.RESNETS.STEM_IN_CHANNELS (int): The number of input channels for the stem block.
            - MODEL.RESNETS.STEM_OUT_CHANNELS (int): The number of output channels for the stem block.
            - MODEL.RESNETS.NORM (str): The normalization method to be used in the backbone.
            - MODEL.RESNETS.OUT_FEATURES (list): The list of feature names to be outputted by the backbone.
            - MODEL.RESNETS.DEPTH (int): The depth of the ResNet backbone.
            - MODEL.RESNETS.NUM_GROUPS (int): The number of groups in each bottleneck block.
            - MODEL.RESNETS.WIDTH_PER_GROUP (int): The width of each group in each bottleneck block.
            - MODEL.RESNETS.RES2_OUT_CHANNELS (int): The number of output channels for the res2 block.
            - MODEL.RESNETS.STRIDE_IN_1X1 (bool): Whether to apply stride in the 1x1 convolution in each bottleneck block.
            - MODEL.RESNETS.RES5_DILATION (int): The dilation value for the res5 block. Must be 1 or 2.

    Returns:
        None

    Raises:
        AssertionError: If the value of 'res5_dilation' attribute is not 1 or 2.
        AssertionError: If 'out_channels' is not 64 for the ResNet18 or ResNet34 models.
        AssertionError: If 'res5_dilation' is not 1 for the ResNet18 or ResNet34 models.
        AssertionError: If 'num_groups' is not 1 for the ResNet18 or ResNet34 models.
    """
    stem = BasicStem(
        in_channels=cfg.MODEL.RESNETS.STEM_IN_CHANNELS,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
    )

    # fmt: off
    norm = cfg.MODEL.RESNETS.NORM  # "BN"
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    depth = cfg.MODEL.RESNETS.DEPTH
    num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation = cfg.MODEL.RESNETS.RES5_DILATION
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []

    for idx, stage_idx in enumerate(range(2, 6)):
        # res5_dilation is used this way as a convention in R-FCN & Deformable Conv paper
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            stage_kargs["block_class"] = BottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features)

def read_config():
    """Reads the visual_backbone.yaml configuration file and returns its contents as a dictionary.

    Returns:
        None.

    Raises:
        FileNotFoundError: If the visual_backbone.yaml file does not exist in the current directory.
        yaml.YAMLError: If there is an error loading the YAML data from the file.
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(curr_dir, 'visual_backbone.yaml'), 'r') as file:
        data = yaml.safe_load(file)
        data = Dict(data)
    return data


def build_resnet_fpn_backbone(cfg):
    """
    Builds a ResNet-FPN backbone based on the provided configuration.

    Args:
        cfg (object): The configuration object containing the model parameters.

    Returns:
        object: The constructed FPN backbone.

    Raises:
        None.

    This function builds a ResNet-FPN backbone using the specified configuration.
    It first constructs the bottom-up ResNet backbone using the provided configuration.
    Then, it retrieves the required input features and output channels from the configuration.
    Finally, it constructs the FPN backbone using the bottom-up backbone, input features, output channels,
    normalization method, top block, and fuse type specified in the configuration.
    The constructed FPN backbone is returned as the result.
    """
    bottom_up = build_resnet_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


class LastLevelMaxPool(nn.Cell):

    """
    The LastLevelMaxPool class represents a neural network cell that performs max pooling on input data.
    This class inherits from nn.Cell and implements the functionality to construct the max pooling operation on input data.

    Attributes:
        num_levels (int): The number of levels in the max pooling operation. Default value is 1.
        in_feature (str): The input feature for the max pooling operation. Default value is 'p5'.

    Methods:
        construct(x): Constructs the max pooling operation on the input data x and returns the result.

    Example:
        ```python
        >>> # Create an instance of LastLevelMaxPool
        >>> last_level_max_pool = LastLevelMaxPool()
        >>> # Perform max pooling on input data
        >>> result = last_level_max_pool.construct(input_data)
        ```
    """
    def __init__(self):
        """
        Initializes an instance of the LastLevelMaxPool class.

        Args:
            self: The instance of the LastLevelMaxPool class being initialized.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def construct(self, x):
        """
        Constructs the last level max pooling operation on the input tensor.

        Args:
            self: An instance of the LastLevelMaxPool class.
            x (Tensor): The input tensor to be max pooled.

        Returns:
            None

        Raises:
            None
        """
        return [ops.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class FPN(nn.Cell):

    """
    This class represents a Feature Pyramid Network (FPN) implemented as a neural network module in MindSpore.
    FPN is a commonly used architecture in computer vision tasks, especially in object detection.

    The FPN class inherits from the nn.Cell class, which is the base class for all neural network modules in MindSpore.

    Attributes:
        bottom_up (nn.Cell): The bottom-up network that extracts features from the input data.
        in_features (tuple): The names of the input features used by the FPN.
        out_features (list): The names of the output features produced by the FPN.
        out_feature_channels (dict): A dictionary mapping the names of the output features to
            their corresponding channel dimensions.
        out_feature_strides (dict): A dictionary mapping the names of the output features to
            their corresponding stride values.
        size_divisibility (int): The size divisibility of the FPN's output features.
        padding_constraints (dict): A dictionary specifying the padding constraints for the FPN.
        square_pad (int): The size of the square padding applied to the FPN's output features.
        fuse_type (str): The type of fusion operation used when combining features.

    Methods:
        __init__:
            Initializes the FPN module with the provided parameters.
        output_shape:
            Returns a dictionary containing the output shape specifications for each output feature.
        construct:
            Constructs the FPN network by passing the input data through the bottom-up network and
            performing lateral connections and fusion operations to generate the output features.
    """
    def __init__(self,
                 bottom_up,
                 in_features,
                 out_channels,
                 norm="",
                 top_block=None,
                 fuse_type="sum",
                 square_pad=0):
        """
        __init__

        Initializes the FPN (Feature Pyramid Network) module.

        Args:
            self: FPN instance
                The FPN instance to initialize.
            bottom_up: object
                The bottom-up network or backbone network that generates the input feature maps.
            in_features: list
                List of strings representing the names of the input feature maps to be used for the FPN.
            out_channels: int
                The number of output channels for the FPN feature maps.
            norm: str, optional
                The normalization type to be applied to the FPN feature maps. Default is an empty string.
            top_block: object, optional
                The top block to be applied to the FPN. Default is None.
            fuse_type: str, optional
                The type of fusion to be used for combining feature maps. Default is 'sum'.
            square_pad: int, optional
                The amount of padding to be applied to the feature maps. Default is 0.

        Returns:
            None.

        Raises:
            AssertionError:
                If the input features are not properly specified.
            AttributeError:
                If the attributes of the FPN instance cannot be set.
            ValueError:
                If the specified padding mode is invalid.
        """
        super(FPN, self).__init__()
        assert in_features, in_features

        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        lateral_convs = []
        output_convs = []

        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=True, pad_mode='valid')
            output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, has_bias=True, pad_mode='pad')
            stage = int(math.log2(strides[idx]))

            setattr(self, "fpn_lateral{}".format(stage), lateral_conv)
            setattr(self, "fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.top_block = top_block
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up

        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        """
        Returns the size divisibility of the object.

        Args:
            self (FPN): The FPN instance itself.

        Returns:
            None.

        Raises:
            None.
        """
        return self._size_divisibility

    @property
    def padding_constraints(self):
        """
        Returns the padding constraints for the FPN class.

        Args:
            self (FPN): An instance of the FPN class.

        Returns:
            None.

        Raises:
            None.
        """
        return {"square_size": self._square_pad}

    def output_shape(self):
        """
        Returns the output shape of the Feature Pyramid Network (FPN) for each feature level.

        Args:
            self (FPN): The instance of the FPN class.

        Returns:
            None.

        Raises:
            None.
        """
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def construct(self, x):
        """
        Constructs the Feature Pyramid Network (FPN) based on the provided input.

        Args:
            self: An instance of the FPN class.
            x: The input tensor of shape (batch_size, channels, height, width).

        Returns:
            None

        Raises:
            None

        This method constructs the FPN by performing the following steps:

        1. Extracts the bottom-up features using the 'bottom_up' function.
        2. Initializes an empty list 'results' to store the intermediate results.
        3. Retrieves the bottom-up feature corresponding to the last specified input feature.
        4. Applies the first lateral convolution to the bottom-up feature and appends the result to 'results'.
        5. Iterates over the remaining lateral and output convolutions.

            a. Retrieves the input feature for the current convolution from the 'bottom_up_features'.
            b. Resizes the previous feature map using nearest neighbor interpolation.
            c. Applies the lateral convolution to the input feature.
            d. Adds the lateral features and the resized top-down features.
            e. If the fusion type is 'avg', averages the resulting features.
            f. Inserts the output of the current convolution at the beginning of 'results'.
        6. If a 'top_block' is specified:

            a. Checks if the 'top_block.in_feature' is present in 'bottom_up_features'.
            b. If present, retrieves the corresponding feature; otherwise, retrieves it from 'results' using the index.
            c. Applies the 'top_block' to the 'top_block_in_feature' after converting it to 'ms.float16' datatype.
            d. Extends 'results' with the output of the 'top_block'.
        7. Asserts that the length of 'self._out_features' is equal to the length of 'results'.
        8. Returns a tuple containing the 'self._out_features' and the corresponding outputs from 'results'.
        """
        bottom_up_features = self.bottom_up(x)

        results = []
        bottom_up_feature = bottom_up_features.get(self.in_features[-1])
        prev_features = self.lateral_convs[0](bottom_up_feature)
        results.append(self.output_convs[0](prev_features))

        for idx, (lateral_conv, output_conv) in enumerate(zip(self.lateral_convs, self.output_convs)):
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                old_shape = list(prev_features.shape)[2:]
                new_size = tuple(2 * i for i in old_shape)
                top_down_features = ops.ResizeNearestNeighbor(size=new_size)(prev_features)
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))
        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature.astype(ms.float16)))

        assert len(self._out_features) == len(results)

        return tuple(list(zip(self._out_features, results)))
