import math
import os

import yaml
from addict import Dict

import mindspore as ms
from mindspore import nn, ops

from .resnet import ShapeSpec, build_resnet_backbone


def read_config():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(curr_dir, 'visual_backbone.yaml'), 'r') as file:
        data = yaml.safe_load(file)
        data = Dict(data)
    return data


def build_resnet_fpn_backbone(cfg):
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
    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def construct(self, x):
        return [ops.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class FPN(nn.Cell):
    def __init__(self,
                 bottom_up,
                 in_features,
                 out_channels,
                 norm="",
                 top_block=None,
                 fuse_type="sum",
                 square_pad=0):
        super(FPN, self).__init__()
        assert in_features, in_features

        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=use_bias)
            output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, has_bias=use_bias,
                                    pad_mode='pad')
            stage = int(math.log2(strides[idx]))
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = nn.CellList(lateral_convs[::-1])
        self.output_convs = nn.CellList(output_convs[::-1])

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
        return self._size_divisibility

    @property
    def padding_constraints(self):
        return {"square_size": self._square_pad}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def construct(self, x):
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
                new_size = tuple([2 * i for i in old_shape])
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

        return tuple([(f, res) for f, res in zip(self._out_features, results)])
