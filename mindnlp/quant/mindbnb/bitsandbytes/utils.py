# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
'''
    utils
'''
import json
import shlex
import subprocess
from typing import Tuple

import mindspore
from mindspore import ops

from mindnlp.core import nn


def outlier_hook(module, input):
    assert isinstance(module, nn.Linear)
    tracer = OutlierTracer.get_instance()
    hvalue = tracer.get_hvalue(module.weight)
    if hvalue not in tracer.hvalue2outlier_idx:
        outlier_idx = find_outlier_dims(module.weight)
        tracer.outliers.append(outlier_idx)
        tracer.hvalues.append(hvalue)
        if len(tracer.outliers) > 1:
            # assign the current layer the outlier idx found from the weight
            # of the previous linear layer
            if tracer.outliers[-1].numel() > 0:
                assert tracer.outliers[-1].max() < module.weight.shape[1]
            tracer.hvalue2outlier_idx[hvalue] = tracer.outliers[-1]

        else:
            # first layer, we cannot use the weight for outlier detection
            # we follow a mixed approach:
            # (1) zscore test of std of hidden dimension
            # (2) magnitude > 6 test
            merged = input[0].view(-1, input[0].shape[-1])
            # (1) zscore test of std of hidden dimension
            outlier_idx = find_outlier_dims(merged, reduction_dim=1, zscore=3)
            # (2) magnitude > 6 test
            dims = (ops.abs([0]) > 6).sum(dim=list(range(len(input[0].shape) - 1)))
            outlier_idx2 = ops.nonzero(dims > 0)[0]
            outlier_idx = ops.cat([outlier_idx, outlier_idx2]).unique()
            tracer.hvalue2outlier_idx[hvalue] = outlier_idx
    else:
        for hook in tracer.hooks:
            hook.remove()


class OutlierTracer:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self, model):
        self.last_w = None
        self.current_outlier_dims = None
        self.hvalues = []
        self.outliers = []
        self.hvalue2outlier_idx = {}
        self.initialized = True
        self.hooks = []

        for n, m in model.named_modules():
            if isinstance(m, nn.Linear):
                self.hooks.append(m.register_forward_pre_hook(outlier_hook))

    def is_initialized(self):
        return getattr(self, "initialized", False)

    def get_hvalue(self, weight):
        return weight.data.storage().data_ptr()

    def get_outliers(self, weight):
        if not self.is_initialized():
            print("Outlier tracer is not initialized...")
            return None
        hvalue = self.get_hvalue(weight)
        if hvalue in self.hvalue2outlier_idx:
            return self.hvalue2outlier_idx[hvalue]
        else:
            return None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance


def find_outlier_dims(weight, reduction_dim=0, zscore=4.0, topk=None, rdm=False):
    if rdm:
        return ops.randint(
            0, weight.shape[1], size=(topk,)
        ).long()

    m = weight.mean(reduction_dim)
    mm = m.mean()
    mstd = m.std()
    zm = (m - mm) / mstd

    std = weight.std(reduction_dim)
    stdm = std.mean()
    stdstd = std.std()

    zstd = (std - stdm) / stdstd

    if topk is not None:
        val, idx = ops.topk(std.abs(), k=topk, dim=0)
    else:
        idx = ops.nonzero(zstd > zscore)[0]

    return idx


def execute_and_return(command_string: str) -> Tuple[str, str]:
    def _decode(subprocess_err_out_tuple):
        return tuple(
            to_decode.decode("UTF-8").strip() for to_decode in subprocess_err_out_tuple
        )

    def execute_and_return_decoded_std_streams(command_string):
        return _decode(
            subprocess.Popen(
                shlex.split(command_string),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).communicate(),
        )

    std_out, std_err = execute_and_return_decoded_std_streams(command_string)
    return std_out, std_err


def replace_linear(
    model,
    linear_replacement,
    skip_modules=("lm_head",),
    copy_weights=False,
    post_processing_function=None,
):

    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear(
                module,
                linear_replacement,
                skip_modules,
                copy_weights,
                post_processing_function,
            )

        if isinstance(module, nn.Linear) and name not in skip_modules:
            old_module = model._modules[name]
            model._modules[name] = linear_replacement(
                module.in_features,
                module.out_features,
                module.bias is not None,
            )
            if copy_weights:
                model._modules[name].weight = old_module.weight
                model._modules[name].bias = old_module.bias

            if post_processing_function is not None:
                func = getattr(module, post_processing_function, None)
                if func is not None:
                    func(module)
    return model


def pack_dict_to_tensor(source_dict):
    json_str = json.dumps(source_dict)
    json_bytes = json_str.encode("utf-8")
    tensor_data = mindspore.tensor(list(json_bytes), dtype=mindspore.uint8)

    return tensor_data


def unpack_tensor_to_dict(tensor_data):
    json_bytes = bytes(tensor_data.cpu().numpy())
    json_str = json_bytes.decode("utf-8")
    unpacked_dict = json.loads(json_str)

    return unpacked_dict


LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING = {
    "row": 0,
    "col32": 1,
    "col_turing": 2,
    "col_ampere": 3,
}
INVERSE_LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING = {
    val: name for (name, val) in LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING.items()
}
