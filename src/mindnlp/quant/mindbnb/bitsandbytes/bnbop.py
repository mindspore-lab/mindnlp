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
    custom ops
'''
# pylint: disable=E0401
import mindspore
from mindspore import ops

# from mindspore.ops import custom_info_register, CustomRegOp, DataType

from bitsandbytes.lib import lib_path


cget_col_row_stats = ops.Custom(
    f"{lib_path}:custom_cget_col_row_stats",
    out_shape=([1]),
    out_dtype=mindspore.int32,
    func_type="aot",
)

cdouble_rowcol_quant = ops.Custom(
    f"{lib_path}:custom_cdouble_rowcol_quant",
    out_shape=([1]),
    out_dtype=mindspore.int32,
    func_type="aot",
)

ctransform_row2col32T = ops.Custom(
    f"{lib_path}:custom_ctransform_row2col32T",
    out_shape=([1]),
    out_dtype=mindspore.int32,
    func_type="aot",
)

ctransform_row2col32 = ops.Custom(
    f"{lib_path}:custom_ctransform_row2col32",
    out_shape=([1]),
    out_dtype=mindspore.int32,
    func_type="aot",
)

ctransform_row2turingT = ops.Custom(
    f"{lib_path}:custom_ctransform_row2turingT",
    out_shape=([1]),
    out_dtype=mindspore.int32,
    func_type="aot",
)

ctransform_row2turing = ops.Custom(
    f"{lib_path}:custom_ctransform_row2turing",
    out_shape=([1]),
    out_dtype=mindspore.int32,
    func_type="aot",
)

ctransform_row2ampereT = ops.Custom(
    f"{lib_path}:custom_ctransform_row2ampereT",
    out_shape=([1]),
    out_dtype=mindspore.int32,
    func_type="aot",
)

ctransform_row2ampere = ops.Custom(
    f"{lib_path}:custom_ctransform_row2ampere",
    out_shape=([1]),
    out_dtype=mindspore.int32,
    func_type="aot",
)

cextractOutliers_turing = ops.Custom(
    f"{lib_path}:custom_cextractOutliers_turing",
    out_shape=([1]),
    out_dtype=mindspore.int32,
    func_type="aot",
)

cextractOutliers_ampere = ops.Custom(
    f"{lib_path}:custom_cextractOutliers_ampere",
    out_shape=([1]),
    out_dtype=mindspore.int32,
    func_type="aot",
)

cigemmlt_turing_32 = ops.Custom(
    f"{lib_path}:custom_cigemmlt_turing_32",
    out_shape=([1]),
    out_dtype=mindspore.int32,
    func_type="aot",
)

cigemmlt_turing_8 = ops.Custom(
    f"{lib_path}:custom_cigemmlt_turing_64",
    out_shape=([1]),
    out_dtype=mindspore.int32,
    func_type="aot",
)

cigemmlt_ampere_32 = ops.Custom(
    f"{lib_path}:custom_cigemmlt_ampere_32",
    out_shape=([1]),
    out_dtype=mindspore.int32,
    func_type="aot",
)

cigemmlt_ampere_8 = ops.Custom(
    f"{lib_path}:custom_cigemmlt_ampere_64",
    out_shape=([1]),
    out_dtype=mindspore.int32,
    func_type="aot",
)

cdequant_mm_int32_fp16 = ops.Custom(
    f"{lib_path}:custom_cdequant_mm_int32_fp16",
    out_shape=([1]),
    out_dtype=mindspore.int32,
    func_type="aot",
)
