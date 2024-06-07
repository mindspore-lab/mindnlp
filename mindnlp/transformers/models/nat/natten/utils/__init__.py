#################################################################################################
# Copyright (c) 2022-2024 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################

from .checks import (
    check_all_args,
    check_backward_tiling_config,
    check_causal_arg,
    check_dilation_arg,
    check_kernel_size_arg,
    check_tiling_config,
    get_num_na_weights,
)
from .misc import get_device_cc
from .tensor import (
    check_additional_keys,
    check_additional_values,
    make_attn_tensor_from_input,
)

__all__ = [
    "check_additional_keys",
    "check_additional_values",
    "check_all_args",
    "check_causal_arg",
    "check_dilation_arg",
    "check_kernel_size_arg",
    "check_tiling_config",
    "check_backward_tiling_config",
    "get_num_na_weights",
    "get_device_cc",
    "make_attn_tensor_from_input",
]
