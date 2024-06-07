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

from .context import (
    are_deterministic_algorithms_enabled,
    disable_autotuner,
    disable_gemm_na,
    disable_tf32,
    disable_tiled_na,
    enable_gemm_na,
    enable_tf32,
    enable_tiled_na,
    get_memory_usage_preference,
    has_bfloat,
    has_cuda,
    has_fna,
    has_fp32_gemm,
    has_fp64_gemm,
    has_fused_na,
    has_gemm,
    has_half,
    has_tf32_gemm,
    is_autotuner_enabled,
    is_autotuner_enabled_for_backward,
    is_autotuner_enabled_for_forward,
    is_autotuner_thorough_for_backward,
    is_autotuner_thorough_for_forward,
    is_fna_enabled,
    is_fused_na_enabled,
    is_gemm_na_enabled,
    is_kv_parallelism_in_fused_na_enabled,
    is_memory_usage_default,
    is_memory_usage_strict,
    is_memory_usage_unrestricted,
    is_tf32_in_gemm_na_enabled,
    is_tiled_na_enabled,
    set_memory_usage_preference,
    use_autotuner,
    use_deterministic_algorithms,
    use_fna,
    use_fused_na,
    use_gemm_na,
    use_kv_parallelism_in_fused_na,
    use_tf32_in_gemm_na,
    use_tiled_na,
)
from .na1d import NeighborhoodAttention1D
from .na2d import NeighborhoodAttention2D
from .na3d import NeighborhoodAttention3D

__all__ = [
    "NeighborhoodAttention1D",
    "NeighborhoodAttention2D",
    "NeighborhoodAttention3D",
    "are_deterministic_algorithms_enabled",
    "use_deterministic_algorithms",
    "use_kv_parallelism_in_fused_na",
    "is_kv_parallelism_in_fused_na_enabled",
    "use_fna",
    "is_fna_enabled",
    "use_fused_na",
    "is_fused_na_enabled",
    "use_autotuner",
    "disable_autotuner",
    "is_autotuner_enabled",
    "is_autotuner_enabled_for_forward",
    "is_autotuner_enabled_for_backward",
    "is_autotuner_thorough_for_forward",
    "is_autotuner_thorough_for_backward",
    "has_cuda",
    "has_half",
    "has_bfloat",
    "has_gemm",
    "has_fna",
    "has_fused_na",
    "has_tf32_gemm",
    "has_fp32_gemm",
    "has_fp64_gemm",
    "use_tf32_in_gemm_na",
    "use_tiled_na",
    "use_gemm_na",
    "is_tf32_in_gemm_na_enabled",
    "is_tiled_na_enabled",
    "is_gemm_na_enabled",
    "set_memory_usage_preference",
    "get_memory_usage_preference",
    "is_memory_usage_default",
    "is_memory_usage_strict",
    "is_memory_usage_unrestricted",
    # To be deprecated
    "enable_tf32",
    "disable_tf32",
    "enable_gemm_na",
    "disable_gemm_na",
    "enable_tiled_na",
    "disable_tiled_na",
]

__version__ = "0.17.1"
