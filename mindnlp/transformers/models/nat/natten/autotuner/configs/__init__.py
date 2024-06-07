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

# FNA/FMHA forward supports 64x64 and 32x128 GEMM configs in all
# use cases. Some architectures (SM80 and SM90 )have more shared
# memory so they can handle 64x128 GEMMs.

from .fna_backward_128x128 import _FNA_BACKWARD_128x128_TILE_SIZES
from .fna_backward_128x64 import _FNA_BACKWARD_128x64_TILE_SIZES
from .fna_backward_64x64 import _FNA_BACKWARD_64x64_TILE_SIZES

# FNA/FMHA backward supports 64x64 GEMM configs in all
# use cases. Some architectures have more shared memory
# so they can handle 128x64 or 128x128 GEMMs, but that
# is also dependent on the GEMM K.

from .fna_forward_32x128 import _FNA_FORWARD_32x128_TILE_SIZES
from .fna_forward_64x128 import _FNA_FORWARD_64x128_TILE_SIZES
from .fna_forward_64x64 import _FNA_FORWARD_64x64_TILE_SIZES

__all__ = [
    "_FNA_BACKWARD_64x64_TILE_SIZES",
    "_FNA_BACKWARD_128x64_TILE_SIZES",
    "_FNA_BACKWARD_128x128_TILE_SIZES",
    "_FNA_FORWARD_32x128_TILE_SIZES",
    "_FNA_FORWARD_64x128_TILE_SIZES",
    "_FNA_FORWARD_64x64_TILE_SIZES",
]
