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


from typing import Dict, List

from ...types import FnaTileShapeType

# TODO: More combinations are possible for
# 2D and 3D (query tile does not have to be smaller
# than KV tile); but that behavior is untested,
# and IIRC was unstable.

_FNA_FORWARD_64x128_TILE_SIZES: Dict[int, List[FnaTileShapeType]] = {
    1: [
        ((64,), (128,)),
    ],
    2: [
        ((32, 2), (64, 2)),
        ((32, 2), (32, 4)),
        ((16, 4), (32, 4)),
        ((16, 4), (16, 8)),
        ((8, 8), (16, 8)),
        ((8, 8), (8, 16)),
        ((4, 16), (8, 16)),
        ((4, 16), (4, 32)),
        ((2, 32), (4, 32)),
        ((2, 32), (2, 64)),
    ],
    3: [
        ((16, 2, 2), (32, 2, 2)),
        ((16, 2, 2), (16, 4, 2)),
        ((16, 2, 2), (16, 2, 4)),
        ((8, 4, 2), (16, 4, 2)),
        ((8, 4, 2), (8, 8, 2)),
        ((8, 4, 2), (8, 4, 4)),
        ((8, 2, 4), (16, 2, 4)),
        ((8, 2, 4), (8, 4, 4)),
        ((8, 2, 4), (8, 2, 8)),
        ((4, 8, 2), (8, 8, 2)),
        ((4, 8, 2), (4, 16, 2)),
        ((4, 8, 2), (4, 8, 4)),
        ((4, 4, 4), (8, 4, 4)),
        ((4, 4, 4), (4, 8, 4)),
        ((4, 4, 4), (4, 4, 8)),
        ((4, 2, 8), (8, 2, 8)),
        ((4, 2, 8), (4, 4, 8)),
        ((4, 2, 8), (4, 2, 16)),
        ((2, 16, 2), (4, 16, 2)),
        ((2, 16, 2), (2, 32, 2)),
        ((2, 16, 2), (2, 16, 4)),
        ((2, 8, 4), (4, 8, 4)),
        ((2, 8, 4), (2, 16, 4)),
        ((2, 8, 4), (2, 8, 8)),
        ((2, 4, 8), (4, 4, 8)),
        ((2, 4, 8), (2, 8, 8)),
        ((2, 4, 8), (2, 4, 16)),
        ((2, 2, 16), (4, 2, 16)),
        ((2, 2, 16), (2, 4, 16)),
        ((2, 2, 16), (2, 2, 32)),
    ],
}
