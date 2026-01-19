import sys
import pytest
import mindspore

from mindnlp.core import ops
from mindnlp.utils.testing_utils import require_mindspore_gpu
from mindnlp.utils import is_triton_available

if is_triton_available():
    from mindnlp.triton import MSDriver
    import triton
    import triton.language as tl



@pytest.mark.skipif(sys.platform != "linux", reason="Test only runs on Linux")
@require_mindspore_gpu
def test_add():

    @triton.jit
    def add_kernel(x_ptr,  # *Pointer* to first input vector.
                y_ptr,  # *Pointer* to second input vector.
                output_ptr,  # *Pointer* to output vector.
                n_elements,  # Size of the vector.
                BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                # NOTE: `constexpr` so it can be used as a shape value.
                ):

        pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    def add(x: mindspore.Tensor, y: mindspore.Tensor):
        # We need to preallocate the output.
        output = ops.empty_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=512)

        return output

    mindspore.set_context(device_target='GPU')
    triton.runtime.driver.set_active(MSDriver())

    size = 98432
    x = mindspore.ops.ones((size,), dtype=mindspore.float32)
    y = mindspore.ops.ones((size,), dtype=mindspore.float32)
    z = add(x, y)
    print(z)
