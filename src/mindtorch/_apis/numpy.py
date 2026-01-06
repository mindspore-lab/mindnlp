import numbers
import numpy as np
import scipy
import mindspore as ms
from mindspore import ops
from mindspore import _Function as Function
import mindtorch

__all__ = []

class EmptyFunction(Function):
    @staticmethod
    def forward(ctx, size, dtype):
        result = ms.Tensor.from_numpy(np.empty(size, mindtorch.dtype2np[dtype]))
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # empty is a creation function, no backward needed
        return None, None

def empty(size, dtype):
    return EmptyFunction.apply(size, dtype)


class NewEmptyFunction(Function):
    @staticmethod
    def forward(ctx, input, size, dtype, device):
        # Use input's dtype if dtype is None
        if dtype is None:
            dtype = input.dtype
        
        # Create empty tensor with the specified size and dtype
        result = ms.Tensor.from_numpy(np.empty(size, mindtorch.dtype2np[dtype]))
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # new_empty is a creation function, no backward needed
        return None, None, None, None

def new_empty(input, size, dtype, device):
    return NewEmptyFunction.apply(input, size, dtype, device)


class OnesFunction(Function):
    @staticmethod
    def forward(ctx, size, dtype):
        if dtype is None:
            dtype = mindtorch.float32
        result = ms.Tensor.from_numpy(np.ones(size, mindtorch.dtype2np[dtype]))
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None

def ones(size, dtype):
    return OnesFunction.apply(size, dtype)


class ZerosFunction(Function):
    @staticmethod
    def forward(ctx, size, dtype):
        result = ms.Tensor.from_numpy(np.zeros(size, mindtorch.dtype2np[dtype]))
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None

def zeros(size, dtype):
    return ZerosFunction.apply(size, dtype)


class ArangeFunction(Function):
    @staticmethod
    def forward(ctx, start, end, step, dtype):
        if step == 0:
            raise ValueError("arange step must not be zero")
        result = ms.Tensor.from_numpy(np.arange(start, end, step, mindtorch.dtype2np[dtype]))
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None

def arange(start, end, step, dtype):
    return ArangeFunction.apply(start, end, step, dtype)


class LinspaceFunction(Function):
    @staticmethod
    def forward(ctx, start, end, steps, dtype):
        result = ms.Tensor.from_numpy(np.linspace(start, end, steps, dtype=mindtorch.dtype2np[dtype]))
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None

def linspace(start, end, steps, dtype):
    return LinspaceFunction.apply(start, end, steps, dtype)


class DivFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        if isinstance(input, numbers.Number):
            other_np = other.asnumpy()
            input_np = np.array(input, dtype=other_np.dtype)
        elif isinstance(other, numbers.Number):
            input_np = input.asnumpy()
            if input_np.dtype == np.int64:
                input_np = input_np.astype(np.int32)
            other_np = np.array(other, dtype=input_np.dtype)
        else:
            input_np = input.asnumpy()
            other_np = other.asnumpy()

        out = np.divide(input_np, other_np)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input if not isinstance(input, numbers.Number) else None,
                            other if not isinstance(other, numbers.Number) else None)
        ctx.is_input_number = isinstance(input, numbers.Number)
        ctx.is_other_number = isinstance(other, numbers.Number)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_tensors
        grad_input = None
        grad_other = None
        
        if ctx.needs_input_grad[0] and not ctx.is_input_number and input is not None:
            if ctx.is_other_number:
                other_val = ctx.saved_tensors[1] if len(ctx.saved_tensors) > 1 and ctx.saved_tensors[1] is not None else 1.0
                grad_input = grad_output.asnumpy() / other_val
        else:
            grad_input = grad_output.asnumpy() / other.asnumpy()
        if not isinstance(grad_input, np.ndarray):
            grad_input = np.array(grad_input)
        grad_input = ms.Tensor.from_numpy(grad_input)
        
        if ctx.needs_input_grad[1] and not ctx.is_other_number and other is not None:
            if ctx.is_input_number:
                input_val = ctx.saved_tensors[0] if len(ctx.saved_tensors) > 0 and ctx.saved_tensors[0] is not None else 0.0
                grad_other = -grad_output.asnumpy() * input_val / (other.asnumpy() ** 2)
            else:
                grad_other = -grad_output.asnumpy() * input.asnumpy() / (other.asnumpy() ** 2)
            if not isinstance(grad_other, np.ndarray):
                grad_other = np.array(grad_other)
            grad_other = ms.Tensor.from_numpy(grad_other)
        
        return grad_input, grad_other

def div(input, other):
    return DivFunction.apply(input, other)


class PowScalarTensorFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        other_np = other.asnumpy()
        out = np.power(input, other_np)
        if out.dtype == np.float64:
            out = out.astype(np.float32)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(other)
        ctx.input = input
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        other, = ctx.saved_tensors
        grad_other = None
        if ctx.needs_input_grad[1]:
            # d/dx (a^x) = a^x * ln(a)
            output = grad_output.asnumpy() * np.power(ctx.input, other.asnumpy()) * np.log(ctx.input)
            if output.dtype == np.float64:
                output = output.astype(np.float32)
            grad_other = ms.Tensor.from_numpy(output)
        return None, grad_other

def pow_scalar_tensor(input, other):
    return PowScalarTensorFunction.apply(input, other)


class MulFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        if not isinstance(input, numbers.Number):
            input_np = input.asnumpy()
        else:
            input_np = input
        if not isinstance(other, numbers.Number):
            other_np = other.asnumpy()
        else:
            other_np = other
        
        out = input_np * other_np
        if out.dtype == np.float64:
            out = out.astype(np.float32)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input if not isinstance(input, numbers.Number) else None,
                            other if not isinstance(other, numbers.Number) else None)
        ctx.is_input_number = isinstance(input, numbers.Number)
        ctx.is_other_number = isinstance(other, numbers.Number)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_tensors
        grad_input = None
        grad_other = None
        
        if ctx.needs_input_grad[0] and not ctx.is_input_number and input is not None:
            if ctx.is_other_number:
                other_val = ctx.saved_tensors[1] if len(ctx.saved_tensors) > 1 and ctx.saved_tensors[1] is not None else None
                if other_val is None:
                    grad_input = grad_output.asnumpy()
                else:
                    grad_input = grad_output.asnumpy() * other_val
            else:
                grad_input = grad_output.asnumpy() * other.asnumpy()
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        
        if ctx.needs_input_grad[1] and not ctx.is_other_number and other is not None:
            if ctx.is_input_number:
                input_val = ctx.saved_tensors[0] if len(ctx.saved_tensors) > 0 and ctx.saved_tensors[0] is not None else None
                if input_val is None:
                    grad_other = grad_output.asnumpy()
                else:
                    grad_other = grad_output.asnumpy() * input_val
            else:
                grad_other = grad_output.asnumpy() * input.asnumpy()
            if not isinstance(grad_other, np.ndarray):
                grad_other = np.array(grad_other)
            grad_other = ms.Tensor.from_numpy(grad_other)
        
        return grad_input, grad_other

def mul(input, other):
    return MulFunction.apply(input, other)


class SubFunction(Function):
    @staticmethod
    def forward(ctx, input, other, alpha=1):
        if not isinstance(input, numbers.Number):
            input_np = input.asnumpy()
        else:
            input_np = input
        if not isinstance(other, numbers.Number):
            other_np = other.asnumpy()
        else:
            other_np = other
        if alpha == 1:
            out = np.subtract(input_np, other_np)
        else:
            out = np.subtract(input_np, other_np * alpha)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input if not isinstance(input, numbers.Number) else None,
                            other if not isinstance(other, numbers.Number) else None)
        ctx.alpha = alpha
        ctx.is_input_number = isinstance(input, numbers.Number)
        ctx.is_other_number = isinstance(other, numbers.Number)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_tensors
        grad_input = None
        grad_other = None
        
        if ctx.needs_input_grad[0] and not ctx.is_input_number:
            grad_input = grad_output.asnumpy()
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        
        if ctx.needs_input_grad[1] and not ctx.is_other_number:
            grad_other = -grad_output.asnumpy() * ctx.alpha
            if not isinstance(grad_other, np.ndarray):
                grad_other = np.array(grad_other)
            grad_other = ms.Tensor.from_numpy(grad_other)
        
        return grad_input, grad_other, None

def sub(input, other, alpha=1):
    return SubFunction.apply(input, other, alpha)

class ClampScalarFunction(Function):
    @staticmethod
    def forward(ctx, input, min_val, max_val):
        out = np.clip(input.asnumpy(), min_val, max_val)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.min_val = min_val
        ctx.max_val = max_val
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Gradient is 0 where input is clamped, 1 otherwise
            input_np = input.asnumpy()
            grad_input = grad_output.asnumpy() * ((input_np > ctx.min_val) & (input_np < ctx.max_val))
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input, None, None

def clamp_scalar(input, min, max):
    return ClampScalarFunction.apply(input, min, max)


class ReluFunction(Function):
    @staticmethod
    def forward(ctx, input):
        input_np = input.numpy()
        out = np.maximum(0, input_np)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            input_np = input.numpy()
            grad_output_np = grad_output.numpy()
            # ReLU backward: grad = grad_output if input > 0, else 0
            grad_input_np = grad_output_np * (input_np > 0).astype(grad_output_np.dtype)
            if not isinstance(grad_input_np, np.ndarray):
                grad_input_np = np.array(grad_input_np)
            grad_input = ms.Tensor.from_numpy(grad_input_np)
        return grad_input

def relu(input):
    return ReluFunction.apply(input)


class AddFunction(Function):
    @staticmethod
    def forward(ctx, input, other, alpha=1):
        if not isinstance(input, numbers.Number):
            input_np = input.asnumpy()
        else:
            input_np = input
        if not isinstance(other, numbers.Number):
            other_np = other.asnumpy()
        else:
            other_np = other
        # Apply alpha scaling: input + alpha * other
        if alpha != 1:
            other_np = other_np * alpha
        out = np.add(input_np, other_np)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input if not isinstance(input, numbers.Number) else None,
                            other if not isinstance(other, numbers.Number) else None)
        ctx.alpha = alpha
        ctx.is_input_number = isinstance(input, numbers.Number)
        ctx.is_other_number = isinstance(other, numbers.Number)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_tensors
        grad_input = None
        grad_other = None
        
        if ctx.needs_input_grad[0] and not ctx.is_input_number:
            grad_input = grad_output.asnumpy()
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        
        if ctx.needs_input_grad[1] and not ctx.is_other_number:
            grad_other = grad_output.asnumpy() * ctx.alpha
            if not isinstance(grad_other, np.ndarray):
                grad_other = np.array(grad_other)
            grad_other = ms.Tensor.from_numpy(grad_other)
        
        return grad_input, grad_other, None

def add(input, other, alpha=1):
    return AddFunction.apply(input, other, alpha)


class CastFunction(Function):
    @staticmethod
    def forward(ctx, input, dtype):
        if input.dtype == dtype:
            return input
        if hasattr(dtype, 'dtype'):
            dtype = dtype.dtype
        out = input.asnumpy().astype(mindtorch.dtype2np[dtype])
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.original_dtype = input.dtype
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Cast gradient back to original dtype
            grad_input = grad_output.asnumpy().astype(mindtorch.dtype2np[ctx.original_dtype])
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input, None

def cast(input, dtype):
    return CastFunction.apply(input, dtype)


class GetitemFunction(Function):
    @staticmethod
    def forward(ctx, input, slice):
        # Track which indices are tensors to preserve dimensions
        tensor_indices_info = []
        # Check if slice contains MindSpore tensors (from _tensor.py conversion)
        if isinstance(slice, tuple):
            new_slice = ()
            for i, s in enumerate(slice):
                if isinstance(s, mindtorch.Tensor):
                    s_np = s.asnumpy()
                    tensor_indices_info.append((i, s_np.shape, s_np.ndim, s_np))
                    new_slice += (s_np,)
                elif isinstance(s, ms.Tensor):
                    # Handle MindSpore tensor (converted from mindtorch.Tensor in _tensor.py)
                    s_np = s.asnumpy()
                    tensor_indices_info.append((i, s_np.shape, s_np.ndim, s_np))
                    new_slice += (s_np,)
                else:
                    new_slice += (s,)
        else:
            if isinstance(slice, mindtorch.Tensor):
                s_np = slice.asnumpy()
                tensor_indices_info.append((0, s_np.shape, s_np.ndim, s_np))
                new_slice = s_np
            elif isinstance(slice, ms.Tensor):
                s_np = slice.asnumpy()
                tensor_indices_info.append((0, s_np.shape, s_np.ndim, s_np))
                new_slice = s_np
            else:
                new_slice = slice
        
        input_np = input.asnumpy() if hasattr(input, 'numpy') else input
        
        # Handle indexing, including 0-dimensional arrays
        try:
            out = input_np[new_slice]
        except IndexError as e:
            # Handle "too many indices for array: array is 0-dimensional" error
            if "0-dimensional" in str(e):
                # For 0D arrays, only empty indexing is valid
                # If we get this error, the input might have become 0D during processing
                # Return the scalar value itself
                if input_np.ndim == 0:
                    out = input_np
                else:
                    # Re-raise if it's not a 0D issue
                    raise
            else:
                raise
        
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        
        # PyTorch behavior: when using tensor indexing, the result should preserve
        # the shape of the index tensor in the output dimensions
        # numpy already handles this correctly for array indexing
        # The issue might be elsewhere in the pipeline
        
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.slice = slice
        ctx.input_shape = input.shape
        ctx.tensor_indices_info = tensor_indices_info
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Create zeros with input shape and scatter gradient
            grad_input = np.zeros(ctx.input_shape, dtype=grad_output.asnumpy().dtype)
            # Scatter gradient back to original positions
            if isinstance(ctx.slice, tuple):
                new_slice = ()
                for s in ctx.slice:
                    if isinstance(s, mindtorch.Tensor):
                        s = s.asnumpy()
                    new_slice += (s,)
            else:
                new_slice = ctx.slice
            grad_input[new_slice] = grad_output.asnumpy()
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input, None

def getitem(input, slice):
    return GetitemFunction.apply(input, slice)


def setitem(input, slice, value):
    out = input.asnumpy()
    out[slice] = value
    return input


class ContiguousFunction(Function):
    @staticmethod
    def forward(ctx, input):
        # Contiguous just returns the input (no-op in numpy)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        # Pass through gradient
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.asnumpy()
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def contiguous(input):
    return ContiguousFunction.apply(input)


class ReshapeFunction(Function):
    @staticmethod
    def forward(ctx, input, shape):
        out = np.reshape(input.asnumpy(), shape)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.input_shape = input.shape
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Reshape gradient back to input shape
            grad_input = np.reshape(grad_output.asnumpy(), ctx.input_shape)
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input, None

def reshape(input, shape):
    return ReshapeFunction.apply(input, shape)


class BitwiseAndScalarFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        out = np.bitwise_and(input.asnumpy(), other)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Bitwise operations are not differentiable
        return None, None

def bitwise_and_scalar(input, other):
    return BitwiseAndScalarFunction.apply(input, other)


class BitwiseAndTensorFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        out = np.bitwise_and(input.asnumpy(), other.asnumpy())
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Bitwise operations are not differentiable
        return None, None

def bitwise_and_tensor(input, other):
    return BitwiseAndTensorFunction.apply(input, other)


class BitwiseOrScalarFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        out = np.bitwise_or(input.asnumpy(), other)
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Bitwise operations are not differentiable
        return None, None

def bitwise_or_scalar(input, other):
    return BitwiseOrScalarFunction.apply(input, other)


class BitwiseOrTensorFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        out = np.bitwise_or(input.asnumpy(), other.asnumpy())
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Bitwise operations are not differentiable
        return None, None

def bitwise_or_tensor(input, other):
    return BitwiseOrTensorFunction.apply(input, other)


class RightShiftFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        out = np.right_shift(input.asnumpy(), other)
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Bitwise operations are not differentiable
        return None, None

def right_shift(input, other):
    return RightShiftFunction.apply(input, other)


class TransposeExtViewFunction(Function):
    @staticmethod
    def forward(ctx, input, dim0, dim1):
        out = np.swapaxes(input.asnumpy(), dim0, dim1)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.dim0 = dim0
        ctx.dim1 = dim1
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Swap back
            grad_input = np.swapaxes(grad_output.asnumpy(), ctx.dim0, ctx.dim1)
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input, None, None

def transpose_ext_view(input, dim0, dim1):
    return TransposeExtViewFunction.apply(input, dim0, dim1)


class ExpandDimsFunction(Function):
    @staticmethod
    def forward(ctx, input, dim):
        out = np.expand_dims(input.asnumpy(), dim)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.dim = dim
        ctx.input_shape = input.shape
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Squeeze the dimension that was expanded
            grad_input = np.squeeze(grad_output.asnumpy(), ctx.dim)
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def expand_dims(input, dim):
    return ExpandDimsFunction.apply(input, dim)


class ExpandDimsViewFunction(Function):
    @staticmethod
    def forward(ctx, input, dim):
        out = np.expand_dims(input.asnumpy(), dim)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.dim = dim
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = np.squeeze(grad_output.asnumpy(), ctx.dim)
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input, None

def expand_dims_view(input, dim):
    return ExpandDimsViewFunction.apply(input, dim)


class EqualFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        # equal returns a scalar bool tensor indicating if two tensors are equal
        # First check shape
        if not isinstance(input, numbers.Number):
            input_np = input.asnumpy()
            input_shape = input.shape
            input_dtype = input.dtype
        else:
            input_np = np.array(input)
            input_shape = input_np.shape
            input_dtype = type(input).__name__
        
        if not isinstance(other, numbers.Number):
            other_np = other.asnumpy()
            other_shape = other.shape
            other_dtype = other.dtype
        else:
            other_np = np.array(other)
            other_shape = other_np.shape
            other_dtype = type(other).__name__
        
        # Check shape
        if input_shape != other_shape:
            out = np.array(False, dtype=bool)
            result = ms.Tensor.from_numpy(out)
            return result
        
        # Check dtype (convert to same dtype for comparison)
        # Use eq for element-wise comparison
        eq_result = np.equal(input_np, other_np)
        
        # Check if all elements are equal
        if eq_result.size == 0:
            # Empty tensors are considered equal
            out = np.array(True, dtype=bool)
        else:
            out = np.array(np.all(eq_result), dtype=bool)
        
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Comparison operations don't need gradients
        return None, None

def equal(input, other):
    return EqualFunction.apply(input, other)


class EqFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        if not isinstance(input, numbers.Number):
            input_np = input.asnumpy()
        else:
            input_np = input
        if not isinstance(other, numbers.Number):
            other_np = other.asnumpy()
        else:
            other_np = other
        out = np.equal(input_np, other_np)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Comparison operations don't need gradients
        return None, None

def eq(input, other):
    return EqFunction.apply(input, other)


class ReduceAllFunction(Function):
    @staticmethod
    def forward(ctx, input, dim, keepdim):
        out = np.all(input.asnumpy(), axis=dim, keepdims=keepdim)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Logical operations don't need gradients
        return None, None, None

def reduce_all(input, dim, keepdim):
    return ReduceAllFunction.apply(input, dim, keepdim)

class ReduceAnyFunction(Function):
    @staticmethod
    def forward(ctx, input, dim, keepdim):
        out = np.any(input.asnumpy(), axis=dim, keepdims=keepdim)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Logical operations don't need gradients
        return None, None, None

def reduce_any(input, dim, keepdim):
    return ReduceAnyFunction.apply(input, dim, keepdim)



class SumFunction(Function):
    @staticmethod
    def forward(ctx, input, dim, keepdim, dtype):
        # Handle case where input might be a tuple (shouldn't happen, but be safe)
        input_np = input.asnumpy()
        if dtype is not None:
            dtype_np = mindtorch.dtype2np[dtype]
        else:
            dtype_np = None
        out = np.sum(input_np, dim, dtype=dtype_np, keepdims=keepdim)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.input_shape = input.shape
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        if len(ctx.saved_tensors) > 0:
            input = ctx.saved_tensors[0]
        else:
            input = None
        grad_input = None
        if ctx.needs_input_grad[0] and input is not None:
            grad_output_np = grad_output.asnumpy()
            # Expand dimensions if keepdim is False
            if not ctx.keepdim and ctx.dim is not None:
                # Expand the reduced dimension back
                if isinstance(ctx.dim, (list, tuple)):
                    for d in sorted(ctx.dim, reverse=True):
                        grad_output_np = np.expand_dims(grad_output_np, d)
                else:
                    grad_output_np = np.expand_dims(grad_output_np, ctx.dim)
            # Broadcast to input shape
            grad_input = np.broadcast_to(grad_output_np, ctx.input_shape)
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input, None, None, None

py_sum = sum
def sum(input, dim=None, keepdim=False, dtype=None):
    return SumFunction.apply(input, dim, keepdim, dtype)


class FullFunction(Function):
    @staticmethod
    def forward(ctx, size, fill_value):
        out = np.full(size, fill_value)
        if out.dtype == np.float64:
            out = out.astype(np.float32)
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None

def full(size, fill_value):
    return FullFunction.apply(size, fill_value)


class FillScalarFunction(Function):
    @staticmethod
    def forward(ctx, size, fill_value, dtype):
        if dtype is not None:
            dtype_np = mindtorch.dtype2np[dtype]
        else:
            dtype_np = None
        out = np.full(size, fill_value, dtype=dtype_np)
        if out.dtype == np.float64:
            out = out.astype(np.float32)
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None

def fill_scalar(size, fill_value, dtype=None):
    return FillScalarFunction.apply(size, fill_value, dtype)


class ZerosLikeFunction(Function):
    @staticmethod
    def forward(ctx, input, dtype=None):
        input_np = input.asnumpy()
        if dtype is not None:
            dtype_np = mindtorch.dtype2np[dtype]
        else:
            dtype_np = input_np.dtype
        out = np.zeros_like(input_np, dtype=dtype_np)
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None

def zeros_like(input, dtype=None):
    return ZerosLikeFunction.apply(input, dtype)

class FullLikeFunction(Function):
    @staticmethod
    def forward(ctx, input, fill_value, dtype=None):
        input_np = input.asnumpy()
        if dtype is not None:
            dtype_np = mindtorch.dtype2np[dtype]
        else:
            dtype_np = input_np.dtype
        out = np.full_like(input_np, fill_value, dtype=dtype_np)
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None

def full_like(input, fill_value, dtype=None):
    return FullLikeFunction.apply(input, fill_value, dtype)

class EmptyLikeFunction(Function):
    @staticmethod
    def forward(ctx, input, dtype=None):
        if dtype is None:
            dtype = input.dtype
        result = ms.Tensor.from_numpy(np.empty(input.shape, dtype=mindtorch.dtype2np[dtype]))
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # empty_like is not differentiable
        return None, None

def empty_like(input, dtype=None):
    return EmptyLikeFunction.apply(input, dtype)


def broadcast_to(input, shape):
    input_np = input.asnumpy()
    # 处理 shape 中为 -1 的情况，自动替换为输入的对应维度
    shape = list(shape)
    input_shape = input_np.shape
    ndim_diff = len(shape) - len(input_shape)
    if ndim_diff < 0:
        raise ValueError("broadcast_to shape length cannot be less than input shape length")
    for i, s in enumerate(shape):
        if s == -1:
            # 替换-1为输入的实际shape
            shape[i] = input_shape[i - ndim_diff]
    shape = tuple(shape)
    out = np.broadcast_to(input_np, shape)
    return ms.Tensor.from_numpy(out)


def uniform_real(size):
    out = np.random.rand(*size).astype(np.float32)
    return ms.Tensor.from_numpy(out)


def normal(shape):
    out = np.random.normal(0., 1., shape).astype(np.float32)
    return ms.Tensor.from_numpy(out)


def normal_float_float(mean, std, size, dtype, generator):
    """Normal distribution with scalar mean/std on numpy backend."""
    # generator is unused for numpy backend
    out = np.random.normal(mean, std, size).astype(mindtorch.dtype2np[dtype])
    return ms.Tensor.from_numpy(out)


def normal_tensor_tensor(mean, std, size, dtype, generator):
    """
    Generates random numbers from a normal distribution with tensor mean and std.
    
    Args:
        mean: Mean tensor (scalar or tensor)
        std: Standard deviation tensor (scalar or tensor)
        size: Output shape
        dtype: Output data type
        generator: Random number generator (optional, for compatibility)
    
    Returns:
        Tensor with random numbers from normal distribution
    """
    # Extract scalar values from mean and std tensors
    # Convert to numpy first to avoid .item() errors on multi-element tensors
    if hasattr(mean, 'asnumpy'):
        mean_np = mean.asnumpy()
    elif hasattr(mean, 'numpy'):
        mean_np = mean.numpy()
    else:
        mean_np = np.asarray(mean)
    
    # Extract scalar value: if 0D or single element, use item(); otherwise use first element
    if mean_np.ndim == 0:
        mean_val = mean_np.item()
    elif mean_np.size == 1:
        mean_val = mean_np.flat[0]
    else:
        # Multiple elements: use first element (or could broadcast, but item() expects single element)
        mean_val = float(mean_np.flat[0])
    
    if hasattr(std, 'asnumpy'):
        std_np = std.asnumpy()
    elif hasattr(std, 'numpy'):
        std_np = std.numpy()
    else:
        std_np = np.asarray(std)
    
    # Extract scalar value: if 0D or single element, use item(); otherwise use first element
    if std_np.ndim == 0:
        std_val = std_np.item()
    elif std_np.size == 1:
        std_val = std_np.flat[0]
    else:
        # Multiple elements: use first element
        std_val = float(std_np.flat[0])
    
    # Handle generator if provided
    if generator is not None:
        seed, offset = generator._step(12)  # pylint: disable=protected-access
        # Handle both scalar and tensor seed
        if hasattr(seed, 'asnumpy'):
            seed_np = seed.asnumpy()
            if seed_np.size == 1:
                np.random.seed(int(seed_np.item() if seed_np.ndim == 0 else seed_np.flat[0]))
            else:
                # Use first element if multiple
                np.random.seed(int(seed_np.flat[0]))
        elif hasattr(seed, 'item'):
            try:
                np.random.seed(seed.item())
            except ValueError:
                # Multiple elements, use first
                seed_np = seed.asnumpy() if hasattr(seed, 'asnumpy') else np.asarray(seed)
                np.random.seed(int(seed_np.flat[0]))
        else:
            np.random.seed(int(seed))
    
    # Generate random numbers from normal distribution
    out = np.random.normal(mean_val, std_val, size)#.astype(mindtorch.dtype2np[dtype])
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    out = out.astype(mindtorch.dtype2np[dtype])
    return ms.Tensor.from_numpy(out)


def upsample_bicubic2d(input, size=None, scale_factor=None, align_corners=False):
    """
    Bicubic upsample for CPU/numpy backend using scipy (order=3).
    Input: NCHW Tensor; Output: NCHW Tensor.
    """
    x_np = input.asnumpy()  # NCHW
    n, c, h, w = x_np.shape

    if size is None:
        if scale_factor is None:
            raise ValueError("Either size or scale_factor must be provided")
        if isinstance(scale_factor, (tuple, list)):
            if len(scale_factor) != 2:
                raise ValueError("scale_factor for 2d upsample must have length 2")
            scale_h, scale_w = scale_factor
        else:
            scale_h = scale_w = scale_factor
        new_h = max(1, int(round(h * scale_h)))
        new_w = max(1, int(round(w * scale_w)))
    else:
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise ValueError("size for 2d upsample must have length 2 (H, W)")
        new_h, new_w = size

    # zoom factors
    zoom_h = new_h / h
    zoom_w = new_w / w

    out_np = np.empty((n, c, new_h, new_w), dtype=x_np.dtype)
    for i in range(n):
        for j in range(c):
            out_np[i, j] = scipy.ndimage.zoom(
                x_np[i, j], zoom=(zoom_h, zoom_w), order=3, mode="reflect", prefilter=True
            )

    # For single-batch single-channel, squeeze to 2D to match PIL expectations in pipelines
    if n == 1 and c == 1:
        return ms.Tensor(out_np[0, 0])
    return ms.Tensor(out_np)

def pad_v3(input_x, padding, mode='constant', value=None):
    pad_op = ops.PadV3(mode=mode, paddings_contiguous=True).set_device('CPU')
    if input_x.dtype == mindtorch.bool:
        input_x = input_x.to(mindtorch.int32)
        value = int(value)
        out = pad_op(input_x, padding, value)
        return cast(out, mindtorch.bool)

    if isinstance(value, (float, int)):
        value = mindtorch.tensor(value, dtype=input_x.dtype)
    return pad_op(input_x, padding, value)


class PadFunction(Function):
    @staticmethod
    def forward(ctx, input, pad, mode='constant', value=None):
        input_np = input.asnumpy()
        
        # Convert PyTorch pad format to NumPy format
        # PyTorch: (pad_left, pad_right, pad_top, pad_bottom, ...) - from last dimension
        # NumPy: ((before_1, after_1), (before_2, after_2), ...) - from first dimension
        if isinstance(pad, (list, tuple)):
            pad = tuple(p if isinstance(p, int) else int(p) for p in pad)
        else:
            pad = (pad,)
        
        # Handle negative padding (slicing)
        new_pad = ()
        input_shape = list(input_np.shape)
        for idx, pad_v in enumerate(pad):
            if not isinstance(pad_v, int):
                pad_v = int(pad_v)
            if pad_v < 0:
                dim = input.ndim - 1 - idx // 2
                # Slice the input
                slices = [slice(None)] * input.ndim
                slices[dim] = slice(0, input_shape[dim] + pad_v)
                input_np = input_np[tuple(slices)]
                input_shape[dim] = input_shape[dim] + pad_v
                pad_v = 0
            new_pad += (pad_v,)
        
        # Check if all padding is zero
        if py_sum(new_pad) == 0:
            result = ms.Tensor.from_numpy(input_np)
            ctx.save_for_backward(input)
            ctx.pad = new_pad
            ctx.input_shape = input_shape
            return result
        
        # Convert to NumPy pad_width format
        # PyTorch pad format: (pad_left, pad_right, pad_top, pad_bottom, ...)
        #   - pairs are for dimensions from last to first
        #   - (pad_left, pad_right) is for last dimension
        #   - (pad_top, pad_bottom) is for second-to-last dimension
        # NumPy pad_width format: ((before_1, after_1), (before_2, after_2), ...)
        #   - pairs are for dimensions from first to last
        num_dims_to_pad = len(new_pad) // 2
        pad_width = []
        
        # Build pad_width from last dimension to first (matching PyTorch order)
        for i in range(num_dims_to_pad):
            # Get the pair for dimension (num_dims_to_pad - 1 - i) from the end
            left_idx = 2 * i
            right_idx = 2 * i + 1
            pad_width.append((new_pad[left_idx], new_pad[right_idx]))
        
        # Reverse to match NumPy's dimension order (first to last)
        pad_width = pad_width[::-1]
        
        # Pad the remaining dimensions (from the beginning) with (0, 0)
        for _ in range(input_np.ndim - num_dims_to_pad):
            pad_width.insert(0, (0, 0))
        
        # Handle different modes
        if mode == 'constant':
            if value is None:
                value = 0
            out = np.pad(input_np, pad_width, mode='constant', constant_values=value)
        elif mode == 'reflect':
            out = np.pad(input_np, pad_width, mode='reflect')
        elif mode == 'replicate':
            out = np.pad(input_np, pad_width, mode='edge')
        elif mode == 'circular':
            # NumPy doesn't have circular mode, use wrap
            out = np.pad(input_np, pad_width, mode='wrap')
        else:
            raise ValueError(f"Unsupported padding mode: {mode}")
        
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.pad = new_pad
        ctx.input_shape = input_shape
        ctx.pad_width = pad_width
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        
        if ctx.needs_input_grad[0]:
            grad_output_np = grad_output.asnumpy()
            
            # Unpad: extract the original region from the padded gradient
            # The pad_width tells us how much was added on each side
            slices = []
            for i, (before, after) in enumerate(ctx.pad_width):
                start = before
                end = grad_output_np.shape[i] - after if after > 0 else None
                slices.append(slice(start, end))
            
            grad_input = grad_output_np[tuple(slices)]
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)

            grad_input = ms.Tensor.from_numpy(grad_input)
        
        return grad_input, None, None, None

def pad(input, pad, mode='constant', value=None):
    return PadFunction.apply(input, pad, mode, value)


class CloneFunction(Function):
    @staticmethod
    def forward(ctx, input):
        # Create a copy of the input tensor
        out = np.copy(input.asnumpy())
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Clone's backward simply passes through the gradient
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.asnumpy()
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def clone(input):
    return CloneFunction.apply(input)


class ConcatFunction(Function):
    @staticmethod
    def forward(ctx, dim, *tensors):
        tensors_list = list(tensors)
        out = np.concatenate([t.asnumpy() for t in tensors_list], dim)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(*tensors_list)
        ctx.dim = dim
        ctx.tensor_shapes = [t.shape for t in tensors_list]
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        tensors = ctx.saved_tensors
        grad_inputs = []
        grad_output_np = grad_output.asnumpy()
        
        # Split gradient along the concatenation dimension
        start_idx = 0
        for i, shape in enumerate(ctx.tensor_shapes):
            end_idx = start_idx + shape[ctx.dim]
            slices = [slice(None)] * grad_output_np.ndim
            slices[ctx.dim] = slice(start_idx, end_idx)
            grad_input = grad_output_np[tuple(slices)]
            if ctx.needs_input_grad[i + 1]:  # +1 because first arg is dim
                grad_inputs.append(ms.Tensor.from_numpy(grad_input))
            else:
                grad_inputs.append(None)
            start_idx = end_idx
        
        return (None,) + tuple(grad_inputs)

def concat(tensors, dim=None, axis=None):
    # Support both dim and axis for compatibility
    if axis is not None:
        dim = axis
    if dim is None:
        dim = 0
    if isinstance(tensors, (list, tuple)):
        return ConcatFunction.apply(dim, *tensors)
    else:
        return ConcatFunction.apply(dim, tensors)


class AbsFunction(Function):
    @staticmethod
    def forward(ctx, input):
        out = np.abs(input.asnumpy())
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.asnumpy() * np.sign(input.asnumpy())
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def abs(input):
    return AbsFunction.apply(input)


class MeanFunction(Function):
    @staticmethod
    def forward(ctx, input, dim, keepdim, dtype):
        out = np.mean(input.asnumpy(), dim, keepdims=keepdim)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.input_shape = input.shape
        # Calculate the number of elements reduced
        if dim is None:
            ctx.num_elements = np.prod(input.shape)
        elif isinstance(dim, (list, tuple)):
            ctx.num_elements = np.prod([input.shape[d] for d in dim])
        else:
            ctx.num_elements = input.shape[dim]
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_output_np = grad_output.asnumpy()
            # Expand dimensions if keepdim is False
            if not ctx.keepdim and ctx.dim is not None:
                if isinstance(ctx.dim, (list, tuple)):
                    for d in sorted(ctx.dim, reverse=True):
                        grad_output_np = np.expand_dims(grad_output_np, d)
                else:
                    grad_output_np = np.expand_dims(grad_output_np, ctx.dim)
            # Broadcast and divide by number of elements
            grad_input = np.broadcast_to(grad_output_np, ctx.input_shape) / ctx.num_elements
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input, None, None, None

def mean_ext(input, dim, keepdim, dtype):
    return MeanFunction.apply(input, dim, keepdim, dtype)

def mean(input, dim=None, keepdim=False, dtype=None):
    return MeanFunction.apply(input, dim, keepdim, dtype)


class MatmulFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        out = np.matmul(input.asnumpy(), other.asnumpy())
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input, other)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_tensors
        grad_input = None
        grad_other = None
        
        if ctx.needs_input_grad[0]:
            # grad_input = grad_output @ other.T
            grad_input = np.matmul(grad_output.asnumpy(), other.asnumpy().T)
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        
        if ctx.needs_input_grad[1]:
            # grad_other = input.T @ grad_output
            grad_other = np.matmul(input.asnumpy().T, grad_output.asnumpy())
            if not isinstance(grad_other, np.ndarray):
                grad_other = np.array(grad_other)
            grad_other = ms.Tensor.from_numpy(grad_other)
        
        return grad_input, grad_other

def matmul_ext(input, other):
    return MatmulFunction.apply(input, other)


def matmul(input, other):
    return MatmulFunction.apply(input, other)


def bmm(input, other):
    """
    Performs batch matrix multiplication of matrices stored in input and other.
    
    Args:
        input: 3D tensor of shape (B, N, M)
        other: 3D tensor of shape (B, M, P)
    
    Returns:
        3D tensor of shape (B, N, P)
    """
    input_np = input.asnumpy()
    other_np = other.asnumpy()
    
    # Use einsum for batch matrix multiplication: 'bij,bjk->bik'
    # This is equivalent to: for each batch i, compute input[i] @ other[i]
    result_np = np.einsum('bij,bjk->bik', input_np, other_np)
    
    if not isinstance(result_np, np.ndarray):
        result_np = np.array(result_np)
    
    return ms.Tensor.from_numpy(result_np)


class MaxFunction(Function):
    @staticmethod
    def forward(ctx, input):
        out = np.max(input.asnumpy())
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        # max is not differentiable (argmax is needed for gradient)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # max is not differentiable without argmax
        return None

def max(input):
    return MaxFunction.apply(input)


class MinFunction(Function):
    @staticmethod
    def forward(ctx, input):
        out = np.min(input.asnumpy())
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        # min is not differentiable (argmin is needed for gradient)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # min is not differentiable without argmin
        return None

def min(input):
    return MinFunction.apply(input)



def randint(from_, to, shape, generator, dtype):
    out = np.random.randint(from_, to, shape, dtype=mindtorch.dtype2np[dtype])

    return ms.Tensor.from_numpy(out)


class IdentityFunction(Function):
    @staticmethod
    def forward(ctx, input):
        out = np.copy(input.asnumpy())
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Identity's backward passes through gradient
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.asnumpy()
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def identity(input):
    return IdentityFunction.apply(input)


# def non_zero()
def isclose(input, other, rtol, atol, equal_nan):
    out = np.isclose(input.asnumpy(), other.asnumpy(), rtol, atol, equal_nan)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return ms.Tensor.from_numpy(out)


def non_zero(input):
    out = np.nonzero(input.asnumpy())
    out = np.stack(out, 1)
    return ms.Tensor.from_numpy(out)


def non_zero_ext(input):
    """Return a tuple of 1-D index tensors, one for each dimension, similar to torch.nonzero(as_tuple=True)."""
    outs = np.nonzero(input.asnumpy())
    return tuple(ms.Tensor.from_numpy(o) for o in outs)


def count_nonzero(input, dims):
    """
    Counts the number of non-zero values in the input tensor along specified dimensions.
    
    Args:
        input: Input tensor
        dims: Dimension or dimensions to reduce. If None, counts all non-zero values.
    
    Returns:
        Tensor containing the count of non-zero values
    """
    input_np = input.asnumpy()
    
    # Count non-zero values
    if dims is None:
        # Count all non-zero values
        count = np.count_nonzero(input_np)
        out = np.array(count, dtype=np.int64)
    else:
        # Count along specified dimensions
        if isinstance(dims, (list, tuple)):
            # Multiple dimensions
            count = np.count_nonzero(input_np, axis=tuple(dims))
        else:
            # Single dimension
            count = np.count_nonzero(input_np, axis=dims)
        
        if not isinstance(count, np.ndarray):
            count = np.array(count)
        # Ensure output is int64
        out = count.astype(np.int64)
    
    return ms.Tensor.from_numpy(out)


class TileFunction(Function):
    @staticmethod
    def forward(ctx, input, dims):
        out = np.tile(input.asnumpy(), dims)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.dims = dims
        ctx.input_shape = input.shape
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Tile backward: sum over repeated dimensions
            grad_output_np = grad_output.asnumpy()
            # Calculate the shape after tiling
            output_shape = grad_output_np.shape
            # Reshape and sum over tiled dimensions
            # This is complex, so we use a simpler approach: reshape and sum
            input_shape = ctx.input_shape
            # Reshape to group tiled dimensions
            reshaped = grad_output_np.reshape(input_shape + tuple(d // s for d, s in zip(output_shape, input_shape)))
            # Sum over the tiled dimensions
            axes = tuple(range(len(input_shape), len(reshaped.shape)))
            grad_input = np.sum(reshaped, axis=axes)
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input, None

def tile(input, dims):
    return TileFunction.apply(input, dims)


class SqueezeFunction(Function):
    @staticmethod
    def forward(ctx, input, dim=None):
        input_np = input.asnumpy()
        # Check if dim is valid and size is 1
        if dim is not None:
            if isinstance(dim, int):
                # Check bounds
                if dim < 0:
                    dim = input.ndim + dim
                if dim < 0 or dim >= input.ndim:
                    return input
                if input.shape[dim] != 1:
                    return input
            # If dim is a tuple, check all dimensions
            elif isinstance(dim, (tuple, list)):
                for d in dim:
                    if isinstance(d, int):
                        d_pos = d if d >= 0 else input.ndim + d
                        if d_pos < 0 or d_pos >= input.ndim or input.shape[d_pos] != 1:
                            return input
        
        # Use numpy squeeze, but handle axis carefully
        try:
            if dim is not None:
                out = np.squeeze(input_np, axis=dim)
            else:
                out = np.squeeze(input_np)
        except (ValueError, np.exceptions.AxisError):
            # If squeeze fails, return input as-is
            return input
        
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.dim = dim
        ctx.input_shape = input.shape
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Expand dimensions back
            grad_output_np = grad_output.asnumpy()
            if ctx.dim is not None:
                if isinstance(ctx.dim, int):
                    grad_input = np.expand_dims(grad_output_np, ctx.dim)
                elif isinstance(ctx.dim, (tuple, list)):
                    grad_input = grad_output_np
                    # Expand in reverse order to maintain indices
                    for d in sorted(ctx.dim, reverse=True):
                        if isinstance(d, int):
                            d_pos = d if d >= 0 else len(ctx.input_shape) + d
                            grad_input = np.expand_dims(grad_input, d_pos)
            else:
                # Expand all dimensions that were squeezed
                grad_input = grad_output_np
                for i, size in enumerate(ctx.input_shape):
                    if size == 1:
                        grad_input = np.expand_dims(grad_input, i)
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input, None

def squeeze(input, dim=None):
    return SqueezeFunction.apply(input, dim)


class IndexSelectFunction(Function):
    @staticmethod
    def forward(ctx, input, dim, index):
        out = np.take(input.asnumpy(), index.asnumpy(), axis=dim)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input, index)
        ctx.dim = dim
        ctx.input_shape = input.shape
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, index = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Scatter gradient back
            grad_output_np = grad_output.asnumpy()
            index_np = index.asnumpy()
            grad_input = np.zeros(ctx.input_shape, dtype=grad_output_np.dtype)
            # Use advanced indexing to scatter
            indices = [slice(None)] * grad_input.ndim
            indices[ctx.dim] = index_np
            grad_input[tuple(indices)] = grad_output_np
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input, None, None

def index_select(input, dim, index):
    return IndexSelectFunction.apply(input, dim, index)


def rand_ext(size, seed, offset, dtype):
    out = np.random.randn(*size)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    out = out.astype(mindtorch.dtype2np[dtype])
    return ms.Tensor.from_numpy(out)



def inplace_uniform(input, from_, to_, generator_):
    seed, _ = generator_._step(12)
    # Handle both scalar and tensor seed
    if hasattr(seed, 'item') and seed.numel() == 1:
        np.random.seed(seed.item())
    elif hasattr(seed, 'numpy'):
        seed_np = seed.asnumpy()
        if seed_np.size == 1:
            np.random.seed(int(seed_np.item()))
        else:
            # Use first element if multiple
            np.random.seed(int(seed_np.flat[0]))
    else:
        np.random.seed(int(seed))
    out = np.random.uniform(from_, to_, input.shape).astype(mindtorch.dtype2np[input.dtype])
    # Directly modify tensor data
    input_np = input.asnumpy()
    # Handle 0-dimensional arrays
    if input_np.ndim == 0:
        input_np[()] = out
    else:
        input_np[:] = out
    return input


def inplace_fill_scalar(input, value):
    out = np.full_like(input.asnumpy(), value)
    # Directly modify tensor data
    input_np = input.asnumpy()
    # Handle 0-dimensional arrays (scalars)
    if input_np.ndim == 0:
        input_np[()] = out
    else:
        input_np[:] = out
    return input


def inplace_fill_tensor(input, value):
    out = np.full_like(input.asnumpy(), value)
    # Directly modify tensor data
    input_np = input.asnumpy()
    # Handle 0-dimensional arrays (scalars)
    if input_np.ndim == 0:
        input_np[()] = out
    else:
        input_np[:] = out
    return input


def inplace_normal(input, mean, std, generator_):
    out = np.random.normal(mean, std, input.shape).astype(mindtorch.dtype2np[input.dtype])
    # Directly modify tensor data
    input_np = input.asnumpy()
    # Handle 0-dimensional arrays (scalars)
    if input_np.ndim == 0:
        input_np[()] = out
    else:
        input_np[:] = out
    return input


def inplace_random(input, from_val=0, to_val=None, seed=None, offset=None):
    # 选择随机数生成器
    rng = np.random
    arr = input.asnumpy()
    if np.issubdtype(arr.dtype, np.floating):
        # 浮点类型处理
        if to_val is None:
            # 默认 [0, 1) 均匀分布
            rnd = rng.random(size=arr.shape).astype(arr.dtype)
        else:
            rnd = (from_val + (to_val - from_val) * rng.random(size=arr.shape)).astype(arr.dtype)
            
    elif np.issubdtype(arr.dtype, np.integer):
        # 整数类型处理
        from_int = int(from_val)
        
        if to_val is None:
            # 默认范围 [0, dtype.max]
            max_val = np.iinfo(arr.dtype).max
            rnd = rng.randint(0, max_val + 1, size=arr.shape).astype(arr.dtype)
        else:
            # 指定范围 [from_int, to_val)
            to_int = int(to_val)
            
            # 验证参数有效性
            if from_int >= to_int:
                raise ValueError(f"Empty range for integers: from={from_int} >= to={to_int}")
                
            # 处理整数边界问题
            dtype_min = np.iinfo(arr.dtype).min
            dtype_max = np.iinfo(arr.dtype).max
            from_int = np.clip(from_int, dtype_min, dtype_max)
            to_int = np.clip(to_int, dtype_min + 1, dtype_max + 1)
            
            rnd = rng.randint(from_int, to_int, size=arr.shape).astype(arr.dtype)
            
    elif arr.dtype == bool:
        # 布尔类型处理 (忽略 from_val/to_val)
        rnd = rng.random(size=arr.shape) > 0.5
    
    else:
        raise TypeError(f"Unsupported data type: {arr.dtype}")
    
    # Directly modify tensor data
    input_np = input.asnumpy()
    # Handle 0-dimensional arrays (scalars)
    if input_np.ndim == 0:
        # For 0-d arrays, assign directly
        input_np[()] = rnd
    else:
        # For multi-dimensional arrays, use slice assignment
        input_np[:] = rnd
    return input


def inplace_copy(input, other):
    # Directly modify tensor data using numpy
    input_np = input.asnumpy()
    other_np = other.asnumpy()
    # Handle 0-dimensional arrays (scalars)
    if input_np.ndim == 0:
        input_np[()] = other_np
    else:
        input_np[:] = other_np
    return input


def inplace_relu(input):
    """
    In-place ReLU operation: applies ReLU (max(0, x)) to input tensor in-place.
    
    Args:
        input: Input tensor to apply ReLU to
    
    Returns:
        The input tensor (modified in-place)
    """
    # Compute ReLU: max(0, x)
    input_np = input.asnumpy()
    out = np.maximum(0, input_np)
    # Directly modify tensor data
    # Handle 0-dimensional arrays (scalars)
    if input_np.ndim == 0:
        input_np[()] = out
    else:
        input_np[:] = out
    return input


def softmax(input, dim):
    softmax_op = ops.Softmax(dim).set_device('CPU')
    return softmax_op(input)



class TopkFunction(Function):
    @staticmethod
    def forward(ctx, input, k, dim, largest, sorted):
        input_np = input.asnumpy()
        
        # If dim is None, use the last dimension
        if dim is None:
            dim = input.ndim - 1
        
        # If not largest, negate the input to get smallest k
        if not largest:
            input_np = -input_np
        
        # Get the shape
        shape = input_np.shape
        
        # Reshape to 2D if needed (flatten all dimensions except the target dim)
        if dim != input.ndim - 1:
            # Transpose to move dim to last position
            dims = list(range(input.ndim))
            dims[dim], dims[-1] = dims[-1], dims[dim]
            input_np = np.transpose(input_np, dims)
            shape = input_np.shape
        
        # Flatten all but last dimension
        flat_shape = (-1, shape[-1])
        input_flat = input_np.reshape(flat_shape)
        
        # Use numpy's argpartition for efficiency
        if sorted:
            # Use argsort for sorted results
            top_indices = np.argsort(input_flat, axis=-1)[:, -k:][:, ::-1]
        else:
            # Use argpartition for unsorted results
            top_indices = np.argpartition(input_flat, -k, axis=-1)[:, -k:]
        
        # Get values using advanced indexing
        batch_indices = np.arange(input_flat.shape[0])[:, None]
        top_values = input_flat[batch_indices, top_indices]
        
        # Reshape back to original shape (except last dim becomes k)
        output_shape = list(shape)
        output_shape[-1] = k
        # Ensure we maintain at least 1D shape (don't squeeze to 0D)
        if len(output_shape) == 0:
            output_shape = [k]
        values_np = top_values.reshape(output_shape)
        indices_np = top_indices.reshape(output_shape)
        
        # If not largest, negate values back
        if not largest:
            values_np = -values_np
        
        # Transpose back if needed
        if dim != input.ndim - 1:
            dims = list(range(input.ndim))
            dims[dim], dims[-1] = dims[-1], dims[dim]
            values_np = np.transpose(values_np, dims)
            indices_np = np.transpose(indices_np, dims)
        
        # Convert to tensors
        values = ms.Tensor.from_numpy(values_np)
        indices = ms.Tensor.from_numpy(indices_np.astype(np.int64))
        
        # Save for backward (topk is not differentiable, but save for consistency)
        return values, indices
    
    @staticmethod
    def backward(ctx, grad_values, grad_indices):
        # topk is not differentiable
        return None, None, None, None, None

def topk(input, k, dim=None, largest=True, sorted=True):
    return TopkFunction.apply(input, k, dim, largest, sorted)


def sort_ext(input, dim, descending, stable):
    return sort(input, dim, descending, stable)


def sort(input, dim, descending, stable):
    """
    Sorts the elements of the input tensor along a given dimension.
    
    Args:
        input: Input tensor
        dim: The dimension to sort along
        descending: If True, sort in descending order
        stable: If True, use stable sorting algorithm
    
    Returns:
        Tuple of (values, indices) where:
        - values: Sorted tensor
        - indices: Indices of the sorted elements
    """
    # INSERT_YOUR_CODE
    # Use numpy implementation to perform sort
    input_np = input.asnumpy() if hasattr(input, 'asnumpy') else np.asarray(input)

    # numpy.sort always returns ascending, so for descending sort, negate, sort ascending, then re-negate, or flip.
    # We need the indices too, so use np.argsort for indices and indexing to get sorted values.

    # Ensure axis is an int and handle None (default is to flatten if None)
    axis = dim
    if axis is None:
        # Flatten input
        flat_values = np.sort(input_np, axis=None)
        flat_indices = np.argsort(input_np, axis=None)
        values = ms.Tensor.from_numpy(flat_values)
        indices = ms.Tensor.from_numpy(flat_indices.astype(np.int64))
        return values, indices

    # When descending, sort ascending then reverse results on that axis
    if descending:
        sort_indices = np.argsort(input_np, axis=axis)
        sort_indices = np.flip(sort_indices, axis=axis)
    else:
        sort_indices = np.argsort(input_np, axis=axis)

    # np.take_along_axis to gather sorted values along the axis
    sorted_values = np.take_along_axis(input_np, sort_indices, axis=axis)
    values = ms.Tensor.from_numpy(sorted_values)
    indices = ms.Tensor.from_numpy(sort_indices.astype(np.int64))
    return values, indices


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, input, decimals):
        out = np.round(input.asnumpy(), decimals)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Round is not differentiable, but we pass gradient through (similar to PyTorch)
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.asnumpy()
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def round(input, decimals):
    return RoundFunction.apply(input, decimals)


def isin(elements, test_elements, assume_unique=False, invert=False):
    # Convert to numpy arrays if needed
    if hasattr(elements, 'numpy'):
        elements_np = elements.asnumpy()
    else:
        elements_np = np.asarray(elements)
    
    if hasattr(test_elements, 'numpy'):
        test_elements_np = test_elements.asnumpy()
    else:
        test_elements_np = np.asarray(test_elements)
    
    out = np.isin(elements_np, test_elements_np, assume_unique=assume_unique, invert=invert)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return ms.Tensor.from_numpy(out)


class LdexpFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        input_np = input.asnumpy()
        if not isinstance(other, numbers.Number):
            other_np = other.asnumpy()
        else:
            other_np = other
        out = np.ldexp(input_np, other_np)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input, other if not isinstance(other, numbers.Number) else None)
        ctx.is_other_number = isinstance(other, numbers.Number)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0] if len(ctx.saved_tensors) > 0 else None
        other = ctx.saved_tensors[1] if len(ctx.saved_tensors) > 1 else None
        
        grad_input = None
        grad_other = None
        
        if ctx.needs_input_grad[0] and input is not None:
            # d/dx (x * 2^exp) = 2^exp
            if ctx.is_other_number:
                exp = ctx.saved_tensors[1] if len(ctx.saved_tensors) > 1 else None
                if exp is not None:
                    grad_input = grad_output.asnumpy() * np.power(2, exp.asnumpy())
                else:
                    grad_input = grad_output.asnumpy()
            else:
                other_np = other.asnumpy() if other is not None else None
                if other_np is not None:
                    grad_input = grad_output.asnumpy() * np.power(2, other_np)
                else:
                    grad_input = grad_output.asnumpy()
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        
        # other (exponent) typically doesn't need gradient
        return grad_input, grad_other

def ldexp(input, other):
    return LdexpFunction.apply(input, other)


class LessFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        if not isinstance(input, numbers.Number):
            input_np = input.asnumpy()
        else:
            input_np = input
        if not isinstance(other, numbers.Number):
            other_np = other.asnumpy()
        else:
            other_np = other
        out = input_np < other_np
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Comparison operations don't need gradients
        return None, None

def less(input, other):
    return LessFunction.apply(input, other)


class CumsumFunction(Function):
    @staticmethod
    def forward(ctx, input, dim, dtype=None):
        input_np = input.asnumpy()
        if dtype is not None:
            dtype_np = mindtorch.dtype2np[dtype]
        else:
            dtype_np = None
        out = np.cumsum(input_np, axis=dim, dtype=dtype_np)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.dim = dim
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Reverse cumsum: flip, cumsum, flip
            grad_output_np = grad_output.asnumpy()
            # Flip along the dimension
            grad_flipped = np.flip(grad_output_np, axis=ctx.dim)
            # Cumsum along the dimension
            grad_cumsum = np.cumsum(grad_flipped, axis=ctx.dim)
            # Flip back
            grad_input = np.flip(grad_cumsum, axis=ctx.dim)
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input, None, None

def cumsum(input, dim, dtype=None):
    return CumsumFunction.apply(input, dim, dtype)


def cumsum_ext(input, dim, dtype):
    if dtype is not None:
        dtype = mindtorch.dtype2np[dtype]
    out = np.cumsum(input.asnumpy(), dim, dtype)

    return ms.Tensor.from_numpy(out)


class GreaterEqualFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        if not isinstance(input, numbers.Number):
            input_np = input.asnumpy()
        else:
            input_np = input
        if not isinstance(other, numbers.Number):
            other_np = other.asnumpy()
        else:
            other_np = other
        out = input_np >= other_np
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Comparison operations don't need gradients
        return None, None

def greater_equal(input, other):
    return GreaterEqualFunction.apply(input, other)


class MaskedFillFunction(Function):
    @staticmethod
    def forward(ctx, input, mask, value):
        input_np = input.asnumpy()
        mask_np = mask.asnumpy() if hasattr(mask, 'numpy') else mask
        
        # Handle scalar value
        value_np = np.array(value, dtype=input_np.dtype)
        # Use np.where to apply mask
        out = np.where(mask_np, value_np, input_np)
        result = ms.Tensor.from_numpy(out)
        
        # Save for backward
        ctx.save_for_backward(input, mask)
        ctx.value = value
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, mask = ctx.saved_tensors
        grad_input = None
        grad_mask = None
        grad_value = None
        
        if ctx.needs_input_grad[0]:
            # Gradient only flows where mask is False (where input was kept)
            mask_np = mask.asnumpy() if hasattr(mask, 'numpy') else mask
            grad_output_np = grad_output.asnumpy()
            # Where mask is True, gradient is 0 (value was used, not input)
            # Where mask is False, gradient passes through
            grad_input = np.where(mask_np, 0, grad_output_np)
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        
        # mask and value don't need gradients (mask is boolean, value is typically a scalar)
        return grad_input, grad_mask, grad_value

def masked_fill(input, mask, value):
    return MaskedFillFunction.apply(input, mask, value)


def inplace_masked_fill(input, mask, value):
    # In-place masked fill operation
    input_np = input.asnumpy()
    mask_np = mask.asnumpy() if hasattr(mask, 'numpy') else mask
    out = np.where(mask_np, value, input_np)
    # Directly modify tensor data
    # Handle 0-dimensional arrays (scalars)
    if input_np.ndim == 0:
        input_np[()] = out
    else:
        input_np[:] = out
    return input


class LogicalNotFunction(Function):
    @staticmethod
    def forward(ctx, input):
        out = np.logical_not(input.asnumpy())
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Logical operations don't need gradients
        return None

def logical_not(input):
    return LogicalNotFunction.apply(input)


class NotEqualFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        if not isinstance(input, numbers.Number):
            input_np = input.asnumpy()
        else:
            input_np = input
        if not isinstance(other, numbers.Number):
            other_np = other.asnumpy()
        else:
            other_np = other
        out = np.not_equal(input_np, other_np)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Comparison operations don't need gradients
        return None, None

def not_equal(input, other):
    return NotEqualFunction.apply(input, other)


class LessEqualFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        if not isinstance(input, numbers.Number):
            input_np = input.asnumpy()
        else:
            input_np = input
        if not isinstance(other, numbers.Number):
            other_np = other.asnumpy()
        else:
            other_np = other
        out = input_np <= other_np
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Comparison operations don't need gradients
        return None, None

def less_equal(input, other):
    return LessEqualFunction.apply(input, other)


class TrilFunction(Function):
    @staticmethod
    def forward(ctx, input, diagonal=0):
        input_np = input.numpy()
        # Use numpy's tril function
        out = np.tril(input_np, k=diagonal)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.diagonal = diagonal
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # tril backward: only pass gradient to lower triangular part
            grad_output_np = grad_output.numpy()
            grad_input_np = np.tril(grad_output_np, k=ctx.diagonal)
            if not isinstance(grad_input_np, np.ndarray):
                grad_input_np = np.array(grad_input_np)
            grad_input = ms.Tensor.from_numpy(grad_input_np)
        return grad_input, None

def tril(input, diagonal=0):
    return TrilFunction.apply(input, diagonal)


def tril_ext(input, diagonal):
    out = np.tril(input.asnumpy(), diagonal)
    return ms.Tensor.from_numpy(out)


def randperm_ext(n, seed, offset, dtype):
    out = np.random.permutation(n)
    return ms.Tensor.from_numpy(out)


def randperm(n, generator, dtype):
    """
    Returns a random permutation of integers from 0 to n - 1.
    
    Args:
        n: The upper bound (exclusive). Must be positive.
        generator: Random number generator (optional, for compatibility)
        dtype: The desired data type of returned tensor. Default: int64.
    
    Returns:
        A random permutation of integers from 0 to n - 1.
    """
    # Extract seed from generator if provided
    if generator is not None:
        seed, offset = generator._step(12)  # pylint: disable=protected-access
        # Handle both scalar and tensor seed
        if hasattr(seed, 'item') and seed.numel() == 1:
            np.random.seed(seed.item())
        elif hasattr(seed, 'numpy'):
            seed_np = seed.asnumpy()
            if seed_np.size == 1:
                np.random.seed(int(seed_np.item()))
            else:
                # Use first element if multiple
                np.random.seed(int(seed_np.flat[0]))
        else:
            np.random.seed(int(seed))
    
    # Generate random permutation
    out = np.random.permutation(n)
    
    # Convert to specified dtype
    if dtype is not None:
        dtype_np = mindtorch.dtype2np[dtype]
        out = out.astype(dtype_np)
    else:
        # Default to int64
        out = out.astype(np.int64)
    
    return ms.Tensor.from_numpy(out)


def embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq):
    out = np.take(weight.asnumpy(), input.asnumpy(), axis=0)
    return ms.Tensor.from_numpy(out)


class RandFunction(Function):
    @staticmethod
    def forward(ctx, size, generator, dtype):
        if dtype is None:
            dtype = mindtorch.float32
        # Handle empty size (0-dimensional tensor)
        if size == [] or size == ():
            result_np = np.random.random()
        else:
            # Generate random numbers in [0, 1) using numpy
            result_np = np.random.random(size)
        result_np = np.array(result_np).astype(mindtorch.dtype2np[dtype])
        result = ms.Tensor.from_numpy(result_np)
        # rand is not differentiable
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # rand is not differentiable
        return None, None, None

def rand(size, generator, dtype):
    return RandFunction.apply(size, generator, dtype)


def rand_like(input, generator, dtype):
    """
    Returns a tensor filled with random numbers from a uniform distribution on [0, 1),
    with the same shape as input.
    
    Args:
        input: Input tensor to match shape
        generator: Random number generator (optional, for compatibility)
        dtype: The desired data type of returned tensor. If None, uses input's dtype.
    
    Returns:
        A tensor with the same shape as input, filled with random numbers
    """
    # Use input's dtype if dtype is None
    if dtype is None:
        dtype = input.dtype
    
    # Call rand with input's shape
    return rand(input.shape, generator, dtype)


def randn(size, generator, dtype):
    out = np.random.randn(*size).astype(mindtorch.dtype2np[dtype])
    return ms.Tensor.from_numpy(out)


class BernoulliFunction(Function):
    @staticmethod
    def forward(ctx, input, generator):
        # Extract seed from generator
        seed, _ = generator._step(12)  # pylint: disable=protected-access
        # Handle both scalar and tensor seed
        if hasattr(seed, 'item') and seed.numel() == 1:
            np.random.seed(seed.item())
        elif hasattr(seed, 'numpy'):
            seed_np = seed.asnumpy()
            if seed_np.size == 1:
                np.random.seed(int(seed_np.item()))
            else:
                # Use first element if multiple
                np.random.seed(int(seed_np.flat[0]))
        else:
            np.random.seed(int(seed))
        
        # Get input probabilities as numpy array
        input_np = input.asnumpy()
        
        # Generate uniform random numbers in [0, 1)
        uniform_random = np.random.random(input_np.shape)
        
        # Bernoulli sampling: 1 if uniform_random < input_np, else 0
        # Convert boolean result to the same dtype as input
        out = (uniform_random < input_np).astype(input_np.dtype)
        
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        # bernoulli is not differentiable
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # bernoulli is not differentiable
        return None, None

def bernoulli(input, generator):
    return BernoulliFunction.apply(input, generator)


def erfinv(input):
    out = scipy.special.erfinv(input)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return ms.Tensor.from_numpy(out)


def inplace_add_ext(input, other, alpha):
    if not isinstance(other, numbers.Number):
        other = other.numpy()
    out = input.numpy() + other * alpha
    # Directly modify tensor data
    input_np = input.numpy()
    # Handle 0-dimensional arrays (scalars)
    if input_np.ndim == 0:
        input_np[()] = out
    else:
        input_np[:] = out
    return input


class InplaceSubFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        input_np = input.asnumpy()
        if not isinstance(other, numbers.Number):
            other_np = other.asnumpy()
        else:
            other_np = other
        out_np = input_np - other_np
        # Directly modify tensor data
        # Handle 0-dimensional arrays (scalars)
        if input_np.ndim == 0:
            input_np[()] = out_np
        else:
            input_np[:] = out_np
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        # inplace operations don't need backward
        return None, None

def inplace_sub(input, other):
    return InplaceSubFunction.apply(input, other)


class InplaceMulFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        input_np = input.asnumpy()
        if not isinstance(other, numbers.Number):
            other_np = other.asnumpy()
        else:
            other_np = other
        out_np = input_np * other_np
        # Directly modify tensor data
        # Handle 0-dimensional arrays (scalars)
        if input_np.ndim == 0:
            input_np[()] = out_np
        else:
            input_np[:] = out_np
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        # inplace operations don't need backward
        return None, None

def inplace_mul(input, other):
    return InplaceMulFunction.apply(input, other)


class InplaceAddFunction(Function):
    @staticmethod
    def forward(ctx, input, other, alpha=1):
        input_np = input.asnumpy()
        if not isinstance(other, numbers.Number):
            other_np = other.asnumpy()
        else:
            other_np = other
        out_np = input_np + other_np * alpha
        # Directly modify tensor data
        # Handle 0-dimensional arrays (scalars)
        if input_np.ndim == 0:
            input_np[()] = out_np
        else:
            input_np[:] = out_np
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        # inplace operations don't need backward
        return None, None, None

def inplace_add(input, other, alpha=1):
    return InplaceAddFunction.apply(input, other, alpha)


class PowTensorScalarFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        input_np = input.asnumpy()
        if input_np.dtype == np.int64:
            input_np = input_np.astype(np.int32)
        out = np.power(input_np, other)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.other = other
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # d/dx (x^a) = a * x^(a-1)
            input_np = input.asnumpy()
            if input_np.dtype == np.int64:
                input_np = input_np.astype(np.int32)
            grad_input = grad_output.asnumpy() * ctx.other * np.power(input_np, ctx.other - 1)
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input, None

def pow_tensor_scalar(input, other):
    return PowTensorScalarFunction.apply(input, other)


class PowFunction(Function):
    @staticmethod
    def forward(ctx, input, exponent):
        # Handle different input types
        if isinstance(input, numbers.Number):
            # Scalar ** Tensor
            if not isinstance(exponent, numbers.Number):
                exponent_np = exponent.asnumpy()
                out = np.power(input, exponent_np)
                if out.dtype == np.float64:
                    out = out.astype(np.float32)
                result = ms.Tensor.from_numpy(out)
                ctx.save_for_backward(exponent)
                ctx.input = input
                ctx.is_input_number = True
                ctx.is_exponent_number = False
                return result
            else:
                # Scalar ** Scalar
                out = np.power(input, exponent)
                if isinstance(out, np.ndarray):
                    if out.dtype == np.float64:
                        out = out.astype(np.float32)
                    result = ms.Tensor.from_numpy(out)
                else:
                    result = ms.Tensor.from_numpy(np.array(out, dtype=np.float32))
                ctx.is_input_number = True
                ctx.is_exponent_number = True
                return result
        else:
            # Tensor ** Tensor or Tensor ** Scalar
            input_np = input.asnumpy()
            if input_np.dtype == np.int64:
                input_np = input_np.astype(np.int32)
            
            if isinstance(exponent, numbers.Number):
                # Tensor ** Scalar
                out = np.power(input_np, exponent)
            else:
                # Tensor ** Tensor
                exponent_np = exponent.asnumpy()
                out = np.power(input_np, exponent_np)
            
            if not isinstance(out, np.ndarray):
                out = np.array(out)
            if out.dtype == np.float64:
                out = out.astype(np.float32)
            
            result = ms.Tensor.from_numpy(out)
            ctx.save_for_backward(input if not isinstance(input, numbers.Number) else None,
                                exponent if not isinstance(exponent, numbers.Number) else None)
            ctx.is_input_number = isinstance(input, numbers.Number)
            ctx.is_exponent_number = isinstance(exponent, numbers.Number)
            if isinstance(input, numbers.Number):
                ctx.input = input
            if isinstance(exponent, numbers.Number):
                ctx.exponent = exponent
            return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0] if len(ctx.saved_tensors) > 0 else None
        exponent = ctx.saved_tensors[1] if len(ctx.saved_tensors) > 1 else None
        
        grad_input = None
        grad_exponent = None
        
        if ctx.is_input_number and ctx.is_exponent_number:
            # Scalar ** Scalar - no gradients
            return None, None
        
        if not ctx.is_input_number and ctx.needs_input_grad[0] and input is not None:
            # Tensor ** (Scalar or Tensor)
            input_np = input.asnumpy()
            if input_np.dtype == np.int64:
                input_np = input_np.astype(np.int32)
            
            if ctx.is_exponent_number:
                # Tensor ** Scalar: d/dx (x^a) = a * x^(a-1)
                grad_input_np = grad_output.asnumpy() * ctx.exponent * np.power(input_np, ctx.exponent - 1)
            else:
                # Tensor ** Tensor: d/dx (x^y) = y * x^(y-1)
                exponent_np = exponent.asnumpy()
                grad_input_np = grad_output.asnumpy() * exponent_np * np.power(input_np, exponent_np - 1)
            
            if grad_input_np.dtype == np.float64:
                grad_input_np = grad_input_np.astype(np.float32)
            grad_input = ms.Tensor.from_numpy(grad_input_np)
        
        if not ctx.is_exponent_number and ctx.needs_input_grad[1] and exponent is not None:
            # (Scalar or Tensor) ** Tensor
            if ctx.is_input_number:
                # Scalar ** Tensor: d/dx (a^x) = a^x * ln(a)
                exponent_np = exponent.asnumpy()
                grad_exponent_np = grad_output.asnumpy() * np.power(ctx.input, exponent_np) * np.log(ctx.input)
            else:
                # Tensor ** Tensor: d/dx (y^x) = y^x * ln(y)
                input_np = input.asnumpy()
                exponent_np = exponent.asnumpy()
                grad_exponent_np = grad_output.asnumpy() * np.power(input_np, exponent_np) * np.log(input_np)
            
            if grad_exponent_np.dtype == np.float64:
                grad_exponent_np = grad_exponent_np.astype(np.float32)
            grad_exponent = ms.Tensor.from_numpy(grad_exponent_np)
        
        return grad_input, grad_exponent

def pow(input, exponent):
    return PowFunction.apply(input, exponent)


class ViewAsComplexFunction(Function):
    @staticmethod
    def forward(ctx, input):
        input_np = input.asnumpy()
        # Check that the last dimension is 2
        if input_np.shape[-1] != 2:
            raise RuntimeError(f"view_as_complex: input tensor must have last dimension of size 2, but got {input_np.shape[-1]}")
        
        # Reshape to remove the last dimension and create complex view
        # The last dimension [real, imag] pairs become complex numbers
        new_shape = input_np.shape[:-1]
        # Use numpy's view to create complex array
        # We need to reshape and view as complex
        complex_np = input_np[..., 0] + 1j * input_np[..., 1]
        
        if not isinstance(complex_np, np.ndarray):
            complex_np = np.array(complex_np)
        
        result = ms.Tensor.from_numpy(complex_np)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # view_as_complex backward is view_as_real
        grad_output_np = grad_output.asnumpy()
        # Convert complex to real by stacking real and imag parts
        grad_real = np.real(grad_output_np)
        grad_imag = np.imag(grad_output_np)
        # Stack along last dimension: [real, imag]
        grad_input_np = np.stack([grad_real, grad_imag], axis=-1)
        if not isinstance(grad_input_np, np.ndarray):
            grad_input_np = np.array(grad_input_np)
        return ms.Tensor.from_numpy(grad_input_np)

def view_as_complex(input):
    return ViewAsComplexFunction.apply(input)


class ViewAsRealFunction(Function):
    @staticmethod
    def forward(ctx, input):
        input_np = input.asnumpy()
        # Check that input is complex
        if not np.iscomplexobj(input_np):
            raise RuntimeError(f"view_as_real: input tensor must be complex, but got {input_np.dtype}")
        
        # Extract real and imaginary parts
        real_part = np.real(input_np)
        imag_part = np.imag(input_np)
        
        # Stack along a new last dimension: [real, imag]
        result_np = np.stack([real_part, imag_part], axis=-1)
        
        if not isinstance(result_np, np.ndarray):
            result_np = np.array(result_np)
        
        result = ms.Tensor.from_numpy(result_np)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # view_as_real backward is view_as_complex
        grad_output_np = grad_output.asnumpy()
        # Extract real and imag parts from last dimension
        grad_real = grad_output_np[..., 0]
        grad_imag = grad_output_np[..., 1]
        # Combine into complex
        grad_input_np = grad_real + 1j * grad_imag
        if not isinstance(grad_input_np, np.ndarray):
            grad_input_np = np.array(grad_input_np)
        return ms.Tensor.from_numpy(grad_input_np)

def view_as_real(input):
    return ViewAsRealFunction.apply(input)


class PolarFunction(Function):
    @staticmethod
    def forward(ctx, abs, angle):
        abs_np = abs.asnumpy()
        angle_np = angle.asnumpy()
        
        # Compute real and imaginary parts
        # polar(abs, angle) = abs * cos(angle) + abs * sin(angle) * j
        real = abs_np * np.cos(angle_np)
        imag = abs_np * np.sin(angle_np)
        
        # Create complex array
        complex_np = real + 1j * imag
        
        if not isinstance(complex_np, np.ndarray):
            complex_np = np.array(complex_np)
        
        result = ms.Tensor.from_numpy(complex_np)
        ctx.save_for_backward(abs, angle)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        abs, angle = ctx.saved_tensors
        grad_output_np = grad_output.asnumpy()
        
        # Extract real and imaginary parts of gradient
        grad_real = np.real(grad_output_np)
        grad_imag = np.imag(grad_output_np)
        
        abs_np = abs.asnumpy()
        angle_np = angle.asnumpy()
        
        # Backward for polar: d/dabs = cos(angle) * grad_real + sin(angle) * grad_imag
        #                    d/dangle = -abs * sin(angle) * grad_real + abs * cos(angle) * grad_imag
        grad_abs = None
        grad_angle = None
        
        if ctx.needs_input_grad[0]:
            grad_abs = grad_real * np.cos(angle_np) + grad_imag * np.sin(angle_np)
            if not isinstance(grad_abs, np.ndarray):
                grad_abs = np.array(grad_abs)
            grad_abs = ms.Tensor.from_numpy(grad_abs)
        
        if ctx.needs_input_grad[1]:
            grad_angle = -abs_np * np.sin(angle_np) * grad_real + abs_np * np.cos(angle_np) * grad_imag
            if not isinstance(grad_angle, np.ndarray):
                grad_angle = np.array(grad_angle)
            grad_angle = ms.Tensor.from_numpy(grad_angle)
        
        return grad_abs, grad_angle

def polar(abs, angle):
    return PolarFunction.apply(abs, angle)


class FlattenFunction(Function):
    @staticmethod
    def forward(ctx, input, start_dim, end_dim):
        input_np = input.asnumpy()
        shape = list(input_np.shape)
        ndim = len(shape)
        
        # Normalize negative indices
        if start_dim < 0:
            start_dim = start_dim + ndim
        if end_dim < 0:
            end_dim = end_dim + ndim
        
        # Calculate flattened size
        flattened_size = 1
        for i in range(start_dim, end_dim + 1):
            flattened_size *= shape[i]
        
        # Create new shape
        new_shape = shape[:start_dim] + [flattened_size] + shape[end_dim + 1:]
        
        # Reshape
        result_np = input_np.reshape(new_shape)
        
        if not isinstance(result_np, np.ndarray):
            result_np = np.array(result_np)
        
        result = ms.Tensor.from_numpy(result_np)
        ctx.start_dim = start_dim
        ctx.end_dim = end_dim
        ctx.original_shape = shape
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Reshape back to original shape
        grad_output_np = grad_output.asnumpy()
        grad_input_np = grad_output_np.reshape(ctx.original_shape)
        if not isinstance(grad_input_np, np.ndarray):
            grad_input_np = np.array(grad_input_np)
        return ms.Tensor.from_numpy(grad_input_np), None, None

def flatten(input, start_dim, end_dim):
    return FlattenFunction.apply(input, start_dim, end_dim)


class FmodScalarFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        out = np.fmod(input.asnumpy(), other)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.other = other
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # fmod(x, y) = x - y * floor(x/y)
            # d/dx fmod(x, y) = 1 (gradient passes through)
            grad_input = grad_output.asnumpy()
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input, None

def fmod_scalar(input, other):
    return FmodScalarFunction.apply(input, other)


class FmodTensorFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        input_np = input.asnumpy()
        other_np = other.asnumpy()
        out = np.fmod(input_np, other_np)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input, other)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_tensors
        grad_input = None
        grad_other = None
        
        if ctx.needs_input_grad[0]:
            # fmod(x, y) = x - y * floor(x/y)
            # d/dx fmod(x, y) = 1 (gradient passes through)
            grad_input = grad_output.asnumpy()
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        
        if ctx.needs_input_grad[1]:
            # d/dy fmod(x, y) = -floor(x/y) (but typically not needed)
            # For simplicity, we can set it to zero or compute it
            grad_other = np.zeros_like(other.asnumpy())
            grad_other = ms.Tensor.from_numpy(grad_other)
        
        return grad_input, grad_other


def fmod_tensor(input, other):
    """
    Computes the element-wise remainder of division (fmod) between two tensors.
    
    Args:
        input: First input tensor
        other: Second input tensor (can be broadcasted)
    
    Returns:
        Tensor with the element-wise remainder
    """
    return FmodTensorFunction.apply(input, other)


def argmax_with_value(input, dim, keepdim):
    indices = np.argmax(input.asnumpy(), dim, keepdims=keepdim)
    values = np.max(input.asnumpy(), dim, keepdims=keepdim)

    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    return ms.Tensor.from_numpy(indices), ms.Tensor.from_numpy(values)


def argmin_with_value(input, dim, keepdim):
    indices = np.argmin(input.asnumpy(), dim, keepdims=keepdim)
    values = np.min(input.asnumpy(), dim, keepdims=keepdim)

    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    return ms.Tensor.from_numpy(indices), ms.Tensor.from_numpy(values)



def argmax_ext(input, dim, keepdim):
    indices = np.argmax(input.asnumpy(), dim, keepdims=keepdim)
    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)
    return ms.Tensor.from_numpy(indices)

class ArgmaxFunction(Function):
    @staticmethod
    def forward(ctx, input, dim=None, keepdim=False):
        input_np = input.asnumpy()
        indices = np.argmax(input_np, axis=dim, keepdims=keepdim)
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        result = ms.Tensor.from_numpy(indices)
        # argmax is not differentiable
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # argmax is not differentiable
        return None, None, None

def argmax(input, dim=None, keepdim=False):
    return ArgmaxFunction.apply(input, dim, keepdim)


class ArgminFunction(Function):
    @staticmethod
    def forward(ctx, input, dim=None, keepdim=False):
        input_np = input.asnumpy()
        indices = np.argmin(input_np, axis=dim, keepdims=keepdim)
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        result = ms.Tensor.from_numpy(indices)
        # argmin is not differentiable
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # argmin is not differentiable
        return None, None, None

def argmin(input, dim=None, keepdim=False):
    return ArgminFunction.apply(input, dim, keepdim)


class LogFunction(Function):
    @staticmethod
    def forward(ctx, input):
        out = np.log(input.asnumpy())
        if input.dtype == ms.int64:
            out = out.astype(np.float32)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.asnumpy() / input.asnumpy()
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def log(input):
    return LogFunction.apply(input)


def eye(n, m, dtype):
    out = np.eye(n, m, dtype=mindtorch.dtype2np[dtype])
    return ms.Tensor.from_numpy(out)


def lin_space_ext(start, end, steps, dtype):
    out = np.linspace(start, end, steps, dtype=mindtorch.dtype2np[dtype])
    return ms.Tensor.from_numpy(out)


def conv2d(input, weight, bias=None, stride=1, padding='valid', dilation=1, groups=1, training=True):
    """
    2D convolution operation implemented using numpy.
    
    Args:
        input: Input tensor of shape (N, C_in, H, W)
        weight: Weight tensor of shape (C_out, C_in/groups, kH, kW)
        bias: Optional bias tensor of shape (C_out,)
        stride: Stride for convolution (int or tuple)
        padding: Padding mode ('valid', 'same') or padding value (int or tuple)
        dilation: Dilation rate (int or tuple)
        groups: Number of groups for grouped convolution
    
    Returns:
        Output tensor of shape (N, C_out, H_out, W_out)
    """
    from scipy import signal
    
    input_np = input.asnumpy()
    weight_np = weight.asnumpy()
    
    # Get weight shape first to determine expected input channels
    C_out, C_in_per_group, kH, kW = weight_np.shape
    expected_C_in = C_in_per_group * groups
    
    # Handle different input dimensions
    if input_np.ndim == 2:
        # Add batch and channel dimensions: (H, W) -> (1, C_in, H, W)
        # For 2D input, assume single channel and expand to expected channels if needed
        if expected_C_in == 1:
            input_np = input_np[np.newaxis, np.newaxis, :]
        else:
            # If weight expects more channels, we need to broadcast or error
            # For now, assume it's a single channel that should be repeated
            input_np = np.tile(input_np[np.newaxis, np.newaxis, :], (1, expected_C_in, 1, 1))
        N = 1
        squeeze_output = True
    elif input_np.ndim == 3:
        # Add batch dimension if missing: (C, H, W) -> (1, C, H, W)
        input_np = input_np[np.newaxis, :]
        N = 1
        squeeze_output = True
    elif input_np.ndim == 4:
        N = input_np.shape[0]
        squeeze_output = False
    else:
        raise ValueError(f"conv2d: input must have 2, 3, or 4 dimensions, got {input_np.ndim}")
    
    N, C_in, H, W = input_np.shape
    
    # Verify input channels match expected
    if C_in != expected_C_in:
        raise ValueError(f"conv2d: input has {C_in} channels but weight expects {expected_C_in} channels (C_in_per_group={C_in_per_group} * groups={groups})")
    C_out, C_in_per_group, kH, kW = weight_np.shape
    
    # Normalize stride and dilation
    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    elif isinstance(stride, (tuple, list)):
        if len(stride) == 2:
            stride_h, stride_w = stride[0], stride[1]
        elif len(stride) == 4:
            stride_h, stride_w = stride[2], stride[3]
        else:
            stride_h, stride_w = stride[0], stride[0]
    else:
        stride_h, stride_w = 1, 1
    
    if isinstance(dilation, int):
        dilation_h, dilation_w = dilation, dilation
    elif isinstance(dilation, (tuple, list)):
        if len(dilation) == 2:
            dilation_h, dilation_w = dilation[0], dilation[1]
        elif len(dilation) == 4:
            dilation_h, dilation_w = dilation[2], dilation[3]
        else:
            dilation_h, dilation_w = dilation[0], dilation[0]
    else:
        dilation_h, dilation_w = 1, 1
    
    # Calculate effective kernel size with dilation
    eff_kH = (kH - 1) * dilation_h + 1
    eff_kW = (kW - 1) * dilation_w + 1
    
    # Handle padding
    if isinstance(padding, str):
        if padding == 'valid':
            pad_h, pad_w = 0, 0
        elif padding == 'same':
            # Calculate padding to maintain output size
            pad_h = max(0, (H - 1) * stride_h + eff_kH - H) // 2
            pad_w = max(0, (W - 1) * stride_w + eff_kW - W) // 2
        else:
            raise ValueError(f"Unsupported padding mode: {padding}")
    elif isinstance(padding, int):
        pad_h, pad_w = padding, padding
    elif isinstance(padding, (tuple, list)):
        if len(padding) == 2:
            pad_h, pad_w = padding[0], padding[1]
        elif len(padding) == 4:
            pad_h, pad_w = padding[0], padding[2]  # (top, bottom, left, right) -> (top, left)
        else:
            raise ValueError(f"padding must be int, 2-tuple, or 4-tuple, got {padding}")
    else:
        pad_h, pad_w = 0, 0
    
    # Pad input if needed
    if pad_h > 0 or pad_w > 0:
        input_padded = np.pad(input_np, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    else:
        input_padded = input_np
    
    # Calculate output dimensions
    H_out = (H + 2 * pad_h - eff_kH) // stride_h + 1
    W_out = (W + 2 * pad_w - eff_kW) // stride_w + 1
    
    # Initialize output
    output_np = np.zeros((N, C_out, H_out, W_out), dtype=input_np.dtype)
    
    # Perform convolution for each batch and output channel
    for n in range(N):
        for c_out in range(C_out):
            # Determine which input channels to use based on groups
            group_id = c_out // (C_out // groups)
            c_in_start = group_id * C_in_per_group
            c_in_end = (group_id + 1) * C_in_per_group
            
            # Sum over input channels in this group
            conv_result = np.zeros((H_out, W_out), dtype=input_np.dtype)
            for c_in in range(c_in_start, c_in_end):
                # Get kernel for this input-output channel pair
                kernel = weight_np[c_out, c_in - c_in_start]
                
                # Apply dilation to kernel
                if dilation_h > 1 or dilation_w > 1:
                    dilated_kernel = np.zeros((eff_kH, eff_kW), dtype=kernel.dtype)
                    dilated_kernel[::dilation_h, ::dilation_w] = kernel
                    kernel = dilated_kernel
                
                # Flip kernel for convolution (cross-correlation)
                kernel_flipped = np.flipud(np.fliplr(kernel))
                
                # Perform 2D convolution using scipy.signal.convolve2d
                conv_channel = signal.convolve2d(
                    input_padded[n, c_in],
                    kernel_flipped,
                    mode='valid'
                )
                
                # Apply stride by subsampling
                if stride_h > 1 or stride_w > 1:
                    conv_channel = conv_channel[::stride_h, ::stride_w]
                
                    # Ensure output size matches (handle edge cases)
                    h_end = H_out if H_out < conv_channel.shape[0] else conv_channel.shape[0]
                    w_end = W_out if W_out < conv_channel.shape[1] else conv_channel.shape[1]
                    h_slice = slice(0, h_end)
                    w_slice = slice(0, w_end)
                    conv_result[h_slice, w_slice] += conv_channel[h_slice, w_slice]
            
            output_np[n, c_out] = conv_result
    
    # Add bias if provided
    if bias is not None:
        bias_np = bias.asnumpy() if hasattr(bias, 'asnumpy') else bias
        output_np = output_np + bias_np[None, :, None, None]
    
    # Remove batch dimension if input was 3D
    if squeeze_output:
        output_np = output_np[0]
    
    return ms.Tensor.from_numpy(output_np)


def upsample_nearest2d(input, output_size, scale_factors):
    """
    Upsample input tensor using nearest neighbor interpolation.
    
    Args:
        input: Input tensor of shape (N, C, H, W)
        output_size: Target output size (H, W) or None
        scale_factors: Scale factors for upsampling (h_scale, w_scale) or None
    
    Returns:
        Upsampled tensor
    """
    from scipy import ndimage
    
    input_np = input.asnumpy()
    N, C, H, W = input_np.shape
    
    if output_size is None:
        # Calculate output size from scale factors
        if scale_factors is None:
            raise ValueError("Either output_size or scale_factors must be provided")
        if isinstance(scale_factors, (list, tuple)):
            h_scale = scale_factors[0]
            w_scale = scale_factors[1] if len(scale_factors) > 1 else scale_factors[0]
        else:
            h_scale = w_scale = scale_factors
        output_h = int(H * h_scale)
        output_w = int(W * w_scale)
        output_size = (output_h, output_w)
    else:
        if isinstance(output_size, (list, tuple)):
            output_h, output_w = output_size[0], output_size[1]
        else:
            output_h = output_w = output_size
    
    # Calculate zoom factors
    zoom_h = output_h / H
    zoom_w = output_w / W
    
    # Upsample each channel using scipy.ndimage.zoom
    output_np = np.zeros((N, C, output_h, output_w), dtype=input_np.dtype)
    for n in range(N):
        for c in range(C):
            output_np[n, c] = ndimage.zoom(input_np[n, c], (zoom_h, zoom_w), order=0, mode='nearest')
    
    return ms.Tensor.from_numpy(output_np)


def upsample_bilinear2d(input, output_size, scale_factors, align_corners):
    """
    Upsample input tensor using bilinear interpolation.
    
    Args:
        input: Input tensor of shape (N, C, H, W)
        output_size: Target output size (H, W)
        scale_factors: Scale factors (not used when output_size is provided)
        align_corners: Whether to align corners
    
    Returns:
        Upsampled tensor
    """
    from scipy import ndimage
    
    input_np = input.asnumpy()
    N, C, H, W = input_np.shape
    
    if isinstance(output_size, (list, tuple)):
        output_h, output_w = output_size[0], output_size[1]
    else:
        output_h = output_w = output_size
    
    # Calculate zoom factors
    zoom_h = output_h / H
    zoom_w = output_w / W
    
    # Upsample each channel using scipy.ndimage.zoom with bilinear interpolation (order=1)
    output_np = np.zeros((N, C, output_h, output_w), dtype=input_np.dtype)
    for n in range(N):
        for c in range(C):
            output_np[n, c] = ndimage.zoom(input_np[n, c], (zoom_h, zoom_w), order=1, mode='nearest')
    
    return ms.Tensor.from_numpy(output_np)

def addcmul(input, tensor1, tensor2, value=1.0):
    """
    Performs the element-wise operation: output = input + value * tensor1 * tensor2
    """
    # addcmul(input, tensor1, tensor2, value) = input + value * tensor1 * tensor2
    return add(input, mul(mul(tensor1, tensor2), value))

def addmm(input, mat1, mat2, beta, alpha):
    """
    Matrix multiply-and-add: beta*input + alpha*(mat1 @ mat2)
    Supports input as 1D bias (broadcast across rows) or same shape as matmul result.
    """
    in_np = input.asnumpy() if hasattr(input, 'asnumpy') else np.asarray(input)
    a = mat1.asnumpy() if hasattr(mat1, 'asnumpy') else np.asarray(mat1)
    b = mat2.asnumpy() if hasattr(mat2, 'asnumpy') else np.asarray(mat2)
    prod = np.matmul(a, b)
    out = alpha * prod + beta * in_np
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return ms.Tensor.from_numpy(out)

def batch_norm(input, weight, bias, running_mean=None, runnning_var=None, training=False, momentum=0.1, epsilon=1e-5):
    """
    Batch Normalization over a mini-batch of inputs.
    Following npu_310b.py implementation pattern.

    Args:
        input: Input tensor of shape (N, C, ...)
        weight: Scale tensor of shape (C,)
        bias: Shift tensor of shape (C,)
        running_mean: Running mean tensor of shape (C,)
        runnning_var: Running variance tensor of shape (C,) (note: typo in parameter name)
        training: Whether in training mode
        momentum: Momentum for running statistics
        epsilon: Small value for numerical stability

    Returns:
        Normalized tensor with same shape as input
    """
    input_np = input.asnumpy()
    ndim = input_np.ndim

    if ndim < 2:
        raise ValueError(f"batch_norm: input must have at least 2 dimensions, got {ndim}")

    N = input_np.shape[0]
    C = input_np.shape[1]
    spatial_shape = input_np.shape[2:]

    # Create default values if None
    if running_mean is None:
        running_mean = ones(C, dtype=input.dtype)
    if runnning_var is None:
        runnning_var = zeros(C, dtype=input.dtype)
    if weight is None:
        weight = ones(C, dtype=input.dtype)
    if bias is None:
        bias = zeros(C, dtype=input.dtype)

    # Convert to numpy
    weight_np = weight.asnumpy()
    bias_np = bias.asnumpy()
    running_mean_np = running_mean.asnumpy()
    running_var_np = runnning_var.asnumpy()

    if training:
        # Compute mean and variance over batch and spatial dimensions for each channel
        axes = (0,) + tuple(range(2, ndim))  # (0,2,3,..)
        mean = np.mean(input_np, axis=axes, keepdims=True)
        var = np.var(input_np, axis=axes, ddof=0, keepdims=True)

        # Update running stats
        mean_squeezed = np.squeeze(mean, axis=axes)
        var_squeezed = np.squeeze(var, axis=axes)
        running_mean_np[:] = (1 - momentum) * running_mean_np + momentum * mean_squeezed
        running_var_np[:] = (1 - momentum) * running_var_np + momentum * var_squeezed
    else:
        # Use running statistics and reshape to (1, C, 1, 1, ...)
        rep = ndim - 2 if ndim > 2 else 0
        expand_shape = (1, C) + (1,) * rep
        mean = running_mean_np.reshape(expand_shape)
        var = running_var_np.reshape(expand_shape)

    # Normalize
    normalized = (input_np - mean) / np.sqrt(var + epsilon)

    # Apply weight and bias: reshape to (1, C, 1, 1, ...)
    rep = ndim - 2 if ndim > 2 else 0
    weight_reshaped = weight_np.reshape((1, C) + (1,) * rep)
    bias_reshaped = bias_np.reshape((1, C) + (1,) * rep)

    output = normalized * weight_reshaped + bias_reshaped

    # Ensure dtype and return
    output = output.astype(input_np.dtype)
    return ms.Tensor.from_numpy(output)

def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    """
    Implements Group Normalization by reshaping and calling batch_norm.
    This function reshapes input of shape (N, C, *spatial) into
    (N * num_groups, C // num_groups, -1) so that batch_norm computes
    mean/var across the correct axes, then reshapes back.
    """
    input_shape = input.shape
    N = input_shape[0]
    C = input_shape[1]

    # compute product of spatial dimensions
    spatial_dims = input_shape[2:]
    spatial_size = 1
    for s in spatial_dims:
        spatial_size *= s

    # reshape to (N * num_groups, C // num_groups, spatial_size)
    assert C % num_groups == 0, "C must be divisible by num_groups"
    channels_per_group = C // num_groups
    input_reshaped = reshape(input, (N * num_groups, channels_per_group, spatial_size if spatial_size != 0 else 1))

    # use batch_norm to compute mean/var over batch and spatial dims for each group-channel
    outputs = batch_norm(input_reshaped, None, None, None, None, True, 0.0, eps)

    # reshape back to original
    out = reshape(outputs, input_shape)

    # apply affine parameters if provided
    affine_param_shape = [1] * input.ndim
    affine_param_shape[1] = C
    affine_param_shape = tuple(affine_param_shape)

    if weight is not None and bias is not None:
        out = add(out, reshape(bias, affine_param_shape))
        out = mul(out, reshape(weight, affine_param_shape))
    elif weight is not None:
        out = mul(out, reshape(weight, affine_param_shape))
    elif bias is not None:
        out = add(out, reshape(bias, affine_param_shape))
    return out

def rms_norm(input, normalized_shape, weight=None, eps=1e-5):
    """Root Mean Square Layer Norm (RMSNorm) for numpy backend.
    normalized_shape is ignored; numpy broadcasting handles shapes.
    """
    if eps is None:
        eps = 1e-5
    x = input.asnumpy()
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    y = x / rms
    if weight is not None:
        w = weight.asnumpy() if hasattr(weight, 'asnumpy') else np.asarray(weight)
        y = y * w
    return ms.Tensor.from_numpy(y)

def split_tensor(tensor, split_size_or_sections, dim):
    """
    Splits a tensor into multiple sub-tensors along the specified dimension.
    
    Args:
        tensor: The input tensor.
        split_size_or_sections: The size of each chunk (int).
        dim: The dimension along which to split.
    
    Returns:
        List of tensors.
    """
    arr = tensor.asnumpy()
    dim_size = arr.shape[dim]
    
    # Calculate split indices
    # If split_size_or_sections=3 and dim_size=10, we get indices [3, 6, 9]
    split_indices = list(range(split_size_or_sections, dim_size, split_size_or_sections))
    
    if len(split_indices) == 0:
        # If no splits needed (split_size >= dim_size), return the whole tensor
        return [ms.Tensor.from_numpy(arr)]
    
    # Split the array
    outs = np.split(arr, split_indices, axis=dim)
    out = [ms.Tensor.from_numpy(o) for o in outs]
    return out

def split_with_size(tensor, split_size_or_sections, dim):
    out = np.array_split(tensor.asnumpy(), np.cumsum(split_size_or_sections[:-1]), dim)
    out = [ms.Tensor.from_numpy(o) for o in out]
    return out


def floor_div(input, other):
    if not isinstance(other, numbers.Number):
        other = other.asnumpy()
    out = np.floor_divide(input.asnumpy(), other)
    if not isinstance(out, np.ndarray):
        out = np.array(out)

    return ms.Tensor.from_numpy(out)


class SinFunction(Function):
    @staticmethod
    def forward(ctx, input):
        out = np.sin(input.asnumpy())
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.asnumpy() * np.cos(input.asnumpy())
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def sin(input):
    return SinFunction.apply(input)


class CosFunction(Function):
    @staticmethod
    def forward(ctx, input):
        out = np.cos(input.asnumpy())
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output.asnumpy() * np.sin(input.asnumpy())
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def cos(input):
    return CosFunction.apply(input)


def triu(input, diagonal):
    out = np.triu(input.asnumpy(), diagonal)
    return ms.Tensor.from_numpy(out)


class SigmoidFunction(Function):
    @staticmethod
    def forward(ctx, input):
        input_np = input.asnumpy()
        out = 1 / (1 + np.exp(-input_np))
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(result)  # Save output for backward
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            grad_input = grad_output.asnumpy() * output.asnumpy() * (1 - output.asnumpy())
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def sigmoid(input):
    return SigmoidFunction.apply(input)


class TanhFunction(Function):
    @staticmethod
    def forward(ctx, input):
        input_np = input.asnumpy()
        out = np.tanh(input_np)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(result)  # Save output for backward
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # tanh'(x) = 1 - tanh(x)^2
            grad_input = grad_output.asnumpy() * (1 - output.asnumpy() * output.asnumpy())
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def tanh(input):
    return TanhFunction.apply(input)


class GeluFunction(Function):
    @staticmethod
    def forward(ctx, input, approximate='none'):
        input_np = input.asnumpy()
        if approximate == 'tanh':
            # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            M_SQRT2 = np.sqrt(2.0)
            M_2_SQRTPI = 2.0 / np.sqrt(np.pi)
            kBeta = M_SQRT2 * M_2_SQRTPI * 0.5
            kKappa = 0.044715
            x_cubed = input_np * input_np * input_np
            inner = kBeta * (input_np + kKappa * x_cubed)
            tanh_inner = np.tanh(inner)
            out = 0.5 * input_np * (1 + tanh_inner)
        else:
            # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
            M_SQRT1_2 = 1.0 / np.sqrt(2.0)
            out = 0.5 * input_np * (1 + scipy.special.erf(input_np * M_SQRT1_2))
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.approximate = approximate
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            input_np = input.asnumpy()
            grad_output_np = grad_output.asnumpy()
            M_SQRT2 = np.sqrt(2.0)
            M_SQRT1_2 = 1.0 / np.sqrt(2.0)
            M_2_SQRTPI = 2.0 / np.sqrt(np.pi)
            
            if ctx.approximate == 'tanh':
                kBeta = M_SQRT2 * M_2_SQRTPI * 0.5
                kKappa = 0.044715
                x_sq = input_np * input_np
                x_cube = x_sq * input_np
                inner = kBeta * (input_np + kKappa * x_cube)
                tanh_inner = np.tanh(inner)
                
                left = 0.5 * input_np
                right = 1 + tanh_inner
                left_derivative = 0.5 * right
                
                tanh_derivative = 1 - tanh_inner * tanh_inner
                inner_derivative = kBeta * (1 + 3 * kKappa * x_sq)
                right_derivative = left * tanh_derivative * inner_derivative
                
                grad_input = grad_output_np * (left_derivative + right_derivative)
            else:
                kAlpha = M_SQRT1_2
                kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5
                cdf = 0.5 * (1 + scipy.special.erf(input_np * kAlpha))
                pdf = kBeta * np.exp(input_np * input_np * -0.5)
                grad_input = grad_output_np * (cdf + input_np * pdf)
            
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input, None

def gelu(input, approximate='none'):
    return GeluFunction.apply(input, approximate)


class SiluFunction(Function):
    @staticmethod
    def forward(ctx, input):
        input_np = input.asnumpy()
        # SiLU(x) = x * sigmoid(x)
        # Use numerically stable sigmoid computation
        # For large positive x, sigmoid(x) ≈ 1, so SiLU(x) ≈ x
        # For large negative x, sigmoid(x) ≈ 0, so SiLU(x) ≈ 0
        # Use: sigmoid(x) = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        # But for very large x, we can use approximation
        sigmoid_x = np.where(input_np >= 0, 
                            1 / (1 + np.exp(-np.clip(input_np, -500, 500))),
                            np.exp(np.clip(input_np, -500, 500)) / (1 + np.exp(np.clip(input_np, -500, 500))))
        # For very large positive values (>= 20), sigmoid is effectively 1, so SiLU(x) = x
        # For very large negative values (<= -20), sigmoid is effectively 0, so SiLU(x) = 0
        # Use direct computation for these cases to avoid numerical errors
        if isinstance(input_np, np.ndarray):
            large_positive = input_np >= 20
            large_negative = input_np <= -20
            out = np.where(large_positive, input_np,
                          np.where(large_negative, 0.0,
                                  input_np * sigmoid_x))
            sigmoid_x = np.where(large_positive, 1.0,
                                np.where(large_negative, 0.0, sigmoid_x))
        else:
            # Scalar case
            if input_np >= 20:
                out = input_np
                sigmoid_x = 1.0
            elif input_np <= -20:
                out = 0.0
                sigmoid_x = 0.0
            else:
                out = input_np * sigmoid_x
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.sigmoid_x = sigmoid_x
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            input_np = input.asnumpy()
            sigmoid_x_np = ctx.sigmoid_x
            # SiLU'(x) = sigmoid(x) + x * sigmoid'(x)
            # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            sigmoid_derivative = sigmoid_x_np * (1 - sigmoid_x_np)
            grad_input = grad_output.asnumpy() * (sigmoid_x_np + input_np * sigmoid_derivative)
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def silu(input):
    return SiluFunction.apply(input)


def swish(input):
    # Swish is an alias for SiLU
    return SiluFunction.apply(input)


class MishFunction(Function):
    @staticmethod
    def forward(ctx, input):
        input_np = input.asnumpy()
        # Mish(x) = x * tanh(softplus(x))
        # softplus(x) = ln(1 + exp(x))
        softplus_x = np.log(1 + np.exp(input_np))
        tanh_softplus = np.tanh(softplus_x)
        out = input_np * tanh_softplus
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            input_np = input.asnumpy()
            # Mish'(x) = tanh(softplus(x)) + x * sech^2(softplus(x)) * sigmoid(x)
            softplus_x = np.log(1 + np.exp(input_np))
            tanh_softplus = np.tanh(softplus_x)
            sigmoid_x = 1 / (1 + np.exp(-input_np))
            sech_squared = 1 - tanh_softplus * tanh_softplus
            grad_input = grad_output.asnumpy() * (tanh_softplus + input_np * sech_squared * sigmoid_x)
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def mish(input):
    return MishFunction.apply(input)

class LeakyReluFunction(Function):
    @staticmethod
    def forward(ctx, input, negative_slope):
        x = input.asnumpy()
        y = np.where(x > 0, x, negative_slope * x)
        result = ms.Tensor.from_numpy(y)
        ctx.save_for_backward(input)
        ctx.negative_slope = negative_slope
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            x = input.asnumpy()
            grad = np.where(x > 0, 1.0, ctx.negative_slope)
            grad_input = ms.Tensor.from_numpy(grad_output.asnumpy() * grad)
        return grad_input, None

def leaky_relu(input, negative_slope):
    return LeakyReluFunction.apply(input, negative_slope)


class NegFunction(Function):
    @staticmethod
    def forward(ctx, input):
        out = -input.asnumpy()
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output.asnumpy()
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def neg(input):
    return NegFunction.apply(input)


def divmod(input, other, rounding_mode):
    if not isinstance(input, numbers.Number):
        input = input.asnumpy()
    if not isinstance(other, numbers.Number):
        other = other.asnumpy()

    if rounding_mode == 'floor':
        out = np.floor_divide(input, other)
    elif rounding_mode == 'trunc':
        out = np.trunc(np.true_divide(input, other)).astype(np.int64)

    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return ms.Tensor.from_numpy(out)


def unstack_ext_view(input, dim):
    arr = input.asnumpy()
    num_splits = arr.shape[dim]
    # Use split to unstack along the specified dimension
    if num_splits == 0:
        return []
    # Split into num_splits parts
    indices = list(range(1, num_splits))
    if len(indices) == 0:
        # If only one element, return it as a list
        outs = [arr]
    else:
        outs = np.split(arr, indices_or_sections=indices, axis=dim)
    # Squeeze the dimension that was split
    result = []
    for out in outs:
        out_squeezed = np.squeeze(out, axis=dim)
        result.append(ms.Tensor.from_numpy(out_squeezed))
    return result

def unstack_view(input, dim):
    # Alias for unstack_ext_view
    return unstack_ext_view(input, dim)


class StackFunction(Function):
    @staticmethod
    def forward(ctx, dim, *tensors):
        tensors_list = list(tensors)
        out = np.stack([t.asnumpy() for t in tensors_list], dim)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(*tensors_list)
        ctx.dim = dim
        ctx.num_tensors = len(tensors_list)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        tensors = ctx.saved_tensors
        grad_inputs = []
        grad_output_np = grad_output.asnumpy()
        
        # Unstack gradient along the stacking dimension
        grad_splits = np.split(grad_output_np, ctx.num_tensors, axis=ctx.dim)
        for i, grad_split in enumerate(grad_splits):
            grad_split = np.squeeze(grad_split, axis=ctx.dim)
            if ctx.needs_input_grad[i]:
                grad_inputs.append(ms.Tensor.from_numpy(grad_split))
            else:
                grad_inputs.append(None)
        
        return (None,) + tuple(grad_inputs)

def stack(tensors, dim):
    if isinstance(tensors, (list, tuple)):
        return StackFunction.apply(dim, *tensors)
    else:
        return StackFunction.apply(dim, tensors)


class SqrtFunction(Function):
    @staticmethod
    def forward(ctx, input):
        if isinstance(input, numbers.Number):
            input_np = np.array(input)
        else:
            input_np = input.asnumpy()
        out = np.sqrt(input_np)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input if not isinstance(input, numbers.Number) else None)
        ctx.is_number = isinstance(input, numbers.Number)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors if not ctx.is_number else (None,)
        grad_input = None
        if ctx.needs_input_grad[0] and not ctx.is_number and input is not None:
            input_np = input.asnumpy()
            grad_input = grad_output.asnumpy() / (2 * np.sqrt(input_np))
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def sqrt(input):
    return SqrtFunction.apply(input)


class TransposeViewFunction(Function):
    @staticmethod
    def forward(ctx, input, dim0, dim1):
        # Swap dimensions dim0 and dim1
        # Handle negative dimensions
        ndim = input.ndim
        # Normalize negative dimensions
        if dim0 < 0:
            dim0 = dim0 + ndim
        if dim1 < 0:
            dim1 = dim1 + ndim
        # Clamp dimensions to valid range (in case tensor was reshaped)
        # If dimension is >= ndim, treat it as the last dimension
        if dim0 >= ndim:
            dim0 = ndim - 1
        if dim1 >= ndim:
            dim1 = ndim - 1
        # Check bounds after normalization and clamping
        if dim0 < 0 or dim0 >= ndim:
            raise IndexError(f"Dimension out of range: dim0={dim0}, ndim={ndim}")
        if dim1 < 0 or dim1 >= ndim:
            raise IndexError(f"Dimension out of range: dim1={dim1}, ndim={ndim}")
        ranks = list(range(ndim))
        ranks[dim0], ranks[dim1] = ranks[dim1], ranks[dim0]
        out = np.transpose(input.asnumpy(), ranks)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.dim0 = dim0
        ctx.dim1 = dim1
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Transpose back
            ranks = list(range(grad_output.ndim))
            ranks[ctx.dim0], ranks[ctx.dim1] = ranks[ctx.dim1], ranks[ctx.dim0]
            grad_input = np.transpose(grad_output.asnumpy(), ranks)
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input, None, None

def transpose_view(input, dim0, dim1):
    return TransposeViewFunction.apply(input, dim0, dim1)


class PermuteFunction(Function):
    @staticmethod
    def forward(ctx, input, dims):
        # dims is a tuple/list specifying the new dimension order
        out = np.transpose(input.asnumpy(), dims)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.dims = dims
        # Calculate inverse permutation for backward
        ctx.inv_dims = tuple(dims.index(i) for i in range(len(dims)))
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Transpose back using inverse permutation
            grad_input = np.transpose(grad_output.asnumpy(), ctx.inv_dims)
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input, None

def permute(input, dims):
    return PermuteFunction.apply(input, dims)


def einsum(equation, operands):
    out = np.einsum(equation, *[o.asnumpy() for o in operands])
    return ms.Tensor.from_numpy(out)


def std(input, dim, correction, keepdim):
    out = np.std(input.asnumpy(), dim, ddof=float(correction), keepdims=keepdim)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return ms.Tensor.from_numpy(out)


def meshgrid(tensors, indexing):
    outs = np.meshgrid(*[t.asnumpy() for t in tensors], indexing=indexing)
    new_outs = ()
    for out in outs:
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        new_outs += (ms.Tensor.from_numpy(out),)
    return new_outs


def repeat_interleave_tensor(input, repeats, dim, _):
    out = np.repeat(input.asnumpy(), repeats, dim)
    return ms.Tensor.from_numpy(out)


def repeat_interleave_int(input, repeats, dim, _):
    out = np.repeat(input.asnumpy(), repeats, dim)
    return ms.Tensor.from_numpy(out)


class GreaterFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        if not isinstance(input, numbers.Number):
            input_np = input.asnumpy()
        else:
            input_np = input
        if not isinstance(other, numbers.Number):
            other_np = other.asnumpy()
        else:
            other_np = other
        out = input_np > other_np
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Comparison operations don't need gradients
        return None, None

def greater(input, other):
    return GreaterFunction.apply(input, other)


def linalg_vector_norm(input, p, dim, keepdim, dtype):
    out = np.linalg.norm(input.asnumpy(), p, dim, keepdim)
    return ms.Tensor.from_numpy(out)


class ExpFunction(Function):
    @staticmethod
    def forward(ctx, input):
        out = np.exp(input.asnumpy())
        if input.dtype == np.int64:
            out = out.astype(np.float32)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(result)  # Save output for backward
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.asnumpy() * output.asnumpy()
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def exp(input):
    return ExpFunction.apply(input)


class Expm1Function(Function):
    @staticmethod
    def forward(ctx, input):
        out = np.expm1(input.asnumpy())
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # d/dx expm1(x) = exp(x)
            grad_input = grad_output.asnumpy() * np.exp(input.asnumpy())
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def expm1(input):
    return Expm1Function.apply(input)


class OnesLikeFunction(Function):
    @staticmethod
    def forward(ctx, input, dtype=None):
        input_np = input.asnumpy()
        if dtype is not None:
            dtype_np = mindtorch.dtype2np[dtype]
        else:
            dtype_np = input_np.dtype
        out = np.ones_like(input_np, dtype=dtype_np)
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None

def ones_like(input, dtype=None):
    return OnesLikeFunction.apply(input, dtype)


def reverse_v2(input, dims):
    out = np.flip(input.asnumpy(), dims)
    return ms.Tensor.from_numpy(out)


class RsqrtFunction(Function):
    @staticmethod
    def forward(ctx, input):
        out = np.reciprocal(np.sqrt(input.asnumpy()))
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # d/dx (1/sqrt(x)) = -1/(2*x^(3/2))
            input_np = input.asnumpy()
            grad_input = -grad_output.asnumpy() / (2 * np.sqrt(input_np) * input_np)
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def rsqrt(input):
    return RsqrtFunction.apply(input)


class BitwiseXorTensorFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        out = np.bitwise_xor(input.asnumpy(), other.asnumpy())
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Bitwise operations are not differentiable
        return None, None

def bitwise_xor_tensor(input, other):
    return BitwiseXorTensorFunction.apply(input, other)


class MinimumFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        input_np = input.asnumpy()
        other_np = other.asnumpy()
        out = np.minimum(input_np, other_np)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input, other)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_tensors
        grad_input = None
        grad_other = None
        
        grad_output_np = grad_output.asnumpy()
        input_np = input.asnumpy()
        other_np = other.asnumpy()
        
        # Gradient flows to the smaller element
        mask = input_np <= other_np
        
        if ctx.needs_input_grad[0]:
            grad_input = np.where(mask, grad_output_np, 0)
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        
        if ctx.needs_input_grad[1]:
            grad_other = np.where(~mask, grad_output_np, 0)
            if not isinstance(grad_other, np.ndarray):
                grad_other = np.array(grad_other)
            grad_other = ms.Tensor.from_numpy(grad_other)
        
        return grad_input, grad_other

def minimum(input, other):
    return MinimumFunction.apply(input, other)

class MaximumFunction(Function):
    @staticmethod
    def forward(ctx, input, other):
        input_np = input.asnumpy()
        other_np = other.asnumpy()
        out = np.maximum(input_np, other_np)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input, other)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_tensors
        grad_input = None
        grad_other = None
        
        grad_output_np = grad_output.asnumpy()
        input_np = input.asnumpy()
        other_np = other.asnumpy()
        
        # Gradient flows to the larger element
        mask = input_np >= other_np
        
        if ctx.needs_input_grad[0]:
            grad_input = np.where(mask, grad_output_np, 0)
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        
        if ctx.needs_input_grad[1]:
            grad_other = np.where(~mask, grad_output_np, 0)
            if not isinstance(grad_other, np.ndarray):
                grad_other = np.array(grad_other)
            grad_other = ms.Tensor.from_numpy(grad_other)
        
        return grad_input, grad_other

def maximum(input, other):
    return MaximumFunction.apply(input, other)


class ProdFunction(Function):
    @staticmethod
    def forward(ctx, input, dim=None, keepdim=False, dtype=None):
        input_np = input.asnumpy()
        if dtype is not None:
            dtype_np = mindtorch.dtype2np[dtype]
        else:
            dtype_np = None
        out = np.prod(input_np, axis=dim, dtype=dtype_np, keepdims=keepdim)
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.input_shape = input.shape
        ctx.prod_value = result
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_output_np = grad_output.asnumpy()
            input_np = input.asnumpy()
            
            if ctx.dim is None:
                # Full reduction
                grad_input = grad_output_np * ctx.prod_value / input_np
            else:
                # Partial reduction
                # Expand dimensions if keepdim is False
                if not ctx.keepdim:
                    grad_output_np = np.expand_dims(grad_output_np, ctx.dim)
                    ctx.prod_value = np.expand_dims(ctx.prod_value, ctx.dim)
                # Broadcast to input shape
                grad_output_np = np.broadcast_to(grad_output_np, ctx.input_shape)
                ctx.prod_value = np.broadcast_to(ctx.prod_value, ctx.input_shape)
                grad_input = grad_output_np * ctx.prod_value / input_np
            
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input, None, None, None

def prod(input, dim=None, keepdim=False, dtype=None):
    return ProdFunction.apply(input, dim, keepdim, dtype)


def prod_ext(input, dim, keepdim, dtype):
    out = np.prod(input.asnumpy(), axis=dim, keepdims=keepdim)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return ms.Tensor.from_numpy(out)


def select(condition, input, other):
    if not isinstance(input, numbers.Number):
        input = input.asnumpy()
    if not isinstance(other, numbers.Number):
        other = other.asnumpy()

    out = np.where(condition.asnumpy(), input, other)
    return ms.Tensor.from_numpy(out)


def dense(input, weight, bias):
    output = np.dot(input.asnumpy(), weight.asnumpy().T)
    if bias is not None:
        output += bias
    return ms.Tensor.from_numpy(output)


def dropout_ext(input, p):
    if p != 0:
        mask = (np.random.rand(*input.shape) < (1 - p))
        out = input.asnumpy() * mask / (1 - p)
        return ms.Tensor.from_numpy(out), ms.Tensor.from_numpy(mask)
    else:
        return input, None


def dropout(input, p, training=True):
    # dropout function - similar to dropout_ext but with training flag
    if not training or p == 0:
        return input
    # Use dropout_ext implementation
    result, _ = dropout_ext(input, p)
    return result


class FloorFunction(Function):
    @staticmethod
    def forward(ctx, input):
        out = np.floor(input.asnumpy())
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        result = ms.Tensor.from_numpy(out)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # floor is not differentiable, but we pass through the gradient
        # (similar to how PyTorch handles it)
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.asnumpy()
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def floor(input):
    return FloorFunction.apply(input)


def chunk(input, chunks, dim):
    out = np.array_split(input.asnumpy(), chunks, dim)
    out = [ms.Tensor.from_numpy(o) for o in out]
    return out


def narrow(input, dim, start, length):
    slices = [slice(None)] * input.ndim
    # 将指定维度的切片修改为 [start: start+length]
    slices[dim] = slice(start, start + length)
    # 应用切片并返回视图
    out = input.asnumpy()[tuple(slices)]
    return ms.Tensor.from_numpy(out)


def roll(input, shifts, dims):
    out = np.roll(input.asnumpy(), shifts, dims)
    return ms.Tensor.from_numpy(out)


def outer(input, other):
    out = np.outer(input.asnumpy(), other.asnumpy())
    return ms.Tensor.from_numpy(out)


def one_hot_ext(tensor, num_classes=-1):
    if num_classes == -1:
        num_classes = np.max(tensor.asnumpy()) + 1  # 自动确定类别数[2](@ref)
    
    out = np.eye(num_classes)[tensor.asnumpy()]
    return ms.Tensor.from_numpy(out)

def one_hot(tensor, num_classes):
    """
    Compute one-hot encoding of integer tensor.
    
    Args:
        tensor: Input tensor of integer indices
        num_classes: Number of classes (depth of one-hot dimension)
    
    Returns:
        One-hot encoded tensor with shape (*, num_classes) where * is the input shape
    """
    if num_classes == -1:
        # Auto-determine num_classes from max value in tensor
        tensor_np = tensor.asnumpy()
        num_classes = int(np.max(tensor_np)) + 1
    
    # Use numpy's eye to create one-hot encoding
    tensor_np = tensor.asnumpy()
    out = np.eye(num_classes, dtype=np.int64)[tensor_np]
    
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return ms.Tensor.from_numpy(out)


def logsumexp(input, dim, keepdim=False):
    """
    Compute log(sum(exp(input))) along the specified dimension.
    Uses numerical stability trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    
    Args:
        input: Input tensor
        dim: Dimension along which to compute logsumexp
        keepdim: Whether to keep the reduced dimension
    
    Returns:
        Tensor with logsumexp computed along the specified dimension
    """
    input_np = input.asnumpy()
    
    # Compute max along the specified dimension with keepdims=True for broadcasting
    input_max_np = np.max(input_np, axis=dim, keepdims=True)
    
    # Subtract max for numerical stability: exp(input - max)
    input_exp_np = np.exp(input_np - input_max_np)
    
    # Sum along the dimension
    input_sumexp_np = np.sum(input_exp_np, axis=dim, keepdims=keepdim)
    
    # Take log
    input_logsumexp_np = np.log(input_sumexp_np)
    
    # Add back the max
    if not keepdim:
        # Squeeze the dimension from input_max_np to match the shape
        input_max_np = np.squeeze(input_max_np, axis=dim)
    
    result_np = input_logsumexp_np + input_max_np
    
    if not isinstance(result_np, np.ndarray):
        result_np = np.array(result_np)
    
    return ms.Tensor.from_numpy(result_np)


def baddbmm(input, batch1, batch2, alpha=1, beta=1):
    """
    Performs batch matrix-matrix product of matrices in batch1 and batch2,
    with input added to the final result.
    
    Formula: output = beta * input + alpha * (batch1 @ batch2)
    
    Args:
        input: Tensor to be added (broadcastable with (B, N, P))
        batch1: First batch of matrices, shape (B, N, M)
        batch2: Second batch of matrices, shape (B, M, P)
        alpha: Multiplier for batch1 @ batch2, default 1
        beta: Multiplier for input, default 1
    
    Returns:
        Tensor of shape (B, N, P)
    """
    # Compute batch matrix multiplication: batch1 @ batch2
    bmm_result = bmm(batch1, batch2)
    
    # Scale and add: beta * input + alpha * bmm_result
    # This matches the cpu.py implementation: add(mul(beta, input), mul(alpha, bmm(batch1, batch2)))
    return add(mul(beta, input), mul(alpha, bmm_result))


class Log1pFunction(Function):
    @staticmethod
    def forward(ctx, input):
        out = np.log1p(input.asnumpy())
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # d/dx log1p(x) = 1/(1+x)
            grad_input = grad_output.asnumpy() / (1 + input.asnumpy())
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def log1p(input):
    return Log1pFunction.apply(input)


def gather(input, indices, _dimension):
    out = np.take(input.asnumpy(), indices.asnumpy(), _dimension)
    return ms.Tensor.from_numpy(out)


class ScatterFunction(Function):
    @staticmethod
    def forward(ctx, input, dim, index, src):
        input_np = input.asnumpy()
        index_np = index.asnumpy()
        src_np = src.asnumpy()
        
        # Create output as a copy of input
        out = np.copy(input_np)
        
        # Scatter operation: out[index[i][j][k]][j][k] = src[i][j][k] for dim=0
        # Use advanced indexing to scatter values
        indices_list = []
        for d in range(input_np.ndim):
            if d == dim:
                indices_list.append(index_np)
            else:
                # Create meshgrid for other dimensions
                shape = list(index_np.shape)
                shape[d] = input_np.shape[d]
                indices_list.append(np.broadcast_to(np.arange(input_np.shape[d]).reshape([-1 if i == d else 1 for i in range(input_np.ndim)]), shape))
        
        # Use advanced indexing to scatter
        out[tuple(indices_list)] = src_np
        
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input, index, src)
        ctx.dim = dim
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, index, src = ctx.saved_tensors
        grad_input = None
        grad_src = None
        
        if ctx.needs_input_grad[0]:
            # Gradient for input: pass through where not scattered, zero where scattered
            grad_input = grad_output.asnumpy().copy()
            # Zero out scattered positions
            indices_list = []
            for d in range(grad_output.ndim):
                if d == ctx.dim:
                    indices_list.append(index.asnumpy())
                else:
                    shape = list(index.asnumpy().shape)
                    shape[d] = grad_output.shape[d]
                    indices_list.append(np.broadcast_to(np.arange(grad_output.shape[d]).reshape([-1 if i == d else 1 for i in range(grad_output.ndim)]), shape))
            grad_input[tuple(indices_list)] = 0
            grad_input = ms.Tensor.from_numpy(grad_input)
        
        if ctx.needs_input_grad[3]:  # src gradient
            # Gradient for src: gather from grad_output
            indices_list = []
            for d in range(grad_output.ndim):
                if d == ctx.dim:
                    indices_list.append(index.asnumpy())
                else:
                    shape = list(index.asnumpy().shape)
                    shape[d] = grad_output.shape[d]
                    indices_list.append(np.broadcast_to(np.arange(grad_output.shape[d]).reshape([-1 if i == d else 1 for i in range(grad_output.ndim)]), shape))
            grad_src = grad_output.asnumpy()[tuple(indices_list)]
            grad_src = ms.Tensor.from_numpy(grad_src)
        
        return grad_input, None, None, grad_src

def scatter(input, dim, index, src):
    return ScatterFunction.apply(input, dim, index, src)


def scatter_add_ext(input, dim, index, src):
    """
    Scatter add operation: adds values from src to input at positions specified by index along dimension dim.
    
    Args:
        input: Input tensor
        dim: Dimension along which to scatter
        index: Indices where to scatter
        src: Source tensor with values to add
    
    Returns:
        Output tensor with values added
    """
    input_np = input.asnumpy()
    index_np = index.asnumpy()
    src_np = src.asnumpy()
    
    # Create output as a copy of input
    out = np.copy(input_np)
    
    # Scatter add operation: out[index[i][j][k]][j][k] += src[i][j][k] for dim=0
    # Use advanced indexing to scatter add values
    indices_list = []
    for d in range(input_np.ndim):
        if d == dim:
            indices_list.append(index_np)
        else:
            # Create meshgrid for other dimensions
            shape = list(index_np.shape)
            shape[d] = input_np.shape[d]
            indices_list.append(np.broadcast_to(np.arange(input_np.shape[d]).reshape([-1 if i == d else 1 for i in range(input_np.ndim)]), shape))
    
    # Use np.add.at for in-place addition (handles duplicate indices correctly)
    np.add.at(out, tuple(indices_list), src_np)
    
    return ms.Tensor.from_numpy(out)


def layer_norm_ext(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    # 确定需要计算均值和方差的轴
    # 从第一个维度开始到 normalized_shape 所涵盖的维度之前的维度会被保留（即 batch 维度等）
    # 我们需要计算所有不在最后 len(normalized_shape) 个维度上的轴的均值和方差
    input = input.asnumpy()
    if weight is not None:
        weight = weight.asnumpy()
    if bias is not None:
        bias = bias.asnumpy()

    start_axis = input.ndim - len(normalized_shape)
    axes = tuple(range(start_axis, input.ndim))
    
    # 计算均值和方差，并保持维度以便广播
    mean = np.mean(input, axis=axes, keepdims=True)
    var = np.var(input, axis=axes, keepdims=True)
    
    # 标准化: (x - mean) / sqrt(var + eps)
    normalized = (input - mean) / np.sqrt(var + eps)
    
    # 应用可学习的缩放和平移参数 (gamma 和 beta)
    if weight is not None:
        normalized = normalized * weight
    if bias is not None:
        normalized = normalized + bias
    
    return (ms.Tensor.from_numpy(normalized),)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    # Use the same implementation as layer_norm_ext
    result = layer_norm_ext(input, normalized_shape, weight, bias, eps)
    # layer_norm_ext returns a tuple, but layer_norm should return a single tensor
    if isinstance(result, tuple):
        return result[0]
    return result


def erf(input):
    out = scipy.special.erf(input.asnumpy())
    return ms.Tensor.from_numpy(out)


def mse_loss_ext(input, target, reduction='mean'):
    if input.shape != target.shape:
        raise ValueError(f"Input and target must have the same shape. Got input: {input.shape}, target: {target.shape}")

    squared_errors = np.square(input - target)

    if reduction == 'mean':
        loss = np.mean(squared_errors)
    elif reduction == 'sum':
        loss = np.sum(squared_errors)
    elif reduction == 'none':
        loss = squared_errors
    else:
        raise ValueError("Reduction must be 'mean', 'sum', or 'none'.")

    if not isinstance(loss, np.ndarray):
        loss = np.array(loss)
    return ms.Tensor.from_numpy(loss)


def mse_loss(input, target, reduction):
    """
    Computes the mean squared error (MSE) loss between input and target.
    
    Args:
        input: Input tensor
        target: Target tensor (must have the same shape as input)
        reduction: Specifies the reduction to apply to the output:
            - 'none': no reduction will be applied
            - 'mean': the sum of the output will be divided by the number of elements
            - 'sum': the output will be summed
    
    Returns:
        Tensor containing the MSE loss
    """
    # Convert to numpy if needed
    if hasattr(input, 'asnumpy'):
        input_np = input.asnumpy()
    else:
        input_np = np.asarray(input)
    
    if hasattr(target, 'asnumpy'):
        target_np = target.asnumpy()
    else:
        target_np = np.asarray(target)
    
    if input_np.shape != target_np.shape:
        raise ValueError(f"Input and target must have the same shape. Got input: {input_np.shape}, target: {target_np.shape}")

    # Compute squared errors: (input - target)^2
    squared_errors = np.square(input_np - target_np)

    # Apply reduction
    if reduction == 'mean':
        loss = np.mean(squared_errors)
    elif reduction == 'sum':
        loss = np.sum(squared_errors)
    elif reduction == 'none':
        loss = squared_errors
    else:
        raise ValueError(f"Reduction must be 'mean', 'sum', or 'none', got '{reduction}'")

    if not isinstance(loss, np.ndarray):
        loss = np.array(loss)
    return ms.Tensor.from_numpy(loss)


class SquareFunction(Function):
    @staticmethod
    def forward(ctx, input):
        out = np.square(input.asnumpy())
        result = ms.Tensor.from_numpy(out)
        ctx.save_for_backward(input)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # d/dx (x^2) = 2*x
            grad_input = grad_output.asnumpy() * 2 * input.asnumpy()
            if not isinstance(grad_input, np.ndarray):
                grad_input = np.array(grad_input)
            grad_input = ms.Tensor.from_numpy(grad_input)
        return grad_input

def square(input):
    return SquareFunction.apply(input)


def lgamma(input):
    out = scipy.special.gammaln(input.asnumpy())
    return ms.Tensor.from_numpy(out)


def gamma(shape, alpha, beta):
    out = np.random.gamma(alpha, 1/beta, shape)
    return ms.Tensor.from_numpy(out)


def gather_d(input, dim, index):
    # Convert to numpy using asnumpy() since all inputs are mindspore.Tensor
    input_np = input.asnumpy()
    index_np = index.asnumpy()
    
    # Build indices for advanced indexing
    indices = []
    for axis in range(input.ndim):
        if axis == dim:
            indices.append(index_np)
        else:
            # Create meshgrid-like index for other dimensions
            shape = list(index_np.shape)
            shape[axis] = input.shape[axis]
            # Create indices for this axis: [0, 1, 2, ..., input.shape[axis]-1]
            axis_indices = np.arange(input.shape[axis])
            # Reshape to broadcast correctly
            for i in range(len(shape)):
                if i != axis:
                    axis_indices = np.expand_dims(axis_indices, i)
            # Broadcast to the full shape
            axis_indices = np.broadcast_to(axis_indices, shape)
            indices.append(axis_indices)
    
    # Use advanced indexing
    out = input_np[tuple(indices)]
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return ms.Tensor.from_numpy(out)


class TakeAlongDimFunction(Function):
    @staticmethod
    def forward(ctx, input, indices, dim=None):
        # Convert to numpy using asnumpy() since all inputs are mindspore.Tensor
        input_np = input.asnumpy()
        indices_np = indices.asnumpy()
        
        # Normalize dim
        if dim is not None:
            if dim < 0:
                dim = dim + input_np.ndim
        
        # Use numpy's take_along_axis directly
        if dim is None:
            # Flatten both tensors
            input_flat = input_np.reshape(-1)
            indices_flat = indices_np.reshape(-1)
            result = np.take_along_axis(input_flat, indices_flat, axis=0)
            # Reshape result back to indices shape
            result = result.reshape(indices_np.shape)
        else:
            result = np.take_along_axis(input_np, indices_np, axis=dim)
        
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        
        # Save for backward
        ctx.save_for_backward(input, indices)
        ctx.dim = dim
        ctx.input_shape = input_np.shape
        ctx.input_ndim = input_np.ndim
        
        return ms.Tensor.from_numpy(result)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, indices = ctx.saved_tensors
        dim = ctx.dim
        
        grad_input = None
        
        if ctx.needs_input_grad[0] and grad_output is not None:
            grad_output_np = grad_output.asnumpy()
            indices_np = indices.asnumpy()
            
            # Initialize grad_input with zeros
            grad_input_np = np.zeros(ctx.input_shape, dtype=grad_output_np.dtype)
            
            # Use put_along_axis to scatter grad_output back to input positions
            if dim is None:
                # Flatten case
                grad_input_flat = grad_input_np.reshape(-1)
                indices_flat = indices_np.reshape(-1)
                grad_output_flat = grad_output_np.reshape(-1)
                # Use put_along_axis to scatter
                np.put_along_axis(grad_input_flat, indices_flat, grad_output_flat, axis=0)
            else:
                # Normalize dim
                if dim < 0:
                    dim = dim + ctx.input_ndim
                # Use put_along_axis to scatter grad_output back
                np.put_along_axis(grad_input_np, indices_np, grad_output_np, axis=dim)
            
            if not isinstance(grad_input_np, np.ndarray):
                grad_input_np = np.array(grad_input_np)
            
            grad_input = ms.Tensor.from_numpy(grad_input_np)
        
        # indices doesn't need gradient
        return grad_input, None, None

def take_along_dim(input, indices, dim=None):
    """
    Take values from input tensor along the specified dimension using indices.
    
    Args:
        input (Tensor): The input tensor.
        indices (Tensor): The indices tensor. Must have the same number of dimensions as input.
        dim (int, optional): The dimension along which to take values. If None, input and indices are flattened.
    
    Returns:
        Tensor: The result tensor with values taken from input along the specified dimension.
    """
    return TakeAlongDimFunction.apply(input, indices, dim)



def log_softmax(input, dim=-1, dtype=None):
    # Support both 'dim' and 'axis' for compatibility
    if isinstance(dim, str) and dim == 'axis':
        dim = -1
    x = input.asnumpy()
    if dtype is not None:
        x = x.astype(mindtorch.dtype2np[dtype])
    x_max = np.max(x, axis=dim, keepdims=True)
    x_shifted = x - x_max
    
    exp_x = np.exp(x_shifted)
    sum_exp_x = np.sum(exp_x, axis=dim, keepdims=True)
    log_sum_exp_x = np.log(sum_exp_x)
    
    out = x_shifted - log_sum_exp_x
    return ms.Tensor.from_numpy(out)


def nllloss(input, target, weight=None, reduction='mean', ignore_index=-100):
    # INSERT_YOUR_CODE
    input_np = input.asnumpy()
    target_np = target.asnumpy()

    # If weight is provided, convert to numpy array
    if weight is not None:
        weight_np = weight.asnumpy()
    else:
        weight_np = None

    # input: (N, C, ...) or (C, ...) if unbatched
    # target: (N, ...) or (...), class indices

    # flatten input and target if necessary
    input_shape = input_np.shape
    target_shape = target_np.shape

    # For multi-dimensional, reshape as needed
    n_classes = input_shape[1] if len(input_shape) > 1 else input_shape[0]

    # Broadcast target to flattened indices where possible
    if input_np.ndim > 2:
        # For shape (N, C, d1, d2, ...)
        n = input_np.shape[0]
        c = input_np.shape[1]
        rest = input_np.shape[2:]
        input_flat = input_np.reshape(n, c, -1)
        target_flat = target_np.reshape(n, -1)
        out_shape = target_np.shape
    elif input_np.ndim == 2:
        n = input_np.shape[0]
        c = input_np.shape[1]
        input_flat = input_np
        target_flat = target_np
        out_shape = target_np.shape
    else:
        # (C,), target is scalar
        input_flat = input_np[None, :]
        target_flat = np.expand_dims(target_np, axis=0)
        out_shape = ()
        n = 1

    # Get value at the class index, use ignore_index to mask
    if ignore_index is not None:
        mask = (target_flat != ignore_index)
    else:
        mask = np.ones_like(target_flat, dtype=bool)

    # For each sample, select the log-prob for the true class.
    # Set to 0 if ignore_index, these will be ignored in reduction
    nll = np.zeros_like(target_flat, dtype=input_np.dtype)
    for ix in np.ndindex(target_flat.shape):
        tgt = target_flat[ix]
        if mask[ix]:
            if tgt < 0 or tgt >= n_classes:
                # Index out of bounds, set as 0 (or could raise)
                nll[ix] = 0.
            else:
                if input_flat.shape == target_flat.shape:
                    # Rare, input is (d0, d1, ...)
                    nll[ix] = -input_flat[ix + (tgt,)]
                elif input_flat.ndim == 2:
                    # input_flat (N, C), target_flat (N,)
                    nll[ix] = -input_flat[ix[0], tgt]
                elif input_flat.ndim == 3:
                    # input_flat (N, C, d), target_flat (N, d)
                    nll[ix] = -input_flat[ix[0], tgt, ix[1]]
                else:
                    # fallback
                    nll[ix] = -input_flat[ix[0], tgt]
        # else: nll[ix] = 0. (already zero)

    # Apply weighting if needed
    if weight_np is not None:
        # Broadcast weights by target (for each position)
        wt = np.zeros_like(target_flat, dtype=input_np.dtype)
        for ix in np.ndindex(target_flat.shape):
            tgt = target_flat[ix]
            if mask[ix] and tgt >= 0 and tgt < len(weight_np):
                wt[ix] = weight_np[tgt]
            else:
                wt[ix] = 0
        nll = nll * wt
        total_weight = np.sum(wt[mask])
    else:
        total_weight = np.sum(mask)

    # Apply reduction
    if reduction == 'none':
        result_np = nll.reshape(out_shape)
    elif reduction == 'sum':
        result_np = np.sum(nll)
    else:  # 'mean' or default
        if total_weight > 0:
            result_np = np.sum(nll) / total_weight
        else:
            result_np = np.sum(nll)  # Avoid div/0, matches torch.nn.functional nll_loss

    # Convert back to tensor
    result = ms.Tensor.from_numpy(np.array(result_np))
    return result


def diag_ext(input, diagonal):
    out = np.diag(input.asnumpy(), diagonal)
    return ms.Tensor.from_numpy(out)

def diag(input, diagonal):
    """
    Extract a diagonal or construct a diagonal matrix.
    
    Args:
        input: Input tensor (1D or 2D)
        diagonal: Diagonal offset (0 = main diagonal, >0 = above, <0 = below)
    
    Returns:
        If input is 1D: returns a 2D diagonal matrix
        If input is 2D: returns a 1D tensor with diagonal elements
    """
    out = np.diag(input.asnumpy(), diagonal)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return ms.Tensor.from_numpy(out)


def sign(input):
    out = np.sign(input.asnumpy())
    return ms.Tensor.from_numpy(out)


def log2(input):
    out = np.log2(input.asnumpy())
    return ms.Tensor.from_numpy(out)


def inplace_zero(input):
    input_np = input.asnumpy()
    other_np = np.zeros_like(input_np)
    # Handle 0-dimensional arrays (scalars)
    if input_np.ndim == 0:
        input_np[()] = other_np
    else:
        input_np[:] = other_np
    return input

def cumprod(input, dim, dtype):
    out = np.cumprod(input.asnumpy(), dim, mindtorch.dtype2np[dtype])
    return ms.Tensor.from_numpy(out)

class MultinomialFunction(Function):
    @staticmethod
    def forward(ctx, input, num_samples, replacement, generator):
        input_np = input.asnumpy()
        
        # Normalize probabilities (input should be probabilities or logits)
        # If input contains negative values, treat as logits and apply softmax
        if np.any(input_np < 0):
            # Logits: apply exp and normalize
            exp_input = np.exp(input_np - np.max(input_np, axis=-1, keepdims=True))
            probs = exp_input / np.sum(exp_input, axis=-1, keepdims=True)
        else:
            # Probabilities: normalize
            probs = input_np / np.sum(input_np, axis=-1, keepdims=True)
        
        # Get shape
        batch_shape = probs.shape[:-1]
        num_categories = probs.shape[-1]
        
        if replacement:
            # With replacement: use cumulative distribution and searchsorted
            # Generate uniform random samples
            if generator is not None:
                seed, offset = generator._step(12)
                # Handle both scalar and tensor seed
                if hasattr(seed, 'item') and seed.numel() == 1:
                    np.random.seed(seed.item())
                elif hasattr(seed, 'numpy'):
                    seed_np = seed.asnumpy()
                    if seed_np.size == 1:
                        np.random.seed(int(seed_np.item()))
                    else:
                        np.random.seed(int(seed_np.flat[0]))
                else:
                    np.random.seed(int(seed))
            uniform_samples = np.random.random(batch_shape + (num_samples,))
            
            # Compute cumulative probabilities
            cum_probs = np.cumsum(probs, axis=-1)
            
            # Use searchsorted to find indices
            # Flatten for easier processing
            cum_probs_flat = cum_probs.reshape(-1, num_categories)
            uniform_flat = uniform_samples.reshape(-1, num_samples)
            samples_flat = np.zeros((cum_probs_flat.shape[0], num_samples), dtype=np.int64)
            
            for i in range(cum_probs_flat.shape[0]):
                samples_flat[i] = np.searchsorted(cum_probs_flat[i], uniform_flat[i], side='right')
            
            samples = samples_flat.reshape(batch_shape + (num_samples,))
        else:
            # Without replacement: use Gumbel-max trick
            if generator is not None:
                seed, offset = generator._step(12)
                # Handle both scalar and tensor seed
                if hasattr(seed, 'item') and seed.numel() == 1:
                    np.random.seed(seed.item())
                elif hasattr(seed, 'numpy'):
                    seed_np = seed.asnumpy()
                    if seed_np.size == 1:
                        np.random.seed(int(seed_np.item()))
                    else:
                        np.random.seed(int(seed_np.flat[0]))
                else:
                    np.random.seed(int(seed))
            
            # Generate Gumbel noise: -log(-log(U)) where U ~ Uniform(0,1)
            uniform = np.random.random(probs.shape)
            gumbel = -np.log(-np.log(uniform + 1e-10) + 1e-10)
            
            # Add log probabilities
            log_probs = np.log(probs + 1e-10)
            vals = log_probs + gumbel
            
            # Get top k indices
            # Flatten for easier processing
            vals_flat = vals.reshape(-1, num_categories)
            topk_indices = np.argpartition(vals_flat, -num_samples, axis=-1)[:, -num_samples:]
            # Sort if needed
            for i in range(vals_flat.shape[0]):
                topk_indices[i] = topk_indices[i][np.argsort(vals_flat[i, topk_indices[i]])[::-1]]
            
            samples = topk_indices.reshape(batch_shape + (num_samples,))
        
        result = ms.Tensor.from_numpy(samples.astype(np.int64))
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # multinomial is not differentiable
        return None, None, None, None

def multinomial(input, num_samples, replacement, generator):
    return MultinomialFunction.apply(input, num_samples, replacement, generator)

def sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False):
    """
    Scaled Dot Product Attention implementation using numpy.
    """
    import math
    
    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    
    query_np = query.asnumpy()
    key_np = key.asnumpy()
    value_np = value.asnumpy()
    
    # Handle GQA (Grouped Query Attention)
    if enable_gqa:
        key_repeats = query.shape[-3] // key.shape[-3]
        value_repeats = query.shape[-3] // value.shape[-3]
        if key_repeats > 1:
            key_np = np.repeat(key_np, key_repeats, axis=-3)
        if value_repeats > 1:
            value_np = np.repeat(value_np, value_repeats, axis=-3)
    
    # Initialize attention bias
    if attn_mask is None:
        attn_bias = np.zeros((L, S), dtype=query_np.dtype)
    else:
        attn_bias = np.zeros_like(attn_mask.asnumpy() if hasattr(attn_mask, 'numpy') else attn_mask)
    
    # Apply causal mask
    if is_causal:
        assert attn_mask is None, "Cannot use both is_causal and attn_mask"
        temp_mask = np.tril(np.ones((L, S), dtype=bool), k=0)
        attn_bias = np.where(temp_mask, attn_bias, -np.inf)
    
    # Apply attention mask
    if attn_mask is not None:
        attn_mask_np = attn_mask.asnumpy() if hasattr(attn_mask, 'numpy') else attn_mask
        if attn_mask_np.dtype == bool:
            attn_bias = np.where(attn_mask_np, attn_bias, -np.inf)
        else:
            attn_bias = attn_bias + attn_mask_np
    
    # Compute attention weights: Q @ K^T * scale
    attn_weight = np.matmul(query_np, np.swapaxes(key_np, -2, -1)) * scale_factor
    attn_weight = attn_weight + attn_bias
    
    # Apply softmax
    # Subtract max for numerical stability
    attn_weight_exp = np.exp(attn_weight - np.max(attn_weight, axis=-1, keepdims=True))
    attn_weight = attn_weight_exp / np.sum(attn_weight_exp, axis=-1, keepdims=True)
    
    # Apply dropout
    if dropout_p > 0.0:
        dropout_mask = np.random.random(attn_weight.shape) > dropout_p
        attn_weight = attn_weight * dropout_mask / (1 - dropout_p)
    
    # Compute output: attn_weight @ V
    output = np.matmul(attn_weight, value_np)
    
    return ms.Tensor.from_numpy(output)


def linalg_qr(input_x, mode):
    """
    Compute the QR decomposition of a matrix.
    
    Args:
        input_x: Input tensor of shape (*, m, n)
        mode: One of 'reduced', 'complete', or 'r'
            - 'reduced': Returns Q of shape (*, m, k) and R of shape (*, k, n) where k = min(m, n)
            - 'complete': Returns Q of shape (*, m, m) and R of shape (*, m, n)
            - 'r': Returns empty Q and R of shape (*, k, n)
    
    Returns:
        Tuple of (Q, R) tensors
    """
    input_np = input_x.asnumpy()
    
    # Handle mode parameter
    if mode == 'complete':
        # numpy.linalg.qr uses 'full' for complete mode
        Q_np, R_np = np.linalg.qr(input_np, mode='full')
        Q = ms.Tensor.from_numpy(Q_np)
        R = ms.Tensor.from_numpy(R_np)
        return Q, R
    elif mode == 'reduced':
        Q_np, R_np = np.linalg.qr(input_np, mode='reduced')
        Q = ms.Tensor.from_numpy(Q_np)
        R = ms.Tensor.from_numpy(R_np)
        return Q, R
    elif mode == 'r':
        # For 'r' mode, compute only R and return empty Q
        _, R_np = np.linalg.qr(input_np, mode='reduced')
        # Create empty Q tensor with appropriate shape
        # Q should be empty, so we create a tensor with shape (0,)
        Q = ms.Tensor.from_numpy(np.array([], dtype=input_np.dtype).reshape(0))
        R = ms.Tensor.from_numpy(R_np)
        return Q, R
    else:
        raise ValueError(f"mode must be one of 'reduced', 'complete', or 'r', got {mode}")


def max_pool2d(input, kernel_size, stride=1, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    """
    Applies a 2D max pooling over an input signal composed of several input planes.
    
    Args:
        input: Input tensor of shape (N, C, H, W)
        kernel_size: Size of the pooling kernel, can be a single number or a tuple (h, w)
        stride: Stride of the pooling operation, can be a single number or a tuple (h, w)
        padding: Padding added to both sides of the input, can be a single number or a tuple (h, w)
        dilation: Spacing between kernel elements, can be a single number or a tuple (h, w)
        ceil_mode: If True, will use ceil instead of floor to compute the output shape
        return_indices: If True, will return the indices along with the outputs
    
    Returns:
        Output tensor, and optionally indices tensor if return_indices=True
    """
    input_np = input.asnumpy()
    
    # Ensure input is 4D: (N, C, H, W)
    if input_np.ndim != 4:
        raise ValueError(f"max_pool2d expects 4D input, got {input_np.ndim}D")
    
    N, C, H, W = input_np.shape
    
    # Normalize parameters to tuples
    if isinstance(kernel_size, (int, numbers.Number)):
        kernel_size = (int(kernel_size), int(kernel_size))
    else:
        kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
    
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, (int, numbers.Number)):
        stride = (int(stride), int(stride))
    else:
        stride = (int(stride[0]), int(stride[1]))
    
    if isinstance(padding, (int, numbers.Number)):
        padding = (int(padding), int(padding))
    else:
        padding = (int(padding[0]), int(padding[1]))
    
    if isinstance(dilation, (int, numbers.Number)):
        dilation = (int(dilation), int(dilation))
    else:
        dilation = (int(dilation[0]), int(dilation[1]))
    
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    
    # Calculate effective kernel size with dilation
    eff_kh = kh + (kh - 1) * (dh - 1)
    eff_kw = kw + (kw - 1) * (dw - 1)
    
    # Calculate output dimensions
    if ceil_mode:
        out_h = int(np.ceil((H + 2 * ph - eff_kh) / sh)) + 1
        out_w = int(np.ceil((W + 2 * pw - eff_kw) / sw)) + 1
    else:
        out_h = int(np.floor((H + 2 * ph - eff_kh) / sh)) + 1
        out_w = int(np.floor((W + 2 * pw - eff_kw) / sw)) + 1
    
    # Adjust output size if needed (handle edge cases)
    if (out_h - 1) * sh >= H + 2 * ph - eff_kh + 1:
        out_h -= 1
    if (out_w - 1) * sw >= W + 2 * pw - eff_kw + 1:
        out_w -= 1
    
    # Pad input
    if ph > 0 or pw > 0:
        input_padded = np.pad(input_np, ((0, 0), (0, 0), (ph, ph), (pw, pw)), 
                              mode='constant', constant_values=-np.inf)
    else:
        input_padded = input_np
    
    # Initialize output
    output = np.zeros((N, C, out_h, out_w), dtype=input_np.dtype)
    indices = None
    if return_indices:
        indices = np.zeros((N, C, out_h, out_w), dtype=np.int64)
    
    # Perform max pooling
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * sh
            w_start = j * sw
            
            # Extract the pooling window with dilation
            window = np.full((N, C, kh, kw), -np.inf, dtype=input_np.dtype)
            window_indices_h = np.zeros((N, C, kh, kw), dtype=np.int64)
            window_indices_w = np.zeros((N, C, kh, kw), dtype=np.int64)
            
            for ki in range(kh):
                for kj in range(kw):
                    h_idx = h_start + ki * dh
                    w_idx = w_start + kj * dw
                    if 0 <= h_idx < input_padded.shape[2] and 0 <= w_idx < input_padded.shape[3]:
                        window[:, :, ki, kj] = input_padded[:, :, h_idx, w_idx]
                        # Convert to original input coordinates (remove padding)
                        orig_h = h_idx - ph
                        orig_w = w_idx - pw
                        window_indices_h[:, :, ki, kj] = orig_h
                        window_indices_w[:, :, ki, kj] = orig_w
            
            # Compute max over the window
            window_flat = window.reshape(N, C, kh * kw)
            max_pos_flat = np.argmax(window_flat, axis=2)
            max_ki = max_pos_flat // kw
            max_kj = max_pos_flat % kw
            
            # Get max values
            batch_idx = np.arange(N)[:, None]
            channel_idx = np.arange(C)[None, :]
            max_vals = window[batch_idx, channel_idx, max_ki, max_kj]
            output[:, :, i, j] = max_vals
            
            # Compute indices if needed
            if return_indices:
                # Get the indices of max values in original input
                max_h = window_indices_h[batch_idx, channel_idx, max_ki, max_kj]
                max_w = window_indices_w[batch_idx, channel_idx, max_ki, max_kj]
                # Store as linear index (row * W + col) in the original input
                # Clamp to valid range
                max_h = np.clip(max_h, 0, H - 1)
                max_w = np.clip(max_w, 0, W - 1)
                indices[:, :, i, j] = max_h * W + max_w
    
    result = ms.Tensor.from_numpy(output)
    
    if return_indices:
        indices_tensor = ms.Tensor.from_numpy(indices)
        return result, indices_tensor
    return result


def _get_unfold_indices(input_shape, dimension, size, step):
    if dimension < 0:
        dimension += len(input_shape)
    indices = []
    for i in range(0, input_shape[dimension] - size + 1, step):
        indices.append(list(range(i, i + size)))

    return indices, dimension


def unfold(input, dimension, size, step):
    _indices, _dimension = _get_unfold_indices(input.shape, dimension, size, step)
    indices = ms.tensor(_indices)
    output = gather(input, indices, _dimension)
    output = transpose_view(output, _dimension + 1, -1)
    return output


def stop_gradient(input):
    """
    Stops gradient computation (no-op for numpy backend).
    For numpy backend, this simply returns the input since numpy doesn't support automatic differentiation.
    
    Args:
        input: Input tensor
    
    Returns:
        The input tensor (no gradient computation)
    """
    # For numpy backend, stop_gradient is a no-op since numpy doesn't support gradients
    return input

def isfinite(input):
    return mindtorch.Tensor(np.isfinite(input.asnumpy()))

class IsInfFunction(Function):
    @staticmethod
    def forward(ctx, input):
        out = np.isinf(input.asnumpy())
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        return ms.Tensor.from_numpy(out)
    
    @staticmethod
    def backward(ctx, grad_output):
        return None

def isinf(input):
    return IsInfFunction.apply(input)

class LogitFunction(Function):
    @staticmethod
    def forward(ctx, input, eps=None):
        x_np = input.asnumpy()
        if eps is not None:
            x_np = np.clip(x_np, eps, 1.0 - eps)
            ctx.eps = eps
        else:
            ctx.eps = None
        ctx.save_for_backward(input)
        out = np.log(x_np / (1.0 - x_np))
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        return ms.Tensor.from_numpy(out)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            x_np = input.asnumpy()
            grad_out_np = grad_output.asnumpy()
            if ctx.eps is not None:
                mask = (x_np > ctx.eps) & (x_np < 1.0 - ctx.eps)
                grad_np = np.zeros_like(x_np, dtype=grad_out_np.dtype)
                grad_np[mask] = grad_out_np[mask] / (x_np[mask] * (1.0 - x_np[mask]))
            else:
                grad_np = grad_out_np / (x_np * (1.0 - x_np))
            if not isinstance(grad_np, np.ndarray):
                grad_np = np.array(grad_np)
            grad_input = ms.Tensor.from_numpy(grad_np)
        return grad_input, None

def logit(input, eps=None):
    return LogitFunction.apply(input, eps)

class Conv3dFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, training=True):
        from scipy import signal
        x = input.asnumpy()
        w = weight.asnumpy()

        # Determine expected input channels from weight
        C_out, C_in_per_group, kD, kH, kW = w.shape
        expected_C_in = C_in_per_group * groups

        # Normalize input to 5D (N, C, D, H, W)
        if x.ndim == 3:
            D, H, W = x.shape
            if expected_C_in == 1:
                x = x[np.newaxis, np.newaxis, :, :, :]
            else:
                x = np.tile(x[np.newaxis, np.newaxis, :, :, :], (1, expected_C_in, 1, 1, 1))
            N = 1
            squeeze_output = True
        elif x.ndim == 4:
            # (C, D, H, W) -> (1, C, D, H, W)
            x = x[np.newaxis, :]
            N = 1
            squeeze_output = True
        elif x.ndim == 5:
            N = x.shape[0]
            squeeze_output = False
        else:
            raise ValueError(f"conv3d: input must have 3, 4, or 5 dimensions, got {x.ndim}")

        N, C_in, D, H, W = x.shape
        if C_in != expected_C_in:
            raise ValueError(f"conv3d: input has {C_in} channels but weight expects {expected_C_in} channels (C_in_per_group={C_in_per_group} * groups={groups})")

        # Normalize stride, padding, dilation to 3-tuples
        if isinstance(stride, int):
            sd, sh, sw = stride, stride, stride
        elif isinstance(stride, (tuple, list)):
            vals = list(stride)
            if len(vals) == 3:
                sd, sh, sw = int(vals[0]), int(vals[1]), int(vals[2])
            else:
                sd, sh, sw = int(vals[-3]), int(vals[-2]), int(vals[-1])
        else:
            sd, sh, sw = 1, 1, 1

        if isinstance(dilation, int):
            dd, dh, dw = dilation, dilation, dilation
        elif isinstance(dilation, (tuple, list)):
            vals = list(dilation)
            if len(vals) == 3:
                dd, dh, dw = int(vals[0]), int(vals[1]), int(vals[2])
            else:
                dd, dh, dw = int(vals[-3]), int(vals[-2]), int(vals[-1])
        else:
            dd, dh, dw = 1, 1, 1

        # Padding
        if isinstance(padding, str):
            pad_mode = padding.lower()
            if pad_mode == 'valid':
                pd, ph, pw = 0, 0, 0
            elif pad_mode == 'same':
                eff_kD = (kD - 1) * dd + 1
                eff_kH = (kH - 1) * dh + 1
                eff_kW = (kW - 1) * dw + 1
                pd = np.maximum(0, (D - 1) * sd + eff_kD - D) // 2
                ph = np.maximum(0, (H - 1) * sh + eff_kH - H) // 2
                pw = np.maximum(0, (W - 1) * sw + eff_kW - W) // 2
            else:
                pd, ph, pw = 0, 0, 0
        elif isinstance(padding, int):
            pd = ph = pw = int(padding)
        elif isinstance(padding, (tuple, list)):
            vals = list(padding)
            if len(vals) == 3:
                pd, ph, pw = int(vals[0]), int(vals[1]), int(vals[2])
            else:
                pd, ph, pw = int(vals[-3]), int(vals[-2]), int(vals[-1])
        else:
            pd, ph, pw = 0, 0, 0

        # Effective kernel sizes with dilation
        eff_kD = (kD - 1) * dd + 1
        eff_kH = (kH - 1) * dh + 1
        eff_kW = (kW - 1) * dw + 1

        # Pad input
        if pd > 0 or ph > 0 or pw > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)), mode='constant')
        else:
            x_padded = x

        # Output dimensions
        D_out = (D + 2 * pd - eff_kD) // sd + 1
        H_out = (H + 2 * ph - eff_kH) // sh + 1
        W_out = (W + 2 * pw - eff_kW) // sw + 1

        out = np.zeros((N, C_out, D_out, H_out, W_out), dtype=x.dtype)

        # Convolution
        for n in range(N):
            for c_out in range(C_out):
                group_id = c_out // (C_out // groups)
                c_in_start = group_id * C_in_per_group
                c_in_end = (group_id + 1) * C_in_per_group

                conv_result = np.zeros((D_out, H_out, W_out), dtype=x.dtype)
                for c_in_idx in range(c_in_start, c_in_end):
                    kernel = w[c_out, c_in_idx - c_in_start]
                    # Apply dilation to kernel
                    if dd > 1 or dh > 1 or dw > 1:
                        dilated_kernel = np.zeros((eff_kD, eff_kH, eff_kW), dtype=kernel.dtype)
                        dilated_kernel[::dd, ::dh, ::dw] = kernel
                        kernel = dilated_kernel
                    # Flip kernel for convolution
                    kernel_flipped = np.flip(kernel, axis=(0, 1, 2))
                    conv_channel = signal.convolve(x_padded[n, c_in_idx], kernel_flipped, mode='valid')
                    # Stride subsampling
                    conv_channel = conv_channel[::sd, ::sh, ::sw]
                    # Align to output shape (handle edge rounding)
                    d_end = D_out if D_out <= conv_channel.shape[0] else conv_channel.shape[0]
                    h_end = H_out if H_out <= conv_channel.shape[1] else conv_channel.shape[1]
                    w_end = W_out if W_out <= conv_channel.shape[2] else conv_channel.shape[2]
                    conv_result[:d_end, :h_end, :w_end] += conv_channel[:d_end, :h_end, :w_end]
                out[n, c_out] = conv_result

        # Bias
        if bias is not None:
            b = bias.asnumpy() if hasattr(bias, 'asnumpy') else np.asarray(bias)
            out = out + b.reshape(1, C_out, 1, 1, 1)

        # Squeeze back if needed
        if squeeze_output:
            out = out[0]
        return ms.Tensor.from_numpy(out)

    @staticmethod
    def backward(ctx, grad_output):
        # Not implementing gradients for numpy backend
        return None, None, None, None, None, None, None

def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, training=True):
    return Conv3dFunction.apply(input, weight, bias, stride, padding, dilation, groups, training)

def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, training=True):
    # Expand to 2D conv: (N,C,L) -> (N,C,1,L), kernel (C_out,C_in/groups,K) -> (C_out,C_in/groups,1,K)
    input_2d = expand_dims(input, 2)
    weight_np = weight.asnumpy()
    weight_2d = ms.Tensor.from_numpy(weight_np.reshape(weight_np.shape[0], weight_np.shape[1], 1, weight_np.shape[2]))
    # Normalize stride, padding, dilation for 2D
    if isinstance(stride, int):
        stride_2d = (1, stride)
    elif isinstance(stride, (tuple, list)):
        stride_2d = (1, stride[0]) if len(stride) >= 1 else (1, 1)
    else:
        stride_2d = (1, 1)
    if isinstance(dilation, int):
        dilation_2d = (1, dilation)
    elif isinstance(dilation, (tuple, list)):
        dilation_2d = (1, dilation[0]) if len(dilation) >= 1 else (1, 1)
    else:
        dilation_2d = (1, 1)
    if isinstance(padding, str):
        padding_2d = padding
    elif isinstance(padding, int):
        padding_2d = (0, padding)
    elif isinstance(padding, (tuple, list)):
        if len(padding) == 1:
            padding_2d = (0, padding[0])
        elif len(padding) == 2:
            padding_2d = (0, padding[1])
        else:
            padding_2d = (0, 0)
    else:
        padding_2d = 0
    out_2d = conv2d(input_2d, weight_2d, bias=bias, stride=stride_2d, padding=padding_2d, dilation=dilation_2d, groups=groups, training=True)
    out_np = out_2d.asnumpy()
    out_1d = np.squeeze(out_np, axis=2)
    return ms.Tensor.from_numpy(out_1d)

def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    x = input.asnumpy()
    w = weight.asnumpy()
    N, C_in, H, W = x.shape
    # Normalize params
    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = int(stride[0]), int(stride[1]) if len(stride) > 1 else int(stride[0])
    if isinstance(padding, int):
        pad_h, pad_w = padding, padding
    else:
        pad_h, pad_w = int(padding[0]), int(padding[1]) if len(padding) > 1 else int(padding[0])
    if isinstance(output_padding, int):
        out_pad_h, out_pad_w = output_padding, output_padding
    else:
        out_pad_h, out_pad_w = int(output_padding[0]), int(output_padding[1]) if len(output_padding) > 1 else int(output_padding[0])
    if isinstance(dilation, int):
        dil_h, dil_w = dilation, dilation
    else:
        dil_h, dil_w = int(dilation[0]), int(dilation[1]) if len(dilation) > 1 else int(dilation[0])
    # Weight shape: (C_in_per_group, C_out_per_group, kH, kW)
    C_in_per_group = w.shape[0]
    C_out_per_group = w.shape[1]
    kH, kW = w.shape[2], w.shape[3]
    # Compute output dims
    H_out = (H - 1) * stride_h - 2 * pad_h + dil_h * (kH - 1) + out_pad_h + 1
    W_out = (W - 1) * stride_w - 2 * pad_w + dil_w * (kW - 1) + out_pad_w + 1
    C_out = C_out_per_group * groups
    out = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)
    # Accumulate
    for n in range(N):
        for g in range(groups):
            in_start = g * C_in_per_group
            in_end = (g + 1) * C_in_per_group
            out_start = g * C_out_per_group
            out_end = (g + 1) * C_out_per_group
            for c_in in range(in_start, in_end):
                k_idx = c_in - in_start
                xi = x[n, c_in]
                for i in range(H):
                    for j in range(W):
                        val = xi[i, j]
                        if val == 0:
                            continue
                        for kh in range(kH):
                            out_i = i * stride_h - pad_h + kh * dil_h
                            if out_i < 0 or out_i >= H_out:
                                continue
                            for kw in range(kW):
                                out_j = j * stride_w - pad_w + kw * dil_w
                                if out_j < 0 or out_j >= W_out:
                                    continue
                                out[n, out_start:out_end, out_i, out_j] += val * w[k_idx, :, kh, kw]
    if bias is not None:
        b = bias.asnumpy() if hasattr(bias, 'asnumpy') else np.asarray(bias)
        out += b.reshape(1, C_out, 1, 1)
    return ms.Tensor.from_numpy(out)

class UpsampleNearest3dFunction(Function):
    @staticmethod
    def forward(ctx, input, output_size=None, scale_factors=None):
        from scipy import ndimage
        x = input.asnumpy()
        if x.ndim != 5:
            raise ValueError(f"upsample_nearest3d expects 5D input (N,C,D,H,W), got {x.ndim}D")
        N, C, D, H, W = x.shape

        if output_size is None:
            if scale_factors is None:
                raise ValueError("Either output_size or scale_factors must be provided")
            if isinstance(scale_factors, (list, tuple)):
                sd = float(scale_factors[0])
                sh = float(scale_factors[1]) if len(scale_factors) > 1 else float(scale_factors[0])
                sw = float(scale_factors[2]) if len(scale_factors) > 2 else float(scale_factors[0])
                out_d = int(np.maximum(1, int(np.round(D * sd))))
                out_h = int(np.maximum(1, int(np.round(H * sh))))
                out_w = int(np.maximum(1, int(np.round(W * sw))))
            else:
                sd = sh = sw = float(scale_factors)
                out_d = int(np.maximum(1, int(np.round(D * sd))))
                out_h = int(np.maximum(1, int(np.round(H * sh))))
                out_w = int(np.maximum(1, int(np.round(W * sw))))
        else:
            if not isinstance(output_size, (list, tuple)) or len(output_size) != 3:
                raise ValueError("output_size for 3d upsample must have length 3 (D, H, W)")
            out_d, out_h, out_w = int(output_size[0]), int(output_size[1]), int(output_size[2])

        zoom_d = out_d / D
        zoom_h = out_h / H
        zoom_w = out_w / W

        out = np.empty((N, C, out_d, out_h, out_w), dtype=x.dtype)
        for n in range(N):
            for c in range(C):
                out[n, c] = ndimage.zoom(x[n, c], (zoom_d, zoom_h, zoom_w), order=0, mode='nearest')
        return ms.Tensor.from_numpy(out)

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None

def upsample_nearest3d(input, output_size=None, scale_factors=None):
    return UpsampleNearest3dFunction.apply(input, output_size, scale_factors)
