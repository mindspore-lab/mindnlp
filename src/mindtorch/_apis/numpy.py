import ctypes
import numbers
import numpy as np
import scipy
from mindspore import ops
from mindspore import _Function as Function
import mindspore as ms
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
        if not isinstance(input, numbers.Number):
            input_np = input.asnumpy()
            if input_np.dtype == np.int64:
                input_np = input_np.astype(np.int32)
        else:
            input_np = input
        if not isinstance(other, numbers.Number):
            other_np = other.asnumpy()
        else:
            other_np = other
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


dyn_shape_op = ops.TensorShape().set_device('CPU')
def tensor_shape(self):
    return dyn_shape_op(self)


class CastFunction(Function):
    @staticmethod
    def forward(ctx, input, dtype):
        if input.dtype == dtype:
            return input
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
        out = input_np[new_slice]
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


broadcast_to_op = ops.Primitive('BroadcastTo').set_device('CPU')
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

def concat(tensors, dim):
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



def randint(from_, to, shape, dtype, generator):
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
    sort_op = ops.Sort(dim, descending).set_device('CPU')
    return sort_op(input)


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, input):
        out = np.round(input.asnumpy())
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

def round(input):
    return RoundFunction.apply(input)


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


def randn(size, generator, dtype):
    out = np.random.randn(*size).astype(mindtorch.dtype2np[dtype])
    return ms.Tensor.from_numpy(out)


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


stop_gradient_op = ops.StopGradient().set_device('CPU')
def stop_gradient(*args):
    return stop_gradient_op(*args)


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


def conv2d(input, weight, bias=None, stride=1, padding='valid', dilation=1, groups=1):
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
    
    # Handle different input dimensions
    if input_np.ndim == 3:
        # Add batch dimension if missing: (C, H, W) -> (1, C, H, W)
        input_np = input_np[np.newaxis, :]
        N = 1
        squeeze_output = True
    else:
        N = input_np.shape[0]
        squeeze_output = False
    
    N, C_in, H, W = input_np.shape
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


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    """
    Group Normalization over a mini-batch of inputs.
    
    Args:
        input: Input tensor with shape (N, C, *)
        num_groups: Number of groups to divide channels into
        weight: Optional scale tensor of shape (C,)
        bias: Optional shift tensor of shape (C,)
        eps: Small value for numerical stability
    
    Returns:
        Normalized tensor with same shape as input
    """
    input_np = input.asnumpy()
    N, C = input_np.shape[0], input_np.shape[1]
    
    # Reshape input to (N, num_groups, -1) for group-wise normalization
    if np.prod(input_np.shape) != 0:
        inp_view = input_np.reshape((N, num_groups, -1))
        # Compute mean and variance for each group
        mean = np.mean(inp_view, axis=-1, keepdims=True)
        var = np.var(inp_view, axis=-1, ddof=0, keepdims=True)
        # Normalize
        Y = (inp_view - mean) / np.sqrt(var + eps)
        # Reshape back to original shape
        Y = Y.reshape(input_np.shape)
    else:
        Y = input_np.copy()
    
    # Apply weight and bias if provided
    if weight is not None:
        weight_np = weight.asnumpy() if hasattr(weight, 'asnumpy') else weight
        # Expand weight to match input dimensions
        if len(Y.shape) > 2:
            expand_dims = [0] + [idx + 2 for idx in range(input_np.ndim - 2)]
            for dim in expand_dims:
                weight_np = np.expand_dims(weight_np, dim)
        Y = Y * weight_np
    
    if bias is not None:
        bias_np = bias.asnumpy() if hasattr(bias, 'asnumpy') else bias
        # Expand bias to match input dimensions
        if len(Y.shape) > 2:
            expand_dims = [0] + [idx + 2 for idx in range(input_np.ndim - 2)]
            for dim in expand_dims:
                bias_np = np.expand_dims(bias_np, dim)
        Y = Y + bias_np
    
    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)
    return ms.Tensor.from_numpy(Y)


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
        dim0 = dim0 if dim0 >= 0 else dim0 + ndim
        dim1 = dim1 if dim1 >= 0 else dim1 + ndim
        # Check bounds
        if dim0 < 0 or dim0 >= ndim or dim1 < 0 or dim1 >= ndim:
            raise IndexError(f"Dimension out of range: dim0={dim0}, dim1={dim1}, ndim={ndim}")
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


def ones_like(input):
    out = np.ones_like(input.asnumpy())
    return ms.Tensor.from_numpy(out)


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
    op = ops.NLLLoss(reduction, ignore_index).set_device('CPU')
    return op(input, target, weight)


def diag_ext(input, diagonal):
    out = np.diag(input.asnumpy(), diagonal)
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