import numbers
import mindspore
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.auto_generate import gen_ops_prim
from mindspore.ops.auto_generate import pyboost_inner_prim
from mindspore._c_expression import _empty_instance
from mindspore.ops.operations.math_ops import NPUGetFloatStatusV2, NPUClearFloatStatusV2
from mindspore.ops.auto_generate.gen_ops_prim import MaxPoolWithIndices, MaxPoolWithMask
import mindtorch
from mindtorch._C import default_generator

op_list = list(filter(lambda s: s.endswith("_op"), dir(gen_ops_prim)))

__all__ = []

for op_name in op_list:
    func_name = op_name.replace('_op', '')
    __all__.append(func_name)
    globals()[func_name] = getattr(gen_ops_prim, op_name).__class__().set_device('Ascend')

def empty(*args, **kwargs):
    return _empty_instance(*args, **kwargs, device='Ascend')

def reduce_any(input, dim, keepdim):
    if dim is None:
        dim = ()
    return pyboost_inner_prim.reduce_any_impl(input, dim, keepdim)

__all__.append('reduce_any')

def reduce_all(input, dim, keepdim):
    if dim is None:
        dim = ()
    return pyboost_inner_prim.reduce_all_impl(input, dim, keepdim)

__all__.append('reduce_all')

broadcast_to_op = ops.Primitive('BroadcastTo').set_device('Ascend')
def broadcast_to(*args):
    return broadcast_to_op(*args)

__all__.append('broadcast_to')

cast_op = ops.Cast().set_device('Ascend')
def cast(*args):
    return cast_op(*args)

__all__.append('cast')

zeros_op = ops.Zeros().set_device('Ascend')
def zeros(*args):
    return zeros_op(*args)

__all__.append('zeros')

def softmax(*args):
    return pyboost_inner_prim.softmax_impl(*args)

__all__.append('softmax')

def dropout_ext(input, p):
    seed, offset = default_generator._step(12)  # pylint: disable=protected-access
    return gen_ops_prim.dropout_ext_op(input, p, seed, offset)

def squeeze(*args):
    return pyboost_inner_prim.squeeze_impl(*args)

__all__.append('squeeze')

ones_op = ops.Ones().set_device('Ascend')
def ones(*args):
    return ones_op(*args)

__all__.append('ones')

def nllloss(*args):
    return pyboost_inner_prim.nllloss_impl(*args)

__all__.append('nllloss')

def repeat_elements(*args):
    return ops.repeat_elements(*args)

__all__.append('repeat_elements')

def concat(*args):
    return pyboost_inner_prim.concat_impl(*args)

__all__.append('concat')

def multinomial_ext(input, num_samples, replacement, generator):
    seed, offset = generator._step(12)  # pylint: disable=protected-access
    return gen_ops_prim.multinomial_ext_op(input, num_samples, replacement, seed, offset)

def isclose(*args):
    return pyboost_inner_prim.isclose_impl(*args)

__all__.append('isclose')

tile_op = ops.Primitive('Tile').set_device('Ascend')
def tile(*args):
    return tile_op(*args)

__all__.append('tile')

def pad_v3(input_x, padding, mode='constant', value=None):
    pad_op = ops.PadV3(mode=mode, paddings_contiguous=True).set_device('Ascend')
    if input_x.dtype == mindtorch.bool:
        input_x = input_x.to(mindtorch.int32)
        value = int(value)
        out = pad_op(input_x, padding, value)
        return cast(out, mindtorch.bool)

    if isinstance(value, (float, int)):
        value = mindtorch.tensor(value, dtype=input_x.dtype)
    return pad_op(input_x, padding, value)

__all__.append('pad_v3')


def inplace_uniform(input, from_, to_, generator_):
    seed, offset = generator_._step(12)
    return gen_ops_prim.inplace_uniform_op(input, from_, to_, seed, offset)

def inplace_normal(input, mean, std, generator_):
    seed, offset = generator_._step(12)
    return gen_ops_prim.inplace_normal_op(input, mean, std, seed, offset)


def binary_cross_entropy_with_logits(*args):
    return pyboost_inner_prim.binary_cross_entropy_with_logits_impl(*args)

__all__.append('binary_cross_entropy_with_logits')

def gather(input_params, input_indices, axis, batch_dims=0):
    gather_op = _get_cache_prim(ops.Gather)(batch_dims).set_device('Ascend')
    return gather_op(input_params, input_indices, axis)

__all__.append('gather')

def randint(low, high, size, dtype, generator):
    seed, offset = generator._step(12)  # pylint: disable=protected-access
    return gen_ops_prim.randint_op(low, high, size, seed, offset, dtype)

def stack_ext(*args):
    return pyboost_inner_prim.stack_ext_impl(*args)

__all__.append('stack_ext')

def argmax_with_value(*args):
    return pyboost_inner_prim.argmax_with_value_impl(*args)

__all__.append('argmax_with_value')

def argmin_with_value(*args):
    return pyboost_inner_prim.argmin_with_value_impl(*args)

__all__.append('argmin_with_value')


right_shift_op = ops.RightShift().set_device('Ascend')
def right_shift(input, other):
    if isinstance(other, numbers.Number):
        other = mindtorch.Tensor(other, input.dtype)
    return right_shift_op(input, other)

tensor_mul = ops.Mul().set_device('Ascend')
tensor_pow = ops.Pow().set_device('Ascend')
def ldexp(input, other):
    out = tensor_mul(input, tensor_pow(2.0, other))
    return out

__all__.append('ldexp')

def reverse_v2(input, dims):
    if isinstance(dims, int):
        dims = (dims,)
    return pyboost_inner_prim.reverse_v2_impl(input, dims)

__all__.append('reverse_v2')

adam_op = ops.Adam().set_device('Ascend')
def raw_adam(param, exp_avg, exp_avg_sq, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
    # var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad
    return adam_op(param, exp_avg, exp_avg_sq, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)

__all__.append('raw_adam')

depend_op = ops.Depend().set_device('Ascend')
def depend(*args):
    return depend_op(*args)

__all__.append('depend')

npu_get_float_status_op = NPUGetFloatStatusV2().set_device('Ascend')
def npu_get_float_status_v2(status):
    return npu_get_float_status_op(status)

__all__.append('npu_get_float_status_v2')

npu_clear_float_status_op = NPUClearFloatStatusV2().set_device('Ascend')
def npu_clear_float_status_v2(status):
    return npu_clear_float_status_op(status)

__all__.append('npu_clear_float_status_v2')

stop_gradient_op = ops.StopGradient().set_device('Ascend')
def stop_gradient(*args):
    return stop_gradient_op(*args)

__all__.append('stop_gradient')

# allfinite_op = AllFinite().set_device('Ascend')
def all_finite(inputs):
    return allfinite_op(inputs)

def rsqrt_fp32(input):
    return rsqrt(input.astype(mindspore.float32))

__all__.append('rsqrt_fp32')

def matmul_ext_fp16(input, other):
    return matmul_ext(input.astype(mindspore.float16), other.astype(mindspore.float16))

__all__.append('matmul_ext_fp16')

def dense_fp16(input, weight, bias):
    input = input.astype(mindspore.float16)
    weight = weight.astype(mindspore.float16)
    if bias is not None:
        bias = bias.astype(mindspore.float16)
    return dense(input, weight, bias)

__all__.append('dense_fp16')

def softmax_fp32(input, dim):
    return softmax(input.astype(mindspore.float32), dim)

__all__.append('softmax_fp32')

def log_softmax_ext_fp32(input, dim, dtype):
    return log_softmax_ext(input.astype(mindspore.float32), dim, dtype)

__all__.append('log_softmax_ext_fp32')

def one_hot_ext(tensor, num_classes):
    on_value = mindtorch.Tensor(1, dtype=tensor.dtype)
    off_value = mindtorch.Tensor(0, dtype=tensor.dtype)

    return pyboost_inner_prim.one_hot_ext_impl(tensor, num_classes, on_value, off_value, -1)

__all__.append('one_hot_ext')

def flash_attention_score(*args, **kwargs):
    return pyboost_inner_prim.flash_attention_score_impl(*args, **kwargs)

__all__.append('flash_attention_score')

def triu(input, diagonal):
    return pyboost_inner_prim.triu_impl(input, diagonal)

__all__.append('triu')

masked_scatter_op = ops.MaskedScatter().set_device('Ascend')
def masked_scatter(input, mask, source):
    return masked_scatter_op(input, mask, source)

__all__.append('masked_scatter')

def roll(*args):
    return pyboost_inner_prim.roll_impl(*args)

__all__.append('roll')

lgamma_op = ops.Lgamma().set_device('Ascend')
def lgamma(input):
    return lgamma_op(input)

__all__.append('lgamma')

def max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices):
    strides = stride if (stride is not None) else kernel_size
    if return_indices:
        max_pool_func_ = _get_cache_prim(MaxPoolWithIndices)(kernel_size, strides, padding, dilation, ceil_mode)
        out, indices = max_pool_func_(input)
    else:
        max_pool_func_ = _get_cache_prim(MaxPoolWithMask)(kernel_size, strides, padding, dilation, ceil_mode)
        out, indices = max_pool_func_(input)
    if return_indices:
        return out, indices
    return out

__all__.append('max_pool2d')

def unique_consecutive(*args):
    return pyboost_inner_prim.unique_consecutive_impl(*args)

__all__.append('unique_consecutive')

def im2col(input, kernel_size, dilation, padding, stride):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding)
    unfold_op = _get_cache_prim(ops.Im2Col)(ksizes=kernel_size, strides=stride, dilations=dilation, pads=padding).set_device('Ascend')
    tmp = unfold_op(input)
    tmp_shape = tmp.shape
    out_shape = tmp_shape[:1] + (-1,) + tmp_shape[-1:]
    out = reshape(tmp, out_shape)
    return out

__all__.append('im2col')

def cdist(x1, x2, p):
    cdist_op = _get_cache_prim(ops.Cdist)(float(p)).set_device('Ascend')
    return cdist_op(x1, x2)

__all__.append('cdist')

def meshgrid(tensors, indexing):
    return pyboost_inner_prim.meshgrid_impl(tensors, indexing)

__all__.append('meshgrid')

def grid_sampler_2d(input, grid, mode, padding_mode, align_corners):
    return pyboost_inner_prim.grid_sampler_2d_impl(input, grid, mode, padding_mode, align_corners)

def grid_sampler_3d(input, grid, mode, padding_mode, align_corners):
    return pyboost_inner_prim.grid_sampler_3d_impl(input, grid, mode, padding_mode, align_corners)

__all__.append('grid_sampler_2d')
__all__.append('grid_sampler_3d')

def rms_norm(x, gamma, epsilon):
    return pyboost_inner_prim.rms_norm_impl(x, gamma, epsilon)[0]

__all__.append('rms_norm')

_complex = ops.Complex().set_device('Ascend')
def view_as_complex(input):
    real_part, imag_part = input.tensor_split(2, -1)
    return _complex(real_part.squeeze(-1), imag_part.squeeze(-1))

__all__.append('view_as_complex')

imag_op = ops.Imag().set_device('Ascend')
def imag(input):
    return imag_op(input)

__all__.append('imag')

def glu(x, axis):
    return pyboost_inner_prim.glu_impl(x, axis)

__all__.append('glu')

def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False):
    ctc_loss_op = _get_cache_prim(ops.CTCLossV2)(blank=blank, reduction="none", zero_infinity=zero_infinity).set_device('Ascend')
    if targets.ndim == 1:
        targets = targets.unsqueeze(-1)
    loss, _ = ctc_loss_op(log_probs, targets, input_lengths, target_lengths)
    if zero_infinity:
        loss = select(isinf(loss), 0., loss)
    if reduction == 'sum':
        loss = sum_ext(loss)
    if reduction == 'mean':
        input_type = loss.dtype
        target_length_t = target_lengths.clip(1., None)
        loss = loss.astype("float32")
        loss = div(loss, target_length_t)
        loss = mean_ext(loss)
        loss = loss.astype(input_type)
    return loss

__all__.append('ctc_loss')

def reduce_max(input, dim, keepdim):
    return pyboost_inner_prim.reduce_max_impl(input, dim, keepdim)

__all__.append('reduce_max')

def elu(input, alpha):
    return pyboost_inner_prim.elu_ext_impl(input, alpha)

__all__.append('elu')

dynamic_rnn_op = ops.DynamicRNN().set_device('Ascend')
def dynamic_rnn(*args):
    return dynamic_rnn_op(*args)

__all__.append('dynamic_rnn')

def cross(input, other, dim=None, *, out=None):
    return pyboost_inner_prim.cross_impl(input, other, dim)

__all__.append('cross')

def logit(input, eps):
    if eps is None:
        eps = -1.0
    logit_ = _get_cache_prim(ops.Logit)(eps).set_device('Ascend')
    return logit_(input)

__all__.append('logit')

def bucketize(input, boundaries, right):
    epsilon_ = 0. if right else 1.e-6
    boundaries = [boundary + epsilon_ for boundary in boundaries]
    bucketize_op = ops.Bucketize(boundaries).set_device('Ascend')
    return bucketize_op(input)

__all__.append('bucketize')

def dropout2d(input, p):
    dropout_2d_op = ops.Dropout2D(1.0 - p)
    return dropout_2d_op(input)

__all__.append('dropout2d')

def bernoulli_ext(input, generator):
    seed, offset = generator._step(12)  # pylint: disable=protected-access
    return gen_ops_prim.bernoulli_ext_op(input, seed, offset)

__all__.append('bernoulli_ext')