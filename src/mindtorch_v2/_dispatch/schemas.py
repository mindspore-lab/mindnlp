from .registry import registry


def _register_unary_ops(names):
    for name in names:
        registry.register_schema(name, f"{name}(Tensor input) -> Tensor")


def _register_binary_ops(names, *, other_type="Tensor"):
    for name in names:
        registry.register_schema(name, f"{name}(Tensor input, {other_type} other) -> Tensor")


def _register_reduction_ops(names):
    for name in names:
        registry.register_schema(name, f"{name}(Tensor input, int[]? dim=None, bool keepdim=False) -> Tensor")


# Mechanism-level baseline set: these ops must always have schema coverage so
# dispatch binding/errors stay consistent while agents add more kernels.
CORE_SCHEMA_OPS = (
    "add",
    "mul",
    "matmul",
    "relu",
    "sum",
    "mean",
    "std",
    "reshape",
    "view",
    "transpose",
    "to",
    "tensor",
    "zeros",
    "ones",
    "empty",
    "arange",
    "linspace",
    "full",
    "randn",
    "rand",
    "add_",
    "mul_",
    "relu_",
    "zero_",
    "uniform_",
    "normal_",
    "fill_",
    "clamp_",
    "copy_",
    "erfinv_",
    "sub_",
    "setitem",
    "all",
    "any",
    "argmax",
    "argmin",
    "count_nonzero",
    "allclose",
    "isclose",
    "equal",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "cumsum",
    "cumprod",
    "cummax",
    "argsort",
    "sort",
    "topk",
    "stack",
    "cat",
    "concat",
    "concatenate",
    "hstack",
    "vstack",
    "row_stack",
    "dstack",
    "column_stack",
    "chunk",
    "split",
    "vsplit",
    "hsplit",
    "dsplit",
    "unbind",
    "nonzero",
    "masked_select",
    "flip",
    "roll",
    "rot90",
    "repeat",
    "repeat_interleave",
    "tile",
    "take",
    "take_along_dim",
    "index_select",
    "gather",
    "scatter",
    "tril",
    "triu",
    "diag",
    "tril_indices",
    "triu_indices",
    "cartesian_prod",
    "pad_sequence",
    "block_diag",
    "abs",
    "neg",
    "exp",
    "log",
    "sqrt",
    "pow",
    "sub",
    "div",
    "true_divide",
    "min",
    "max",
    "fmin",
    "fmax",
    "where",
    "atan",
    "atan2",
    "asin",
    "acos",
    "lerp",
    "addcmul",
    "addcdiv",
    "logaddexp",
    "logaddexp2",
    "hypot",
    "remainder",
    "fmod",
    "squeeze",
    "unsqueeze",
    "permute",
    "var",
    "norm",
    "prod",
    "floor_divide",
    "rms_norm",
    "conv2d",
    "conv1d",
    "conv_transpose2d",
    "conv_transpose1d",
    "max_pool2d",
    "avg_pool2d",
    "adaptive_avg_pool2d",
)


def register_schemas():
    registry.register_schema("add", "add(Tensor input, Tensor other, *, Scalar alpha=1) -> Tensor")
    registry.register_error_overrides(
        "add",
        {
            "missing": "{name}() received an invalid combination of arguments - got {got}, but expected (Tensor input, Tensor other, *, Number alpha = 1, Tensor out = None)",
        },
    )

    _register_binary_ops(("mul", "matmul"))
    _register_unary_ops(("relu",))

    registry.register_schema("sum", "sum(Tensor input, int[]? dim=None, bool keepdim=False, Dtype? dtype=None) -> Tensor")
    registry.register_error_overrides(
        "sum",
        {
            "missing": "{name}() received an invalid combination of arguments - got {got}, but expected one of:\n * (Tensor input, *, torch.dtype dtype = None)\n * (Tensor input, tuple of ints dim, bool keepdim = False, *, torch.dtype dtype = None, Tensor out = None)\n * (Tensor input, tuple of names dim, bool keepdim = False, *, torch.dtype dtype = None, Tensor out = None)\n",
            "unexpected": "{name}() received an invalid combination of arguments - got {got}, but expected one of:\n * (Tensor input, *, torch.dtype dtype = None)\n * (Tensor input, tuple of ints dim, bool keepdim = False, *, torch.dtype dtype = None, Tensor out = None)\n * (Tensor input, tuple of names dim, bool keepdim = False, *, torch.dtype dtype = None, Tensor out = None)\n",
        },
    )
    registry.register_schema("mean", "mean(Tensor input, int[]? dim=None, bool keepdim=False, Dtype? dtype=None) -> Tensor")
    registry.register_schema("std", "std(Tensor input, int[]? dim=None, bool keepdim=False, bool unbiased=True) -> Tensor")

    registry.register_schema("reshape", "reshape(Tensor(a) input, int[] shape) -> Tensor(a)")
    registry.register_error_overrides(
        "reshape",
        {
            "missing": '{name}() missing 2 required positional argument: "input", "shape"',
        },
    )
    registry.register_schema("view", "view(Tensor(a) input, int[] shape) -> Tensor(a)")
    registry.register_error_overrides(
        "view",
        {
            "missing": "{name}() received an invalid combination of arguments - got {got}, but expected one of:\n * (torch.dtype dtype)\n * (tuple of ints size)\n",
        },
    )
    registry.register_schema("transpose", "transpose(Tensor(a) input, int dim0, int dim1) -> Tensor(a)")
    registry.register_error_overrides(
        "transpose",
        {
            "missing": "{name}() received an invalid combination of arguments - got {got}, but expected one of:\n * (Tensor input, int dim0, int dim1)\n * (Tensor input, name dim0, name dim1)\n",
        },
    )
    registry.register_schema("squeeze", "squeeze(Tensor(a) input, int? dim=None) -> Tensor(a)")
    registry.register_schema("unsqueeze", "unsqueeze(Tensor(a) input, int dim) -> Tensor(a)")
    registry.register_schema("permute", "permute(Tensor(a) input, int[] dims) -> Tensor(a)")

    registry.register_schema("to", "to(Tensor input, Device? device=None, Dtype? dtype=None, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor")
    registry.register_schema("tensor", "tensor(Tensor data, *, Dtype? dtype=None, Device? device=None, bool requires_grad=False) -> Tensor")
    registry.register_schema("zeros", "zeros(int[] size, *, Dtype? dtype=None, Device? device=None, MemoryFormat? memory_format=None) -> Tensor")
    registry.register_schema("ones", "ones(int[] size, *, Dtype? dtype=None, Device? device=None, MemoryFormat? memory_format=None) -> Tensor")
    registry.register_schema("empty", "empty(int[] size, *, Dtype? dtype=None, Device? device=None, MemoryFormat? memory_format=None) -> Tensor")
    registry.register_schema("arange", "arange(Scalar start, Scalar end, Scalar step=1, Dtype? dtype=None) -> Tensor")
    registry.register_schema("linspace", "linspace(Scalar start, Scalar end, int steps, Dtype? dtype=None) -> Tensor")
    registry.register_schema("full", "full(int[] size, Scalar fill_value, Dtype? dtype=None) -> Tensor")
    registry.register_schema("logspace", "logspace(Scalar start, Scalar end, int steps, Dtype? dtype=None) -> Tensor")
    registry.register_schema("eye", "eye(int n, int? m=None, Dtype? dtype=None, Tensor? out=None) -> Tensor")
    registry.register_schema("range", "range(Scalar start, Scalar end, Scalar step=1, Dtype? dtype=None) -> Tensor")
    registry.register_schema("randn", "randn(int[] size, *, Dtype? dtype=None, MemoryFormat? memory_format=None) -> Tensor")
    registry.register_schema("rand", "rand(int[] size, *, Dtype? dtype=None, MemoryFormat? memory_format=None) -> Tensor")

    registry.register_schema("add_", "add_(Tensor(a!) self, Tensor other) -> Tensor")
    registry.register_schema("mul_", "mul_(Tensor(a!) self, Tensor other) -> Tensor")
    registry.register_schema("relu_", "relu_(Tensor(a!) self) -> Tensor")
    registry.register_schema("zero_", "zero_(Tensor(a!) self) -> Tensor")
    registry.register_schema("uniform_", "uniform_(Tensor(a!) self, float low=0.0, float high=1.0) -> Tensor")
    registry.register_schema("normal_", "normal_(Tensor(a!) self, float mean=0.0, float std=1.0) -> Tensor")
    registry.register_schema("fill_", "fill_(Tensor(a!) self, Scalar value) -> Tensor")
    registry.register_schema("clamp_", "clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor")
    registry.register_schema("copy_", "copy_(Tensor(a!) self, Tensor src) -> Tensor")
    registry.register_schema("erfinv_", "erfinv_(Tensor(a!) self) -> Tensor")
    registry.register_schema("sub_", "sub_(Tensor(a!) self, Tensor other) -> Tensor")
    registry.register_schema("getitem", "getitem(Tensor self, Any key) -> Tensor")
    registry.register_schema("setitem", "setitem(Tensor(a!) self, Any key, Any value) -> Tensor")

    _register_reduction_ops(("all", "any", "argmax", "argmin", "count_nonzero"))
    registry.register_schema("amin", "amin(Tensor input, int[]? dim=None, bool keepdim=False) -> Tensor")
    registry.register_schema("amax", "amax(Tensor input, int[]? dim=None, bool keepdim=False) -> Tensor")

    registry.register_schema("allclose", "allclose(Tensor input, Tensor other, *, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool")
    registry.register_schema("isclose", "isclose(Tensor input, Tensor other, *, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor")
    registry.register_schema("equal", "equal(Tensor input, Tensor other) -> bool")

    _register_binary_ops(("eq", "ne", "lt", "le", "gt", "ge"), other_type="Any")

    registry.register_schema("cumsum", "cumsum(Tensor input, int dim=0) -> Tensor")
    registry.register_schema("cumprod", "cumprod(Tensor input, int dim=0) -> Tensor")
    registry.register_schema("cummax", "cummax(Tensor input, int dim=0) -> (Tensor, Tensor)")
    registry.register_schema("argsort", "argsort(Tensor input, int dim=-1, bool descending=False, bool stable=False) -> Tensor")
    registry.register_schema("sort", "sort(Tensor input, int dim=-1, bool descending=False, bool stable=False) -> (Tensor, Tensor)")
    registry.register_schema("topk", "topk(Tensor input, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor, Tensor)")

    registry.register_schema("stack", "stack(Tensor[] tensors, int dim=0) -> Tensor")
    registry.register_schema("cat", "cat(Tensor[] tensors, int dim=0) -> Tensor")
    registry.register_schema("concat", "concat(Tensor[] tensors, int dim=0) -> Tensor")
    registry.register_schema("concatenate", "concatenate(Tensor[] tensors, int dim=0) -> Tensor")
    registry.register_schema("hstack", "hstack(Tensor[] tensors) -> Tensor")
    registry.register_schema("vstack", "vstack(Tensor[] tensors) -> Tensor")
    registry.register_schema("row_stack", "row_stack(Tensor[] tensors) -> Tensor")
    registry.register_schema("dstack", "dstack(Tensor[] tensors) -> Tensor")
    registry.register_schema("column_stack", "column_stack(Tensor[] tensors) -> Tensor")

    registry.register_schema("chunk", "chunk(Tensor input, int chunks, int dim=0) -> Tensor[]")
    registry.register_schema("split", "split(Tensor input, Any split_size_or_sections, int dim=0) -> Tensor[]")
    registry.register_schema("vsplit", "vsplit(Tensor input, Any split_size_or_sections) -> Tensor[]")
    registry.register_schema("hsplit", "hsplit(Tensor input, Any split_size_or_sections) -> Tensor[]")
    registry.register_schema("dsplit", "dsplit(Tensor input, Any split_size_or_sections) -> Tensor[]")
    registry.register_schema("unbind", "unbind(Tensor input, int dim=0) -> Tensor[]")

    registry.register_schema("nonzero", "nonzero(Tensor input, bool as_tuple=False) -> Any")
    registry.register_schema("masked_select", "masked_select(Tensor input, Tensor mask) -> Tensor")
    registry.register_schema("flip", "flip(Tensor input, int[] dims) -> Tensor")
    registry.register_schema("roll", "roll(Tensor input, Any shifts, Any dims=None) -> Tensor")
    registry.register_schema("rot90", "rot90(Tensor input, int k=1, int[] dims) -> Tensor")
    registry.register_schema("repeat", "repeat(Tensor input, int[] repeats) -> Tensor")
    registry.register_schema("repeat_interleave", "repeat_interleave(Tensor input, Any repeats, int? dim=None) -> Tensor")
    registry.register_schema("tile", "tile(Tensor input, int[] dims) -> Tensor")

    registry.register_schema("take", "take(Tensor input, Tensor index) -> Tensor")
    registry.register_schema("take_along_dim", "take_along_dim(Tensor input, Tensor indices, int dim) -> Tensor")
    registry.register_schema("index_select", "index_select(Tensor input, int dim, Tensor index) -> Tensor")
    registry.register_schema("gather", "gather(Tensor input, int dim, Tensor index) -> Tensor")
    registry.register_schema("scatter", "scatter(Tensor input, int dim, Tensor index, Any src) -> Tensor")

    registry.register_schema("tril", "tril(Tensor input, int diagonal=0) -> Tensor")
    registry.register_schema("triu", "triu(Tensor input, int diagonal=0) -> Tensor")
    registry.register_schema("diag", "diag(Tensor input, int diagonal=0) -> Tensor")
    registry.register_schema("tril_indices", "tril_indices(int row, int col, int offset=0, Dtype? dtype=None, Device? device=None, Any layout=None) -> Tensor")
    registry.register_schema("triu_indices", "triu_indices(int row, int col, int offset=0, Dtype? dtype=None, Device? device=None, Any layout=None) -> Tensor")
    registry.register_schema("cartesian_prod", "cartesian_prod(Tensor[] tensors) -> Tensor")
    registry.register_schema("pad_sequence", "pad_sequence(Tensor[] seqs, bool batch_first=False, float padding_value=0.0, str padding_side=right) -> Tensor")
    registry.register_schema("block_diag", "block_diag(Tensor[] tensors) -> Tensor")

    _register_unary_ops((
        "abs", "neg", "exp", "log", "sqrt", "atan", "asin", "acos",
        "sin", "cos", "tan", "tanh", "sigmoid", "floor", "ceil", "round", "trunc", "frac",
        "log2", "log10", "exp2", "rsqrt", "reciprocal", "sign", "signbit", "isnan", "isinf", "isfinite",
        "sinh", "cosh", "asinh", "acosh", "atanh", "erf", "erfc", "softplus",
        "relu6", "contiguous", "gelu", "silu", "mish",
    ))
    registry.register_schema("hardtanh", "hardtanh(Tensor input, Scalar min_val=-1.0, Scalar max_val=1.0) -> Tensor")
    registry.register_schema("softmax", "softmax(Tensor input, int dim=-1, Dtype? dtype=None) -> Tensor")
    registry.register_schema("log_softmax", "log_softmax(Tensor input, int dim=-1, Dtype? dtype=None) -> Tensor")
    registry.register_schema("dropout", "dropout(Tensor input, float p=0.5, bool training=True) -> Tensor")
    registry.register_schema("leaky_relu", "leaky_relu(Tensor input, float negative_slope=0.01) -> Tensor")
    registry.register_schema("elu", "elu(Tensor input, float alpha=1.0) -> Tensor")
    _register_binary_ops(("pow", "sub", "div", "true_divide", "min", "max", "fmin", "fmax", "atan2", "logaddexp", "logaddexp2", "hypot", "remainder", "fmod"), other_type="Any")
    registry.register_schema("lerp", "lerp(Tensor input, Tensor other, Any weight) -> Tensor")
    registry.register_schema("addcmul", "addcmul(Tensor input, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor")
    registry.register_schema("addcdiv", "addcdiv(Tensor input, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor")
    registry.register_schema("where", "where(Tensor cond, Tensor x, Tensor y) -> Tensor")
    registry.register_schema("clamp", "clamp(Tensor input, Any min_val=None, Any max_val=None) -> Tensor")
    registry.register_schema("clamp_min", "clamp_min(Tensor input, Scalar min_val) -> Tensor")
    registry.register_schema("clamp_max", "clamp_max(Tensor input, Scalar max_val) -> Tensor")
    registry.register_schema("batch_norm", "batch_norm(Tensor input, Any running_mean=None, Any running_var=None, Any weight=None, Any bias=None, bool training=False, float momentum=0.1, float eps=1e-05) -> Tensor")
    registry.register_schema("group_norm", "group_norm(Tensor input, int num_groups, Any weight=None, Any bias=None, float eps=1e-05) -> Tensor")
    registry.register_schema("layer_norm", "layer_norm(Tensor input, int[] normalized_shape, Any weight=None, Any bias=None, float eps=1e-05) -> Tensor")
    registry.register_schema("embedding", "embedding(Tensor input, Tensor weight, int? padding_idx=None, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor")
    registry.register_schema("pad", "pad(Tensor input, int[] pad, str mode=constant, Scalar value=0) -> Tensor")
    registry.register_schema("prelu", "prelu(Tensor input, Tensor weight) -> Tensor")
    registry.register_schema("max_", "max_(Tensor input, Tensor other) -> Tensor")
    registry.register_schema("min_", "min_(Tensor input, Tensor other) -> Tensor")
    registry.register_schema("linalg_qr", "linalg_qr(Tensor input, str mode=reduced) -> (Tensor, Tensor)")

    # View / selection ops
    registry.register_schema("narrow", "narrow(Tensor(a) input, int dim, int start, int length) -> Tensor(a)")
    registry.register_schema("select", "select(Tensor(a) input, int dim, int index) -> Tensor(a)")
    registry.register_schema("expand", "expand(Tensor(a) input, int[] sizes) -> Tensor(a)")
    registry.register_schema("unfold", "unfold(Tensor(a) input, int dimension, int size, int step) -> Tensor(a)")

    # Masked ops
    registry.register_schema("masked_fill", "masked_fill(Tensor input, Tensor mask, Scalar value) -> Tensor")
    registry.register_schema("masked_fill_", "masked_fill_(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor")
    registry.register_schema("masked_scatter_", "masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor")

    # Index ops (in-place)
    registry.register_schema("index_put_", "index_put_(Tensor(a!) self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor")
    registry.register_schema("index_copy_", "index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor")
    registry.register_schema("index_fill_", "index_fill_(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor")
    registry.register_schema("index_add_", "index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source, Scalar alpha=1.0) -> Tensor")

    # Index ops (out-of-place)
    registry.register_schema("index_put", "index_put(Tensor input, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor")

    # Scatter ops (in-place)
    registry.register_schema("scatter_", "scatter_(Tensor(a!) self, int dim, Tensor index, Any src) -> Tensor")
    registry.register_schema("scatter_add_", "scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor")

    registry.register_schema("var", "var(Tensor input, int[]? dim=None, bool unbiased=True, bool keepdim=False) -> Tensor")
    registry.register_schema("norm", "norm(Tensor input, Any p=2, int[]? dim=None, bool keepdim=False) -> Tensor")
    registry.register_schema("prod", "prod(Tensor input, int? dim=None, bool keepdim=False) -> Tensor")
    registry.register_schema("floor_divide", "floor_divide(Tensor input, Any other) -> Tensor")
    registry.register_schema("rms_norm", "rms_norm(Tensor input, int[] normalized_shape, Any weight=None, float eps=1e-6) -> Tensor")

    # Conv operations
    registry.register_schema("conv2d", "conv2d(Tensor input, Tensor weight, Tensor? bias=None, Any stride=None, Any padding=None, Any dilation=None, int groups=1) -> Tensor")
    registry.register_schema("conv1d", "conv1d(Tensor input, Tensor weight, Tensor? bias=None, Any stride=None, Any padding=None, Any dilation=None, int groups=1) -> Tensor")
    registry.register_schema("conv_transpose2d", "conv_transpose2d(Tensor input, Tensor weight, Tensor? bias=None, Any stride=None, Any padding=None, Any output_padding=None, int groups=1, Any dilation=None) -> Tensor")
    registry.register_schema("conv_transpose1d", "conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, Any stride=None, Any padding=None, Any output_padding=None, int groups=1, Any dilation=None) -> Tensor")

    # Pooling operations
    registry.register_schema("max_pool2d", "max_pool2d(Tensor input, Any kernel_size, Any stride, Any padding=None, Any dilation=None, bool ceil_mode=False, bool return_indices=False) -> Tensor")
    registry.register_schema("avg_pool2d", "avg_pool2d(Tensor input, Any kernel_size, Any stride, Any padding=None, bool ceil_mode=False, bool count_include_pad=True, Any divisor_override=None) -> Tensor")
    registry.register_schema("adaptive_avg_pool2d", "adaptive_avg_pool2d(Tensor input, Any output_size) -> Tensor")

    # P1 ops from upstream
    registry.register_schema("addmm", "addmm(Tensor input, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor")
    registry.register_schema("upsample_nearest2d", "upsample_nearest2d(Tensor input, Any output_size) -> Tensor")
    registry.register_schema("upsample_bilinear2d", "upsample_bilinear2d(Tensor input, Any output_size, bool align_corners=False, Any scales_h=None, Any scales_w=None) -> Tensor")
    registry.register_schema("one_hot", "one_hot(Tensor input, int num_classes=-1) -> Tensor")

    # Batch 1 ops
    registry.register_schema("div_", "div_(Tensor(a!) self, Tensor other) -> Tensor")
    _register_binary_ops(("logical_and", "logical_or"), other_type="Any")
    _register_unary_ops(("logical_not",))

    # New math ops
    registry.register_schema("log1p", "log1p(Tensor input) -> Tensor")
    registry.register_schema("expm1", "expm1(Tensor input) -> Tensor")
    registry.register_schema("maximum", "maximum(Tensor input, Any other) -> Tensor")
    registry.register_schema("minimum", "minimum(Tensor input, Any other) -> Tensor")
    registry.register_schema("dot", "dot(Tensor input, Tensor other) -> Tensor")
    registry.register_schema("outer", "outer(Tensor input, Tensor other) -> Tensor")
    registry.register_schema("inner", "inner(Tensor input, Tensor other) -> Tensor")
    registry.register_schema("mv", "mv(Tensor input, Tensor other) -> Tensor")
    registry.register_schema("cross", "cross(Tensor input, Tensor other, int dim=-1) -> Tensor")
    registry.register_schema("tensordot", "tensordot(Tensor input, Tensor other, Any dims=2) -> Tensor")
    registry.register_schema("einsum", "einsum(str equation, Tensor[] tensors) -> Tensor")
    registry.register_schema("mm", "mm(Tensor input, Tensor other) -> Tensor")
    registry.register_schema("bmm", "bmm(Tensor input, Tensor other) -> Tensor")
    registry.register_schema("moveaxis", "moveaxis(Tensor input, Any source, Any destination) -> Tensor")
    registry.register_schema("hardswish", "hardswish(Tensor input) -> Tensor")
    registry.register_schema("hardsigmoid", "hardsigmoid(Tensor input) -> Tensor")
    registry.register_schema("softsign", "softsign(Tensor input) -> Tensor")
    registry.register_schema("selu", "selu(Tensor input) -> Tensor")
    registry.register_schema("celu", "celu(Tensor input, float alpha=1.0) -> Tensor")
    registry.register_schema("threshold", "threshold(Tensor input, Scalar threshold, Scalar value) -> Tensor")
    registry.register_schema("instance_norm", "instance_norm(Tensor input, Any weight=None, Any bias=None, Any running_mean=None, Any running_var=None, bool use_input_stats=True, float momentum=0.1, float eps=1e-5, bool cudnn_enabled=False) -> Tensor")
    registry.register_schema("normalize", "normalize(Tensor input, float p=2.0, int dim=1, float eps=1e-12) -> Tensor")

    # New logical ops
    registry.register_schema("logical_xor", "logical_xor(Tensor input, Any other) -> Tensor")

    # New bitwise ops
    registry.register_schema("bitwise_and", "bitwise_and(Tensor input, Any other) -> Tensor")
    registry.register_schema("bitwise_or", "bitwise_or(Tensor input, Any other) -> Tensor")
    registry.register_schema("bitwise_xor", "bitwise_xor(Tensor input, Any other) -> Tensor")
    registry.register_schema("bitwise_not", "bitwise_not(Tensor input) -> Tensor")

    # New random in-place op
    registry.register_schema("randint_", "randint_(Tensor(a!) self, int low, int? high=None) -> Tensor")

    # New creation ops
    registry.register_schema("randint", "randint(int low, int? high=None, int[]? size=None, Dtype? dtype=None, bool requires_grad=False) -> Tensor")
    registry.register_schema("randperm", "randperm(int n, Dtype? dtype=None, bool requires_grad=False) -> Tensor")

    # New shape ops
    registry.register_schema("flatten", "flatten(Tensor input, int start_dim=0, int end_dim=-1) -> Tensor")
    registry.register_schema("unflatten", "unflatten(Tensor input, int dim, int[] sizes) -> Tensor")
    registry.register_schema("broadcast_to", "broadcast_to(Tensor input, int[] shape) -> Tensor")
    registry.register_schema("movedim", "movedim(Tensor input, Any source, Any destination) -> Tensor")
    registry.register_schema("diagonal", "diagonal(Tensor input, int offset=0, int dim1=0, int dim2=1) -> Tensor")

    # New search ops
    registry.register_schema("unique", "unique(Tensor input, bool sorted=True, bool return_inverse=False, bool return_counts=False, int? dim=None) -> Any")
    registry.register_schema("searchsorted", "searchsorted(Tensor sorted_sequence, Any values, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor")
    registry.register_schema("kthvalue", "kthvalue(Tensor input, int k, int dim=-1, bool keepdim=False) -> (Tensor, Tensor)")
    registry.register_schema("median", "median(Tensor input, int? dim=None, bool keepdim=False) -> Any")

    # New GROUP C ops for Tensor API alignment
    registry.register_schema("logsumexp", "logsumexp(Tensor input, int dim, bool keepdim=False) -> Tensor")
    registry.register_schema("trace", "trace(Tensor input) -> Tensor")
    registry.register_schema("det", "det(Tensor input) -> Tensor")
    registry.register_schema("matrix_power", "matrix_power(Tensor input, int n) -> Tensor")
    registry.register_schema("dist", "dist(Tensor input, Tensor other, Any p=2) -> Tensor")
    registry.register_schema("renorm", "renorm(Tensor input, Any p, int dim, Scalar maxnorm) -> Tensor")
    registry.register_schema("nansum", "nansum(Tensor input, int? dim=None, bool keepdim=False) -> Tensor")
    registry.register_schema("nanmean", "nanmean(Tensor input, int? dim=None, bool keepdim=False) -> Tensor")
    registry.register_schema("argwhere", "argwhere(Tensor input) -> Tensor")
    registry.register_schema("baddbmm", "baddbmm(Tensor input, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor")
    registry.register_schema("cummin", "cummin(Tensor input, int dim) -> (Tensor, Tensor)")

    # Optimizer step ops (fused per-parameter kernels)
    registry.register_schema("_sgd_step", "_sgd_step(Tensor param, Tensor grad, Any buf, float lr, float momentum, float dampening, float weight_decay, bool nesterov, bool maximize) -> Tensor")
    registry.register_schema("_adam_step", "_adam_step(Tensor param, Tensor grad, Tensor exp_avg, Tensor exp_avg_sq, Any max_exp_avg_sq, int step, float lr, float beta1, float beta2, float eps, float weight_decay, bool amsgrad, bool maximize) -> Tensor")
    registry.register_schema("_adamw_step", "_adamw_step(Tensor param, Tensor grad, Tensor exp_avg, Tensor exp_avg_sq, Any max_exp_avg_sq, int step, float lr, float beta1, float beta2, float eps, float weight_decay, bool amsgrad, bool maximize) -> Tensor")
    registry.register_schema("_adagrad_step", "_adagrad_step(Tensor param, Tensor grad, Tensor state_sum, int step, float lr, float lr_decay, float weight_decay, float eps, bool maximize) -> Tensor")
    registry.register_schema("_rmsprop_step", "_rmsprop_step(Tensor param, Tensor grad, Tensor square_avg, Any grad_avg, Any buf, int step, float lr, float alpha, float eps, float weight_decay, float momentum, bool centered, bool maximize) -> Tensor")
    registry.register_schema("_adadelta_step", "_adadelta_step(Tensor param, Tensor grad, Tensor square_avg, Tensor acc_delta, float lr, float rho, float eps, float weight_decay, bool maximize) -> Tensor")
    registry.register_schema("_adamax_step", "_adamax_step(Tensor param, Tensor grad, Tensor exp_avg, Tensor exp_inf, int step, float lr, float beta1, float beta2, float eps, float weight_decay, bool maximize) -> Tensor")
    registry.register_schema("_nadam_step", "_nadam_step(Tensor param, Tensor grad, Tensor exp_avg, Tensor exp_avg_sq, int step, float lr, float beta1, float beta2, float eps, float weight_decay, float mu, float mu_next, float mu_product, float mu_product_next, bool maximize) -> Tensor")
    registry.register_schema("_radam_step", "_radam_step(Tensor param, Tensor grad, Tensor exp_avg, Tensor exp_avg_sq, int step, float lr, float beta1, float beta2, float eps, float weight_decay, bool maximize) -> Tensor")
    registry.register_schema("_asgd_step", "_asgd_step(Tensor param, Tensor grad, Tensor ax, int step, float lr, float lambd, float alpha, float t0, float weight_decay, bool maximize) -> Tensor")
    registry.register_schema("_rprop_step", "_rprop_step(Tensor param, Tensor grad, Tensor prev, Tensor step_sizes, float lr, float etaminus, float etaplus, float step_size_min, float step_size_max, bool maximize) -> Tensor")
    registry.register_schema("_sparse_adam_step", "_sparse_adam_step(Tensor param, Tensor grad, Tensor exp_avg, Tensor exp_avg_sq, int step, float lr, float beta1, float beta2, float eps) -> Tensor")
