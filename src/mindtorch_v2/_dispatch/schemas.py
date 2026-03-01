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
        "log2", "log10", "exp2", "rsqrt", "sign", "signbit", "isnan", "isinf", "isfinite",
        "sinh", "cosh", "asinh", "acosh", "atanh", "erf", "erfc", "softplus",
        "relu6", "contiguous", "gelu", "silu", "mish",
    ))
    registry.register_schema("hardtanh", "hardtanh(Tensor input, Scalar min_val=-1.0, Scalar max_val=1.0) -> Tensor")
    registry.register_schema("softmax", "softmax(Tensor input, int dim=-1, Dtype? dtype=None) -> Tensor")
    registry.register_schema("log_softmax", "log_softmax(Tensor input, int dim=-1, Dtype? dtype=None) -> Tensor")
    registry.register_schema("dropout", "dropout(Tensor input, float p=0.5, bool training=True) -> Tensor")
    registry.register_schema("leaky_relu", "leaky_relu(Tensor input, float negative_slope=0.01) -> Tensor")
    registry.register_schema("elu", "elu(Tensor input, float alpha=1.0) -> Tensor")
    _register_binary_ops(("pow", "sub", "div", "true_divide", "min", "max", "fmin", "fmax", "atan2", "lerp", "addcmul", "addcdiv", "logaddexp", "logaddexp2", "hypot", "remainder", "fmod"), other_type="Any")
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
