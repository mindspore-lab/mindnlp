"""Metal Shading Language (MSL) kernel source strings for GPU compute.

All kernels are compiled once at first use via MetalKernelDispatcher.
Naming convention: <op>_f32 / <op>_f16 for each dtype variant.
"""

# ---------------------------------------------------------------------------
# Helper: generate element-wise kernels for both f32 and f16
# ---------------------------------------------------------------------------

_UNARY_TEMPLATE = """
kernel void {name}_{suffix}(device const {type}* a [[buffer(0)]],
                             device {type}* out      [[buffer(1)]],
                             constant uint& N        [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {{
    if (id < N) {{
        {type} x = a[id];
        out[id] = {expr};
    }}
}}
"""

_BINARY_TEMPLATE = """
kernel void {name}_{suffix}(device const {type}* a [[buffer(0)]],
                             device const {type}* b [[buffer(1)]],
                             device {type}* out      [[buffer(2)]],
                             constant uint& N        [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {{
    if (id < N) out[id] = {expr};
}}
"""

_BINARY_SCALAR_TEMPLATE = """
kernel void {name}_scalar_{suffix}(device const {type}* a  [[buffer(0)]],
                                    constant {type}& scalar [[buffer(1)]],
                                    device {type}* out      [[buffer(2)]],
                                    constant uint& N        [[buffer(3)]],
                                    uint id [[thread_position_in_grid]]) {{
    if (id < N) out[id] = {expr};
}}
"""


def _gen_unary(name, expr, types=("float", "half")):
    """Generate unary kernel source for given types."""
    parts = []
    for t in types:
        suffix = "f32" if t == "float" else "f16"
        parts.append(_UNARY_TEMPLATE.format(name=name, suffix=suffix, type=t, expr=expr))
    return "".join(parts)


def _gen_binary(name, expr, types=("float", "half")):
    """Generate binary kernel source for given types."""
    parts = []
    for t in types:
        suffix = "f32" if t == "float" else "f16"
        parts.append(_BINARY_TEMPLATE.format(name=name, suffix=suffix, type=t, expr=expr))
        scalar_expr = expr.replace("b[id]", "scalar")
        parts.append(_BINARY_SCALAR_TEMPLATE.format(name=name, suffix=suffix, type=t, expr=scalar_expr))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Build the full MSL source
# ---------------------------------------------------------------------------

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

# --- Binary element-wise ---
_BINARY_OPS = (
    ("add", "a[id] + b[id]"),
    ("sub", "a[id] - b[id]"),
    ("mul", "a[id] * b[id]"),
    ("div", "a[id] / b[id]"),
    ("maximum", "max(a[id], b[id])"),
    ("minimum", "min(a[id], b[id])"),
    ("pow", "pow(a[id], b[id])"),
    ("fmod", "fmod(a[id], b[id])"),
    ("remainder", "a[id] - b[id] * floor(a[id] / b[id])"),
)

# --- Unary element-wise ---
_UNARY_OPS = (
    ("neg", "-x"),
    ("abs", "abs(x)"),
    ("sqrt", "sqrt(x)"),
    ("rsqrt", "rsqrt(x)"),
    ("exp", "exp(x)"),
    ("log", "log(x)"),
    ("log2", "log2(x)"),
    ("sin", "sin(x)"),
    ("cos", "cos(x)"),
    ("tanh", "tanh(x)"),
    ("sigmoid", "1.0 / (1.0 + exp(-x))"),
    ("sign", "sign(x)"),
    ("floor", "floor(x)"),
    ("ceil", "ceil(x)"),
    ("round", "rint(x)"),
    ("relu", "max(x, ({type})0)"),
    ("gelu", "0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)))"),
    ("silu", "x / (1.0f + exp(-x))"),
)

# --- Reduction kernels (two-pass parallel) ---
_REDUCTION_SOURCE = """
// ---- sum reduction ----
kernel void sum_partial_f32(device const float* input [[buffer(0)]],
                            device float* partials     [[buffer(1)]],
                            constant uint& N           [[buffer(2)]],
                            uint gid [[thread_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]],
                            uint group_id [[threadgroup_position_in_grid]],
                            uint group_size [[threads_per_threadgroup]]) {
    threadgroup float shared[256];
    float val = (gid < N) ? input[gid] : 0.0f;
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) partials[group_id] = shared[0];
}

kernel void sum_final_f32(device const float* partials [[buffer(0)]],
                          device float* output          [[buffer(1)]],
                          constant uint& N              [[buffer(2)]],
                          uint lid [[thread_position_in_threadgroup]],
                          uint group_size [[threads_per_threadgroup]]) {
    threadgroup float shared[256];
    float val = (lid < N) ? partials[lid] : 0.0f;
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) output[0] = shared[0];
}

// ---- max reduction ----
kernel void max_partial_f32(device const float* input [[buffer(0)]],
                            device float* partials     [[buffer(1)]],
                            constant uint& N           [[buffer(2)]],
                            uint gid [[thread_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]],
                            uint group_id [[threadgroup_position_in_grid]],
                            uint group_size [[threads_per_threadgroup]]) {
    threadgroup float shared[256];
    float val = (gid < N) ? input[gid] : -INFINITY;
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = max(shared[lid], shared[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) partials[group_id] = shared[0];
}

kernel void max_final_f32(device const float* partials [[buffer(0)]],
                          device float* output          [[buffer(1)]],
                          constant uint& N              [[buffer(2)]],
                          uint lid [[thread_position_in_threadgroup]],
                          uint group_size [[threads_per_threadgroup]]) {
    threadgroup float shared[256];
    float val = (lid < N) ? partials[lid] : -INFINITY;
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = max(shared[lid], shared[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) output[0] = shared[0];
}

// ---- min reduction ----
kernel void min_partial_f32(device const float* input [[buffer(0)]],
                            device float* partials     [[buffer(1)]],
                            constant uint& N           [[buffer(2)]],
                            uint gid [[thread_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]],
                            uint group_id [[threadgroup_position_in_grid]],
                            uint group_size [[threads_per_threadgroup]]) {
    threadgroup float shared[256];
    float val = (gid < N) ? input[gid] : INFINITY;
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = min(shared[lid], shared[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) partials[group_id] = shared[0];
}

kernel void min_final_f32(device const float* partials [[buffer(0)]],
                          device float* output          [[buffer(1)]],
                          constant uint& N              [[buffer(2)]],
                          uint lid [[thread_position_in_threadgroup]],
                          uint group_size [[threads_per_threadgroup]]) {
    threadgroup float shared[256];
    float val = (lid < N) ? partials[lid] : INFINITY;
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = min(shared[lid], shared[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) output[0] = shared[0];
}

// ---- argmax reduction ----
kernel void argmax_partial_f32(device const float* input [[buffer(0)]],
                               device float* partial_vals [[buffer(1)]],
                               device uint* partial_idxs   [[buffer(2)]],
                               constant uint& N            [[buffer(3)]],
                               uint gid [[thread_position_in_grid]],
                               uint lid [[thread_position_in_threadgroup]],
                               uint group_id [[threadgroup_position_in_grid]],
                               uint group_size [[threads_per_threadgroup]]) {
    threadgroup float s_val[256];
    threadgroup uint s_idx[256];
    float val = (gid < N) ? input[gid] : -INFINITY;
    uint idx = gid;
    s_val[lid] = val;
    s_idx[lid] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s && s_val[lid + s] > s_val[lid]) {
            s_val[lid] = s_val[lid + s];
            s_idx[lid] = s_idx[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        partial_vals[group_id] = s_val[0];
        partial_idxs[group_id] = s_idx[0];
    }
}

kernel void argmax_final_f32(device const float* partial_vals [[buffer(0)]],
                             device const uint* partial_idxs   [[buffer(1)]],
                             device uint* output               [[buffer(2)]],
                             constant uint& N                  [[buffer(3)]],
                             uint lid [[thread_position_in_threadgroup]],
                             uint group_size [[threads_per_threadgroup]]) {
    threadgroup float s_val[256];
    threadgroup uint s_idx[256];
    float val = (lid < N) ? partial_vals[lid] : -INFINITY;
    uint idx = (lid < N) ? partial_idxs[lid] : 0;
    s_val[lid] = val;
    s_idx[lid] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s && s_val[lid + s] > s_val[lid]) {
            s_val[lid] = s_val[lid + s];
            s_idx[lid] = s_idx[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) output[0] = s_idx[0];
}

// ---- argmin reduction ----
kernel void argmin_partial_f32(device const float* input [[buffer(0)]],
                               device float* partial_vals [[buffer(1)]],
                               device uint* partial_idxs   [[buffer(2)]],
                               constant uint& N            [[buffer(3)]],
                               uint gid [[thread_position_in_grid]],
                               uint lid [[thread_position_in_threadgroup]],
                               uint group_id [[threadgroup_position_in_grid]],
                               uint group_size [[threads_per_threadgroup]]) {
    threadgroup float s_val[256];
    threadgroup uint s_idx[256];
    float val = (gid < N) ? input[gid] : INFINITY;
    uint idx = gid;
    s_val[lid] = val;
    s_idx[lid] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s && s_val[lid + s] < s_val[lid]) {
            s_val[lid] = s_val[lid + s];
            s_idx[lid] = s_idx[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        partial_vals[group_id] = s_val[0];
        partial_idxs[group_id] = s_idx[0];
    }
}

kernel void argmin_final_f32(device const float* partial_vals [[buffer(0)]],
                             device const uint* partial_idxs   [[buffer(1)]],
                             device uint* output               [[buffer(2)]],
                             constant uint& N                  [[buffer(3)]],
                             uint lid [[thread_position_in_threadgroup]],
                             uint group_size [[threads_per_threadgroup]]) {
    threadgroup float s_val[256];
    threadgroup uint s_idx[256];
    float val = (lid < N) ? partial_vals[lid] : INFINITY;
    uint idx = (lid < N) ? partial_idxs[lid] : 0;
    s_val[lid] = val;
    s_idx[lid] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s && s_val[lid + s] < s_val[lid]) {
            s_val[lid] = s_val[lid + s];
            s_idx[lid] = s_idx[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) output[0] = s_idx[0];
}

// ---- fill kernel ----
kernel void fill_f32(device float* out    [[buffer(0)]],
                     constant float& val  [[buffer(1)]],
                     constant uint& N     [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    if (id < N) out[id] = val;
}

kernel void fill_f16(device half* out    [[buffer(0)]],
                     constant half& val  [[buffer(1)]],
                     constant uint& N    [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    if (id < N) out[id] = val;
}

// ---- copy kernel ----
kernel void copy_f32(device const float* src [[buffer(0)]],
                     device float* dst       [[buffer(1)]],
                     constant uint& N        [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    if (id < N) dst[id] = src[id];
}

kernel void copy_f16(device const half* src [[buffer(0)]],
                     device half* dst       [[buffer(1)]],
                     constant uint& N       [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    if (id < N) dst[id] = src[id];
}

// ---- softmax (per-row, last dim) ----
kernel void softmax_f32(device const float* input [[buffer(0)]],
                        device float* output       [[buffer(1)]],
                        constant uint& rows        [[buffer(2)]],
                        constant uint& cols        [[buffer(3)]],
                        uint2 pos [[thread_position_in_grid]]) {
    uint row = pos.y;
    uint col = pos.x;
    if (row >= rows || col >= cols) return;

    // Find max in this row
    float max_val = -INFINITY;
    for (uint c = 0; c < cols; c++) {
        max_val = max(max_val, input[row * cols + c]);
    }

    // Compute exp(x - max)
    float exp_val = exp(input[row * cols + col] - max_val);

    // Sum of exp in this row
    float sum_exp = 0.0f;
    for (uint c = 0; c < cols; c++) {
        sum_exp += exp(input[row * cols + c] - max_val);
    }

    output[row * cols + col] = exp_val / sum_exp;
}
"""

# --- In-place binary kernels ---
_INPLACE_BINARY_TEMPLATE = """
kernel void {name}_inplace_{suffix}(device {type}* a       [[buffer(0)]],
                                     device const {type}* b [[buffer(1)]],
                                     constant uint& N       [[buffer(2)]],
                                     uint id [[thread_position_in_grid]]) {{
    if (id < N) a[id] {op}= b[id];
}}

kernel void {name}_inplace_scalar_{suffix}(device {type}* a      [[buffer(0)]],
                                            constant {type}& scalar [[buffer(1)]],
                                            constant uint& N        [[buffer(2)]],
                                            uint id [[thread_position_in_grid]]) {{
    if (id < N) a[id] {op}= scalar;
}}
"""

_INPLACE_UNARY_SOURCE = """
kernel void relu_inplace_f32(device float* a [[buffer(0)]],
                              constant uint& N [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    if (id < N && a[id] < 0.0f) a[id] = 0.0f;
}

kernel void relu_inplace_f16(device half* a [[buffer(0)]],
                              constant uint& N [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    if (id < N && a[id] < (half)0.0f) a[id] = (half)0.0f;
}
"""


def _build_msl_source():
    """Assemble the complete MSL source string."""
    parts = [_HEADER]

    # Binary ops
    for name, expr in _BINARY_OPS:
        parts.append(_gen_binary(name, expr))

    # Unary ops
    for name, expr in _UNARY_OPS:
        # Handle {type} placeholder in expressions like relu
        for t, suffix in (("float", "f32"), ("half", "f16")):
            e = expr.replace("{type}", t)
            parts.append(_UNARY_TEMPLATE.format(name=name, suffix=suffix, type=t, expr=e))

    # Reductions and special kernels
    parts.append(_REDUCTION_SOURCE)

    # In-place binary ops
    for name, op in (("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/")):
        for t, suffix in (("float", "f32"), ("half", "f16")):
            parts.append(_INPLACE_BINARY_TEMPLATE.format(
                name=name, op=op, suffix=suffix, type=t))

    parts.append(_INPLACE_UNARY_SOURCE)

    return "".join(parts)


MSL_SOURCE = _build_msl_source()
