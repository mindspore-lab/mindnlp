# NPU Op Gap Fill Plan — 2026-03-08

## Overview

93 ops registered for CPU but missing from NPU. This document categorizes each op
by implementation strategy based on ACLNN large-kernel availability and 910B hardware
testing results.

---

## 910B ACLNN Probe Results Summary

| ACLNN kernel | ret code | Status | Notes |
|---|---|---|---|
| aclnnAminmax | 0 | **WORKING** | Upgrade existing composite |
| aclnnBincount | 0 | **WORKING** | Upgrade existing composite |
| aclnnDigamma | 0 | **WORKING** | New op |
| aclnnErfinv | 0 | **WORKING** | Already bound; new NPU registration |
| aclnnLgamma | 0 | **WORKING** | New op |
| aclnnSinc | 0 | **WORKING** | New op |
| aclnnInverse | 0 | **WORKING** | New op (linalg_inv) |
| aclnnLinalgVectorNorm | 0 | **WORKING** | New op |
| aclnnAdaptiveAvgPool3d | 0 | **WORKING** | New op |
| aclnnUpsampleBicubic2d | 0 | **WORKING** | New op |
| aclnnUpsampleLinear1d | 0 | **WORKING** | New op |
| aclnnApplyAdamW (V1) | 0 | **WORKING** | All-tensor params, new op |
| aclnnApplyAdamWV2 | 0 | **WORKING** | Float params, alternative |
| aclnnHeaviside | 561103 | **BROKEN** | Keep existing composite |
| aclnnNanMedian | 161001 | **BROKEN** | Not supported on 910B |
| aclnnLogit | 161001 | **BROKEN** | Not supported on 910B |
| aclnnSlogdet | 161002 | **BROKEN** | Not implemented for Ascend910 |
| aclnnAdaptiveMaxPool2d | 161002 | **BROKEN** | Not implemented |
| aclnnAvgPool3d | 561000 | **BROKEN** | Binary not found on 910B |
| aclnnUpsampleNearest1d | 161002 | **BROKEN** | Not implemented |
| aclnnConvolution (5D) | 161002 | **BROKEN** | 3D conv not supported |

**Previously known broken** (from MEMORY.md):
aclnnEinsum(161002), aclnnStd(161002), aclnnInstanceNorm(161002),
aclnnReduceNansum(161002), aclnnSquare(161002), aclnnIsNegInf(161002),
aclnnIsPosInf(161002), aclnnIsInf(161001),
aclnnAdaptiveAvgPool2d(561103 contamination)

---

## Implementation Categories

### Category 1: ACLNN Large Kernel — New Ops (11 ops, 13 with bmm/mm)

These ops have ACLNN large kernels confirmed working on 910B. Need: add ACLNN
bindings + register NPU dispatch.

| Op | ACLNN kernel | Signature |
|---|---|---|
| `special_digamma` | aclnnDigamma | (self, out) |
| `special_erfinv` | aclnnErfinv | (self, out) — already bound |
| `special_gammaln` | aclnnLgamma | (self, out) |
| `special_sinc` | aclnnSinc | (self, out) |
| `linalg_inv` | aclnnInverse | (self, out) |
| `linalg_vector_norm` | aclnnLinalgVectorNorm | (self, ord, dim[], keepdim, dtype, out) |
| `adaptive_avg_pool3d` | aclnnAdaptiveAvgPool3d | (self, outputSize[], out) |
| `upsample_bicubic2d` | aclnnUpsampleBicubic2d | (self, outSize[], alignCorners, scalesH, scalesW, out) |
| `upsample_linear1d` | aclnnUpsampleLinear1d | (self, outSize[], alignCorners, scales, out) |
| `_adam_step` | aclnnApplyAdamW | all-tensor-param variant |
| `_adamw_step` | aclnnApplyAdamWV2 | float-param variant |
| `bmm` | aclnnBatchMatMul | already bound |
| `mm` | aclnnMm | already bound |

### Category 2: ACLNN Upgrade — Existing Composites (2 ops)

These ops already have composite NPU implementations but have working ACLNN
large kernels. Should upgrade to direct ACLNN calls.

| Op | Current impl | Target ACLNN |
|---|---|---|
| `aminmax` (in ops.py) | composite(amin+amax) | aclnnAminmax |
| `bincount` (in ops.py) | composite(scatter_add) | aclnnBincount |

### Category 3: Composite Implementation (58+ ops)

No ACLNN large kernel available. Implement as compositions of existing working
NPU primitives (add, mul, sum, div, gather, sort, etc.).

#### 3a. Optimizer Ops (10 ops)
Can be implemented as composites of add, mul, div, sqrt, pow:
- `_adadelta_step`
- `_adagrad_step`
- `_adamax_step`
- `_asgd_step`
- `_nadam_step`
- `_radam_step`
- `_rmsprop_step`
- `_rprop_step`
- `_sgd_step`
- `_sparse_adam_step`

#### 3b. Pool/Conv 1D+3D (9 ops)
- `adaptive_avg_pool1d` — composite: reshape to 2D + adaptive_avg_pool2d + reshape back
- `adaptive_max_pool1d` — composite: similar 1D→2D lifting
- `adaptive_max_pool2d` — ACLNN broken; composite of index tricks or max_pool2d
- `avg_pool1d` — composite: reshape + avg_pool2d + reshape
- `avg_pool3d` — ACLNN broken (561000); composite of nested 2D ops or CPU fallback
- `max_pool1d` — composite: reshape + max_pool2d + reshape
- `max_pool3d` — composite: similar to avg_pool3d
- `conv3d` — ACLNN broken; CPU fallback (complex)
- `conv_transpose3d` — CPU fallback (complex)

#### 3c. Linalg Ops Without ACLNN (27 ops)
Most require complex decompositions. Priority order:
- **P0** (used by models): `linalg_norm` (→vector_norm+matrix_norm), `linalg_det` (→slogdet+exp)
- **P1** (useful): `linalg_matrix_norm`, `linalg_multi_dot`, `linalg_matrix_power`
- **P2** (advanced): cholesky, eig, svd, lu, solve, etc. — CPU fallback acceptable

| Op | Strategy |
|---|---|
| `linalg_cholesky` | CPU fallback |
| `linalg_cond` | CPU fallback |
| `linalg_det` | composite: slogdet(sign ignored) + exp(logdet) — but slogdet also broken |
| `linalg_eig` | CPU fallback |
| `linalg_eigh` | CPU fallback |
| `linalg_eigvals` | CPU fallback |
| `linalg_eigvalsh` | CPU fallback |
| `linalg_householder_product` | CPU fallback |
| `linalg_lstsq` | CPU fallback |
| `linalg_lu` | CPU fallback |
| `linalg_lu_factor` | CPU fallback |
| `linalg_lu_solve` | CPU fallback |
| `linalg_matrix_exp` | CPU fallback |
| `linalg_matrix_norm` | composite: vector_norm over matrix dims |
| `linalg_matrix_power` | composite: repeated mm |
| `linalg_matrix_rank` | CPU fallback |
| `linalg_multi_dot` | composite: chain of mm |
| `linalg_norm` | composite: vector_norm or matrix_norm |
| `linalg_pinv` | CPU fallback |
| `linalg_slogdet` | ACLNN broken; CPU fallback |
| `linalg_solve` | CPU fallback |
| `linalg_solve_triangular` | CPU fallback |
| `linalg_svd` | CPU fallback |
| `linalg_svdvals` | CPU fallback |
| `linalg_tensorinv` | CPU fallback |
| `linalg_tensorsolve` | CPU fallback |
| `linalg_vander` | composite: arange + pow broadcasting |

#### 3d. Special Functions Without ACLNN (16 ops)
| Op | Strategy |
|---|---|
| `special_entr` | composite: -x * log(x), where(x>0, ..., 0) |
| `special_erfcx` | composite: erfc(x) * exp(x^2) |
| `special_gammainc` | CPU fallback (regularized incomplete gamma) |
| `special_gammaincc` | CPU fallback |
| `special_i0` | composite: polynomial approx or CPU fallback |
| `special_i0e` | composite: i0(x) * exp(-abs(x)) or CPU fallback |
| `special_i1` | CPU fallback |
| `special_i1e` | CPU fallback |
| `special_log_ndtr` | composite: log(0.5 * erfc(-x/sqrt(2))) |
| `special_logit` | ACLNN broken; composite: log(x/(1-x)) with clamp |
| `special_multigammaln` | composite: sum of lgamma terms |
| `special_ndtr` | composite: 0.5 * erfc(-x/sqrt(2)) |
| `special_ndtri` | CPU fallback (inverse normal CDF) |
| `special_polygamma` | CPU fallback |
| `special_xlog1py` | composite: x * log1p(y) with special cases |
| `special_xlogy` | composite: x * log(y) with special cases |
| `special_zeta` | CPU fallback (Hurwitz zeta) |

### Category 4: FFT Ops — CPU Fallback Only (16 ops)

No ACLNN kernels exist for any FFT operation. All must use CPU fallback.

| Op |
|---|
| `fft_fft`, `fft_ifft`, `fft_rfft`, `fft_irfft` |
| `fft_fft2`, `fft_ifft2`, `fft_rfft2`, `fft_irfft2` |
| `fft_fftn`, `fft_ifftn`, `fft_rfftn`, `fft_irfftn` |
| `fft_hfft`, `fft_ihfft` |
| `fft_fftshift`, `fft_ifftshift` |

---

## Priority Order for Implementation

### Phase 1: Highest Impact (immediate)
Quick wins — direct ACLNN large kernel wrappers:
1. **bmm** + **mm** — critical for all models (just register NPU dispatch for existing ACLNN)
2. **special_digamma, special_erfinv, special_gammaln, special_sinc** — 4 unary ops
3. **linalg_inv** + **linalg_vector_norm** — 2 linalg ops
4. **upsample_bicubic2d** + **upsample_linear1d** — 2 upsample ops

### Phase 2: ACLNN Upgrades + More New Ops
5. Upgrade **aminmax** + **bincount** composites to ACLNN
6. **adaptive_avg_pool3d** — ACLNN new op
7. **_adam_step** + **_adamw_step** — optimizer ops with ACLNN

### Phase 3: Composite Pool/Conv 1D
8. **adaptive_avg_pool1d, avg_pool1d, max_pool1d, adaptive_max_pool1d** — 1D→2D lifting
9. **adaptive_max_pool2d** — composite (ACLNN broken)

### Phase 4: Optimizer Composites
10. All remaining optimizer ops (10 ops) — pure arithmetic composites

### Phase 5: Special Functions
11. Composite special functions (entr, erfcx, logit, ndtr, xlogy, etc.)
12. CPU fallback special functions

### Phase 6: Linalg
13. Composite linalg (norm, multi_dot, matrix_power, vander, det)
14. CPU fallback linalg (23+ ops)

### Phase 7: FFT + 3D Conv
15. All 16 FFT ops — CPU fallback
16. conv3d, conv_transpose3d, avg_pool3d, max_pool3d — CPU fallback

---

## ACLNN Function Signatures Reference

### Unary ops (digamma, lgamma, sinc, erfinv)
```c
aclnnStatus aclnn<Op>GetWorkspaceSize(
    const aclTensor* self, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);
```

### linalg_vector_norm
```c
aclnnStatus aclnnLinalgVectorNormGetWorkspaceSize(
    const aclTensor* self, const aclScalar* ord,
    const aclIntArray* dim, bool keepdim, int32_t dtype,
    aclTensor* out, uint64_t* ws, aclOpExecutor** exec);
```

### adaptive_avg_pool3d
```c
aclnnStatus aclnnAdaptiveAvgPool3dGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* outputSize,
    aclTensor* out, uint64_t* ws, aclOpExecutor** exec);
```

### upsample_bicubic2d
```c
aclnnStatus aclnnUpsampleBicubic2dGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* outputSize,
    bool alignCorners, double scalesH, double scalesW,
    aclTensor* out, uint64_t* ws, aclOpExecutor** exec);
```

### upsample_linear1d
```c
aclnnStatus aclnnUpsampleLinear1dGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* outputSize,
    bool alignCorners, double scales,
    aclTensor* out, uint64_t* ws, aclOpExecutor** exec);
```

### inverse
```c
aclnnStatus aclnnInverseGetWorkspaceSize(
    const aclTensor* self, aclTensor* out,
    uint64_t* ws, aclOpExecutor** exec);
```

### aminmax
```c
aclnnStatus aclnnAminmaxGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* dim, bool keepDim,
    aclTensor* minOut, aclTensor* maxOut,
    uint64_t* ws, aclOpExecutor** exec);
```

### bincount
```c
aclnnStatus aclnnBincountGetWorkspaceSize(
    const aclTensor* self, const aclTensor* weights,
    int64_t minlength, aclTensor* out,
    uint64_t* ws, aclOpExecutor** exec);
```

### ApplyAdamW (V1 — all-tensor params)
```c
aclnnStatus aclnnApplyAdamWGetWorkspaceSize(
    aclTensor* var, aclTensor* m, aclTensor* v,
    const aclTensor* beta1_power, const aclTensor* beta2_power,
    const aclTensor* lr, const aclTensor* weight_decay,
    const aclTensor* beta1, const aclTensor* beta2,
    const aclTensor* epsilon, const aclTensor* grad,
    const aclTensor* max_grad_norm,  // can be NULL
    bool amsgrad, bool maximize,
    uint64_t* ws, aclOpExecutor** exec);
```

### ApplyAdamWV2 (float params)
```c
aclnnStatus aclnnApplyAdamWV2GetWorkspaceSize(
    aclTensor* var, aclTensor* m, aclTensor* v,
    aclTensor* maxGradNormOptRef,  // can be NULL
    const aclTensor* grad, const aclTensor* step,
    float lr, float beta1, float beta2,
    float weightDecay, float eps,
    bool amsgrad, bool maximize,
    uint64_t* ws, aclOpExecutor** exec);
```

---

## Broken ACLNN Ops — Full Record for 910B

| ACLNN symbol | Error | Root cause |
|---|---|---|
| aclnnSquare | 161002 | Binary not for Ascend910 |
| aclnnIsNegInf | 161002 | Not supported on device |
| aclnnIsPosInf | 161002 | Not supported on device |
| aclnnIsInf | 161001 | Binary config missing |
| aclnnEinsum | 161002 | Param validation |
| aclnnStd | 161002 | Param validation |
| aclnnInstanceNorm | 161002 | Param validation |
| aclnnReduceNansum | 161002 | Param validation |
| aclnnAdaptiveAvgPool2d | 561103 | cubeMathType contamination |
| aclnnHeaviside | 561103 | Contamination or 910B bug |
| aclnnNanMedian | 161001 | Not supported on device |
| aclnnLogit | 161001 | Not supported on device |
| aclnnSlogdet | 161002 | Not implemented |
| aclnnAdaptiveMaxPool2d | 161002 | Not implemented |
| aclnnAvgPool3d | 561000 | Binary not found |
| aclnnUpsampleNearest1d | 161002 | Not implemented |
| aclnnConvolution (5D) | 161002 | 3D conv not supported |
| aclnnVar (all-reduce) | 161002 | Fails without explicit dim |
| aclnnMatmul (cubeMath=0) | 161002 | Must use cubeMathType=1 |
