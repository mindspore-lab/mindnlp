# MindNLP Session History

Historical session logs moved from CLAUDE.md to reduce context size.

---

## mindtorch v1 Sessions (2025-01)

### Session: 2025-01-15 - BERT Model Test Fix

**Test Case**: `tests/transformers/tests/models/bert/test_modeling_bert.py::BertModelTest::test_model`

**Bugs Fixed**:

| # | Error Type | File Modified | Fix Description |
|---|------------|---------------|-----------------|
| 1 | TypeError (broadcast_to scalar) | `src/mindtorch/_apis/cpu.py:104` | Convert scalar to tensor before broadcast |
| 2 | RuntimeError (device mismatch) | `src/mindtorch/_op_prim/gpu/legacy.py` | Removed `gather__call__` setattr override |
| 3 | TypeError (reduce_any axis=None) | `src/mindtorch/_apis/cpu.py:218` | Handle None axis by converting to all dims |

**Result**: PASSED (1 passed, 2 warnings in 4.08s)

### Session: 2025-01-17 - Transformer Models 'A' Testing & Fixes

**Test Scope**: All transformer models starting with 'a'

**Bugs Fixed in cpu.py**: 8 fixes (isinstance slice, Bucketize boundaries, init attribute, TensorScatterUpdate dtype, avg_pool1d padding/kernel_size, conv1d args, Concat dtype)

**Results**: 648 tests passing, 146 failing across 11 models (aimv2, albert, align, altclip, apertus, arcee, aria, audio_spectrogram_transformer, auto, autoformer, aya_vision)

**PR Created**: #2392

### Session: 2025-01-19 - 'A' Class Models Round 2

**Bugs Fixed in cpu.py**: 10 fixes (GetitemFunction init, avg_pool1d padding, strided_slice_update, reduce_any, concat dtype, tensor_scatter_update, bucketize, isinstance slice, inplace_copy, split_tensor)

**Total**: 648 tests passing, 146 tests failing

### Session: 2025-01-19 - Qwen Model Series Testing

**Test Scope**: All 12 qwen model variants

**Bugs Fixed**: 8 in cpu.py + 1 in _tensor.py (inplace_sub, CloneFunction, isinf, avg_pool1d, conv1d/conv3d training, repeat_interleave_tensor, _as_index, .data property)

**Total**: 100+ test failures fixed. PR #2393 created.

### Session: 2025-01-21 - Qwen Models Round 3

**Bugs Fixed in cpu.py**: 4 fixes (strided_slice_update bounds, split_tensor num=0, multi-dim boolean indexing, NumPy-based setitem)

**Key Achievement**: qwen3_vl pass rate improved from 44.9% to 89.7%

---

## mindtorch_v2 Sessions (2026-01 to 2026-02)

### Session: 2026-01-27 - mindtorch_v2 A-Class Models Testing

**Infrastructure Created**: `tests/run_test_v2.py` with torch_proxy

**APIs Added**: 20 new APIs/modules (cumsum, padding modules, dtype functions, LRScheduler, loss functions, etc.)

**Results**: AlbertModel and ASTModel forward passes working. AltCLIP/Align partial. Aria/AyaVision/Autoformer blocked.

### Session: 2026-02-06 - mindtorch_v2 Comprehensive Testing (Albert, BERT, GPT-2)

**Results**:

| Model | Architecture | Pass Rate | Status |
|-------|-------------|-----------|--------|
| Albert | Encoder | 98.2% (54/55) | Excellent |
| BERT | Encoder | 79.1% (110/139) | Good |
| GPT-2 | Decoder | 44.3% (62/140) | Functional |

**Bugs Fixed**: `torch.addmm`, `Tensor.view()` contiguity, Ascend `where()` boolean condition

**Commits**: `0db45246`, `85cb15d2`

**Key Findings**:
- Encoder-only models: production-ready (79-98%)
- Decoder-only models: functional for non-generation workloads (44%)
- Main gaps: text generation, gradient checkpointing, model offloading, SafeTensors edge cases
