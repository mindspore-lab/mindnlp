# Phase 5: BERT Validation Plan

**Date**: 2026-01-26
**Status**: Ready for Implementation
**Goal**: Run BERT forward + backward on mindtorch_v2 and fix gaps until tests pass

## Current State Summary

**Implemented in Phases 1-4:**
- Storage-based Tensor with view semantics
- Dispatch system with CPU backend (NumPy)
- Autograd engine with backward functions for core ops
- nn.Module, Parameter, Linear, Embedding, LayerNorm, Dropout
- Activations: ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax

## Phase 5 Tasks

### Task 1: Create mindtorch_v2 Test Infrastructure

**Goal**: Set up test runner that uses mindtorch_v2 instead of mindtorch v1

**Steps**:
1. Create `tests/mindtorch_v2/` directory for v2-specific tests
2. Create `tests/mindtorch_v2/conftest.py` with fixtures that import from mindtorch_v2
3. Create `tests/mindtorch_v2/test_basic.py` with smoke tests for:
   - Tensor creation and basic ops
   - Autograd forward/backward
   - nn.Module forward pass
4. Run basic tests to verify infrastructure works

**Verification**: `python -m pytest tests/mindtorch_v2/test_basic.py -v`

### Task 2: Implement Missing Tensor Operations

**Goal**: Add tensor operations required by BERT that are missing

**Steps**:
1. Add `cat` / `torch.cat` - concatenate tensors along dimension
2. Add `stack` / `torch.stack` - stack tensors along new dimension
3. Add `split` / `torch.split` - split tensor into chunks
4. Add `chunk` / `torch.chunk` - split tensor into specified number of chunks
5. Add `clone` - create a copy of tensor with new storage
6. Add `repeat` - repeat tensor along dimensions
7. Add `masked_fill` / `masked_fill_` - fill tensor where mask is True
8. Add `where` / `torch.where` - select elements based on condition
9. Add backward functions for new ops

**Verification**: Create unit tests for each new op in `tests/mindtorch_v2/test_ops.py`

### Task 3: Implement Missing Reduction/Math Operations

**Goal**: Add reduction and math ops needed by BERT attention

**Steps**:
1. Add `var` - variance reduction
2. Add `std` - standard deviation reduction
3. Add `clamp` / `clamp_` - clamp values to range
4. Add `rsqrt` - reciprocal square root
5. Add `reciprocal` - 1/x element-wise
6. Add `baddbmm` - batched matrix multiply with add
7. Add `bmm` - batched matrix multiply
8. Add `einsum` - Einstein summation (needed for some attention implementations)
9. Add backward functions for new ops

**Verification**: Create unit tests in `tests/mindtorch_v2/test_math.py`

### Task 4: Implement Missing nn Modules

**Goal**: Add nn modules required by BERT

**Steps**:
1. Add `nn.Conv1d` - 1D convolution (some BERT variants use this)
2. Add `nn.BatchNorm1d` - batch normalization
3. Add `nn.CrossEntropyLoss` - loss function
4. Add `nn.MSELoss` - loss function
5. Add `nn.BCEWithLogitsLoss` - loss function
6. Add `nn.ParameterList` - list of parameters
7. Add `nn.init` module with initialization functions:
   - `xavier_uniform_`, `xavier_normal_`
   - `kaiming_uniform_`, `kaiming_normal_`
   - `normal_`, `uniform_`, `zeros_`, `ones_`

**Verification**: Create unit tests in `tests/mindtorch_v2/test_nn.py`

### Task 5: Implement AdamW Optimizer

**Goal**: Add AdamW optimizer for training

**Steps**:
1. Create `mindtorch_v2/optim/__init__.py`
2. Create `mindtorch_v2/optim/optimizer.py` with base Optimizer class:
   - `zero_grad()` method
   - `step()` method
   - `param_groups` attribute
   - `state` dict for optimizer state
3. Create `mindtorch_v2/optim/adamw.py` with AdamW implementation:
   - Standard AdamW algorithm with weight decay
   - Support for `lr`, `betas`, `eps`, `weight_decay`
   - First and second moment tracking
4. Add `optim.SGD` as fallback optimizer

**Verification**:
```python
# Test optimizer on simple model
model = nn.Linear(10, 2)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
for _ in range(10):
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Task 6: Create BERT Model Shim

**Goal**: Create a minimal BERT-like model for testing forward/backward

**Steps**:
1. Create `tests/mindtorch_v2/models/bert_simple.py` with:
   - `BertEmbeddings` - token + position + segment embeddings
   - `BertSelfAttention` - multi-head self-attention
   - `BertAttention` - self-attention + output projection
   - `BertIntermediate` - feed-forward intermediate layer
   - `BertOutput` - feed-forward output with residual
   - `BertLayer` - full transformer layer
   - `BertEncoder` - stack of layers
   - `BertModel` - full model
2. Use only mindtorch_v2 imports (no torch dependency)
3. Initialize with small config (2 layers, 4 heads, hidden=64)

**Verification**:
```python
config = BertConfig(hidden_size=64, num_attention_heads=4, num_hidden_layers=2)
model = BertModel(config)
outputs = model(input_ids, attention_mask)
```

### Task 7: Run BERT Forward Pass

**Goal**: Execute BERT forward pass and fix any failures

**Steps**:
1. Create random input tensors (batch=2, seq_len=16)
2. Run forward pass through BERT model
3. Identify and fix failures:
   - Missing ops → implement in Task 2/3
   - Shape mismatches → fix tensor operations
   - Dtype issues → add proper dtype handling
4. Iterate until forward pass completes without error

**Verification**: Forward pass produces output tensor of expected shape

### Task 8: Run BERT Backward Pass

**Goal**: Execute BERT backward pass and fix any failures

**Steps**:
1. Compute loss from BERT output (sum or mean)
2. Call `loss.backward()`
3. Identify and fix failures:
   - Missing backward functions → implement
   - Gradient shape mismatches → fix backward implementations
   - Accumulation issues → fix autograd engine
4. Verify all parameters have gradients

**Verification**:
```python
loss.backward()
for name, param in model.named_parameters():
    assert param.grad is not None, f"No gradient for {name}"
```

### Task 9: Run BERT Training Loop

**Goal**: Complete training loop with optimizer

**Steps**:
1. Create optimizer with model parameters
2. Run 10 training iterations:
   - Forward pass
   - Backward pass
   - Optimizer step
   - Zero gradients
3. Verify loss decreases (or at least doesn't explode)
4. Fix any issues with gradient updates

**Verification**: Training loop completes without error, loss values are finite

### Task 10: Integration with transformers Tests

**Goal**: Run actual transformers BERT tests with mindtorch_v2

**Steps**:
1. Create import shim that makes mindtorch_v2 available as `torch`
2. Modify test runner to use mindtorch_v2 backend
3. Run: `python tests/run_test.py -vs tests/transformers/tests/models/bert/test_modeling_bert.py::BertModelTest::test_model`
4. Fix failures iteratively
5. Track pass/fail count

**Verification**: `BertModelTest::test_model` passes

## Success Criteria

Phase 5 is complete when:
1. [ ] All basic mindtorch_v2 tests pass
2. [ ] Simple BERT model forward pass works
3. [ ] Simple BERT model backward pass works
4. [ ] Training loop with AdamW completes
5. [ ] At least one transformers BERT test passes

## Notes

- **Development on macOS/CPU**: All implementations use NumPy backend
- **No MindSpore ops**: Avoid using `mindspore.ops` directly; use primitives via `_op_prim`
- **Follow PyTorch semantics**: When in doubt, match PyTorch behavior exactly
- **Test incrementally**: Run tests after each implementation to catch issues early

## Dependencies Between Tasks

```
Task 1 (infrastructure)
    ↓
Task 2 (tensor ops) ←──────┐
    ↓                      │
Task 3 (math ops) ←────────┤
    ↓                      │
Task 4 (nn modules) ←──────┤
    ↓                      │
Task 5 (optimizer)         │
    ↓                      │
Task 6 (BERT model) ───────┘
    ↓
Task 7 (forward pass)
    ↓
Task 8 (backward pass)
    ↓
Task 9 (training loop)
    ↓
Task 10 (integration)
```

Tasks 2-5 can be done in parallel. Task 6 depends on 2-4. Tasks 7-10 are sequential.
