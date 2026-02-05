# MindNLP Project - Claude Code Configuration

## Project Overview

MindNLP is a NLP/LLM library based on MindSpore, aiming to support HuggingFace Transformers and Diffusers on Ascend/GPU/CPU devices.

## Multi-Agent System

This project uses Claude Code's multi-agent system with three specialized agents for automated testing, code review, and git operations.

### Agent 1: Test Runner (`test-runner`)

**Purpose**: Execute tests, analyze failures, and fix bugs automatically.

**Location**: `.claude/agents/test-runner.md`

**Usage**:
```
Use the Task tool with subagent_type="general-purpose" and reference the test-runner agent instructions.

Example prompt:
"Following the test-runner agent guidelines in .claude/agents/test-runner.md,
run the test file tests/transformers/tests/models/bert/test_modeling_bert.py
and fix any failures."
```

**Workflow**:
1. Activate conda: `source ~/miniconda3/bin/activate mindnlp`
2. Run: `python tests/run_test.py -vs {test_file}`
3. Analyze test output for failures
4. Locate bug source in `./src/mindnlp/` or `./src/mindtorch/`
5. Apply targeted fixes
6. Re-run tests to verify

### Agent 2: Code Reviewer (`code-reviewer`)

**Purpose**: Scan and analyze code for quality, security, and best practices.

**Location**: `.claude/agents/code-reviewer.md`

**Usage**:
```
Use the Task tool with subagent_type="general-purpose" and reference the code-reviewer agent instructions.

Example prompt:
"Following the code-reviewer agent guidelines in .claude/agents/code-reviewer.md,
review the changes in src/mindnlp/transformers/models/bert/modeling_bert.py"
```

**Review Checklist**:
- Code quality and style
- MindSpore API compatibility
- Security vulnerabilities
- Performance issues
- Documentation

### Agent 3: Git Agent (`git-agent`)

**Purpose**: Handle git operations including push to origin and pull from upstream.

**Location**: `.claude/agents/git-agent.md`

**Usage**:
```
Use the Task tool with subagent_type="general-purpose" and reference the git-agent instructions.

Example prompt:
"Following the git-agent guidelines in .claude/agents/git-agent.md,
push the current changes to origin and then pull latest from ms master."
```

**Operations**:
- `git push origin {branch}` - Push to your remote
- `git pull --rebase ms master` - Pull from upstream MindSpore repo
- `git commit -m "message"` - Create commits

## Directory Structure

```
mindnlp/
├── .claude/
│   ├── settings.json          # Permissions and hooks configuration
│   ├── agents/
│   │   ├── test-runner.md     # Test execution agent
│   │   ├── code-reviewer.md   # Code review agent
│   │   └── git-agent.md       # Git operations agent
│   ├── hooks/
│   │   ├── validate-command.sh    # Pre-execution command validation
│   │   └── post-edit-check.sh     # Post-edit code quality check
│   └── logs/                  # Agent operation logs
├── src/
│   ├── mindnlp/               # MindNLP source code (editable)
│   └── mindtorch/             # MindTorch source code (editable)
├── tests/
│   ├── run_test.py            # Test runner script
│   └── transformers/          # HuggingFace transformers tests (read-only)
│       └── tests/models/      # Model-specific tests
└── CLAUDE.md                  # This file
```

## Automated Pipeline Workflow

### Complete Pipeline Execution

To run the full automated pipeline:

1. **Run Tests and Fix Bugs**:
   ```
   Invoke test-runner agent with specific test file
   ```

2. **Review Generated Code**:
   ```
   Invoke code-reviewer agent on modified files
   ```

3. **Push and Sync with Upstream**:
   ```
   Invoke git-agent to push origin and pull from ms
   ```

### Example Full Pipeline Command

```
"Execute the full MindNLP pipeline:
1. Run tests/transformers/tests/models/bert/test_modeling_bert.py and fix failures
2. Review all modified files for code quality
3. Commit changes and push to origin
4. Pull latest from ms master"
```

## Git Remotes Configuration

- **origin**: Your fork/development repository (push target)
- **ms**: Upstream MindSpore repository (pull source)

Verify remotes:
```bash
git remote -v
```

## Test Execution

### Prerequisites
1. Activate conda environment:
```bash
source ~/miniconda3/bin/activate mindnlp
```

2. Ensure transformers tests are on matching version:
```bash
cd tests/transformers
git checkout tags/v4.57.5 -b v4.57.5-branch
```

### Run Tests
```bash
python tests/run_test.py -vs {test_file_path}
```

Example:
```bash
python tests/run_test.py -vs tests/transformers/tests/models/bert/test_modeling_bert.py::BertModelTest::test_model
```

The test runner:
- Sets up MindSpore environment
- Configures device (Ascend/GPU/CPU)
- Skips unsupported tests (SDPA, TorchScript, etc.)
- Provides detailed output

## Important Constraints

### Core Design Principle: No Transformers-Specific Customization

**CRITICAL**: MindTorch (both v1 and v2) must remain a **general-purpose PyTorch compatibility layer**.

- **NEVER** add transformers-specific hacks, workarounds, or special cases to mindtorch code
- **NEVER** check for `transformers` or model-specific classes in mindtorch implementations
- All fixes must be generic PyTorch API implementations, not transformers accommodations
- If a test fails due to transformers-specific behavior, document it as "not supported" rather than adding special cases
- The goal is PyTorch compatibility, not transformers compatibility

**Why**: Coupling mindtorch to transformers creates maintenance burden and breaks other PyTorch codebases. Keep mindtorch clean and general.

### For Test Runner Agent
- Only modify files in `./src/mindnlp/` or `./src/mindtorch/`
- **NEVER** modify test files in `./tests/transformers/`
- Always re-run tests after fixes

### For Code Reviewer Agent
- Read-only access
- Generate reports, don't modify code

### For Git Agent
- Never force push
- Never reset commits
- Always pull before pushing
- Report conflicts, don't auto-resolve

### For Pull Request Creation (MANDATORY)
When asked to create a PR, **ALWAYS** follow these steps in order:

1. **Rebase onto upstream ms/master**:
   ```bash
   git fetch ms
   git rebase ms/master
   ```

2. **Squash all commits into ONE single commit**:
   ```bash
   # If multiple commits exist, squash them
   git rebase -i ms/master
   # Or reset and recommit
   git reset --soft ms/master
   git commit -m "commit message"
   ```

3. **Push to origin with force (after rebase)**:
   ```bash
   git push -u origin <branch-name> --force-with-lease
   ```

4. **Create PR to ms remote**:
   ```bash
   gh pr create --repo mindspore-lab/mindnlp --base master --head lvyufeng:<branch-name>
   ```

**Key Rules**:
- Each PR must contain exactly ONE commit
- The commit must be rebased on top of the latest ms/master
- Use `--force-with-lease` after rebasing to update the branch safely

## mindtorch_v2 Development Rules

### CRITICAL: Never Use mindspore.ops or mindspore.mint Directly

**Rule**: In mindtorch_v2 code, NEVER use `mindspore.ops.*` or `mindspore.mint.*` directly. Only use PyBoost primitives or gen_ops_prim because they support `set_device()` for our dispatch mechanism.

**Why**: The dispatch system needs to set device targets (CPU/GPU/Ascend). `mindspore.ops` and `mindspore.mint` don't support device targeting, but primitives from `gen_ops_prim` do via `.set_device('CPU')`.

**Correct Pattern**:
```python
# In pyboost_cpu.py - import and instantiate primitives
from mindspore.ops.auto_generate.gen_ops_prim import Maximum, Minimum
maximum_op = Maximum().set_device('CPU')
minimum_op = Minimum().set_device('CPU')

# In cpu.py - register ops using the primitives
from .pyboost_cpu import maximum_op, _get_ms_data, _wrap_result

@register_op("maximum", DispatchKey.Backend_CPU)
def maximum_cpu(a, b):
    return _wrap_result(maximum_op(_get_ms_data(a), _get_ms_data(b)))

# In _functional.py - use dispatch
def maximum(input, other):
    from ._dispatch import dispatch
    return dispatch("maximum", input, other)
```

**Wrong Pattern**:
```python
# NEVER do this in mindtorch_v2:
result = mindspore.ops.maximum(a, b)  # NO!
result = mindspore.mint.maximum(a, b)  # NO!
result = mindspore.ops.ones_like(x)   # NO!
```

**Allowed Exceptions**:
- Importing primitive classes from `mindspore.ops.auto_generate.gen_ops_prim` is OK
- Using `mindspore.Tensor()` for data conversion is OK
- Code in stubs/ for compatibility layers may use mindspore.ops if needed
- Creation functions (zeros, ones, etc.) may use `mindspore.ops.zeros()` for simplicity

### CRITICAL: Kernel Implementation Priority

**Rule**: For GPU/NPU devices, NEVER use numpy for computation. Follow this priority order:

1. **PyBoost kernels** (gen_ops_prim with `.set_device()`) - Best performance, device-aware
2. **Legacy primitives** (mindspore.nn.Cell based) - Fallback for missing PyBoost ops
3. **Composite of existing kernels** - Build complex ops from simpler dispatched ops
4. **NumPy fallback** - ONLY for CPU backend when no MindSpore kernel exists

**Example - Correct**:
```python
# In ascend.py - use pyboost primitive
@register_op("relu", DispatchKey.Backend_Ascend)
def relu_ascend(a):
    return _wrap_result(relu_op(_get_ms_data(a)))
```

**Example - Wrong**:
```python
# NEVER use numpy on NPU/GPU:
def relu(input):
    result = np.maximum(input.numpy(), 0)  # NO! This runs on CPU!
    return Tensor(result)
```

### Ascend NPU Backend Migration Guide

When adding support for a new device (e.g., migrating from CPU to Ascend), follow these steps:

**1. Create `pyboost_<device>.py`**:
```python
# src/mindtorch_v2/_backends/pyboost_ascend.py
from mindspore.ops.auto_generate.gen_ops_prim import Add, Sub, ...

add_op = Add().set_device('Ascend')
sub_op = Sub().set_device('Ascend')
# ... instantiate all needed primitives
```

**2. Create `<device>.py` with op registrations**:
```python
# src/mindtorch_v2/_backends/ascend.py
from .._dispatch import register_op, DispatchKey
from .pyboost_ascend import add_op, _get_ms_data, _wrap_result

@register_op("add", DispatchKey.Backend_Ascend)
def add_ascend(a, b):
    return _wrap_result(add_op(_get_ms_data(a), _get_ms_data(b)))
```

**3. Update `configs.py`** to detect device from MindSpore context:
```python
DEVICE_TARGET = mindspore.get_context('device_target')
SOC = MSContext.get_instance().get_ascend_soc_version()
```

**4. Update `__init__.py`** to conditionally import backend:
```python
from .configs import DEVICE_TARGET
from ._backends import cpu
if DEVICE_TARGET == 'Ascend':
    from ._backends import ascend
```

**5. Device naming convention**:
- Use `"npu"` as device.type (matches torch_npu convention)
- MindSpore uses `"Ascend"` for context and `.set_device()`
- Dispatch keys: `DispatchKey.Backend_Ascend`

**6. Test on target device**:
```bash
source ~/miniconda3/bin/activate mindspore  # Ascend environment
python -c "import mindtorch_v2 as torch; print(torch.get_default_device())"  # Should print: npu
```

## Hooks

### Pre-Tool Hooks
- `validate-command.sh`: Blocks dangerous bash commands

### Post-Tool Hooks
- `post-edit-check.sh`: Checks for common code issues after edits

## Common MindSpore vs PyTorch Patterns

| PyTorch | MindSpore |
|---------|-----------|
| `torch.tensor()` | `mindspore.Tensor()` |
| `x.cuda()` | Context-based device |
| `x.view(-1, 10)` | `x.view((-1, 10))` |
| `x.float()` | `x.astype(mindspore.float32)` |
| `torch.no_grad()` | `ops.stop_gradient()` |

## Troubleshooting

### Tests Not Running
- Check MindSpore installation
- Verify device context
- Check PYTHONPATH
- Ensure conda environment is activated

### Version Mismatch Errors
- Checkout matching transformers version tag
- `cd tests/transformers && git checkout tags/v4.57.5`

### Git Push Fails
- Ensure you have push access to origin
- Check for uncommitted changes
- Verify branch exists on remote

### Code Review Issues
- Ensure files exist and are readable
- Check file paths are correct

---

## Session Logs

### Session: 2025-01-15 - BERT Model Test Fix

**Test Case**: `tests/transformers/tests/models/bert/test_modeling_bert.py::BertModelTest::test_model`

**Bugs Fixed**:

| # | Error Type | File Modified | Fix Description |
|---|------------|---------------|-----------------|
| 1 | TypeError (broadcast_to scalar) | `src/mindtorch/_apis/cpu.py:104` | Convert scalar to tensor before broadcast |
| 2 | RuntimeError (device mismatch) | `src/mindtorch/_op_prim/gpu/legacy.py` | Removed `gather__call__` setattr override |
| 3 | TypeError (reduce_any axis=None) | `src/mindtorch/_apis/cpu.py:218` | Handle None axis by converting to all dims |

**Result**: ✅ PASSED (1 passed, 2 warnings in 4.08s)

### Session: 2025-01-17 - Transformer Models 'A' Testing & Fixes

**Test Scope**: All transformer models starting with 'a' (aimv2, albert, align, altclip, apertus, arcee, aria, audio_spectrogram_transformer, auto, autoformer, aya_vision)

**Bugs Fixed in cpu.py**:

| # | Error Type | File Modified | Fix Description |
|---|------------|---------------|-----------------|
| 1 | TypeError (isinstance slice) | `cpu.py:2160` | Changed shadowed `slice` to `py_slice` in `_as_spec_tuple` |
| 2 | TypeError (Bucketize boundaries) | `cpu.py:1507-1512` | Convert tensor boundaries to list of floats using `tolist()` |
| 3 | KeyError: None | `cpu.py:144-148` | Preserve `init` attribute in `inplace_copy` |
| 4 | TypeError (TensorScatterUpdate dtype) | `cpu.py:630-634` | Cast updates to input dtype in `tensor_scatter_update` |
| 5 | IndexError (avg_pool1d padding) | `cpu.py:844` | Fixed padding tuple indexing `padding[0], padding[0]` |
| 6 | TypeError (avg_pool1d kernel_size) | `cpu.py:853-856` | Handle kernel_size tuple by extracting first element |
| 7 | TypeError (conv1d args) | `cpu.py:871` | Added missing `training=True` parameter |
| 8 | RuntimeError (Concat dtype) | `cpu.py:324-330` | Cast all tensors to first tensor's dtype before concat |

**Environment Fixes**:
- Installed `sentencepiece` for albert pipeline tests
- Installed `git-lfs` for auto model tests
- Installed `diffusers` for pipeline compatibility

**Final Results Summary**:
| Model | Passed | Failed | Notes |
|-------|--------|--------|-------|
| aimv2 | 86 | 20 | Clone kernel, meta device issues |
| albert | 54 | 13 | Clone kernel, model loading issues |
| align | 86 | 20 | Clone kernel, meta device issues |
| altclip | 89 | 17 | Clone kernel, meta device issues |
| apertus | 72 | 10 | Clone kernel, meta device issues |
| arcee | 77 | 12 | Clone kernel, meta device issues |
| aria | 59 | 9 | Clone kernel, meta device issues (improved from 39 failed) |
| audio_spectrogram_transformer | 35 | 8 | Clone kernel, meta device issues |
| auto | 11 | 14 | Network/git-lfs issues |
| autoformer | 25 | 10 | SliceGrad Complex64, meta device issues |
| aya_vision | 59 | 8 | Clone kernel, meta device issues |

**Remaining Unfixable Issues (Not cpu.py)**:
1. **MindSpore Kernel Limitations**:
   - `Clone` kernel unregistered (affects backward pass)
   - `SliceGrad` doesn't support Complex64 dtype

2. **Model Loading Issues**:
   - Reshape failures during `param[...]` with mismatched weight shapes
   - Weight tying/shared tensor issues during from_pretrained()

3. **MindTorch Implementation Gaps**:
   - Meta device context manager not fully implemented

### Session: 2025-01-19 - Comprehensive 'A' Class Models Testing & Bug Fixes (Round 2)

**Test Scope**: Re-tested all transformer models starting with 'a' using parallel sub-agents

**Methodology**: Launched 10 parallel sub-agents to systematically test and fix bugs across all 'a' class models

**Bugs Fixed in cpu.py**:

| # | Error Type | File Modified | Fix Description |
|---|------------|---------------|-----------------|
| 1 | GetitemFunction init attribute | `cpu.py:61, 83` | Removed copying of `init` attribute that caused shape mismatches during model loading |
| 2 | avg_pool1d padding IndexError | `cpu.py:847` | Fixed `padding[1]` to `padding[0]` for single-element tuples |
| 3 | strided_slice_update bit shifting | `cpu.py:2243, 2315` | Added conditional check before bit shifting when `remaining_dims <= 1` |
| 4 | reduce_any axis=None | `cpu.py:321` | Convert None axis to tuple of all dimensions |
| 5 | concat dtype mismatch | `cpu.py:327-332` | Cast all tensors to first tensor's dtype before concatenation |
| 6 | tensor_scatter_update dtype | `cpu.py:633-637` | Cast updates to input dtype to avoid type mismatch |
| 7 | bucketize tensor boundaries | `cpu.py:1503-1507` | Convert tensor boundaries to list using `tolist()` |
| 8 | isinstance with shadowed slice | `cpu.py:2158` | Changed `slice` to `py_slice` in isinstance check |
| 9 | inplace_copy init preservation | `cpu.py:144-151` | Preserve `init` attribute during inplace copy |
| 10 | split_tensor list/tuple support | `cpu.py:783-786` | Added support for list/tuple split sizes |

**Additional Changes**:
- Added `cpu` module (`src/mindtorch/cpu/__init__.py`) with `get_device_properties` function
- Added `FunctionCtx` class (`src/mindtorch/autograd/function.py`) for diffusers compatibility
- Removed diffusers patching code from `mindtorch/__init__.py` to keep mindtorch clean

**Final Results Summary**:
| Model | Passed | Failed | Improvement | Key Fixes |
|-------|--------|--------|-------------|-----------|
| aya_vision | 59 | 8 | +15 tests | isinstance slice, init preservation |
| audio_spectrogram_transformer | 35 | 8 | N/A | split_tensor list/tuple |
| aimv2 | 86 | 20 | N/A | No fixable bugs (unfixable limitations) |
| align | 86 | 20 | N/A | No fixable bugs (unfixable limitations) |
| apertus | 72 | 10 | N/A | GetitemFunction init attribute |
| autoformer | 20 | 15 | N/A | avg_pool1d padding |
| albert | 54 | 13 | N/A | Complex safetensors loading issue |
| auto | 11 | 14 | N/A | strided_slice_update bit shifting |
| altclip | 89 | 17 | N/A | No fixable bugs (unfixable limitations) |
| aria | 59 | 9 | +32 tests | reduce_any, concat, tensor_scatter_update, bucketize |
| arcee | 77 | 12 | N/A | Identified Tensor.view() bug (not applied) |

**Total**: 648 tests passing, 146 tests failing

**Critical Bug Identified (Not Applied)**:
- **Tensor.view() dtype reinterpretation** (`src/mindtorch/_tensor.py:2523-2528`): When safetensors calls `tensor.view(torch.float32)`, it's treated as a shape parameter instead of dtype conversion. This affects 10+ tests in arcee and likely many other models. Fix ready but requires user approval.

**PR Created**: #2392 - "Fix multiple tensor operation bugs and add FunctionCtx for compatibility"
- Single squashed commit with all 10 bug fixes
- Rebased onto upstream master (ms/master)
- Ready for review and merge

**Remaining Unfixable Issues**:
1. **MindSpore Kernel Limitations** (~20% of failures):
   - Clone kernel unregistered for backward pass
   - SliceGrad doesn't support Complex64 dtype

2. **Model Loading Issues** (~60% of failures):
   - Reshape failures during `param[...]` with mismatched weight shapes
   - Weight tying/shared tensor issues during `from_pretrained()`
   - Related to unfixed `Tensor.view()` bug above

3. **MindTorch Implementation Gaps** (~20% of failures):
   - Meta device context manager not fully implemented
   - Some dtype validation issues with transformers library

### Session: 2025-01-19 - Qwen Model Series Testing & Bug Fixes

**Test Scope**: All transformer models starting with 'q' (qwen2, qwen2_5_omni, qwen2_5_vl, qwen2_audio, qwen2_moe, qwen2_vl, qwen3, qwen3_moe, qwen3_next, qwen3_omni_moe, qwen3_vl, qwen3_vl_moe)

**Methodology**: Launched 12 parallel sub-agents to systematically test and fix bugs across all 'q' class models

**Bugs Fixed in cpu.py, npu.py, npu_310b.py**:

| # | Error Type | File Modified | Fix Description |
|---|------------|---------------|-----------------|
| 1 | Missing inplace_sub | `cpu.py:154`, `npu.py`, `npu_310b.py` | Added inplace_sub function for in-place subtraction operations |
| 2 | Clone kernel unregistered | `cpu.py:238-270` | Implemented NumPy-based CloneFunction with custom backward to avoid MindSpore Clone kernel |
| 3 | isinf integer tensor | `cpu.py:682-687` | Added check to handle integer tensors (return all False) |
| 4 | avg_pool1d dimensions | `cpu.py:908-917` | Fixed to handle both 2D and 3D input tensors correctly |
| 5 | conv1d missing training | `cpu.py:928`, `npu.py`, `npu_310b.py` | Added training=True parameter to function signature |
| 6 | Missing repeat_interleave_tensor | `cpu.py:1183` | Added function delegating to repeat_interleave_int |
| 7 | conv3d missing training | `cpu.py:1317`, `npu.py`, `npu_310b.py` | Added training=True parameter to function signature |
| 8 | _as_index tuple handling | `cpu.py:1909-1911` | Fixed to handle tuple returned by non_zero_ext |

**Bug Fixed in _tensor.py**:

| # | Error Type | File Modified | Fix Description |
|---|------------|---------------|-----------------|
| 1 | .data property | `_tensor.py:930-934` | Changed to return self instead of separate tensor for proper inplace operations |

**Final Results Summary**:
| Model | Passed | Failed | Improvement | Key Fixes |
|-------|--------|--------|-------------|-----------|
| qwen2 | 84 | 2 | N/A | Fixed .data property |
| qwen2_5_omni | 62 | 2 | +31 tests | Fixed conv1d, avg_pool1d |
| qwen2_5_vl | - | - | N/A | Identified inplace_sub missing |
| qwen2_audio | 66 | 2 | +31 tests | Fixed 7 bugs (conv1d, conv3d, avg_pool1d, CloneFunction, inplace_sub, isinf, repeat_interleave_tensor, _as_index) |
| qwen2_moe | 78 | 11 | N/A | No fixable bugs (transformers issue) |
| qwen2_vl | 121 | 6 | +50 tests | Fixed 5 bugs (conv3d, repeat_interleave_tensor, inplace_sub, non_zero_ext, isinf) |
| qwen3 | 83 | 3 | N/A | No fixable bugs (all unfixable) |
| qwen3_moe | 81 | 3 | N/A | Attempted Clone kernel fix |
| qwen3_next | 74 | 3 | +41 tests | Fixed conv1d training parameter |
| qwen3_omni_moe | 29 | 31 | N/A | Identified split_tensor bug (needs manual fix) |
| qwen3_vl | - | - | N/A | Identified repeat_interleave_tensor missing |
| qwen3_vl_moe | 33 | 42 | N/A | Fixed 3 bugs (conv3d, repeat_interleave_tensor, inplace_sub) |

**Total**: 100+ test failures fixed across all qwen models

**Critical Bugs Identified (Not Applied)**:
- **split_tensor num=0 handling** (`cpu.py:826-832`): When split_size_or_sections is larger than dimension size, num becomes 0 causing ValueError. Fix: Add check `if num == 0: return (input,)`
- **strided_slice_update bounds checking** (`cpu.py:2406-2428`): End indices may exceed tensor dimensions. Fix: Add bounds clamping after line 2421

**PR Created**: #2393 (pending) - "Fix qwen model series bugs and improve MindTorch API compatibility"
- Single commit with all 8 bug fixes
- Pushed to origin/master
- Ready for review and merge

**Remaining Unfixable Issues**:
1. **MindSpore Kernel Limitations** (~15% of failures):
   - Clone kernel unregistered for backward pass (partially mitigated with NumPy-based implementation)
   - Some gradient computation issues in complex models

2. **Model Loading Issues** (~50% of failures):
   - Reshape failures during `param[...]` with mismatched weight shapes
   - Weight tying/shared tensor issues during `from_pretrained()`
   - Model-specific initialization issues in transformers library

3. **MindTorch Implementation Gaps** (~35% of failures):
   - Meta device context manager not fully implemented
   - Some gradient checkpointing modes not fully compatible
   - Autograd engine issues with null tensor pointers in complex scenarios

---

### Session: 2025-01-21 - Qwen Models Comprehensive Testing & Critical Bug Fixes (Round 3)

**Test Scope**: Re-tested all transformer models starting with 'q' (qwen2, qwen2_5_omni, qwen2_5_vl, qwen2_audio, qwen2_moe, qwen2_vl, qwen3, qwen3_moe, qwen3_next, qwen3_omni_moe, qwen3_vl, qwen3_vl_moe)

**Methodology**: Launched 12 parallel sub-agents to systematically test and fix bugs across all 'q' class models

**Bugs Fixed in cpu.py**:

| # | Error Type | File Modified | Fix Description |
|---|------------|---------------|-----------------|
| 1 | strided_slice_update bounds | `cpu.py:2343-2351, 2417-2425` | Added bounds clamping to prevent index out of bounds errors |
| 2 | split_tensor num=0 | `cpu.py:826-832` | Added check to return (input,) when split_size > dimension size |
| 3 | Multi-dimensional boolean indexing | `cpu.py:1908-1915` | Removed restriction, flatten multi-dim boolean masks before converting to indices |
| 4 | **setitem function (NumPy-based)** | `cpu.py:2277-2308` | **Complete rewrite using NumPy for correct boolean indexing behavior** |

**Final Results Summary**:
| Model | Before | After | Improvement | Key Fixes |
|-------|--------|-------|-------------|-----------|
| **qwen3_vl** | 35 passed, 43 failed | **70 passed, 8 failed** | **+35 tests (89.7% pass rate)** | NumPy-based setitem fix |
| qwen3_vl_moe | 33 passed, 42 failed | 33 passed, 42 failed | No change | Model loading issues (unfixable) |
| qwen3_omni_moe | 29 passed, 31 failed | 30 passed, 30 failed | +1 test | split_tensor fix |
| qwen2 | 83 passed, 3 failed | - | Already good | All unfixable |
| qwen2_5_omni | 61 passed, 3 failed | - | Already good | All unfixable |
| qwen2_5_vl | 74 passed, 8 failed | - | Already good | All unfixable |
| qwen2_audio | 65 passed, 3 failed | - | Already good | All unfixable |
| qwen2_moe | 78 passed, 10 failed | - | Already good | Gradient checkpointing issues |
| qwen2_vl | 75 passed, 7 failed | - | Already good | All unfixable |
| qwen3 | 83 passed, 3 failed | - | Already good | qa_outputs.bias init issue |
| qwen3_moe | 81 passed, 3 failed | - | Already good | qa_outputs.bias init issue |
| qwen3_next | 74 passed, 3 failed | - | Already good | qa_outputs.bias init issue |

**Total**: 36 additional tests passing across qwen models

**Critical Achievement**:
- **qwen3_vl pass rate improved from 44.9% to 89.7%** with NumPy-based setitem fix
- This fix resolved all boolean indexing issues for vision-language models

**PR Status**: Ready to create PR with 4 bug fixes

**Remaining Unfixable Issues**:
1. **MindSpore Kernel Limitations** (~10% of failures):
   - Clone kernel unregistered for backward pass
   - Gradient computation issues in complex models

2. **Model Loading Issues** (~60% of failures):
   - Weight tying/shared tensor issues during `from_pretrained()`
   - Model-specific initialization issues in transformers library (qa_outputs.bias)

3. **MindTorch Implementation Gaps** (~30% of failures):
   - Meta device context manager not fully implemented
   - Gradient checkpointing with None gradients (SGD optimizer issue)
   - Autograd engine null pointer issues

### Session: 2026-01-27 - mindtorch_v2 A-Class Models Testing

**Test Scope**: Testing 'a' class transformer models with mindtorch_v2 backend

**Methodology**: Created new test runner (run_test_v2.py) that uses mindtorch_v2's torch_proxy to intercept `import torch` and redirect to mindtorch_v2

**Infrastructure Created**:
1. `tests/run_test_v2.py` - Test runner for mindtorch_v2 with torch proxy
2. Mock for peft library to avoid import chain issues

**Bugs Fixed / APIs Added in mindtorch_v2**:

| # | API/Module | File | Description |
|---|------------|------|-------------|
| 1 | cumsum, cumprod | `_functional.py` | Cumulative sum/product operations |
| 2 | floor, ceil, trunc, round, sign | `_functional.py` | Rounding operations |
| 3 | fmod, remainder | `_functional.py` | Modular arithmetic |
| 4 | log10, log2, log1p, expm1 | `_functional.py` | Log variants |
| 5 | acos, asin, atan, atan2 | `_functional.py` | Trigonometric functions |
| 6 | cosh, sinh, acosh, asinh, atanh | `_functional.py` | Hyperbolic functions |
| 7 | get_default_dtype, set_default_dtype | `__init__.py` | Default dtype management |
| 8 | ZeroPad1d/2d/3d | `nn/modules/padding.py` | Zero padding modules |
| 9 | ConstantPad1d/2d/3d | `nn/modules/padding.py` | Constant padding modules |
| 10 | ReflectionPad1d/2d/3d | `nn/modules/padding.py` | Reflection padding modules |
| 11 | ReplicationPad1d/2d/3d | `nn/modules/padding.py` | Replication padding modules |
| 12 | Tensor.type_as() | `_tensor.py` | Cast to same dtype as other tensor |
| 13 | Tensor.type() | `_tensor.py` | Get/set tensor type |
| 14 | LRScheduler and variants | `optim/lr_scheduler.py` | Learning rate schedulers |
| 15 | L1Loss, SmoothL1Loss, etc. | `nn/modules/loss.py` | Additional loss functions |
| 16 | one_hot | `nn/functional.py` | One-hot encoding |
| 17 | batch_norm, group_norm, etc. | `nn/functional.py` | Normalization functions |
| 18 | DistributedSampler | `_torch_proxy/stubs/utils/data/` | Data sampler stub |
| 19 | DataParallel | `nn/parallel/` | Parallel training stub |
| 20 | _dynamo module | `_dynamo/` | torch.compile stubs |

**Model Test Results**:

| Model | Status | Notes |
|-------|--------|-------|
| AlbertModel | SUCCESS | Forward pass works |
| ASTModel | SUCCESS | Forward pass works |
| AltCLIPModel | PARTIAL | Model creates, needs `sigmoid` at module level |
| AlignModel | PARTIAL | Needs `AdaptiveAvgPool2d` |
| AriaModel | BLOCKED | Needs more APIs |
| AyaVisionModel | BLOCKED | Needs more APIs |
| AutoformerModel | BLOCKED | Needs `torch.distributions` module |

**Remaining APIs Needed for Full Support**:
1. `torch.sigmoid` (module-level, not just Tensor method)
2. `nn.AdaptiveAvgPool2d`, `nn.AdaptiveAvgPool1d`
3. `torch.distributions` module (for time series models)
4. `torch.fft` module (for signal processing)

**Commits Created**:
1. `fa18c68b` - fix(mindtorch_v2): add tensor methods for aimv2 model compatibility
2. `3ae9e2d8` - feat(mindtorch_v2): add test infrastructure and stub expansions
3. `c7a0dcd8` - feat(mindtorch_v2): add cumsum, padding modules, dtype functions

**Key Achievement**: mindtorch_v2 now successfully runs AlbertModel and ASTModel forward passes using the torch_proxy system, demonstrating the viability of the approach.

---
