# A-Class Models Testing with mindtorch_v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test all transformer models starting with 'a' using mindtorch_v2 backend and fix any errors encountered.

**Architecture:** Run unit tests for each 'a' class model sequentially, analyze failures, fix bugs in mindtorch_v2 implementation (primarily in `_tensor.py`, `_functional.py`, `_backends/cpu.py`), and verify fixes by re-running tests.

**Tech Stack:** MindSpore, mindtorch_v2, pytest, HuggingFace transformers tests

---

## Prerequisites

### Environment Setup

```bash
source ~/miniconda3/bin/activate mindnlp
cd /Users/lvyufeng/Projects/mindnlp
```

### Models to Test (11 total)

| # | Model | Test File |
|---|-------|-----------|
| 1 | aimv2 | `tests/transformers/tests/models/aimv2/test_modeling_aimv2.py` |
| 2 | albert | `tests/transformers/tests/models/albert/test_modeling_albert.py` |
| 3 | align | `tests/transformers/tests/models/align/test_modeling_align.py` |
| 4 | altclip | `tests/transformers/tests/models/altclip/test_modeling_altclip.py` |
| 5 | apertus | `tests/transformers/tests/models/apertus/test_modeling_apertus.py` |
| 6 | arcee | `tests/transformers/tests/models/arcee/test_modeling_arcee.py` |
| 7 | aria | `tests/transformers/tests/models/aria/test_modeling_aria.py` |
| 8 | audio_spectrogram_transformer | `tests/transformers/tests/models/audio_spectrogram_transformer/test_modeling_audio_spectrogram_transformer.py` |
| 9 | auto | `tests/transformers/tests/models/auto/test_modeling_auto.py` |
| 10 | autoformer | `tests/transformers/tests/models/autoformer/test_modeling_autoformer.py` |
| 11 | aya_vision | `tests/transformers/tests/models/aya_vision/test_modeling_aya_vision.py` |

### Key Files for Bug Fixes

- `src/mindtorch_v2/_tensor.py` - Tensor class implementation
- `src/mindtorch_v2/_functional.py` - Functional operations
- `src/mindtorch_v2/_backends/cpu.py` - CPU backend operations
- `src/mindtorch_v2/_creation.py` - Tensor creation functions
- `src/mindtorch_v2/_autograd/` - Autograd engine
- `src/mindtorch_v2/nn/` - Neural network modules

---

## Task 1: Test aimv2 Model

**Files:**
- Test: `tests/transformers/tests/models/aimv2/test_modeling_aimv2.py`
- Fix: `src/mindtorch_v2/*.py` (as needed)

**Step 1: Run aimv2 tests**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/aimv2/test_modeling_aimv2.py 2>&1 | tee /tmp/aimv2_test.log
```

**Step 2: Analyze failures**

Read the test output log and identify:
- Error types (TypeError, AttributeError, NotImplementedError, etc.)
- Stack traces pointing to mindtorch_v2 files
- Missing operations or incorrect implementations

**Step 3: Fix identified bugs**

For each bug found:
1. Locate the source in mindtorch_v2
2. Implement the fix
3. Document the fix

**Step 4: Re-run tests to verify**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/aimv2/test_modeling_aimv2.py
```

**Step 5: Commit if tests pass**

```bash
git add src/mindtorch_v2/
git commit -m "fix(mindtorch_v2): fix aimv2 model compatibility issues"
```

---

## Task 2: Test albert Model

**Files:**
- Test: `tests/transformers/tests/models/albert/test_modeling_albert.py`
- Fix: `src/mindtorch_v2/*.py` (as needed)

**Step 1: Run albert tests**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/albert/test_modeling_albert.py 2>&1 | tee /tmp/albert_test.log
```

**Step 2: Analyze failures**

Read test output and identify bugs in mindtorch_v2.

**Step 3: Fix identified bugs**

Implement fixes in appropriate mindtorch_v2 files.

**Step 4: Re-run tests to verify**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/albert/test_modeling_albert.py
```

**Step 5: Commit if tests pass**

```bash
git add src/mindtorch_v2/
git commit -m "fix(mindtorch_v2): fix albert model compatibility issues"
```

---

## Task 3: Test align Model

**Files:**
- Test: `tests/transformers/tests/models/align/test_modeling_align.py`
- Fix: `src/mindtorch_v2/*.py` (as needed)

**Step 1: Run align tests**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/align/test_modeling_align.py 2>&1 | tee /tmp/align_test.log
```

**Step 2: Analyze failures**

Read test output and identify bugs in mindtorch_v2.

**Step 3: Fix identified bugs**

Implement fixes in appropriate mindtorch_v2 files.

**Step 4: Re-run tests to verify**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/align/test_modeling_align.py
```

**Step 5: Commit if tests pass**

```bash
git add src/mindtorch_v2/
git commit -m "fix(mindtorch_v2): fix align model compatibility issues"
```

---

## Task 4: Test altclip Model

**Files:**
- Test: `tests/transformers/tests/models/altclip/test_modeling_altclip.py`
- Fix: `src/mindtorch_v2/*.py` (as needed)

**Step 1: Run altclip tests**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/altclip/test_modeling_altclip.py 2>&1 | tee /tmp/altclip_test.log
```

**Step 2: Analyze failures**

Read test output and identify bugs in mindtorch_v2.

**Step 3: Fix identified bugs**

Implement fixes in appropriate mindtorch_v2 files.

**Step 4: Re-run tests to verify**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/altclip/test_modeling_altclip.py
```

**Step 5: Commit if tests pass**

```bash
git add src/mindtorch_v2/
git commit -m "fix(mindtorch_v2): fix altclip model compatibility issues"
```

---

## Task 5: Test apertus Model

**Files:**
- Test: `tests/transformers/tests/models/apertus/test_modeling_apertus.py`
- Fix: `src/mindtorch_v2/*.py` (as needed)

**Step 1: Run apertus tests**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/apertus/test_modeling_apertus.py 2>&1 | tee /tmp/apertus_test.log
```

**Step 2: Analyze failures**

Read test output and identify bugs in mindtorch_v2.

**Step 3: Fix identified bugs**

Implement fixes in appropriate mindtorch_v2 files.

**Step 4: Re-run tests to verify**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/apertus/test_modeling_apertus.py
```

**Step 5: Commit if tests pass**

```bash
git add src/mindtorch_v2/
git commit -m "fix(mindtorch_v2): fix apertus model compatibility issues"
```

---

## Task 6: Test arcee Model

**Files:**
- Test: `tests/transformers/tests/models/arcee/test_modeling_arcee.py`
- Fix: `src/mindtorch_v2/*.py` (as needed)

**Step 1: Run arcee tests**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/arcee/test_modeling_arcee.py 2>&1 | tee /tmp/arcee_test.log
```

**Step 2: Analyze failures**

Read test output and identify bugs in mindtorch_v2.

**Step 3: Fix identified bugs**

Implement fixes in appropriate mindtorch_v2 files.

**Step 4: Re-run tests to verify**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/arcee/test_modeling_arcee.py
```

**Step 5: Commit if tests pass**

```bash
git add src/mindtorch_v2/
git commit -m "fix(mindtorch_v2): fix arcee model compatibility issues"
```

---

## Task 7: Test aria Model

**Files:**
- Test: `tests/transformers/tests/models/aria/test_modeling_aria.py`
- Fix: `src/mindtorch_v2/*.py` (as needed)

**Step 1: Run aria tests**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/aria/test_modeling_aria.py 2>&1 | tee /tmp/aria_test.log
```

**Step 2: Analyze failures**

Read test output and identify bugs in mindtorch_v2.

**Step 3: Fix identified bugs**

Implement fixes in appropriate mindtorch_v2 files.

**Step 4: Re-run tests to verify**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/aria/test_modeling_aria.py
```

**Step 5: Commit if tests pass**

```bash
git add src/mindtorch_v2/
git commit -m "fix(mindtorch_v2): fix aria model compatibility issues"
```

---

## Task 8: Test audio_spectrogram_transformer Model

**Files:**
- Test: `tests/transformers/tests/models/audio_spectrogram_transformer/test_modeling_audio_spectrogram_transformer.py`
- Fix: `src/mindtorch_v2/*.py` (as needed)

**Step 1: Run audio_spectrogram_transformer tests**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/audio_spectrogram_transformer/test_modeling_audio_spectrogram_transformer.py 2>&1 | tee /tmp/ast_test.log
```

**Step 2: Analyze failures**

Read test output and identify bugs in mindtorch_v2.

**Step 3: Fix identified bugs**

Implement fixes in appropriate mindtorch_v2 files.

**Step 4: Re-run tests to verify**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/audio_spectrogram_transformer/test_modeling_audio_spectrogram_transformer.py
```

**Step 5: Commit if tests pass**

```bash
git add src/mindtorch_v2/
git commit -m "fix(mindtorch_v2): fix audio_spectrogram_transformer model compatibility issues"
```

---

## Task 9: Test auto Model

**Files:**
- Test: `tests/transformers/tests/models/auto/test_modeling_auto.py`
- Fix: `src/mindtorch_v2/*.py` (as needed)

**Step 1: Run auto tests**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/auto/test_modeling_auto.py 2>&1 | tee /tmp/auto_test.log
```

**Step 2: Analyze failures**

Read test output and identify bugs in mindtorch_v2.

**Step 3: Fix identified bugs**

Implement fixes in appropriate mindtorch_v2 files.

**Step 4: Re-run tests to verify**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/auto/test_modeling_auto.py
```

**Step 5: Commit if tests pass**

```bash
git add src/mindtorch_v2/
git commit -m "fix(mindtorch_v2): fix auto model compatibility issues"
```

---

## Task 10: Test autoformer Model

**Files:**
- Test: `tests/transformers/tests/models/autoformer/test_modeling_autoformer.py`
- Fix: `src/mindtorch_v2/*.py` (as needed)

**Step 1: Run autoformer tests**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/autoformer/test_modeling_autoformer.py 2>&1 | tee /tmp/autoformer_test.log
```

**Step 2: Analyze failures**

Read test output and identify bugs in mindtorch_v2.

**Step 3: Fix identified bugs**

Implement fixes in appropriate mindtorch_v2 files.

**Step 4: Re-run tests to verify**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/autoformer/test_modeling_autoformer.py
```

**Step 5: Commit if tests pass**

```bash
git add src/mindtorch_v2/
git commit -m "fix(mindtorch_v2): fix autoformer model compatibility issues"
```

---

## Task 11: Test aya_vision Model

**Files:**
- Test: `tests/transformers/tests/models/aya_vision/test_modeling_aya_vision.py`
- Fix: `src/mindtorch_v2/*.py` (as needed)

**Step 1: Run aya_vision tests**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/aya_vision/test_modeling_aya_vision.py 2>&1 | tee /tmp/aya_vision_test.log
```

**Step 2: Analyze failures**

Read test output and identify bugs in mindtorch_v2.

**Step 3: Fix identified bugs**

Implement fixes in appropriate mindtorch_v2 files.

**Step 4: Re-run tests to verify**

```bash
python tests/run_test.py -vs tests/transformers/tests/models/aya_vision/test_modeling_aya_vision.py
```

**Step 5: Commit if tests pass**

```bash
git add src/mindtorch_v2/
git commit -m "fix(mindtorch_v2): fix aya_vision model compatibility issues"
```

---

## Task 12: Final Summary and PR

**Step 1: Generate test summary**

Create a summary of all test results:

```bash
echo "# A-Class Models Test Summary" > /tmp/a_class_summary.md
echo "" >> /tmp/a_class_summary.md
echo "| Model | Passed | Failed | Notes |" >> /tmp/a_class_summary.md
echo "|-------|--------|--------|-------|" >> /tmp/a_class_summary.md
```

**Step 2: Update CLAUDE.md session log**

Add a new session entry documenting:
- All bugs fixed
- Test results per model
- Remaining unfixable issues

**Step 3: Create PR (if requested)**

```bash
git fetch ms
git rebase ms/master
git reset --soft ms/master
git commit -m "fix(mindtorch_v2): improve a-class model compatibility

- Fix tensor operations for model compatibility
- Add missing functional operations
- Improve autograd support

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
git push -u origin HEAD --force-with-lease
gh pr create --repo mindspore-lab/mindnlp --base master --head lvyufeng:$(git branch --show-current)
```

---

## Common Bug Patterns to Watch For

### 1. Missing Tensor Operations
- `tensor.view()`, `tensor.reshape()`, `tensor.permute()`
- `tensor.contiguous()`, `tensor.clone()`
- Indexing operations (boolean, advanced indexing)

### 2. Missing Functional Operations
- `torch.cat()`, `torch.stack()`, `torch.split()`
- `torch.softmax()`, `torch.layer_norm()`
- Reduction ops: `sum`, `mean`, `max`, `min`

### 3. Autograd Issues
- Missing backward pass implementations
- Gradient accumulation problems
- `requires_grad` propagation

### 4. Dtype Mismatches
- Float32 vs Float16 casting
- Integer tensor operations
- Complex number support

### 5. Device Handling
- CPU/GPU tensor placement
- Device transfer operations

---

## Constraints

1. **NEVER modify test files** in `tests/transformers/` - only fix mindtorch_v2
2. **Document all fixes** in commit messages and CLAUDE.md
3. **Re-run tests after each fix** to verify
4. **Commit frequently** - one logical fix per commit
