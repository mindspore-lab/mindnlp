# A-Class Models Testing with mindtorch_v2

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test all transformer models starting with 'a' using mindtorch_v2 and fix any API compatibility issues.

**Architecture:** Run tests for each model sequentially, identify missing APIs or bugs in mindtorch_v2, fix them in `src/mindtorch_v2/`, and verify fixes. Models are tested in alphabetical order with increasing complexity.

**Tech Stack:** mindtorch_v2, pytest, transformers library, MindSpore backend

---

## Models to Test (11 total)

| # | Model | Test File | Passed | Failed | Skipped | Notes |
|---|-------|-----------|--------|--------|---------|-------|
| 1 | albert | `test_modeling_albert.py` | 81 | 0 | 101 | ✅ BASELINE |
| 2 | audio_spectrogram_transformer | `test_modeling_audio_spectrogram_transformer.py` | 64 | 3 | 109 | Gradient checkpointing |
| 3 | aimv2 | `test_modeling_aimv2.py` | 172 | 14 | 225 | SDPA precision |
| 4 | align | `test_modeling_align.py` | 105 | 1 | 306 | retain_grad |
| 5 | altclip | `test_modeling_altclip.py` | 106 | 0 | 306 | ✅ PERFECT |
| 6 | apertus | `test_modeling_apertus.py` | 95 | 16 | 117 | Generation tests |
| 7 | arcee | `test_modeling_arcee.py` | 98 | 18 | 116 | RoPE, generation |
| 8 | aria | `test_modeling_aria.py` | 39 | 56 | 73 | Vision-language |
| 9 | autoformer | `test_modeling_autoformer.py` | 26 | 9 | 142 | FFT/complex |
| 10 | aya_vision | `test_modeling_aya_vision.py` | 50 | 43 | 129 | Vision-language |
| 11 | auto | `test_modeling_auto.py` | 11 | 7 | 13 | Network/integration |

**Total: 847 passed, 167 failed across all models**

---

## Task 1: Test audio_spectrogram_transformer Model

**Files:**
- Test: `tests/transformers/tests/models/audio_spectrogram_transformer/test_modeling_audio_spectrogram_transformer.py`
- Fix: `src/mindtorch_v2/` (as needed)

**Step 1: Run test and capture output**

Run:
```bash
source ~/miniconda3/bin/activate mindnlp && python tests/run_test_v2.py -vs tests/transformers/tests/models/audio_spectrogram_transformer/test_modeling_audio_spectrogram_transformer.py 2>&1 | tee /tmp/ast_test.log
```

**Step 2: Analyze failures**

Look for error patterns:
- `AttributeError: module 'torch' has no attribute 'X'` → Add missing API
- `NotImplementedError` → Implement missing function
- `TypeError` → Fix function signature or behavior

**Step 3: Fix identified issues**

For each missing API, add it to the appropriate file in `src/mindtorch_v2/`.

**Step 4: Re-run tests and verify**

Run the same test command and confirm pass rate improves.

---

## Task 2: Test aimv2 Model

**Files:**
- Test: `tests/transformers/tests/models/aimv2/test_modeling_aimv2.py`
- Fix: `src/mindtorch_v2/` (as needed)

**Step 1: Run test and capture output**

Run:
```bash
source ~/miniconda3/bin/activate mindnlp && python tests/run_test_v2.py -vs tests/transformers/tests/models/aimv2/test_modeling_aimv2.py 2>&1 | tee /tmp/aimv2_test.log
```

**Step 2: Analyze failures**

Look for error patterns in the output.

**Step 3: Fix identified issues**

Add missing APIs or fix bugs in mindtorch_v2.

**Step 4: Re-run tests and verify**

---

## Task 3: Test align Model

**Files:**
- Test: `tests/transformers/tests/models/align/test_modeling_align.py`
- Fix: `src/mindtorch_v2/` (as needed)

**Step 1: Run test and capture output**

Run:
```bash
source ~/miniconda3/bin/activate mindnlp && python tests/run_test_v2.py -vs tests/transformers/tests/models/align/test_modeling_align.py 2>&1 | tee /tmp/align_test.log
```

**Step 2: Analyze failures**

**Step 3: Fix identified issues**

**Step 4: Re-run tests and verify**

---

## Task 4: Test altclip Model

**Files:**
- Test: `tests/transformers/tests/models/altclip/test_modeling_altclip.py`
- Fix: `src/mindtorch_v2/` (as needed)

**Step 1: Run test and capture output**

Run:
```bash
source ~/miniconda3/bin/activate mindnlp && python tests/run_test_v2.py -vs tests/transformers/tests/models/altclip/test_modeling_altclip.py 2>&1 | tee /tmp/altclip_test.log
```

**Step 2: Analyze failures**

**Step 3: Fix identified issues**

**Step 4: Re-run tests and verify**

---

## Task 5: Test apertus Model

**Files:**
- Test: `tests/transformers/tests/models/apertus/test_modeling_apertus.py`
- Fix: `src/mindtorch_v2/` (as needed)

**Step 1: Run test and capture output**

Run:
```bash
source ~/miniconda3/bin/activate mindnlp && python tests/run_test_v2.py -vs tests/transformers/tests/models/apertus/test_modeling_apertus.py 2>&1 | tee /tmp/apertus_test.log
```

**Step 2: Analyze failures**

**Step 3: Fix identified issues**

**Step 4: Re-run tests and verify**

---

## Task 6: Test arcee Model

**Files:**
- Test: `tests/transformers/tests/models/arcee/test_modeling_arcee.py`
- Fix: `src/mindtorch_v2/` (as needed)

**Step 1: Run test and capture output**

Run:
```bash
source ~/miniconda3/bin/activate mindnlp && python tests/run_test_v2.py -vs tests/transformers/tests/models/arcee/test_modeling_arcee.py 2>&1 | tee /tmp/arcee_test.log
```

**Step 2: Analyze failures**

**Step 3: Fix identified issues**

**Step 4: Re-run tests and verify**

---

## Task 7: Test aria Model

**Files:**
- Test: `tests/transformers/tests/models/aria/test_modeling_aria.py`
- Fix: `src/mindtorch_v2/` (as needed)

**Step 1: Run test and capture output**

Run:
```bash
source ~/miniconda3/bin/activate mindnlp && python tests/run_test_v2.py -vs tests/transformers/tests/models/aria/test_modeling_aria.py 2>&1 | tee /tmp/aria_test.log
```

**Step 2: Analyze failures**

**Step 3: Fix identified issues**

**Step 4: Re-run tests and verify**

---

## Task 8: Test autoformer Model

**Files:**
- Test: `tests/transformers/tests/models/autoformer/test_modeling_autoformer.py`
- Fix: `src/mindtorch_v2/` (as needed)

**Step 1: Run test and capture output**

Run:
```bash
source ~/miniconda3/bin/activate mindnlp && python tests/run_test_v2.py -vs tests/transformers/tests/models/autoformer/test_modeling_autoformer.py 2>&1 | tee /tmp/autoformer_test.log
```

**Step 2: Analyze failures**

**Step 3: Fix identified issues**

**Step 4: Re-run tests and verify**

---

## Task 9: Test aya_vision Model

**Files:**
- Test: `tests/transformers/tests/models/aya_vision/test_modeling_aya_vision.py`
- Fix: `src/mindtorch_v2/` (as needed)

**Step 1: Run test and capture output**

Run:
```bash
source ~/miniconda3/bin/activate mindnlp && python tests/run_test_v2.py -vs tests/transformers/tests/models/aya_vision/test_modeling_aya_vision.py 2>&1 | tee /tmp/aya_vision_test.log
```

**Step 2: Analyze failures**

**Step 3: Fix identified issues**

**Step 4: Re-run tests and verify**

---

## Task 10: Test auto Model (Integration)

**Files:**
- Test: `tests/transformers/tests/models/auto/test_modeling_auto.py`
- Fix: `src/mindtorch_v2/` (as needed)

**Step 1: Run test and capture output**

Run:
```bash
source ~/miniconda3/bin/activate mindnlp && python tests/run_test_v2.py -vs tests/transformers/tests/models/auto/test_modeling_auto.py 2>&1 | tee /tmp/auto_test.log
```

**Step 2: Analyze failures**

**Step 3: Fix identified issues**

**Step 4: Re-run tests and verify**

---

## Task 11: Final Summary and Commit

**Step 1: Run all tests to get final status**

Run:
```bash
source ~/miniconda3/bin/activate mindnlp && for model in albert audio_spectrogram_transformer aimv2 align altclip apertus arcee aria autoformer aya_vision auto; do
    echo "=== Testing $model ==="
    python tests/run_test_v2.py --tb=no -q tests/transformers/tests/models/$model/test_modeling_*.py 2>&1 | tail -5
done
```

**Step 2: Document results**

Update this plan with final pass/fail counts for each model.

**Step 3: Commit all fixes**

```bash
git add src/mindtorch_v2/
git commit -m "feat(mindtorch_v2): add APIs for a-class transformer models

- Add missing torch APIs identified during testing
- Fix compatibility issues with transformers library

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Common Error Patterns and Fixes

### Missing Module-Level Functions
If `torch.X` is missing, add to `src/mindtorch_v2/__init__.py` or `src/mindtorch_v2/_functional.py`.

### Missing nn.Module Classes
Add to `src/mindtorch_v2/nn/modules/` in the appropriate file.

### Missing nn.functional Functions
Add to `src/mindtorch_v2/nn/functional.py`.

### Tensor Method Missing
Add to `src/mindtorch_v2/_tensor.py` in the Tensor class.

### Stub Module Missing
Add to `src/mindtorch_v2/_torch_proxy/stubs/`.

---

## Verification Command

Quick verification after fixes:
```bash
source ~/miniconda3/bin/activate mindnlp && python tests/run_test_v2.py -vs tests/transformers/tests/models/{MODEL}/test_modeling_{MODEL}.py::*Test::test_model 2>&1 | tail -20
```

Replace `{MODEL}` with the model name being tested.
