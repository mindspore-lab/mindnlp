# Tensor print design (torch-like repr)

## Goals
- Add torch-like `Tensor.__repr__` and `Tensor.__str__` output.
- Match PyTorch default output rules (omit dtype/device when default float32/cpu).
- Support global `set_printoptions` / `get_printoptions` akin to torch.
- NPU tensors print real values by copying to CPU; meta tensors print placeholders.

## Scope
- Python-only formatting.
- No new dependencies.
- No changes to tensor math/dispatch semantics.

## API
- New module: `mindtorch_v2._printing`.
- Public functions: `set_printoptions(**kwargs)`, `get_printoptions()`.
- `Tensor.__repr__` and `Tensor.__str__` call a shared formatter.

## Formatting behavior
- Prefix: `tensor(...)` (lowercase), aligned to torch.
- Data rendering uses numpy `array2string` with per-call options:
  - `precision`, `threshold`, `edgeitems`, `linewidth`, `sci_mode`.
  - Do not mutate global numpy print options.
- Suffix rules:
  - Omit `dtype` if dtype is default float32.
  - Omit `device` if `cpu`.
  - Include `requires_grad=True` if `requires_grad`.
  - Include `grad_fn=<Name>` if present.
- Meta tensors:
  - No data access; render `tensor(..., device='meta', dtype=...)`.
- NPU tensors:
  - Copy to CPU via `to("cpu")` and format data from CPU view.

## Error handling
- Meta tensors never raise on print.
- Pending tensors flush pipeline before formatting; failures propagate.
- NPU copy errors propagate.

## Tests (TDD)
- CPU float32 prints without dtype/device.
- Non-default dtype prints dtype.
- NPU prints device and value.
- Meta prints placeholder + dtype/device.
- `print(t)` equals `repr(t)`.
- `set_printoptions` affects output (precision).

## Files
- `src/mindtorch_v2/_printing.py` (new)
- `src/mindtorch_v2/_tensor.py` (add `__repr__`/`__str__`)
- `src/mindtorch_v2/__init__.py` (export printoptions)
- `tests/mindtorch_v2/test_tensor_print.py` (new)
