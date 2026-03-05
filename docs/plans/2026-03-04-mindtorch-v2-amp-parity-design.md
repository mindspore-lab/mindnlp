# MindTorch v2 AMP Parity Design (Torch+CUDA baseline)

## Context
MindTorch v2 has a functional AMP stack but is not yet aligned to the current torch+cuda behavior. This plan targets parity with the locally installed torch version, using torch_npu only as reference. The focus is on API surface alignment, autocast semantics, GradScaler behavior, and custom op autocast registration. Missing backend functionality should be represented by placeholders that preserve public APIs and enable future implementation without breaking compatibility.

## Goals
- Match torch.amp autocast context manager semantics, state APIs, and cache behavior.
- Align GradScaler state machine, error handling, and optimizer contracts with torch.
- Provide torch.library.register_autocast equivalent for custom ops.
- Preserve dispatch integration with the autocast policy lists from torch.

## Non-Goals
- Implement CUDA kernels or full hardware performance parity.
- Implement full torch_function mode tracing or FX export internals beyond API stubs.

## Autocast Runtime and State
- Autocast context manager will mirror torch.amp.autocast: validation of device_type, dtype selection via get_autocast_dtype when dtype is None, and cache_enabled defaulting to current cache state.
- Enter/exit semantics will track previous enabled/dtype/cache values, increment nesting on enter, decrement on exit, and clear cache when nesting drops to zero.
- Add internal `_enter_autocast` and `_exit_autocast` functions to match torch export/FX usage. If torch_function modes are not implemented, provide no-op placeholders that preserve API shape and leave clear TODOs.
- State APIs will align with torch signatures: `is_autocast_enabled()` with and without device_type, `set_autocast_enabled`, `get_autocast_dtype`, `set_autocast_dtype`, `is_autocast_cache_enabled`, `set_autocast_cache_enabled`, `clear_autocast_cache`, and nesting helpers.

## GradScaler Parity
- Implement torch-like GradScaler behavior in Python: lazy init of scale and growth tracker, per-optimizer states with OptState transitions, sparse grad handling, and found_inf tracking per device.
- Honor optimizer contracts such as `_step_supports_amp_scaling` and warnings for closure usage, matching torch error patterns where tests require exact strings.
- Provide API parity for `get_scale`, `state_dict`, `load_state_dict`, `is_enabled`, and pickling constraints. If backend helpers are missing, add placeholders that preserve API shape.

## Custom Op Autocast Registration
- Add a `register_autocast` API mirroring torch.library.register_autocast, supporting `op`, `device_type`, and `cast_inputs` arguments.
- The registration wrapper will cast inputs, disable autocast for the internal call, and then call the underlying op. If dispatch key exclusion is not fully implemented, provide a placeholder with a TODO.

## Testing and Verification
- Extend contract tests for autocast context, cache behavior, `_enter_autocast`/`_exit_autocast`, and GradScaler stage errors.
- Add coverage for `register_autocast` API signature and basic input casting behavior.
- Keep existing AMP policy coverage tests aligned to torch testing lists.

## Placeholders and Future Work
- TorchFunction mode and export integration should be stubbed with clear TODOs for future tracing support.
- Device-specific dtype validation beyond cpu/cuda/npu can be added later when those backends are implemented.

