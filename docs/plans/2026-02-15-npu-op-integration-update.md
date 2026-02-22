# NPU ACLNN 算子接入总结（补充版）

目标: 只在 MindTorch v2 内新增 NPU 算子，避免任何 MindSpore 依赖；NPU 路径禁止使用 NumPy；行为完全对齐 PyTorch。

## 接入步骤（通用流程）
1. 确认算子在 CPU 已有实现，行为对齐 PyTorch。
2. 在 `src/mindtorch_v2/_backends/npu/aclnn.py` 添加 ACLNN ctypes 绑定:
   - 使用 `_unary_call`/`_binary_call` 等通用包装。
   - 输出 dtype 与 PyTorch 对齐（如 `isfinite`/`isinf`/`isnan`/`signbit` 为 `bool`）。
3. 在 `src/mindtorch_v2/_backends/npu/ops.py` 添加 NPU 前端封装:
   - 统一检查 NPU 设备与 dtype。
   - 通过 `npu_runtime._alloc_device` 分配输出，使用 `npu_typed_storage_from_ptr` 包装。
   - 避免 NumPy，所有执行仅走 ACLNN/已有 NPU 算子。
4. 在 `tests/mindtorch_v2/test_ops_npu.py` 添加覆盖测试:
   - 先写测试，再实现（TDD）。
   - 使用 CPU/NPU 对比，确保语义一致。
5. 运行目标测试文件，确认通过后再继续扩展。

## 已遇到的特殊情况与处理
- `aclnnIsInf` 不可用（错误码 `161001`）:
  - 退化实现: 使用 `isfinite` + `recip` 的逻辑组合。
  - 规则: `isinf = (~isfinite(x)) & isfinite(1/x)`。
- `aclnnIsPosInf`/`aclnnIsNegInf` 不可用（错误码 `161002`）:
  - 暂不直接使用，依赖 `isinf` 组合逻辑。
- `aclnnEqScalar` 对 `+/-inf` 返回异常（全 False）:
  - 不用于 inf 检测。
- `hardtanh` 某些环境 ACLNN 不支持（错误码 `561103`）:
  - 退化实现: 使用 `clamp(x, min, max)`。
- `isfinite`/`isinf`/`isnan`/`signbit` 输出必须为 `bool`:
  - 输出分配和 `npu_typed_storage_from_ptr` 必须使用 `bool` dtype。

## 性能与安全约束
- 禁止在 NPU 执行路径引入 NumPy。
- 避免不必要的 `try/except`，仅在 ACLNN 不可用时做明确降级。
- 禁止新增 MindSpore 依赖。

## 本轮新增的算子（示例）
- `clamp`（scalar/tensor）
- `cosh`/`sinh`/`erf`/`erfc`/`softplus`
- `relu6`/`hardtanh`
- `isfinite`/`isinf`/`isnan`/`signbit`
- `logical_not`/`logical_and`/`logical_or`/`logical_xor`
- `eq`/`ne`（tensor + scalar）

