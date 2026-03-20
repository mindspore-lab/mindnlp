# Wizard Merge 测试套件

## 测试文件

| 文件 | 说明 |
|------|------|
| `test_merge.py` | 核心合并逻辑（Task DAG、Executor、MergeConfiguration） |
| `test_ckpt_io.py` | .ckpt 读写与 bf16 精度安全 |
| `test_bf16_ckpt_safety.py` | BF16 字节级精度验证 |
| `test_bf16_cpu_full_method_matrix.py` | BF16 下全方法矩阵测试 |
| `test_merge_dtype_matrix.py` | 合并 dtype 组合矩阵 |
| `test_dtype_policy.py` | dtype 策略单元测试 |
| `test_safe_ops.py` | 安全张量运算 |
| `test_preflight.py` | 合并前校验 |
| `test_config_validation_matrix.py` | 配置校验矩阵 |
| `test_cli_run_yaml_compat.py` | CLI run_yaml 兼容性 |
| `test_mergekit_recipe_compat.py` | MergeKit 配方兼容性 |
| `test_basic_merges_parity.py` | 基础合并一致性 |
| `test_base_capabilities_parity.py` | 基础能力一致性 |
| `test_vlm_merges_parity.py` | VLM 合并一致性 |
| `test_mixed_weight_format_compat.py` | 混合权重格式兼容 |
| `test_tokenizer_mergekit_parity.py` | Tokenizer 合并一致性 |

## 运行测试

```bash
# 在项目根目录执行

# 运行全部 Wizard 测试
pytest tests/mindnlp/wizard/ -v

# 运行单个文件
pytest tests/mindnlp/wizard/test_ckpt_io.py -v

# 仅运行快速测试（排除需要真实模型的测试）
pytest tests/mindnlp/wizard/ -v -k "not parity"

# 查看覆盖率
pytest tests/mindnlp/wizard/ --cov=mindnlp.wizard --cov-report=html
```

## 环境说明

- 测试通过 `conftest.py` 自动配置 `sys.path`，无需完整安装 mindnlp
- `conftest.py` 会 stub `torch_npu` 以避免 Ascend 环境依赖
- 部分一致性测试（`*_parity.py`）需要网络访问以下载模型权重

## 预期结果

```
tests/mindnlp/wizard/test_merge.py           ............  PASSED
tests/mindnlp/wizard/test_ckpt_io.py         ........      PASSED
tests/mindnlp/wizard/test_bf16_ckpt_safety.py  ......      PASSED
tests/mindnlp/wizard/test_safe_ops.py        ....          PASSED
tests/mindnlp/wizard/test_preflight.py       ......        PASSED
tests/mindnlp/wizard/test_dtype_policy.py    ........      PASSED
...
```
