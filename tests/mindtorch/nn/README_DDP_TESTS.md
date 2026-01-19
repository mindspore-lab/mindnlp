# DistributedDataParallel (DDP) Tests

This directory contains tests for the `DistributedDataParallel` module in mindtorch.

## Running the Tests

Since DDP requires multiprocess execution, these tests must be run using `mpirun` or similar multiprocess launcher.

### Using mpirun

```bash
# Run with 2 processes
mpirun -n 2 python tests/mindtorch/nn/test_distributed.py

# Run with 4 processes
mpirun -n 4 python tests/mindtorch/nn/test_distributed.py

# Run specific test
mpirun -n 2 python -m pytest tests/mindtorch/nn/test_distributed.py::test_ddp_basic -v
```

### Using pytest with mpirun

```bash
# Run all DDP tests
mpirun -n 2 python -m pytest tests/mindtorch/nn/test_distributed.py -v

# Run with more verbose output
mpirun -n 2 python -m pytest tests/mindtorch/nn/test_distributed.py -v -s
```

### Environment Setup

Make sure you have:
1. MindSpore distributed environment properly configured
2. HCCL or other distributed backend available
3. Sufficient devices/processes for the test

### Test Cases

1. **test_ddp_basic**: Basic DDP functionality test
   - Creates a simple model
   - Wraps with DDP
   - Runs forward and backward
   - Verifies gradients are computed

2. **test_ddp_parameter_sync**: Parameter synchronization test
   - Tests that DDP synchronizes parameters across ranks
   - Verifies parameter consistency

3. **test_ddp_gradient_sync**: Gradient synchronization test
   - Tests that DDP synchronizes gradients across ranks
   - Verifies gradient averaging works correctly

### Error Handling

The tests are designed to catch and report errors:
- Each test function returns `True` on success, `False` on failure
- Errors are printed with full traceback
- The main function exits with code 1 if any test fails

### Notes

- These tests require a distributed environment (HCCL for Ascend, NCCL for GPU, etc.)
- The tests will automatically initialize the distributed process group if not already initialized
- Cleanup is performed automatically after each test

