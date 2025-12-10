# mypy: allow-untyped-defs

import mindtorch


def is_available():
    return hasattr(mindtorch._C, "_faulty_agent_init")


if is_available() and not mindtorch._C._faulty_agent_init():
    raise RuntimeError("Failed to initialize mindtorch.distributed.rpc._testing")

if is_available():
    # Registers FAULTY_TENSORPIPE RPC backend.
    from mindtorch._C._distributed_rpc_testing import (
        FaultyTensorPipeAgent,
        FaultyTensorPipeRpcBackendOptions,
    )

    from . import faulty_agent_backend_registry
