# mypy: allow-untyped-defs

from mindnlp import core


def is_available():
    return hasattr(core._C, "_faulty_agent_init")


if is_available() and not core._C._faulty_agent_init():
    raise RuntimeError("Failed to initialize core.distributed.rpc._testing")

if is_available():
    # Registers FAULTY_TENSORPIPE RPC backend.
    from core._C._distributed_rpc_testing import (
        FaultyTensorPipeAgent,
        FaultyTensorPipeRpcBackendOptions,
    )

    from . import faulty_agent_backend_registry
