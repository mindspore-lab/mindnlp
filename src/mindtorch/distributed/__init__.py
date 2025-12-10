# mypy: allow-untyped-defs
import logging
import pdb
import sys
import traceback
import typing

import mindtorch


log = logging.getLogger(__name__)


def is_available() -> bool:
    """
    Return ``True`` if the distributed package is available.

    Otherwise,
    ``mindtorch.distributed`` does not expose any other APIs. Currently,
    ``mindtorch.distributed`` is available on Linux, MacOS and Windows. Set
    ``USE_DISTRIBUTED=1`` to enable it when building PyTorch from source.
    Currently, the default value is ``USE_DISTRIBUTED=1`` for Linux and Windows,
    ``USE_DISTRIBUTED=0`` for MacOS.
    """
    return True

# Custom Runtime Errors thrown from the distributed package
DistError = RuntimeError
DistBackendError = RuntimeError
DistNetworkError = RuntimeError
DistStoreError = RuntimeError

if is_available():
    from .c10d import (
        # _broadcast_coalesced,
        # _compute_bucket_assignment_by_size,
        # _ControlCollectives,
        # _DEFAULT_FIRST_BUCKET_BYTES,
        # _make_nccl_premul_sum,
        # _register_builtin_comm_hook,
        # _register_comm_hook,
        # _StoreCollectives,
        # _test_python_store,
        # _verify_params_across_processes,
        # Backend as _Backend,
        # BuiltinCommHookType,
        # DebugLevel,
        # FileStore,
        # get_debug_level,
        # GradBucket,
        # Logger,
        PrefixStore,
        ProcessGroup as ProcessGroup,
        # Reducer,
        # set_debug_level,
        # set_debug_level_from_env,
        Store,
        TCPStore,
        Work as _Work,
    )


    from .device_mesh import DeviceMesh, init_device_mesh

    # Variables prefixed with underscore are not auto imported
    # See the comment in `distributed_c10d.py` above `_backend` on why we expose
    # this.
    from .distributed_c10d import *  # noqa: F403
    from .distributed_c10d import (
        _all_gather_base,
        _coalescing_manager,
        _CoalescingManager,
        _create_process_group_wrapper,
        _get_process_group_name,
        _rank_not_in_group,
        _reduce_scatter_base,
        get_node_local_rank,
    )
    from .remote_device import _remote_device
    # from .rendezvous import (
    #     _create_store_from_options,
    #     register_rendezvous_handler,
    #     rendezvous,
    # )

    # set_debug_level_from_env()

else:
    # This stub is sufficient to get
    #   python test/test_public_bindings.py -k test_correct_module_names
    # working even when USE_DISTRIBUTED=0.  Feel free to add more
    # stubs as necessary.
    # We cannot define stubs directly because they confuse pyre

    class _ProcessGroupStub:
        pass

    sys.modules["mindtorch.distributed"].ProcessGroup = _ProcessGroupStub  # type: ignore[attr-defined]