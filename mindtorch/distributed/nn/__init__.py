import mindtorch

from .functional import *  # noqa: F403


if mindtorch.distributed.rpc.is_available():
    from .api.remote_module import RemoteModule
