from mindnlp import core

from .functional import *  # noqa: F403


if core.distributed.rpc.is_available():
    from .api.remote_module import RemoteModule
