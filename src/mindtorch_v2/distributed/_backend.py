import os


class Backend(str):
    UNDEFINED = "undefined"
    GLOO = "gloo"
    NCCL = "nccl"
    MPI = "mpi"
    UCC = "ucc"
    HCCL = "hccl"

    backend_list = [UNDEFINED, GLOO, NCCL, MPI, UCC, HCCL]

    default_device_backend_map = {
        "cpu": GLOO,
        "cuda": NCCL,
        "npu": HCCL,
    }

    backend_capability = {
        GLOO: ["cpu", "cuda"],
        NCCL: ["cuda"],
        HCCL: ["npu"],
    }

    _plugins = {}

    def __new__(cls, name):
        return super().__new__(cls, name.lower())

    @classmethod
    def register_backend(cls, name, func, extended_api=False, devices=None):
        name = name.lower()
        cls._plugins[name] = func
        if name not in cls.backend_list:
            cls.backend_list.append(name)
        if devices:
            if isinstance(devices, str):
                devices = [devices]
            cls.backend_capability[name] = devices


class GroupMember:
    WORLD = None  # set to default PG after init_process_group
    NON_GROUP_MEMBER = object()


class Store:
    """Base class for distributed stores (API compatibility)."""
    def set(self, key, value): raise NotImplementedError
    def get(self, key): raise NotImplementedError
    def wait(self, keys, timeout=None): raise NotImplementedError


class PrefixStore(Store):
    def __init__(self, prefix, store):
        self._prefix = prefix
        self._store = store

    def set(self, key, value):
        self._store.set(f"{self._prefix}/{key}", value)

    def get(self, key):
        return self._store.get(f"{self._prefix}/{key}")

    def wait(self, keys, timeout=None):
        self._store.wait([f"{self._prefix}/{k}" for k in keys], timeout)


def is_nccl_available():
    return False


def is_gloo_available():
    return True


def is_mpi_available():
    return False


def is_ucc_available():
    return False


def is_hccl_available():
    try:
        from ._hccl.hccl_loader import ensure_hccl
        ensure_hccl()
        return True
    except Exception:
        return False


def is_backend_available(backend):
    backend = backend.lower()
    checks = {
        "nccl": is_nccl_available,
        "gloo": is_gloo_available,
        "mpi": is_mpi_available,
        "ucc": is_ucc_available,
        "hccl": is_hccl_available,
    }
    fn = checks.get(backend)
    return fn() if fn else backend in Backend._plugins


def is_torchelastic_launched():
    return "TORCHELASTIC_RUN_ID" in os.environ or "ELASTIC_RUN_ID" in os.environ


def get_default_backend_for_device(device_type):
    return Backend.default_device_backend_map.get(device_type, Backend.UNDEFINED)
