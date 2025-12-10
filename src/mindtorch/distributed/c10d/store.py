from typing import List, Optional, Callable
from abc import ABC, abstractmethod
from datetime import timedelta
try:
    from mindspore.mint.distributed.distributed import TCPStore as MsTCPStore
except:
    MsTCPStore = None

class Store:
    kDefaultTimeout = 300  # in seconds
    kNoTimeout = 0  # No timeout

    def __init__(self, timeout: Optional[int] = kDefaultTimeout):
        self.timeout_ = timeout

    def set(self, key: str, value: str):
        self.set_bytes(key, value.encode())

    @abstractmethod
    def set_bytes(self, key: str, value: List[int]):
        pass

    def compare_set(self, key: str, current_value: str, new_value: str) -> str:
        current_bytes = current_value.encode()
        new_bytes = new_value.encode()
        value = self.compare_set_bytes(key, current_bytes, new_bytes)
        return value.decode()

    @abstractmethod
    def compare_set_bytes(self, key: str, current_value: List[int], new_value: List[int]) -> List[int]:
        pass

    def get_to_str(self, key: str) -> str:
        value = self.get(key)
        return bytes(value).decode()

    @abstractmethod
    def get(self, key: str) -> List[int]:
        pass

    @abstractmethod
    def add(self, key: str, value: int) -> int:
        pass

    @abstractmethod
    def delete_key(self, key: str) -> bool:
        pass

    @abstractmethod
    def check(self, keys: List[str]) -> bool:
        pass

    @abstractmethod
    def get_num_keys(self) -> int:
        pass

    @abstractmethod
    def wait(self, keys: List[str], timeout: Optional[int] = None):
        pass

    def get_timeout(self) -> int:
        return self.timeout_

    def set_timeout(self, timeout: int):
        self.timeout_ = timeout

    def watch_key(self, key: str, callback: Callable[[Optional[str], Optional[str]], None]):
        raise NotImplementedError("watchKey is deprecated, no implementation supports it.")

    def append(self, key: str, value: List[int]):
        expected = value
        current = []
        current = self.compare_set_bytes(key, current, expected)
        while current != expected:
            expected = current + value
            current = self.compare_set_bytes(key, current, expected)

    def multi_get(self, keys: List[str]) -> List[List[int]]:
        result = []
        for key in keys:
            result.append(self.get(key))
        return result

    def multi_set(self, keys: List[str], values: List[List[int]]):
        for i in range(len(keys)):
            self.set_bytes(keys[i], values[i])

    def has_extended_api(self) -> bool:
        return False

class StoreTimeoutGuard:
    def __init__(self, store: Store, timeout: int):
        self.store_ = store
        self.old_timeout_ = store.get_timeout()
        store.set_timeout(timeout)

    def __del__(self):
        self.store_.set_timeout(self.old_timeout_)

    def __copy__(self):
        raise NotImplementedError("Copying not allowed")

    def __move__(self):
        raise NotImplementedError("Moving not allowed")

class TCPStore(Store):
    def __init__(
        self,
        host_name: str,
        port: int,
        world_size: Optional[int] = None,
        is_master: bool = False,
        timeout: timedelta = timedelta(seconds=300),
        wait_for_workers: bool = True,
        multi_tenant: bool = False,
        master_listen_fd: Optional[int] = None,
        use_libuv: bool = True
    ) -> None:
        super().__init__(timeout)
        self.ms_store = MsTCPStore(host_name, port, world_size, is_master, timeout, wait_for_workers, multi_tenant, master_listen_fd, use_libuv)

    @property
    def host(self) -> str:
        return self.ms_store.host

    @property
    def port(self) -> int:
        return self.ms_store.port

    def set(self, key: str, value: str) -> None:
        self.ms_store.set(key, value)

    def add(self, key: str, value: int) -> int:
        return self.ms_store.add(key, value)

    def get(self, key: str) -> bytes:
        return self.ms_store.get(key)

    def delete_key(self, key: str) -> bool:
        return self.ms_store.delete_key(key)

class FileStore(Store):
    def __init__(self, path: str, numWorkers: int = ...): ...
