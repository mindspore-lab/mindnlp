from typing import List
from .store import Store

class PrefixStore(Store):
    def __init__(self, prefix: str, store: Store):
        self.prefix_ = prefix
        self.store_ = store

    def join_key(self, key: str) -> str:
        return f"{self.prefix_}/{key}"

    def join_keys(self, keys: List[str]) -> List[str]:
        return [self.join_key(key) for key in keys]

    def set(self, key: str, value: List[int]):
        self.store_.set(self.join_key(key), value)

    def compare_set(self, key: str, expected_value: List[int], desired_value: List[int]) -> List[int]:
        return self.store_.compare_set(self.join_key(key), expected_value, desired_value)

    def get(self, key: str) -> List[int]:
        return self.store_.get(self.join_key(key))

    def add(self, key: str, value: int) -> int:
        return self.store_.add(self.join_key(key), value)

    def delete_key(self, key: str) -> bool:
        return self.store_.delete_key(self.join_key(key))

    def get_num_keys(self) -> int:
        return self.store_.get_num_keys()

    def check(self, keys: List[str]) -> bool:
        return self.store_.check(self.join_keys(keys))

    def wait(self, keys: List[str]):
        self.store_.wait(self.join_keys(keys))

    def wait_with_timeout(self, keys: List[str], timeout: int):
        self.store_.wait(self.join_keys(keys), timeout)

    def get_timeout(self) -> int:
        return self.store_.get_timeout()

    def set_timeout(self, timeout: int):
        self.store_.set_timeout(timeout)

    def append(self, key: str, value: List[int]):
        self.store_.append(self.join_key(key), value)

    def multi_get(self, keys: List[str]) -> List[List[int]]:
        return self.store_.multi_get(self.join_keys(keys))

    def multi_set(self, keys: List[str], values: List[List[int]]):
        self.store_.multi_set(self.join_keys(keys), values)

    def has_extended_api(self) -> bool:
        return self.store_.has_extended_api()

    def get_underlying_store(self) -> Store:
        return self.store_

    def get_underlying_non_prefix_store(self) -> Store:
        store = self.store_
        while isinstance(store, PrefixStore):
            store = store.get_underlying_store()
        if store is None:
            raise ValueError("Underlying Non-PrefixStore shouldn't be null.")
        return store