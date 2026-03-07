from datetime import timedelta

import pytest

import mindtorch_v2.distributed as dist


def test_monitored_barrier_rejects_wait_all_ranks_for_hccl(monkeypatch) -> None:
    class _FakePG:
        def size(self):
            return 2

        def rank(self):
            return 0

        def barrier(self):
            class _W:
                def wait(self):
                    return True

            return _W()

    fake_pg = _FakePG()
    monkeypatch.setattr(dist, "_default_pg", fake_pg)
    monkeypatch.setitem(dist._pg_map, fake_pg, (dist.Backend("hccl"), object()))

    with pytest.raises(NotImplementedError, match="wait_all_ranks"):
        dist.monitored_barrier(group=fake_pg, wait_all_ranks=True)


def test_monitored_barrier_passes_timeout_to_store_wait(monkeypatch) -> None:
    class _FakeStore:
        def __init__(self):
            self.wait_timeout = None

        def set(self, key, value):
            return None

        def wait(self, keys, timeout=None):
            self.wait_timeout = timeout
            return None

    class _FakePG:
        def size(self):
            return 2

        def rank(self):
            return 1

        def barrier(self):
            class _W:
                def wait(self):
                    return True

            return _W()

    fake_pg = _FakePG()
    fake_store = _FakeStore()
    monkeypatch.setattr(dist, "_default_pg", fake_pg)
    monkeypatch.setitem(dist._pg_map, fake_pg, (dist.Backend("gloo"), fake_store))

    dist.monitored_barrier(group=fake_pg, timeout=timedelta(seconds=7))
    assert fake_store.wait_timeout == 7

