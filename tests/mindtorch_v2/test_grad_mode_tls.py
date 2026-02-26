import threading

import mindtorch_v2 as torch


def test_no_grad_is_thread_local():
    parent_states = []
    child_states = []

    with torch.no_grad():
        parent_states.append(torch.is_grad_enabled())

        def _worker():
            child_states.append(torch.is_grad_enabled())
            with torch.no_grad():
                child_states.append(torch.is_grad_enabled())
            child_states.append(torch.is_grad_enabled())

        t = threading.Thread(target=_worker)
        t.start()
        t.join()

        parent_states.append(torch.is_grad_enabled())

    assert parent_states == [False, False]
    assert child_states == [True, False, True]


def test_set_grad_enabled_is_thread_local():
    observed = []

    def _worker():
        observed.append(torch.is_grad_enabled())
        with torch.set_grad_enabled(False):
            observed.append(torch.is_grad_enabled())
        observed.append(torch.is_grad_enabled())

    with torch.set_grad_enabled(False):
        assert torch.is_grad_enabled() is False
        t = threading.Thread(target=_worker)
        t.start()
        t.join()
        assert torch.is_grad_enabled() is False

    assert observed == [True, False, True]
