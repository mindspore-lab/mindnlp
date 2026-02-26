import threading

import mindtorch_v2 as torch
from mindtorch_v2._dispatch.functionalize import is_functionalize_enabled
from mindtorch_v2._dispatch.pipeline import current_pipeline


def test_pipeline_context_is_thread_local():
    seen = {}

    with torch.pipeline():
        assert current_pipeline() is not None

        def worker():
            seen["pipe"] = current_pipeline()

        t = threading.Thread(target=worker)
        t.start()
        t.join()

    assert seen["pipe"] is None


def test_functionalize_context_is_thread_local():
    seen = {}

    with torch.functionalize():
        assert is_functionalize_enabled() is True

        def worker():
            seen["enabled"] = is_functionalize_enabled()

        t = threading.Thread(target=worker)
        t.start()
        t.join()

    assert seen["enabled"] is False
