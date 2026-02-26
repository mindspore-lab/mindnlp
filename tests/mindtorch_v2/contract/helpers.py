import torch as pt


def assert_torch_error(fn_mt, fn_torch):
    try:
        fn_torch()
    except Exception as exc_torch:
        torch_exc = exc_torch
    else:
        torch_exc = None

    try:
        fn_mt()
    except Exception as exc_mt:
        mt_exc = exc_mt
    else:
        mt_exc = None

    assert type(mt_exc) is type(torch_exc)
    assert str(mt_exc) == str(torch_exc)
