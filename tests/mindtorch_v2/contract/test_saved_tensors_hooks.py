import pytest

import mindtorch_v2 as torch
from mindtorch_v2._autograd.graph import saved_tensors_hooks


def test_saved_tensors_hooks_pack_unpack_counts():
    counts = {"pack": 0, "unpack": 0}

    def pack(t):
        counts["pack"] += 1
        return t

    def unpack(t):
        counts["unpack"] += 1
        return t

    x = torch.ones((2, 2)).requires_grad_()
    with saved_tensors_hooks(pack, unpack):
        y = x * x
        z = y + y
    z.sum().backward()
    assert counts == {"pack": 2, "unpack": 2}


def test_saved_tensors_hooks_retain_graph_unpack_twice():
    counts = {"pack": 0, "unpack": 0}

    def pack(t):
        counts["pack"] += 1
        return t

    def unpack(t):
        counts["unpack"] += 1
        return t

    x = torch.ones((2, 2)).requires_grad_()
    with saved_tensors_hooks(pack, unpack):
        y = x * x
        z = y + y
    out = z.sum()
    out.backward(retain_graph=True)
    out.backward(retain_graph=True)
    assert counts == {"pack": 2, "unpack": 4}


def test_saved_tensors_hooks_freed_without_retain_graph():
    counts = {"pack": 0, "unpack": 0}

    def pack(t):
        counts["pack"] += 1
        return t

    def unpack(t):
        counts["unpack"] += 1
        return t

    x = torch.ones((2, 2)).requires_grad_()
    with saved_tensors_hooks(pack, unpack):
        y = x * x
        z = y + y
    out = z.sum()
    out.backward()
    with pytest.raises(
        RuntimeError,
        match=(
            r"Trying to backward through the graph a second time .* retain_graph=True"
        ),
    ):
        out.backward()
