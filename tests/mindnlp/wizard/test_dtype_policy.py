import mindspore

from mindnlp.wizard.merge.dtype_policy import (
    cast_back,
    cast_to_work,
    choose_work_dtype,
    needs_safe_path,
)


def test_choose_work_dtype_promotes_half_on_cpu(monkeypatch):
    monkeypatch.setattr(mindspore, "get_context", lambda *_args, **_kwargs: "CPU")
    assert choose_work_dtype(mindspore.bfloat16) == mindspore.float32
    assert choose_work_dtype(mindspore.float16) == mindspore.float32


def test_choose_work_dtype_keeps_fp32_on_cpu(monkeypatch):
    monkeypatch.setattr(mindspore, "get_context", lambda *_args, **_kwargs: "CPU")
    assert choose_work_dtype(mindspore.float32) == mindspore.float32
    assert needs_safe_path(mindspore.float32) is False


def test_choose_work_dtype_keeps_half_on_ascend(monkeypatch):
    monkeypatch.setattr(mindspore, "get_context", lambda *_args, **_kwargs: "Ascend")
    assert choose_work_dtype(mindspore.bfloat16) == mindspore.bfloat16
    assert choose_work_dtype(mindspore.float16) == mindspore.float16


def test_cast_roundtrip():
    tensor = mindspore.Tensor([1.0, 2.0], dtype=mindspore.float16)
    work = cast_to_work(tensor, mindspore.float32)
    assert work.dtype == mindspore.float32
    out = cast_back(work, mindspore.float16)
    assert out.dtype == mindspore.float16

