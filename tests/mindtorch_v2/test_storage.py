import mindtorch_v2 as torch


def test_typed_untyped_basic():
    t = torch.tensor([1.0, 2.0])
    st = t.storage()
    ust = t.untyped_storage()
    assert st.dtype.name == "float32"
    assert st.nbytes() == 8
    assert ust.nbytes() == st.nbytes()


def test_cpu_share_memory():
    t = torch.tensor([1.0, 2.0])
    ust = t.untyped_storage()
    ust.share_memory_()
    assert ust.is_shared() is True


def test_cpu_from_file(tmp_path):
    path = tmp_path / "storage.bin"
    path.write_bytes(b"\x00" * 16)
    ust = torch.UntypedStorage.from_file(str(path), shared=False)
    assert ust.nbytes() == 16
    assert ust.filename() == str(path)


def test_npu_storage_no_cpu_copy():
    if not torch.npu.is_available():
        return
    t = torch.tensor([1.0, 2.0], device="npu")
    st = t.storage()
    assert st.device.type == "npu"
    assert st.untyped_storage().data_ptr() != 0


def test_meta_storage_no_data_ptr():
    t = torch.tensor([1.0, 2.0], device="meta")
    try:
        _ = t.untyped_storage().data_ptr()
    except RuntimeError:
        pass
    else:
        raise AssertionError("meta storage should not expose data_ptr")

def test_pending_storage_basic():
    from mindtorch_v2._dtype import float32
    from mindtorch_v2._storage import PendingStorage

    storage = PendingStorage((2, 3), float32, "cpu")
    assert storage.nbytes() == 2 * 3 * 4
    try:
        storage.data_ptr()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected data_ptr to raise")
