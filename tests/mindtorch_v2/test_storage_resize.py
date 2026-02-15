import pytest
import mindtorch_v2 as torch


def test_typed_storage_resize_cpu_elements():
    t = torch.tensor([1.0, 2.0, 3.0])
    storage = t.storage()
    storage.resize_(5)
    assert storage.size() == 5
    assert storage.data[:3].tolist() == [1.0, 2.0, 3.0]


def test_untyped_storage_resize_cpu_bytes():
    t = torch.tensor([1.0, 2.0])
    untyped = t.storage().untyped_storage()
    old_bytes = bytes(untyped.buffer())
    untyped.resize_(len(old_bytes) + 4)
    new_bytes = bytes(untyped.buffer()[:len(old_bytes)])
    assert new_bytes == old_bytes


def test_meta_storage_resize_changes_size():
    storage = torch.Tensor(
        torch._storage.meta_typed_storage_from_shape((2, 3), torch.float32, device="meta"),
        (2, 3),
        (3, 1),
    ).storage()
    storage.resize_(10)
    assert storage.size() == 10


def test_pinned_storage_resize_raises():
    try:
        storage = torch._storage.pinned_cpu_typed_storage_from_numpy(
            torch.tensor([1.0, 2.0]).numpy(),
            torch.float32,
        )
    except RuntimeError:
        pytest.skip("Pinned memory not available")
    with pytest.raises(NotImplementedError):
        storage.resize_(8)


def test_file_backed_storage_resize_raises(tmp_path):
    path = tmp_path / "storage.bin"
    path.write_bytes(b"\x00" * 8)
    untyped = torch.UntypedStorage.from_file(str(path))
    with pytest.raises(RuntimeError, match="Trying to resize storage that is not resizable"):
        untyped.resize_(16)


def test_shared_storage_resize_raises():
    t = torch.tensor([1.0, 2.0])
    untyped = t.storage().untyped_storage().share_memory_()
    with pytest.raises(RuntimeError, match="Trying to resize storage that is not resizable"):
        untyped.resize_(16)


def test_npu_storage_resize_prefix_preserved():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    t = torch.ones((4,), device="npu")
    storage = t.storage()
    storage.resize_(8)
    out = torch.Tensor(storage, (8,), (1,)).to("cpu").numpy().tolist()
    assert out[:4] == [1.0, 1.0, 1.0, 1.0]
