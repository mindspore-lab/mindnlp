import pytest

import multiprocessing as py_mp

import mindtorch_v2 as torch


def _identity_collate(sample):
    return sample


def test_mindtorch_multiprocessing_module_exists_and_exports_context_api():
    import mindtorch_v2.multiprocessing as mt_mp

    assert hasattr(mt_mp, "get_context")
    assert hasattr(mt_mp, "Queue")
    assert hasattr(mt_mp, "Process")
    assert hasattr(mt_mp, "Event")


def test_mindtorch_multiprocessing_start_methods_match_python():
    import mindtorch_v2.multiprocessing as mt_mp

    assert set(mt_mp.get_all_start_methods()) == set(py_mp.get_all_start_methods())


def test_dataloader_default_context_uses_mindtorch_multiprocessing_module():
    from mindtorch_v2.utils.data import DataLoader

    loader = DataLoader([0, 1, 2, 3], num_workers=1)
    mp_ctx = loader._mp_context
    assert mp_ctx.__class__.__module__.startswith("multiprocessing.context")




def test_non_leaf_requires_grad_tensor_queue_transfer_rejected():
    from multiprocessing.reduction import ForkingPickler

    base = torch.tensor([1.0], requires_grad=True)
    non_leaf = base + 1.0

    with pytest.raises(RuntimeError, match='non-leaf tensor which requires_grad'):
        ForkingPickler.dumps(non_leaf)



def test_multiprocessing_sharing_strategy_switch_and_restore():
    import mindtorch_v2.multiprocessing as mt_mp

    original = mt_mp.get_sharing_strategy()
    all_strategies = mt_mp.get_all_sharing_strategies()

    if 'file_system' in all_strategies:
        mt_mp.set_sharing_strategy('file_system')
        assert mt_mp.get_sharing_strategy() == 'file_system'

    if 'file_descriptor' in all_strategies:
        mt_mp.set_sharing_strategy('file_descriptor')
        assert mt_mp.get_sharing_strategy() == 'file_descriptor'

    mt_mp.set_sharing_strategy(original)
    assert mt_mp.get_sharing_strategy() == original


def test_dataloader_sharing_strategy_file_system_shared_contract():
    import mindtorch_v2.multiprocessing as mt_mp
    from mindtorch_v2.utils.data import DataLoader, Dataset

    class TensorDataset(Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, idx):
            return torch.tensor([float(idx), float(idx + 1)], dtype=torch.float32)

    original = mt_mp.get_sharing_strategy()
    if 'file_system' not in mt_mp.get_all_sharing_strategies():
        return

    mt_mp.set_sharing_strategy('file_system')
    loader = DataLoader(TensorDataset(), batch_size=2, num_workers=2)
    batch = next(iter(loader))
    assert batch.storage().is_shared() is True
    mt_mp.set_sharing_strategy(original)


def test_dataloader_sharing_strategy_file_descriptor_shared_contract():
    import mindtorch_v2.multiprocessing as mt_mp
    from mindtorch_v2.utils.data import DataLoader, Dataset

    class TensorDataset(Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, idx):
            return torch.tensor([float(idx), float(idx + 1)], dtype=torch.float32)

    original = mt_mp.get_sharing_strategy()
    if 'file_descriptor' not in mt_mp.get_all_sharing_strategies():
        return

    mt_mp.set_sharing_strategy('file_descriptor')
    loader = DataLoader(TensorDataset(), batch_size=2, num_workers=2)
    batch = next(iter(loader))
    assert batch.storage().is_shared() is True
    mt_mp.set_sharing_strategy(original)


def test_multiprocessing_file_system_cleanup_contract():
    import os

    import mindtorch_v2.multiprocessing as mt_mp

    original = mt_mp.get_sharing_strategy()
    if "file_system" not in mt_mp.get_all_sharing_strategies():
        return

    mt_mp.set_sharing_strategy("file_system")
    try:
        storage = torch.tensor([1.0, 2.0], dtype=torch.float32).storage().untyped_storage()
        storage.share_memory_(strategy="file_system")
        filename = storage.filename()
        assert filename is not None
        assert os.path.exists(filename)
        assert mt_mp.shared_files_count() >= 1

        mt_mp.cleanup_shared_files()

        assert os.path.exists(filename) is False
        assert mt_mp.shared_files_count() == 0
    finally:
        mt_mp.set_sharing_strategy(original)


def test_dataloader_sharing_strategy_file_system_repeated_creation_cleanup_stable():
    import gc

    import mindtorch_v2.multiprocessing as mt_mp
    from mindtorch_v2.utils.data import DataLoader, Dataset

    class TensorDataset(Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, idx):
            return torch.tensor([float(idx), float(idx + 1)], dtype=torch.float32)

    original = mt_mp.get_sharing_strategy()
    if "file_system" not in mt_mp.get_all_sharing_strategies():
        return

    mt_mp.set_sharing_strategy("file_system")
    try:
        for _ in range(5):
            loader = DataLoader(TensorDataset(), batch_size=2, num_workers=2)
            batch = next(iter(loader))
            assert batch.storage().is_shared() is True

        assert mt_mp.shared_files_count() >= 1
        gc.collect()
        mt_mp.cleanup_shared_files()
        assert mt_mp.shared_files_count() == 0
    finally:
        mt_mp.set_sharing_strategy(original)
