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
