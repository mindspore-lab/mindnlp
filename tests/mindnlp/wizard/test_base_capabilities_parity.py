import os
import tempfile

import mindspore
import numpy as np
import pytest
import safetensors.numpy
import torch

from mindnlp.wizard.merge.io.loader import TensorLoader
from mindnlp.wizard.merge.io.lazy_tensor_loader import LazyTensorLoader
from mindnlp.wizard.merge.io.lazy_unpickle import DeferredLoad
from mindnlp.wizard.merge.io.tasks import LoaderCache
from mindnlp.wizard.merge.plan import MergePlanner
from mindnlp.wizard.merge.config import ConfigReader, InputModelDefinition, MergeConfiguration
from mindnlp.wizard.merge.options import MergeOptions
from mindnlp.wizard.merge.common import ModelReference
from mindnlp.wizard.merge.architecture.base import WeightInfo
from mindnlp.wizard.merge.io.tensor_writer import TensorWriter


class TestTensorWriterParity:
    def test_safetensors_write(self):
        with tempfile.TemporaryDirectory() as d:
            writer = TensorWriter(d, safe_serialization=True)
            writer.save_tensor("steve", mindspore.Tensor(np.random.randn(4).astype(np.float32)))
            writer.finalize()
            assert os.path.exists(os.path.join(d, "model.safetensors"))

    def test_bin_write_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(ValueError, match="Unsupported output_format 'bin'"):
                TensorWriter(d, safe_serialization=False)

    def test_ckpt_write(self):
        with tempfile.TemporaryDirectory() as d:
            writer = TensorWriter(d, output_format="ckpt")
            writer.save_tensor("timothan", mindspore.Tensor(np.random.randn(4).astype(np.float32)))
            writer.finalize()
            assert os.path.exists(os.path.join(d, "mindspore_model.ckpt"))

    def test_duplicate_tensor(self):
        with tempfile.TemporaryDirectory() as d:
            writer = TensorWriter(d, safe_serialization=True)
            jim = mindspore.Tensor(np.random.randn(4).astype(np.float32))
            writer.save_tensor("jim", jim)
            writer.save_tensor("jimbo", jim)
            writer.finalize()
            assert os.path.exists(os.path.join(d, "model.safetensors"))

    def test_async_writer(self):
        with tempfile.TemporaryDirectory() as d:
            writer = TensorWriter(
                d, safe_serialization=True, use_async=True, max_shard_size=1, max_write_threads=2
            )
            for i in range(4):
                writer.save_tensor(f"t{i + 1}", mindspore.Tensor(np.random.randn(16).astype(np.float32)))
            writer.finalize()
            assert all(
                os.path.exists(
                    os.path.join(d, f"model-{i + 1:05d}-of-00004.safetensors")
                )
                for i in range(4)
            )


class TestLazyUnpickleParity:
    def test_lazy_unpickle(self):
        with tempfile.TemporaryDirectory() as d:
            data = {
                "a": torch.tensor([1, 2, 3]),
                "b": torch.tensor([4, 5, 6]),
            }
            path = os.path.join(d, "pytorch_model.bin")
            torch.save(data, path)

            loader = LazyTensorLoader.from_disk(d)
            for name in data:
                assert name in loader.index.tensor_paths
                tensor = loader.get_tensor(name)
                np.testing.assert_array_equal(
                    tensor.asnumpy(),
                    data[name].numpy(),
                )

    def test_lazy_unpickle_forwards_device_map_location(self, monkeypatch):
        captured = {"map_location": None}
        origin_execute = DeferredLoad.execute

        def _wrapped_execute(self, reader, map_location=None):
            captured["map_location"] = map_location
            return origin_execute(self, reader, map_location=map_location)

        monkeypatch.setattr(DeferredLoad, "execute", _wrapped_execute)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pytorch_model.bin")
            torch.save({"a": torch.tensor([1.0, 2.0])}, path)

            loader = TensorLoader.get(
                path,
                use_lazy_unpickle=True,
                device="CPU",
            )
            tensor = loader.get_tensor("a")
            np.testing.assert_array_equal(
                tensor.asnumpy(),
                np.array([1.0, 2.0], dtype=np.float32),
            )
            assert captured["map_location"] == "CPU"


class _NoAsNumpyTensor:
    """Test helper: ensure size accounting does not call asnumpy()."""

    nbytes = 16

    def asnumpy(self):
        raise AssertionError("asnumpy() should not be called in save_tensor")


class TestTensorWriterCopyPath:
    def test_save_tensor_size_accounting_avoids_asnumpy(self):
        with tempfile.TemporaryDirectory() as d:
            writer = TensorWriter(d, safe_serialization=True)
            writer.save_tensor("x", _NoAsNumpyTensor())
            assert writer.current_shard_size == 16


class TestReadToNpuDevicePath:
    def test_planner_to_loader_device_propagation(self, monkeypatch):
        class _DummyMergeMethod:
            def parameters(self):
                return []

            def tensor_parameters(self):
                return []

            def make_task(self, output_weight, tensors, **kwargs):
                return tensors

        class _DummyLoader:
            def __init__(self):
                self.last_device = None
                self.index = type("Idx", (), {"tensor_paths": {"w": "dummy"}})()

            def get_tensor(self, name, device="CPU", aliases=None, raise_on_missing=True):
                self.last_device = device
                return mindspore.Tensor(np.array([1.0], dtype=np.float32))

        dummy_loader = _DummyLoader()

        # Ensure planner builds GatherTensors with the merge option device.
        import mindnlp.wizard.merge.plan as plan_mod

        monkeypatch.setattr(plan_mod.merge_methods, "get", lambda _name: _DummyMergeMethod())
        monkeypatch.setattr(LoaderCache, "get", lambda self, _model: dummy_loader)

        model = ModelReference.parse("dummy-model")
        cfg = MergeConfiguration(
            merge_method="dummy",
            models=[InputModelDefinition(model=model)],
        )
        options = MergeOptions(read_to_npu=True, device="Ascend")
        planner = MergePlanner(
            config=cfg,
            arch_info=object(),
            options=options,
            out_model_config=object(),
        )

        planner.plan_tensor(
            weight=WeightInfo(name="w"),
            weights_in=[WeightInfo(name="w")],
            models=[model],
            cfg_reader=ConfigReader(config=cfg, t=0),
        )

        gather = planner._tensors[0][1]
        args = gather.arguments()
        loaded = {k: task.execute() for k, task in args.items()}
        res = gather.execute(**loaded)

        assert model in res
        assert dummy_loader.last_device == "Ascend"


class TestLazyTensorLoaderDevicePath:
    def test_safetensors_path_propagates_device(self, monkeypatch):
        import mindnlp.wizard.merge.io._device as device_mod
        import mindnlp.wizard.merge.io.loader as loader_mod

        captured = {"device": None}
        origin_move = device_mod.move_tensor_to_device

        def _wrapped_move(tensor, device, **kwargs):
            captured["device"] = device
            return origin_move(tensor, device, **kwargs)

        monkeypatch.setattr(loader_mod, "move_tensor_to_device", _wrapped_move)

        with tempfile.TemporaryDirectory() as d:
            st_path = os.path.join(d, "model.safetensors")
            safetensors.numpy.save_file(
                {"a": np.array([1.0, 2.0], dtype=np.float32)},
                st_path,
                metadata={"format": "np"},
            )

            loader = LazyTensorLoader.from_disk(d, lazy_loader=False)
            tensor = loader.get_tensor("a", device="Ascend:0")
            np.testing.assert_array_equal(
                tensor.asnumpy(),
                np.array([1.0, 2.0], dtype=np.float32),
            )
            assert captured["device"] == "Ascend:0"

    def test_bin_lazy_path_propagates_map_location(self, monkeypatch):
        captured = {"map_location": None}
        origin_execute = DeferredLoad.execute

        def _wrapped_execute(self, reader, map_location=None):
            captured["map_location"] = map_location
            return origin_execute(self, reader, map_location=map_location)

        monkeypatch.setattr(DeferredLoad, "execute", _wrapped_execute)

        with tempfile.TemporaryDirectory() as d:
            bin_path = os.path.join(d, "pytorch_model.bin")
            torch.save({"a": torch.tensor([3.0, 4.0])}, bin_path)

            loader = LazyTensorLoader.from_disk(d, lazy_loader=True)
            tensor = loader.get_tensor("a", device="Ascend:0")
            np.testing.assert_array_equal(
                tensor.asnumpy(),
                np.array([3.0, 4.0], dtype=np.float32),
            )
            assert captured["map_location"] == "Ascend:0"
