# Copyright 2026 MindSpore Wizard Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Comprehensive unit tests for the Wizard MindSpore merge package."""

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import immutables
import mindspore
import mindspore.ops as ops
import numpy as np
import pytest
import yaml

from mindnlp.wizard.merge.graph import (
    Executor,
    ExecutionSchedule,
    Task,
    TaskHandle,
    TaskUniverse,
    build_schedule,
    _parse_device,
)
from mindnlp.wizard.merge.multigpu_executor import MultiDeviceExecutor
from mindnlp.wizard.merge.common import (
    ImmutableMap,
    ModelPath,
    ModelReference,
    dtype_from_name,
    get_accelerator_count,
    parse_kmb,
)
from mindnlp.wizard.merge.config import (
    ConfigReader,
    ConditionalParameter,
    InputModelDefinition,
    InputSliceDefinition,
    MergeConfiguration,
    OutputSliceDefinition,
    evaluate_setting,
)
from mindnlp.wizard.merge.io.lazy_tensor_loader import ShardedTensorIndex
from mindnlp.wizard.merge.io.tensor_writer import TensorWriter
from mindnlp.wizard.merge.io.tasks import LoadTensor, LoaderCache
from mindnlp.wizard.merge.merge_methods.registry import REGISTERED_MERGE_METHODS
from mindnlp.wizard.merge.merge_methods.slerp import lerp, slerp, normalize
from mindnlp.wizard.merge.sparsify import (
    SparsificationMethod,
    RescaleNorm,
    magnitude,
    bernoulli,
    magnitude_outliers,
    sparsify,
)
from mindnlp.wizard.merge.options import MergeOptions
from mindnlp.wizard.merge.merge import _write_execution_report


# ===================================================================
# Helper task classes for graph tests
# ===================================================================

class ConstantTask(Task[int]):
    value: int

    def arguments(self) -> Dict[str, Task]:
        return {}

    def execute(self, **kwargs) -> int:
        return self.value


class AddTask(Task[int]):
    a: Task[int]
    b: Task[int]

    def arguments(self) -> Dict[str, Task]:
        return {"a": self.a, "b": self.b}

    def execute(self, a: int, b: int, **kwargs) -> int:
        return a + b


class MultiplyTask(Task[int]):
    a: Task[int]
    b: Task[int]

    def arguments(self) -> Dict[str, Task]:
        return {"a": self.a, "b": self.b}

    def execute(self, a: int, b: int, **kwargs) -> int:
        return a * b


class TensorAddTask(Task[mindspore.Tensor]):
    a: Task[mindspore.Tensor]
    b: Task[mindspore.Tensor]

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"a": self.a, "b": self.b}

    def execute(self, a: mindspore.Tensor, b: mindspore.Tensor, **kwargs):
        return a + b


class TensorConstTask(Task[mindspore.Tensor]):
    data: Tuple[float, ...]

    def arguments(self) -> Dict[str, Task]:
        return {}

    def execute(self, **kwargs) -> mindspore.Tensor:
        return mindspore.Tensor(list(self.data), dtype=mindspore.float32)


class LabeledConstTask(Task[int]):
    value: int
    label: str

    def arguments(self) -> Dict[str, Task]:
        return {}

    def execute(self, **kwargs) -> int:
        return self.value

    def group_label(self) -> Optional[str]:
        return self.label


class CostHintTask(Task[int]):
    value: int
    read_cost: float = 0.0
    compute_cost: float = 0.0
    write_cost: float = 0.0

    def arguments(self) -> Dict[str, Task]:
        return {}

    def execute(self, **kwargs) -> int:
        return self.value

    def group_label(self) -> Optional[str]:
        return "same-group"

    def cost_hint(self):
        return {
            "read": self.read_cost,
            "compute": self.compute_cost,
            "write": self.write_cost,
        }


class CostJoinTask(Task[int]):
    left: Task[int]
    right: Task[int]

    def arguments(self) -> Dict[str, Task]:
        return {"left": self.left, "right": self.right}

    def execute(self, left: int, right: int, **kwargs) -> int:
        return left + right


# ===================================================================
# 1. graph.py tests
# ===================================================================

class TestTaskFrozen:

    def test_frozen_and_hashable(self):
        t = ConstantTask(value=42)
        with pytest.raises(Exception):
            t.value = 99

    def test_equal_tasks_share_hash(self):
        t1 = ConstantTask(value=1)
        t2 = ConstantTask(value=1)
        assert t1 == t2 and hash(t1) == hash(t2)


class TestTaskUniverse:

    def test_add_task_returns_handle(self):
        u = TaskUniverse()
        t = ConstantTask(value=5)
        h = u.add_task(t)
        assert isinstance(h, TaskHandle)
        assert h.task() is t

    def test_deduplication(self):
        u = TaskUniverse()
        t = ConstantTask(value=5)
        h1 = u.add_task(t)
        h2 = u.add_task(t)
        assert h1 == h2
        assert len(u.tasks) == 1

    def test_recursive_add(self):
        c1 = ConstantTask(value=2)
        c2 = ConstantTask(value=3)
        add = AddTask(a=c1, b=c2)
        u = TaskUniverse()
        h = u.add_task(add, recursive=True)
        assert len(u.tasks) == 3
        assert h.task() is add

    def test_get_handle(self):
        u = TaskUniverse()
        t = ConstantTask(value=7)
        u.add_task(t)
        assert u.get_handle(t) is not None

    def test_get_handle_missing(self):
        u = TaskUniverse()
        assert u.get_handle(ConstantTask(value=7)) is None


class TestBuildSchedule:

    def test_empty_targets(self):
        sched = build_schedule([], {})
        assert sched.tasks == []

    def test_topological_ordering(self):
        c1 = ConstantTask(value=1)
        c2 = ConstantTask(value=2)
        add = AddTask(a=c1, b=c2)
        u = TaskUniverse()
        h_add = u.add_task(add)
        sched = build_schedule([h_add], {})
        task_types = [type(th.task()).__name__ for th in sched.tasks]
        add_idx = task_types.index("AddTask")
        for dep_type in ("ConstantTask",):
            assert task_types.index(dep_type) < add_idx

    def test_cached_values_skip(self):
        c1 = ConstantTask(value=1)
        c2 = ConstantTask(value=2)
        add = AddTask(a=c1, b=c2)
        u = TaskUniverse()
        h_add = u.add_task(add)
        h_c1 = u.get_handle(c1)
        sched = build_schedule([h_add], {h_c1: 99})
        scheduled_tasks = [th.task() for th in sched.tasks]
        assert c1 not in scheduled_tasks
        assert add in scheduled_tasks


class TestExecutor:

    def test_simple_dag(self):
        c1 = ConstantTask(value=3)
        c2 = ConstantTask(value=4)
        add = AddTask(a=c1, b=c2)
        ex = Executor(targets=[add], math_device="CPU", storage_device="CPU")
        results = dict(ex.run(quiet=True))
        assert results[add] == 7

    def test_diamond_dag(self):
        c = ConstantTask(value=2)
        a1 = AddTask(a=c, b=c)
        a2 = MultiplyTask(a=c, b=c)
        final = AddTask(a=a1, b=a2)
        ex = Executor(targets=[final], math_device="CPU")
        results = dict(ex.run(quiet=True))
        assert results[final] == (2 + 2) + (2 * 2)

    def test_tensor_dag(self):
        t1 = TensorConstTask(data=(1.0, 2.0, 3.0))
        t2 = TensorConstTask(data=(4.0, 5.0, 6.0))
        add = TensorAddTask(a=t1, b=t2)
        ex = Executor(targets=[add], math_device="CPU", storage_device="CPU")
        results = dict(ex.run(quiet=True))
        out = results[add]
        assert isinstance(out, mindspore.Tensor)
        np.testing.assert_allclose(out.asnumpy(), [5.0, 7.0, 9.0])

    def test_cached_values(self):
        c1 = ConstantTask(value=10)
        c2 = ConstantTask(value=20)
        add = AddTask(a=c1, b=c2)
        u = TaskUniverse()
        h_add = u.add_task(add)
        h_c1 = u.get_handle(c1)
        h_c2 = u.get_handle(c2)
        ex = Executor(
            targets=[h_add],
            cached_values={h_c1: 100, h_c2: 200},
        )
        results = dict(ex.run(quiet=True))
        assert results[add] == 300


class TestParseDevice:

    def test_cpu(self):
        target, dev_id = _parse_device("CPU")
        assert target == "CPU" and dev_id is None

    def test_ascend_with_id(self):
        target, dev_id = _parse_device("Ascend:0")
        assert target == "Ascend" and dev_id == 0


# ===================================================================
# 2. common.py tests
# ===================================================================

class TestModelPath:

    def test_from_string_no_revision(self):
        mp = ModelPath.model_validate("my-org/my-model")
        assert mp.path == "my-org/my-model"
        assert mp.revision is None

    def test_from_string_with_revision(self):
        mp = ModelPath.model_validate("my-org/my-model@main")
        assert mp.path == "my-org/my-model"
        assert mp.revision == "main"

    def test_str_roundtrip(self):
        assert str(ModelPath(path="foo/bar")) == "foo/bar"
        assert str(ModelPath(path="foo/bar", revision="dev")) == "foo/bar@dev"

    def test_invalid_multiple_at(self):
        with pytest.raises(RuntimeError, match="multiple @"):
            ModelPath.model_validate("a@b@c")

    def test_frozen(self):
        mp = ModelPath(path="x")
        with pytest.raises(Exception):
            mp.path = "y"


class TestModelReference:

    def test_from_string_model_only(self):
        mr = ModelReference.model_validate("my-org/model")
        assert mr.model.path == "my-org/model"
        assert mr.lora is None

    def test_from_string_model_plus_lora(self):
        mr = ModelReference.model_validate("base-model+lora-adapter")
        assert mr.model.path == "base-model"
        assert mr.lora.path == "lora-adapter"

    def test_str_roundtrip(self):
        assert str(ModelReference(model=ModelPath(path="abc"))) == "abc"
        mr = ModelReference(
            model=ModelPath(path="abc"), lora=ModelPath(path="lora")
        )
        assert str(mr) == "abc+lora"

    def test_parse_classmethod(self):
        mr = ModelReference.parse("org/m@v1+org/l@v2")
        assert mr.model.path == "org/m" and mr.model.revision == "v1"
        assert mr.lora.path == "org/l" and mr.lora.revision == "v2"

    def test_equality(self):
        mr1 = ModelReference.model_validate("model-a")
        mr2 = ModelReference.model_validate("model-a")
        mr3 = ModelReference.model_validate("model-b")
        assert mr1 == mr2
        assert mr1 != mr3


class TestDtypeFromName:

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("float16", mindspore.float16),
            ("float32", mindspore.float32),
            ("bfloat16", mindspore.bfloat16),
            ("float64", mindspore.float64),
            ("int32", mindspore.int32),
        ],
    )
    def test_valid(self, name, expected):
        assert dtype_from_name(name) == expected

    def test_with_prefix(self):
        assert dtype_from_name("torch.float16") == mindspore.float16
        assert dtype_from_name("mindspore.float32") == mindspore.float32
        assert dtype_from_name("ms.bfloat16") == mindspore.bfloat16

    def test_none_returns_none(self):
        assert dtype_from_name(None) is None

    def test_unimplemented(self):
        with pytest.raises(RuntimeError, match="Unimplemented"):
            dtype_from_name("complex128")


class TestImmutableMap:

    def test_basic_operations(self):
        im = ImmutableMap(immutables.Map({"a": 1, "b": 2}))
        assert im["a"] == 1 and im["b"] == 2 and len(im) == 2

    def test_keys_items_values(self):
        im = ImmutableMap(immutables.Map({"x": 10, "y": 20}))
        assert set(im.keys()) == {"x", "y"}
        assert set(im.values()) == {10, 20}
        assert set(im.items()) == {("x", 10), ("y", 20)}

    def test_iteration(self):
        im = ImmutableMap(immutables.Map({"a": 1}))
        assert list(im) == ["a"]


class TestParseKmb:

    @pytest.mark.parametrize("inp,expected", [
        (42, 42), ("100", 100), ("5k", 5000),
        ("2M", 2_000_000), ("1B", 1_000_000_000),
    ])
    def test_values(self, inp, expected):
        assert parse_kmb(inp) == expected


# ===================================================================
# 3. config.py tests
# ===================================================================

class TestEvaluateSetting:

    def test_scalar(self):
        assert evaluate_setting("w", 0.5) == 0.5
        assert evaluate_setting("w", 1) == 1

    def test_gradient_list(self):
        assert abs(evaluate_setting("w", [0.0, 1.0], t=0.0) - 0.0) < 1e-6
        assert abs(evaluate_setting("w", [0.0, 1.0], t=1.0) - 1.0) < 1e-6
        assert abs(evaluate_setting("w", [0.0, 1.0], t=0.5) - 0.5) < 1e-6
        assert abs(evaluate_setting("w", [0.0, 0.5, 1.0], t=0.5) - 0.5) < 1e-6

    def test_conditional_matching(self):
        cond = ConditionalParameter(value=0.7, filter="attn")
        assert evaluate_setting("model.self_attn.q_proj", [cond]) == 0.7

    def test_conditional_wildcard(self):
        cond = ConditionalParameter(value=0.3, filter="*")
        assert evaluate_setting("any.tensor", [cond]) == 0.3

    def test_conditional_no_match(self):
        cond = ConditionalParameter(value=0.7, filter="attn")
        assert evaluate_setting("model.mlp.gate", [cond]) is None

    def test_conditional_none_filter(self):
        cond = ConditionalParameter(value=0.9, filter=None)
        assert evaluate_setting("anything", [cond]) == 0.9


class TestMergeConfiguration:

    def _base_config(self):
        return MergeConfiguration(
            merge_method="linear",
            models=[
                InputModelDefinition(model=ModelReference.parse("model-a")),
                InputModelDefinition(model=ModelReference.parse("model-b")),
            ],
        )

    def test_valid_models_config(self):
        cfg = self._base_config()
        assert cfg.merge_method == "linear"
        assert len(cfg.models) == 2

    def test_exactly_one_of_models_slices_modules(self):
        with pytest.raises(Exception, match="Exactly one"):
            MergeConfiguration(merge_method="linear")

    def test_referenced_models(self):
        cfg = MergeConfiguration(
            merge_method="linear",
            base_model=ModelReference.parse("base"),
            models=[
                InputModelDefinition(model=ModelReference.parse("m1")),
                InputModelDefinition(model=ModelReference.parse("m2")),
            ],
        )
        names = {str(r) for r in cfg.referenced_models()}
        assert {"base", "m1", "m2"} <= names

    def test_to_yaml_roundtrip(self):
        cfg = self._base_config()
        yaml_str = cfg.to_yaml()
        cfg2 = MergeConfiguration.model_validate(yaml.safe_load(yaml_str))
        assert cfg2.merge_method == cfg.merge_method
        assert len(cfg2.models) == len(cfg.models)

    def test_tokenizer_conflict(self):
        with pytest.raises(Exception, match="Cannot specify both"):
            MergeConfiguration(
                merge_method="linear",
                models=[InputModelDefinition(model=ModelReference.parse("a"))],
                tokenizer_source="base", tokenizer={"k": "v"},
            )

    def test_method_requires_base_model(self):
        with pytest.raises(RuntimeError, match="requires base_model"):
            MergeConfiguration(
                merge_method="slerp",
                models=[
                    InputModelDefinition(model=ModelReference.parse("m1")),
                    InputModelDefinition(model=ModelReference.parse("m2")),
                ],
            )



class TestConfigReader:

    def _make_reader(self):
        cfg = MergeConfiguration(
            merge_method="linear",
            models=[
                InputModelDefinition(
                    model=ModelReference.parse("m1"), parameters={"weight": 0.6}
                ),
                InputModelDefinition(
                    model=ModelReference.parse("m2"), parameters={"weight": 0.4}
                ),
            ],
            parameters={"weight": 0.5},
        )
        return ConfigReader(config=cfg, t=0.5)

    def test_global_parameter(self):
        assert self._make_reader().parameter("weight") == 0.5

    def test_parameter_default(self):
        assert self._make_reader().parameter("nonexistent", default=42) == 42

    def test_parameter_required_missing(self):
        with pytest.raises(RuntimeError, match="Missing required"):
            self._make_reader().parameter("missing", required=True)

    def test_for_tensor(self):
        tr = self._make_reader().for_tensor("layer.0.weight")
        assert tr.tensor_name == "layer.0.weight"

    def test_with_t(self):
        assert self._make_reader().with_t(0.9).t == 0.9

    def test_4_level_priority(self):
        m1 = ModelReference.parse("m1")
        cfg = MergeConfiguration(
            merge_method="linear",
            models=[InputModelDefinition(model=m1)],
            parameters={"weight": 0.1},
        )
        slice_def = OutputSliceDefinition(
            sources=[InputSliceDefinition(model=m1, layer_range=(0, 1), parameters={"weight": 0.9})],
            parameters={"weight": 0.5},
        )
        reader = ConfigReader(config=cfg, t=0.0, slice_out=slice_def)
        assert reader.parameter("weight", model=m1) == 0.9  # source-level
        assert reader.parameter("weight") == 0.5  # slice-level

        reader2 = ConfigReader(
            config=cfg, t=0.0,
            slice_out=OutputSliceDefinition(
                sources=[InputSliceDefinition(model=m1, layer_range=(0, 1))],
            ),
        )
        assert reader2.parameter("weight") == 0.1  # falls through to global

    def test_base_model_resolution(self):
        base = ModelReference.parse("base-model")
        cfg = MergeConfiguration(
            merge_method="slerp",
            base_model=base,
            models=[InputModelDefinition(model=ModelReference.parse("m"))],
        )
        assert ConfigReader(config=cfg, t=0.0).base_model == base

        slice_base = ModelReference.parse("slice-base")
        slice_def = OutputSliceDefinition(
            sources=[InputSliceDefinition(model=ModelReference.parse("m"), layer_range=(0, 1))],
            base_model=slice_base,
        )
        reader = ConfigReader(config=cfg, t=0.0, slice_out=slice_def)
        assert reader.base_model == slice_base


# ===================================================================
# 4. IO tests
# ===================================================================

class TestTensorWriterRoundTrip:

    def test_single_shard_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            t1 = mindspore.Tensor(np.random.randn(4, 4).astype(np.float32))
            t2 = mindspore.Tensor(np.random.randn(8).astype(np.float32))
            writer = TensorWriter(tmpdir, max_shard_size=0)
            writer.save_tensor("weight_a", t1)
            writer.save_tensor("weight_b", t2)
            writer.finalize()

            st_file = os.path.join(tmpdir, "model.safetensors")
            assert os.path.exists(st_file)

            idx = ShardedTensorIndex.from_file(st_file)
            assert "weight_a" in idx.tensor_paths
            assert "weight_b" in idx.tensor_paths

    def test_multi_shard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            t1 = mindspore.Tensor(np.ones((4, 4), dtype=np.float32))
            t2 = mindspore.Tensor(np.zeros((4, 4), dtype=np.float32))
            writer = TensorWriter(tmpdir, max_shard_size=64)
            writer.save_tensor("a", t1)
            writer.save_tensor("b", t2)
            writer.finalize()
            assert writer.shards_written >= 2

            idx_file = os.path.join(tmpdir, "model.safetensors.index.json")
            index_data = json.load(open(idx_file))
            assert {"a", "b"} <= set(index_data["weight_map"].keys())


class TestShardedTensorIndex:

    def test_from_file_and_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            t = mindspore.Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
            writer = TensorWriter(tmpdir, max_shard_size=0)
            writer.save_tensor("vec", t)
            writer.finalize()

            idx = ShardedTensorIndex.from_file(os.path.join(tmpdir, "model.safetensors"))
            assert idx.is_safetensors and "vec" in idx.tensor_paths
            assert len(idx.shards) == 1 and idx.base_path == tmpdir

            idx2 = ShardedTensorIndex.from_disk(tmpdir)
            assert "vec" in idx2.tensor_paths

    def test_from_disk_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="Unable to find"):
                ShardedTensorIndex.from_disk(tmpdir)


# ===================================================================
# 5. merge_methods tests
# ===================================================================

class TestLinearMerge:

    def test_weighted_average(self):
        t1 = mindspore.Tensor([2.0, 4.0, 6.0], dtype=mindspore.float32)
        t2 = mindspore.Tensor([10.0, 20.0, 30.0], dtype=mindspore.float32)
        w1, w2 = 0.3, 0.7
        tensors = ops.stack([t1, t2], axis=0)
        weights = mindspore.Tensor([w1, w2], dtype=mindspore.float32).unsqueeze(-1)
        result = (weights * tensors).sum(axis=0)
        expected = w1 * t1.asnumpy() + w2 * t2.asnumpy()
        np.testing.assert_allclose(result.asnumpy(), expected, atol=1e-5)

    def test_equal_weights_is_average(self):
        t1 = mindspore.Tensor([1.0, 2.0, 3.0, 4.0], dtype=mindspore.float32)
        t2 = mindspore.Tensor([5.0, 6.0, 7.0, 8.0], dtype=mindspore.float32)
        tensors = ops.stack([t1, t2], axis=0)
        weights = mindspore.Tensor([0.5, 0.5], dtype=mindspore.float32).unsqueeze(-1)
        result = (weights * tensors).sum(axis=0)
        np.testing.assert_allclose(result.asnumpy(), (t1.asnumpy() + t2.asnumpy()) / 2.0, atol=1e-5)


class TestSlerpMerge:

    def test_lerp_endpoints(self):
        v0 = mindspore.Tensor([1.0, 0.0], dtype=mindspore.float32)
        v1 = mindspore.Tensor([0.0, 1.0], dtype=mindspore.float32)
        np.testing.assert_allclose(lerp(0.0, v0, v1).asnumpy(), v0.asnumpy(), atol=1e-6)
        np.testing.assert_allclose(lerp(1.0, v0, v1).asnumpy(), v1.asnumpy(), atol=1e-6)

    def test_lerp_midpoint(self):
        v0 = mindspore.Tensor([0.0, 0.0], dtype=mindspore.float32)
        v1 = mindspore.Tensor([2.0, 4.0], dtype=mindspore.float32)
        np.testing.assert_allclose(lerp(0.5, v0, v1).asnumpy(), [1.0, 2.0], atol=1e-6)

    def test_slerp_endpoints(self):
        v0 = mindspore.Tensor([1.0, 0.0, 0.0, 0.0], dtype=mindspore.float32)
        v1 = mindspore.Tensor([0.0, 1.0, 0.0, 0.0], dtype=mindspore.float32)
        np.testing.assert_allclose(slerp(0.0, v0, v1).asnumpy(), v0.asnumpy(), atol=1e-4)
        np.testing.assert_allclose(slerp(1.0, v0, v1).asnumpy(), v1.asnumpy(), atol=1e-4)

    def test_slerp_preserves_norm(self):
        v0 = mindspore.Tensor([3.0, 0.0, 0.0, 0.0], dtype=mindspore.float32)
        v1 = mindspore.Tensor([0.0, 3.0, 0.0, 0.0], dtype=mindspore.float32)
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result_norm = float(ops.norm(slerp(t, v0, v1)).asnumpy())
            assert abs(result_norm - 3.0) < 0.01, f"Norm at t={t}: {result_norm}"

    def test_slerp_collinear_falls_back_to_lerp(self):
        v0 = mindspore.Tensor([1.0, 2.0, 3.0, 4.0], dtype=mindspore.float32)
        v1 = v0 * 1.001
        np.testing.assert_allclose(
            slerp(0.5, v0, v1).asnumpy(), lerp(0.5, v0, v1).asnumpy(), atol=1e-3
        )

    def test_normalize_helper(self):
        v = mindspore.Tensor([3.0, 4.0], dtype=mindspore.float32)
        assert abs(float(ops.norm(normalize(v, eps=1e-8)).asnumpy()) - 1.0) < 1e-5
        zero = mindspore.Tensor([0.0, 0.0], dtype=mindspore.float32)
        np.testing.assert_allclose(normalize(zero, eps=1e-8).asnumpy(), [0.0, 0.0], atol=1e-8)


class TestMergeMethodRegistry:

    EXPECTED_METHODS = {
        "linear", "slerp", "nuslerp", "passthrough", "model_stock",
        "arcee_fusion", "karcher", "task_arithmetic", "ties",
        "dare_ties", "dare_linear", "breadcrumbs", "breadcrumbs_ties",
        "della", "della_linear",
    }

    def test_all_static_methods_registered(self):
        for name in self.EXPECTED_METHODS:
            assert name in REGISTERED_MERGE_METHODS, f"'{name}' not registered"
        assert len(REGISTERED_MERGE_METHODS) >= 15

    def test_get_method(self):
        from mindnlp.wizard.merge.merge_methods import get
        assert get("linear").name() == "linear"
        with pytest.raises(RuntimeError, match="Unimplemented"):
            get("nonexistent_method")

    def test_method_has_name(self):
        for name, method in REGISTERED_MERGE_METHODS.items():
            assert method.name() == name


class TestGeneralizedTaskArithmetic:

    def test_get_mask(self):
        from mindnlp.wizard.merge.merge_methods.generalized_task_arithmetic import get_mask
        delta = mindspore.Tensor(
            [[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]], dtype=mindspore.float32
        )
        assert get_mask(delta, method="sum").shape == delta.shape
        delta2 = mindspore.Tensor(
            [[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0]], dtype=mindspore.float32
        )
        assert get_mask(delta2, method="count").shape == delta2.shape

    def test_gta_methods_in_registry(self):
        for name in ("task_arithmetic", "ties"):
            assert name in REGISTERED_MERGE_METHODS
            assert REGISTERED_MERGE_METHODS[name].name() == name


# ===================================================================
# 6. sparsify tests
# ===================================================================

class TestMagnitudeSparsification:

    def test_density_1_is_identity(self):
        t = mindspore.Tensor([1.0, -2.0, 3.0, -4.0], dtype=mindspore.float32)
        np.testing.assert_allclose(magnitude(t, density=1.0).asnumpy(), t.asnumpy())

    def test_retains_correct_count(self):
        t = mindspore.Tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=mindspore.float32
        )
        result = magnitude(t, density=0.5)
        assert (result.asnumpy() != 0).sum() == 4

    def test_retains_largest_values(self):
        t = mindspore.Tensor([1.0, 2.0, 3.0, 4.0], dtype=mindspore.float32)
        r = magnitude(t, density=0.5).asnumpy()
        assert r[2] != 0 and r[3] != 0
        assert r[0] == 0 and r[1] == 0

    def test_works_with_negative_values(self):
        t = mindspore.Tensor([-10.0, 1.0, -0.5, 0.1], dtype=mindspore.float32)
        assert magnitude(t, density=0.25).asnumpy()[0] != 0

    def test_2d_tensor(self):
        t = mindspore.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=mindspore.float32)
        result = magnitude(t, density=0.5)
        assert result.shape == (2, 2)
        assert (result.asnumpy() != 0).sum() == 2


class TestRandomSparsification:

    def test_density_1_is_identity(self):
        t = mindspore.Tensor([1.0, 2.0, 3.0, 4.0], dtype=mindspore.float32)
        np.testing.assert_allclose(bernoulli(t, density=1.0).asnumpy(), t.asnumpy())

    def test_output_shape(self):
        t = mindspore.Tensor(np.random.randn(4, 4).astype(np.float32))
        assert bernoulli(t, density=0.5).shape == t.shape

    def test_some_zeros(self):
        t = mindspore.Tensor(np.ones(1000, dtype=np.float32))
        r = bernoulli(t, density=0.5).asnumpy()
        zero_count = (r == 0).sum()
        assert 100 < zero_count < 900, "Expected reasonable sparsity"


class TestMagnitudeOutliers:

    def test_density_1_is_identity(self):
        t = mindspore.Tensor([1.0, 2.0, 3.0, 4.0], dtype=mindspore.float32)
        np.testing.assert_allclose(magnitude_outliers(t, density=1.0).asnumpy(), t.asnumpy())

    def test_removes_outliers(self):
        t = mindspore.Tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 100.0], dtype=mindspore.float32
        )
        r = magnitude_outliers(t, density=0.5, gamma=0.125).asnumpy()
        assert r[-1] == 0.0, "Outlier should be removed"

    def test_output_shape(self):
        t = mindspore.Tensor(np.random.randn(4, 4).astype(np.float32))
        assert magnitude_outliers(t, density=0.5, gamma=0.01).shape == (4, 4)


class TestSparsifyDispatch:

    @pytest.mark.parametrize("method", [
        SparsificationMethod.magnitude,
        SparsificationMethod.random,
    ])
    def test_dispatch_basic(self, method):
        t = mindspore.Tensor([1.0, 2.0, 3.0, 4.0], dtype=mindspore.float32)
        assert sparsify(t, density=0.5, method=method).shape == t.shape

    def test_dispatch_magnitude_outliers(self):
        t = mindspore.Tensor(np.random.randn(16).astype(np.float32))
        assert sparsify(t, density=0.5, method=SparsificationMethod.magnitude_outliers, gamma=0.01).shape == t.shape

    @pytest.mark.parametrize("norm,norm_fn", [
        (RescaleNorm.l1, lambda x: float(x.abs().sum().asnumpy())),
        (RescaleNorm.l2, lambda x: float(ops.norm(x.astype(mindspore.float32)).asnumpy())),
    ])
    def test_rescale_norm(self, norm, norm_fn):
        t = mindspore.Tensor(
            [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], dtype=mindspore.float32
        )
        result = magnitude(t, density=0.5, rescale_norm=norm)
        assert abs(norm_fn(t) - norm_fn(result)) / norm_fn(t) < 0.01


class TestBf16CpuMergeMethodRegression:
    def _bf(self, values):
        return mindspore.Tensor(np.array(values, dtype=np.float32)).astype(
            mindspore.bfloat16
        )

    def test_sce_merge_bf16_cpu(self):
        from mindnlp.wizard.merge.merge_methods.sce import sce_merge

        mindspore.set_context(device_target="CPU")
        out = sce_merge(
            [self._bf([1, 2, 3, 4]), self._bf([2, 3, 4, 5])],
            self._bf([0.5, 1.0, 1.5, 2.0]),
            select_topk=0.5,
        )
        assert out.dtype == mindspore.bfloat16
        assert out.shape == (4,)

    def test_ram_merge_bf16_cpu(self):
        from mindnlp.wizard.merge.merge_methods.ram import ram_merge

        mindspore.set_context(device_target="CPU")
        out = ram_merge(
            [self._bf([1, 2, 3, 4]), self._bf([2, 3, 4, 5])],
            self._bf([0.5, 1.0, 1.5, 2.0]),
        )
        assert out.dtype == mindspore.bfloat16
        assert out.shape == (4,)

    def test_multislerp_merge_bf16_cpu(self):
        from mindnlp.wizard.merge.merge_methods.multislerp import multislerp

        mindspore.set_context(device_target="CPU")
        out = multislerp(
            [self._bf([1, 2, 3, 4]), self._bf([2, 3, 4, 5])],
            [0.4, 0.6],
            base_tensor=self._bf([0.1, 0.1, 0.1, 0.1]),
        )
        assert out.dtype == mindspore.bfloat16
        assert out.shape == (4,)

    @pytest.mark.xfail(
        reason="MindSpore CPU auto-promotes bf16 to float32 in ops.uniform/bernoulli",
        strict=False,
    )
    def test_bernoulli_and_della_bf16_cpu(self):
        mindspore.set_context(device_target="CPU")
        src = self._bf([[1.0, 2.0], [3.0, 4.0]])
        out_rand = bernoulli(src, density=0.5)
        out_della = sparsify(
            src,
            density=0.5,
            method=SparsificationMethod.della_magprune,
            epsilon=0.1,
        )
        assert out_rand.dtype == mindspore.bfloat16
        assert out_della.dtype == mindspore.bfloat16
        assert out_rand.shape == src.shape
        assert out_della.shape == src.shape


class TestSparsificationMethodEnum:

    def test_enum_values(self):
        assert SparsificationMethod.magnitude.value == "magnitude"
        assert SparsificationMethod.random.value == "random"
        assert SparsificationMethod.magnitude_outliers.value == "magnitude_outliers"
        assert RescaleNorm.l1.value == "l1"
        assert RescaleNorm.l2.value == "l2"


# ===================================================================
# 7. hardening regression tests
# ===================================================================

class TestMergeOptionsDeviceDetection:
    def test_auto_device_probe_failure_warns_and_falls_back(self, monkeypatch, caplog):
        import mindnlp.wizard.merge.common as common_mod

        def _raise_probe():
            raise RuntimeError("probe failed")

        monkeypatch.setattr(common_mod, "get_accelerator_type", _raise_probe)
        caplog.set_level(logging.WARNING)
        opts = MergeOptions(device="auto")
        assert opts.device == "CPU"
        assert "Automatic device detection failed" in caplog.text

    def test_auto_device_probe_failure_strict_mode_raises(self, monkeypatch):
        import mindnlp.wizard.merge.common as common_mod

        def _raise_probe():
            raise RuntimeError("probe failed")

        monkeypatch.setattr(common_mod, "get_accelerator_type", _raise_probe)
        with pytest.raises(RuntimeError, match="Automatic device detection failed"):
            MergeOptions(device="auto", strict_device_detect=True)

    def test_max_tensor_mem_gb_validation(self):
        with pytest.raises(ValueError, match="max_tensor_mem_gb must be > 0"):
            MergeOptions(max_tensor_mem_gb=0)

    def test_split_pieces_validation(self):
        with pytest.raises(ValueError, match="split_pieces must be >= 1"):
            MergeOptions(split_pieces=0)

class TestAcceleratorCountFallbacks:
    def test_explicit_device_with_index_returns_one(self):
        assert get_accelerator_count("GPU:2") == 1

    def test_uses_runtime_probe_when_available(self, monkeypatch):
        import mindnlp.wizard.merge.common as common_mod

        monkeypatch.setattr(common_mod, "_default_accelerator", lambda: "GPU")
        monkeypatch.setattr(common_mod, "_probe_device_count", lambda _target: 4)
        assert get_accelerator_count() == 4

    def test_probe_failure_falls_back_to_one_with_warning(self, monkeypatch, caplog):
        import mindnlp.wizard.merge.common as common_mod

        monkeypatch.setattr(common_mod, "_default_accelerator", lambda: "GPU")
        monkeypatch.setattr(common_mod, "_probe_device_count", lambda _target: 0)
        caplog.set_level(logging.WARNING)
        assert get_accelerator_count() == 1
        assert "defaulting to 1" in caplog.text


class TestGraphTensorMove:
    def test_move_tensors_uses_move_to_for_mindspore_tensor(self, monkeypatch):
        moved = {"target": None}
        tensor = mindspore.Tensor(np.array([1.0], dtype=np.float32))

        def _fake_move_to(self, target):
            moved["target"] = target
            return self

        monkeypatch.setattr(
            mindspore.Tensor,
            "move_to",
            _fake_move_to,
            raising=False,
        )
        out = Executor._move_tensors(tensor, "Ascend:0")
        assert moved["target"] == "Ascend"
        assert isinstance(out, mindspore.Tensor)


class TestExtractLoraExecutorSelection:
    def test_build_executor_uses_multi_device_executor_when_multi_npu(self, monkeypatch):
        from mindnlp.wizard.merge.scripts import extract_lora as extract_mod

        called = {"multi": False}

        class _FakeMulti:
            def __init__(self, tasks, storage_device=None):
                called["multi"] = True
                self.tasks = tasks
                self.storage_device = storage_device

        monkeypatch.setattr(extract_mod, "MultiDeviceExecutor", _FakeMulti)
        opts = MergeOptions(multi_npu=True, low_cpu_memory=False)
        ex = extract_mod._build_executor([], opts, "CPU")
        assert called["multi"] is True
        assert ex.storage_device == "CPU"

    def test_build_executor_uses_executor_when_not_multi_npu(self):
        from mindnlp.wizard.merge.scripts import extract_lora as extract_mod

        opts = MergeOptions(multi_npu=False, device="Ascend")
        ex = extract_mod._build_executor([], opts, "CPU")
        assert isinstance(ex, Executor)


class TestMultiDeviceLocalityAssignment:
    def test_same_locality_islands_prefer_same_device(self, monkeypatch):
        import mindnlp.wizard.merge.multigpu_executor as mg_mod

        monkeypatch.setattr(mg_mod, "get_accelerator_type", lambda: "Ascend")

        t1 = LabeledConstTask(value=1, label="model-00001-of-00002")
        t2 = LabeledConstTask(value=2, label="model-00001-of-00002")
        t3 = LabeledConstTask(value=3, label="model-00002-of-00002")
        ex = MultiDeviceExecutor(targets=[t1, t2, t3], num_devices=2, storage_device="CPU")

        task_to_device = {}
        for device, handles in ex.device_assignments.items():
            for handle in handles:
                task_to_device[handle.task().value] = device

        assert task_to_device[1] == task_to_device[2]
        metrics = ex.metrics_snapshot()
        assert "island_assignment" in metrics

    def test_locality_key_uses_explicit_shard_suffix(self, monkeypatch):
        import mindnlp.wizard.merge.multigpu_executor as mg_mod

        monkeypatch.setattr(mg_mod, "get_accelerator_type", lambda: "Ascend")
        task = LabeledConstTask(value=1, label="foo/bar::model-00003-of-00008")
        ex = MultiDeviceExecutor(targets=[task], num_devices=1, storage_device="CPU")
        handle = next(iter(ex.targets))
        key = ex._task_locality_key(handle)
        assert key == "model-00003"


class TestLoadTensorLocalityLabel:
    def test_group_label_prefers_shard_name(self, monkeypatch):
        model = ModelReference.parse("dummy/model")

        class _FakeIndex:
            tensor_paths = {"w": "/tmp/model-00004-of-00016.safetensors"}

        class _FakeLoader:
            index = _FakeIndex()

        monkeypatch.setattr(LoaderCache, "get", lambda self, _model: _FakeLoader())
        task = LoadTensor(model=model, tensor="w")
        label = task.group_label()
        assert "model-025pct" in label


class TestIoMoveWarning:
    class _DummyTensor:
        def move_to(self, _target):
            raise RuntimeError("boom")

    def test_device_move_failure_warns(self, caplog):
        import mindnlp.wizard.merge.io._device as device_mod

        device_mod._MOVE_WARNED_TARGETS.clear()
        caplog.set_level(logging.WARNING)
        tensor = self._DummyTensor()
        res = device_mod.move_tensor_to_device(tensor, "Ascend:0")
        assert res is tensor
        assert "Failed to move tensor to Ascend" in caplog.text

    def test_device_move_warns_once(self, caplog):
        import mindnlp.wizard.merge.io._device as device_mod

        device_mod._MOVE_WARNED_TARGETS.clear()
        caplog.set_level(logging.WARNING)
        tensor = self._DummyTensor()
        device_mod.move_tensor_to_device(tensor, "Ascend:0")
        caplog.clear()
        device_mod.move_tensor_to_device(tensor, "Ascend:0")
        assert "Failed to move" not in caplog.text


class TestExecutionReport:
    def test_writes_island_assignment_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_execution_report(
                out_path=tmpdir,
                metrics={
                    "executor": "multi_device",
                    "task_count": 2,
                    "tasks": [
                        {"task": "A", "wait_ms": 0.0, "run_ms": 1.0},
                        {"task": "B", "wait_ms": 0.0, "run_ms": 2.0},
                    ],
                    "queue_depth_samples": [0, 1],
                    "backpressure_trigger_count": 0,
                    "rss_peak_mb": 10.0,
                    "npu_used_peak_mb": None,
                    "island_assignment": [
                        {
                            "device": "Ascend:0",
                            "task_count": 5,
                            "dominant_locality_key": "model-050pct",
                        }
                    ],
                },
                metadata={},
            )
            md_path = os.path.join(tmpdir, "wizard_execution_report.md")
            assert os.path.exists(md_path)
            content = open(md_path, "r", encoding="utf-8").read()
            assert "## Island Assignment" in content
            assert "Ascend:0" in content
