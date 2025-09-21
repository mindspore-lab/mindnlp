# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor

import mindtorch
from mindtorch.distributed.tensor._dtensor_spec import DTensorSpec
from mindtorch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementStrategy,
    StrategyType,
)
from mindtorch.distributed.tensor._ops.utils import register_op_strategy
from mindtorch.distributed.tensor.placement_types import Replicate


aten = mindtorch.ops.aten


# @register_op_strategy(aten.slice_backward.default)
# def slice_backward_rules(op_schema: OpSchema) -> StrategyType:
#     """
#     slice_backward is a new_zeros + slice_scatter, we only allow replication
#     on the input/output for now since new_zeros would produce replication
#     """
#     mesh = op_schema.get_mesh_from_args(validate=False)
#     replicate_spec = DTensorSpec(mesh, tuple([Replicate()] * mesh.ndim))
#     return OpStrategy([PlacementStrategy(replicate_spec)])
