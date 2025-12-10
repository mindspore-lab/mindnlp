# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""mindmindtorch nn"""
# mypy: allow-untyped-defs
from mindtorch.nn.parameter import (  # usort: skip
    Buffer as Buffer,
    Parameter as Parameter,
    UninitializedBuffer as UninitializedBuffer,
    UninitializedParameter as UninitializedParameter,
)
from mindtorch.nn.modules import *  # usort: skip # noqa: F403
from mindtorch.nn import (
    attention as attention,
    functional as functional,
    init as init,
    modules as modules,
    parallel as parallel,
    parameter as parameter,
    utils as utils,
)
from mindtorch.nn.parallel import DataParallel as DataParallel


def factory_kwargs(kwargs):
    r"""Return a canonicalized dict of factory kwargs.

    Given kwargs, returns a canonicalized dict of factory kwargs that can be directly passed
    to factory functions like mindtorch.empty, or errors if unrecognized kwargs are present.

    This function makes it simple to write code like this::

        class MyModule(nn.Module):
            def __init__(self, **kwargs):
                factory_kwargs = mindtorch.nn.factory_kwargs(kwargs)
                self.weight = Parameter(mindtorch.empty(10, **factory_kwargs))

    Why should you use this function instead of just passing `kwargs` along directly?

    1. This function does error validation, so if there are unexpected kwargs we will
    immediately report an error, instead of deferring it to the factory call
    2. This function supports a special `factory_kwargs` argument, which can be used to
    explicitly specify a kwarg to be used for factory functions, in the event one of the
    factory kwargs conflicts with an already existing argument in the signature (e.g.
    in the signature ``def f(dtype, **kwargs)``, you can specify ``dtype`` for factory
    functions, as distinct from the dtype argument, by saying
    ``f(dtype1, factory_kwargs={"dtype": dtype2})``)
    """
    if kwargs is None:
        return {}
    simple_keys = {"device", "dtype", "memory_format"}
    expected_keys = simple_keys | {"factory_kwargs"}
    if not kwargs.keys() <= expected_keys:
        raise TypeError(f"unexpected kwargs {kwargs.keys() - expected_keys}")

    # guarantee no input kwargs is untouched
    r = dict(kwargs.get("factory_kwargs", {}))
    for k in simple_keys:
        if k in kwargs:
            if k in r:
                raise TypeError(
                    f"{k} specified twice, in **kwargs and in factory_kwargs"
                )
            r[k] = kwargs[k]

    return r
