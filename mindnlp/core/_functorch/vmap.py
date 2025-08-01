import contextlib
import functools
import itertools
import os
import threading
from functools import partial
from typing import Any, Callable, Optional, Union

from mindnlp import core
from mindnlp.core.utils._pytree import (
    _broadcast_to_and_flatten,
    tree_flatten,
    tree_map_,
    tree_unflatten,
    TreeSpec,
)

in_dims_t = Union[int, tuple]
out_dims_t = Union[int, tuple[int, ...]]

def _vmap_decrement_nesting():
    return 0

def _vmap_increment_nesting(*args, **kwargs):
    return 0

def _get_name(func: Callable):
    if hasattr(func, "__name__"):
        return func.__name__

    if isinstance(func, functools.partial):
        return f"functools.partial({_get_name(func.func)}, ...)"

    # Not all callables have __name__, in fact, only static functions/methods
    # do.  A callable created via nn.Module, to name one example, doesn't have a
    # __name__.
    return repr(func)


def _check_int_or_none(x, func, out_dims):
    if isinstance(x, int):
        return
    if x is None:
        return
    raise ValueError(
        f"vmap({_get_name(func)}, ..., out_dims={out_dims}): `out_dims` must be "
        f"an int, None or a python collection of ints representing where in the outputs the "
        f"vmapped dimension should appear."
    )

def _validate_and_get_batch_size(
    flat_in_dims: list[Optional[int]], flat_args: list
) -> int:
    batch_sizes = [
        arg.size(in_dim)
        for in_dim, arg in zip(flat_in_dims, flat_args)
        if in_dim is not None
    ]
    if len(batch_sizes) == 0:
        raise ValueError("vmap: Expected at least one Tensor to vmap over")
    if batch_sizes and any(size != batch_sizes[0] for size in batch_sizes):
        raise ValueError(
            f"vmap: Expected all tensors to have the same size in the mapped "
            f"dimension, got sizes {batch_sizes} for the mapped dimension"
        )
    return batch_sizes[0]


def _check_randomness_arg(randomness):
    if randomness not in ["error", "different", "same"]:
        raise RuntimeError(
            f"Only allowed values for randomness are 'error', 'different', or 'same'. Got {randomness}"
        )

def _check_out_dims_is_int_or_int_pytree(out_dims: out_dims_t, func: Callable) -> None:
    if isinstance(out_dims, int):
        return
    tree_map_(partial(_check_int_or_none, func=func, out_dims=out_dims), out_dims)


def _process_batched_inputs(
    in_dims: in_dims_t, args: tuple, func: Callable
) -> tuple[int, list[Any], list[Any], TreeSpec]:
    if not isinstance(in_dims, int) and not isinstance(in_dims, tuple):
        raise ValueError(
            f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
            f"expected `in_dims` to be int or a (potentially nested) tuple "
            f"matching the structure of inputs, got: {type(in_dims)}."
        )
    if len(args) == 0:
        raise ValueError(
            f"vmap({_get_name(func)})(<inputs>): got no inputs. Maybe you forgot to add "
            f"inputs, or you are trying to vmap over a function with no inputs. "
            f"The latter is unsupported."
        )

    flat_args, args_spec = tree_flatten(args)
    flat_in_dims = _broadcast_to_and_flatten(in_dims, args_spec)
    if flat_in_dims is None:
        raise ValueError(
            f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
            f"in_dims is not compatible with the structure of `inputs`. "
            f"in_dims has structure {tree_flatten(in_dims)[1]} but inputs "
            f"has structure {args_spec}."
        )

    for i, (arg, in_dim) in enumerate(zip(flat_args, flat_in_dims)):
        if not isinstance(in_dim, int) and in_dim is not None:
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for an input but in_dim must be either "
                f"an integer dimension or None."
            )
        if isinstance(in_dim, int) and not isinstance(arg, core.Tensor):
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for an input but the input is of type "
                f"{type(arg)}. We cannot vmap over non-Tensor arguments, "
                f"please use None as the respective in_dim"
            )
        if in_dim is not None and (in_dim < -arg.dim() or in_dim >= arg.dim()):
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"Got in_dim={in_dim} for some input, but that input is a Tensor "
                f"of dimensionality {arg.dim()} so expected in_dim to satisfy "
                f"-{arg.dim()} <= in_dim < {arg.dim()}."
            )
        if in_dim is not None and in_dim < 0:
            flat_in_dims[i] = in_dim % arg.dim()

    return (
        _validate_and_get_batch_size(flat_in_dims, flat_args),
        flat_in_dims,
        flat_args,
        args_spec,
    )

def get_chunk_sizes(total_elems, chunk_size):
    n_chunks = n_chunks = total_elems // chunk_size
    chunk_sizes = [chunk_size] * n_chunks
    # remainder chunk
    remainder = total_elems % chunk_size
    if remainder != 0:
        chunk_sizes.append(remainder)
    return chunk_sizes


def _get_chunked_inputs(flat_args, flat_in_dims, batch_size, chunk_size):
    split_idxs = (batch_size,)
    if chunk_size is not None:
        chunk_sizes = get_chunk_sizes(batch_size, chunk_size)
        split_idxs = tuple(itertools.accumulate(chunk_sizes))

    flat_args_chunks = tuple(
        (
            t.tensor_split(split_idxs, dim=in_dim)
            if in_dim is not None
            else [
                t,
            ]
            * len(split_idxs)
        )
        for t, in_dim in zip(flat_args, flat_in_dims)
    )

    # transpose chunk dim and flatten structure
    # chunks_flat_args is a list of flatten args
    chunks_flat_args = zip(*flat_args_chunks)
    return chunks_flat_args

@contextlib.contextmanager
def vmap_increment_nesting(batch_size, randomness):
    try:
        vmap_level = _vmap_increment_nesting(batch_size, randomness)
        yield vmap_level
    finally:
        _vmap_decrement_nesting()

def _add_batch_dim(arg, in_dim, vmap_level):
    return arg.squeeze(in_dim)

def is_batchedtensor(input):
    return False

def _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim):
    return batched_output

def _create_batched_inputs(
    flat_in_dims: list[Any], flat_args: list[Any], vmap_level: int, args_spec
) -> tuple:
    # See NOTE [Ignored _remove_batch_dim, _add_batch_dim]
    batched_inputs = [
        arg if in_dim is None else _add_batch_dim(arg, in_dim, vmap_level)
        for in_dim, arg in zip(flat_in_dims, flat_args)
    ]
    return tree_unflatten(batched_inputs, args_spec)


def _maybe_remove_batch_dim(name, batched_output, vmap_level, batch_size, out_dim):
    if out_dim is None:
        if isinstance(batched_output, core.Tensor) and is_batchedtensor(
            batched_output
        ):
            raise ValueError(
                f"vmap({name}, ...): `{name}` can not return a "
                f"BatchedTensor when out_dim is None"
            )
        return batched_output

    # out_dim is non None
    if not isinstance(batched_output, core.Tensor):
        raise ValueError(
            f"vmap({name}, ...): `{name}` must only return "
            f"Tensors, got type {type(batched_output)}. "
            "Did you mean to set out_dims= to None for output?"
        )

    out = _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)
    return out

def _unwrap_batched(
    batched_outputs: Union[core.Tensor, tuple[core.Tensor, ...]],
    out_dims: out_dims_t,
    vmap_level: int,
    batch_size: int,
    func: Callable,
) -> tuple:
    flat_batched_outputs, output_spec = tree_flatten(batched_outputs)

    def incompatible_error():
        raise ValueError(
            f"vmap({_get_name(func)}, ..., out_dims={out_dims})(<inputs>): "
            f"out_dims is not compatible with the structure of `outputs`. "
            f"out_dims has structure {tree_flatten(out_dims)[1]} but outputs "
            f"has structure {output_spec}."
        )

    if isinstance(batched_outputs, core.Tensor):
        # Some weird edge case requires us to spell out the following
        # see test_out_dims_edge_case
        if isinstance(out_dims, int):
            flat_out_dims = [out_dims]
        elif isinstance(out_dims, tuple) and len(out_dims) == 1:
            flat_out_dims = out_dims
        elif out_dims is None:
            flat_out_dims = [out_dims]
        else:
            incompatible_error()
    else:
        flat_out_dims = _broadcast_to_and_flatten(out_dims, output_spec)
        if flat_out_dims is None:
            incompatible_error()

    flat_outputs = [
        _maybe_remove_batch_dim(
            _get_name(func), batched_output, vmap_level, batch_size, out_dim
        )
        for batched_output, out_dim in zip(flat_batched_outputs, flat_out_dims)
    ]
    return tree_unflatten(flat_outputs, output_spec)


def _flat_vmap(
    func, batch_size, flat_in_dims, flat_args, args_spec, out_dims, randomness, **kwargs
):
    with vmap_increment_nesting(batch_size, randomness) as vmap_level:
        batched_inputs = _create_batched_inputs(
            flat_in_dims, flat_args, vmap_level, args_spec
        )
        batched_outputs = func(*batched_inputs, **kwargs)
        return _unwrap_batched(batched_outputs, out_dims, vmap_level, batch_size, func)


def _flatten_chunks_output(chunks_output_):
    # chunks_output is a list of chunked outputs
    # flatten chunked outputs:
    flat_chunks_output = []
    arg_spec = None
    for output in chunks_output_:
        flat_output, arg_specs = tree_flatten(output)
        flat_chunks_output.append(flat_output)
        if arg_spec is None:
            arg_spec = arg_specs

    # transpose chunk dim and flatten structure
    # flat_output_chunks is flat list of chunks
    flat_output_chunks = list(zip(*flat_chunks_output))
    return flat_output_chunks, arg_spec


def _concat_chunked_outputs(out_dims, arg_spec, flat_output_chunks):
    # concat chunks on out_dim
    flat_out_dims = _broadcast_to_and_flatten(out_dims, arg_spec)
    assert len(flat_out_dims) == len(flat_output_chunks)
    flat_output = []
    for idx, out_dim in enumerate(flat_out_dims):
        flat_output.append(core.cat(flat_output_chunks[idx], dim=out_dim))
        # release tensors
        flat_output_chunks[idx] = None

    return flat_output


def _chunked_vmap(
    func, flat_in_dims, chunks_flat_args, args_spec, out_dims, randomness, **kwargs
):
    chunks_output = []
    rs = core.get_rng_state() if randomness == "same" else None
    for flat_args in chunks_flat_args:
        batch_size = _validate_and_get_batch_size(flat_in_dims, flat_args)

        # The way we compute split the input in `_get_chunked_inputs`,
        # we may get a tensor with `0` batch-size. We skip any computation
        # in that case.
        # Eg.
        # >>> chunk_size = 1
        # >>> batch_size = 6
        # >>> t = torch.zeros(batch_size, 1)
        # >>> t.tensor_split([1, 2, 3, 4, 5, 6])
        # (tensor([[0.]]), tensor([[0.]]), tensor([[0.]]), tensor([[0.]]),
        #  tensor([[0.]]), tensor([[0.]]), tensor([], size=(0, 1)))
        if batch_size == 0:
            continue

        if rs is not None:
            core.set_rng_state(rs)
        chunks_output.append(
            _flat_vmap(
                func,
                batch_size,
                flat_in_dims,
                flat_args,
                args_spec,
                out_dims,
                randomness,
                **kwargs,
            )
        )

    flat_output_chunks, arg_spec = _flatten_chunks_output(chunks_output)

    # chunked output tensors are held by both `flat_output_chunks` and `chunks_output`.
    # eagerly remove the reference from `chunks_output`.
    del chunks_output

    # concat chunks on out_dim
    flat_output = _concat_chunked_outputs(out_dims, arg_spec, flat_output_chunks)

    # finally unflatten the output
    return tree_unflatten(flat_output, arg_spec)


def vmap_impl(func, in_dims, out_dims, randomness, chunk_size, *args, **kwargs):
    _check_out_dims_is_int_or_int_pytree(out_dims, func)
    batch_size, flat_in_dims, flat_args, args_spec = _process_batched_inputs(
        in_dims, args, func
    )

    if chunk_size is not None:
        chunks_flat_args = _get_chunked_inputs(
            flat_args, flat_in_dims, batch_size, chunk_size
        )
        return _chunked_vmap(
            func,
            flat_in_dims,
            chunks_flat_args,
            args_spec,
            out_dims,
            randomness,
            **kwargs,
        )

    # If chunk_size is not specified.
    return _flat_vmap(
        func,
        batch_size,
        flat_in_dims,
        flat_args,
        args_spec,
        out_dims,
        randomness,
        **kwargs,
    )
