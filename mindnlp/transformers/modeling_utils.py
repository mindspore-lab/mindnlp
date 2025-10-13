import types

from mindspore.communication import GlobalComm

import mindtorch
from mindtorch import nn, distributed as dist
from ..utils import logging

logger = logging.get_logger(__name__)


def replace_submodule(model, submodule_path, new_module):
    parent_path, _, child_name = submodule_path.rpartition(".")

    parent_module = model.get_submodule(parent_path) if parent_path else model

    setattr(parent_module, child_name, new_module)


def send_forward(self, *args, **kwargs):
    output = self._forward(*args, **kwargs)
    dist.isend(output[0], self.dist)
    return output


def receive_forward(self, *args, **kwargs):
    hidden_states = args[0]
    dist.recv(hidden_states, src=self.src)
    output = self._forward(*((hidden_states,) + args[1:]), **kwargs)
    return output


def broadcast_forward(self, *args, **kwargs):
    output = self._forward(*args, **kwargs)
    dist.broadcast(output, src=self.src)
    dist.barrier()
    return output


class DecoderLayerIdentity(nn.Module):
    def __init__(self, layer_idx, config):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_key_value_heads = config.num_key_value_heads
        self.attention_type = config.layer_types[layer_idx]

    def forward(self, *args, **kwargs):
        past_key_value = kwargs.get("past_key_value", None)
        hidden_states = args[0]
        bs, seq_len, _ = hidden_states.shape

        if past_key_value is not None:
            past_key_value.update(
                mindtorch.empty(
                    bs,
                    self.num_key_value_heads,
                    seq_len,
                    0,
                    dtype=hidden_states.dtype,
                    device="meta",
                ),
                mindtorch.empty(
                    bs,
                    self.num_key_value_heads,
                    seq_len,
                    0,
                    dtype=hidden_states.dtype,
                    device="meta",
                ),
                self.layer_idx,
            )

        return mindtorch.empty(
            *hidden_states.shape, dtype=hidden_states.dtype, device="meta"
        )


class EmbeddingIndentity(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = dtype

    def forward(self, input):
        return mindtorch.empty(
            input.shape + (self.embedding_dim,), dtype=self.dtype, device="meta"
        )


class LinearIndetity(nn.Module):
    def __init__(self, in_features, out_features, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

    def forward(self, input):
        return mindtorch.empty(
            input.shape[:-1] + (self.out_features,), dtype=self.dtype, device="meta"
        )


def construct_pipeline_parallel_model(model, device_map):
    current_device = dist.get_rank()
    last_device = dist.get_world_size() - 1
    no_split_modules = model._get_no_split_modules(device_map)
    reversed_device_map = {}
    for scope_name, device in device_map.items():
        if device not in reversed_device_map:
            reversed_device_map[device] = [scope_name]
        else:
            reversed_device_map[device].append(scope_name)

        if device != current_device:
            submodule = model.get_submodule(scope_name)
            if isinstance(submodule, nn.Embedding):
                new_embedding = EmbeddingIndentity(
                    submodule.num_embeddings, submodule.embedding_dim, model.dtype
                )
                replace_submodule(model, scope_name, new_embedding)
            elif isinstance(submodule, nn.Linear):
                new_linear = LinearIndetity(
                    submodule.in_features, submodule.out_features, model.dtype
                )
                replace_submodule(model, scope_name, new_linear)
            elif submodule.__class__.__name__ in no_split_modules:
                new_layer = DecoderLayerIdentity(
                    submodule.self_attn.layer_idx, submodule.self_attn.config
                )
                replace_submodule(model, scope_name, new_layer)
            else:
                # new_layer = nn.Identity()
                # replace_submodule(model, scope_name, new_layer)
                pass

    if current_device < last_device:
        current_last_layer = model.get_submodule(
            reversed_device_map[current_device][-1]
        )
        current_last_layer._forward = current_last_layer.forward
        current_last_layer.forward = types.MethodType(send_forward, current_last_layer)
        current_last_layer.dist = current_device + 1

    if current_device > 0:
        current_first_layer = model.get_submodule(
            reversed_device_map[current_device][0]
        )
        current_first_layer._forward = current_first_layer.forward
        current_first_layer.forward = types.MethodType(
            receive_forward, current_first_layer
        )
        current_first_layer.src = current_device - 1

    model_last_layer = model.get_submodule(next(reversed(device_map)))
    model_last_layer._forward = model_last_layer.forward
    model_last_layer.forward = types.MethodType(broadcast_forward, model_last_layer)
    model_last_layer.src = last_device

    return model


def find_usefull_files(shared_files, shared_meta, model_params):
    files_path = "/".join(shared_files[0].split("/")[:-1])
    usefull_files = set()

    for param_name, file_name in shared_meta["weight_map"].items():
        if param_name in model_params:
            usefull_files.add(file_name)
        # else:
        #     shared_meta['all_checkpoint_keys'].remove(param_name)

    usefull_files = [files_path + "/" + file for file in usefull_files]

    return usefull_files, shared_meta


def _load_pretrained_model_wrapper(fn):
    def wrapper(
        cls,
        model,
        state_dict,
        checkpoint_files,
        pretrained_model_name_or_path,
        **kwargs,
    ):
        device_map = kwargs.get("device_map", None)
        sharded_metadata = kwargs.get("sharded_metadata", None)

        # if device_map is not None and not initialize distribute module, raise Error.
        if device_map is not None:
            if (
                all([isinstance(d, int) for d in device_map.values()])
                and len(set(device_map.values())) > 1
            ):
                if not GlobalComm.INITED:
                    raise RuntimeError(
                        f"to use transformers with multi-gpu/npu, please use `msrun/mpirun` "
                        f"with {len(set(device_map.values()))} devices to launch multiprocess."
                    )

                model = construct_pipeline_parallel_model(model, device_map)
                checkpoint_files, sharded_metadata = find_usefull_files(
                    checkpoint_files, sharded_metadata, model.state_dict().keys()
                )

                rank = dist.get_rank()
                world_size = dist.get_world_size()

                dist.barrier()

                for target_rank in range(world_size):
                    if rank == target_rank:

                        model = fn(
                            cls,
                            model,
                            state_dict,
                            checkpoint_files,
                            pretrained_model_name_or_path,
                            **kwargs,
                        )

                    dist.barrier()
                return model

        return fn(
            cls,
            model,
            state_dict,
            checkpoint_files,
            pretrained_model_name_or_path,
            **kwargs,
        )

    return wrapper


def _get_resolved_checkpoint_files_wrapper(fn):
    def wrapper(*args, **kwargs):
        if GlobalComm.INITED and dist.get_world_size() > 1:
            rank = dist.get_rank()

            dist.barrier()

            if rank == 0:
                outs = fn(*args, **kwargs)
            else:
                outs = None

            dist.barrier()

            if rank != 0:
                outs = fn(*args, **kwargs)

            dist.barrier()
            return outs

        else:
            return fn(*args, **kwargs)

    return wrapper
