"""modeling utils for parallel inference"""
# pylint: disable=unnecessary-comprehension
import os
import re
import inspect
import warnings
import types
from subprocess import Popen, PIPE
from collections import OrderedDict, defaultdict
from typing import Optional, Dict, Union, List, Tuple, Set
import mindspore
from mindspore.communication import get_group_size, get_rank
try:
    from mindspore.communication.comm_func import isend, irecv, broadcast
except:
    from mindnlp.parallel.comm_func import isend, irecv, broadcast

from ...core import nn, ops
from ...utils import logging

logger = logging.get_logger(__name__)


def get_gpus_free_memory():
    nvidia_smi = "nvidia-smi"

    # Get ID, processing and memory utilization for all GPUs
    try:
        p = Popen([nvidia_smi,"--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], stdout=PIPE)
        stdout, stderror = p.communicate()
    except:
        return []
    output = stdout.decode('UTF-8')

    lines = output.split(os.linesep)

    numDevices = len(lines)-1
    GPUs = []
    for g in range(numDevices):
        line = lines[g]

        vals = line.split(', ')

        for i in range(2):
            if (i == 0):
                deviceIds = int(vals[i])
            elif (i == 1):
                memFree = int(vals[i])

        GPUs.append((deviceIds, memFree))
    return GPUs

def get_npus_free_memory():
    try:
        p = Popen(['npu-smi',"info"], stdout=PIPE)
        stdout, stderror = p.communicate()
    except:
        return []
    output = stdout.decode('UTF-8')
    lines = output.split(os.linesep)
    table_data = []
    i = 6
    while True:
        if lines[i] == "+---------------------------+---------------+----------------------------------------------------+":
            break
        row1 = lines[i].split()
        row1 = [i for i in row1 if i not in ('|', '/')]
        row2 = lines[i + 1].split()
        row2 = [i for i in row2 if i not in ('|', '/')]
        table_data.append((row1[0], (int(row2[-1]) - int(row2[-2]) // 1000)))
        i += 3

    return table_data


def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.

    Example:

    ```py
    >>> dtype_byte_size(mindspore.float32)
    4
    ```
    """
    if dtype == mindspore.bool_:
        return 1 / 8
    # elif dtype == CustomDtype.INT2:
    #     return 1 / 4
    # elif dtype == CustomDtype.INT4:
    #     return 1 / 2
    # elif dtype == CustomDtype.FP8:
        # return 1
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def convert_file_size_to_int(size: Union[int, str]):
    """
    Converts a size expressed as a string with digits an unit (like `"5MB"`) to an integer (in bytes).

    Args:
        size (`int` or `str`): The size to convert. Will be directly returned if an `int`.

    Example:

    ```py
    >>> convert_file_size_to_int("1MiB")
    1048576
    ```
    """
    mem_size = -1
    err_msg = (
        f"`size` {size} is not in a valid format. Use an integer for bytes, or a string with an unit (like '5.0GB')."
    )
    try:
        if isinstance(size, int):
            mem_size = size
        elif size.upper().endswith("GIB"):
            mem_size = int(float(size[:-3]) * (2**30))
        elif size.upper().endswith("MIB"):
            mem_size = int(float(size[:-3]) * (2**20))
        elif size.upper().endswith("KIB"):
            mem_size = int(float(size[:-3]) * (2**10))
        elif size.upper().endswith("GB"):
            int_size = int(float(size[:-2]) * (10**9))
            mem_size = int_size // 8 if size.endswith("b") else int_size
        elif size.upper().endswith("MB"):
            int_size = int(float(size[:-2]) * (10**6))
            mem_size = int_size // 8 if size.endswith("b") else int_size
        elif size.upper().endswith("KB"):
            int_size = int(float(size[:-2]) * (10**3))
            mem_size = int_size // 8 if size.endswith("b") else int_size
    except ValueError:
        raise ValueError(err_msg)

    if mem_size < 0:
        raise ValueError(err_msg)
    return mem_size


def get_module_leaves(module_sizes):
    module_children = {}
    for module in module_sizes:
        if module == "" or "." not in module:
            continue
        parent = module.rsplit(".", 1)[0]
        module_children[parent] = module_children.get(parent, 0) + 1
    leaves = [module for module in module_sizes if module_children.get(module, 0) == 0 and module != ""]
    return leaves

def get_balanced_memory(
    model: nn.Module,
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
    no_split_module_classes: Optional[List[str]] = None,
    dtype: Optional[Union[str, mindspore.dtype.TensorType]] = None,
    special_dtypes: Optional[Dict[str, str]] = None,
    low_zero: bool = False,
):
    """
    Compute a `max_memory` dictionary for [`infer_auto_device_map`] that will balance the use of each available GPU.

    <Tip>

    All computation is done analyzing sizes and dtypes of the model parameters. As a result, the model can be on the
    meta device (as it would if initialized within the `init_empty_weights` context manager).

    </Tip>

    Args:
        model (`nn.Module`):
            The model to analyze.
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available if unset.
            Example: `max_memory={0: "1GB"}`.
        no_split_module_classes (`List[str]`, *optional*):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        dtype (`str` or `mindspore.dtype.TensorType`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        special_dtypes (`Dict[str, str]`, *optional*):
            If provided, special dtypes to consider for some specific weights (will override dtype used as default for
            all weights).
        low_zero (`bool`, *optional*):
            Minimizes the number of weights on GPU 0, which is convenient when it's used for other operations (like the
            Transformers generate function).
    """
    # Get default / clean up max_memory
    user_not_set_max_memory = max_memory is None
    max_memory = get_max_memory(max_memory)

    num_devices = len([d for d in max_memory if max_memory[d] > 0])

    if num_devices == 0:
        return max_memory

    if num_devices == 1:
        # We cannot do low_zero on just one GPU, but we will still reserve some memory for the buffer
        low_zero = False
        # If user just asked us to handle memory usage, we should avoid OOM
        if user_not_set_max_memory:
            for key in max_memory.keys():
                if isinstance(key, int):
                    max_memory[key] *= 0.9  # 90% is a good compromise
                    logger.info(
                        f"We will use 90% of the memory on device {key} for storing the model, and 10% for the buffer to avoid OOM. "
                        "You can set `max_memory` in to a higher value to use more memory (at your own risk)."
                    )
                    break  # only one device

    module_sizes = compute_module_sizes(model, dtype=dtype, special_dtypes=special_dtypes)
    per_gpu = module_sizes[""] // (num_devices - 1 if low_zero else num_devices)

    # We can't just set the memory to model_size // num_devices as it will end being too small: each GPU will get
    # slightly less layers and some layers will end up offload at the end. So this function computes a buffer size to
    # add which is the biggest of:
    # - the size of no split block (if applicable)
    # - the mean of the layer sizes
    if no_split_module_classes is None:
        no_split_module_classes = []
    elif not isinstance(no_split_module_classes, (list, tuple)):
        no_split_module_classes = [no_split_module_classes]

    # Identify the size of the no_split_block modules
    if len(no_split_module_classes) > 0:
        no_split_children = {}
        for name, size in module_sizes.items():
            if name == "":
                continue
            submodule = model
            for submodule_name in name.split("."):
                submodule = getattr(submodule, submodule_name)
            class_name = submodule.__class__.__name__
            if class_name in no_split_module_classes and class_name not in no_split_children:
                no_split_children[class_name] = size

            if set(no_split_children.keys()) == set(no_split_module_classes):
                break
        buffer = max(no_split_children.values()) if len(no_split_children) > 0 else 0
    else:
        buffer = 0

    # Compute mean of final modules. In the first dict of module sizes, leaves are the parameters
    leaves = get_module_leaves(module_sizes)
    module_sizes = {n: v for n, v in module_sizes.items() if n not in leaves}
    # Once removed, leaves are the final modules.
    leaves = get_module_leaves(module_sizes)
    mean_leaves = int(sum(module_sizes[n] for n in leaves) / max(len(leaves), 1))
    buffer = int(1.25 * max(buffer, mean_leaves))
    per_gpu += buffer

    # Sorted list of GPUs id (we may have some gpu ids not included in the our max_memory list - let's ignore them)
    gpus_idx_list = list(
        sorted(
            device_id for device_id, device_mem in max_memory.items() if isinstance(device_id, int) and device_mem > 0
        )
    )
    # The last device is left with max_memory just in case the buffer is not enough.
    for idx in gpus_idx_list[:-1]:
        max_memory[idx] = min(max_memory[0] if low_zero and idx == 0 else per_gpu, max_memory[idx])

    if low_zero:
        min_zero = max(0, module_sizes[""] - sum(max_memory[i] for i in range(1, num_devices)))
        max_memory[0] = min(min_zero, max_memory[0])

    return max_memory

def infer_auto_device_map(
    model: nn.Module,
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
    no_split_module_classes: Optional[List[str]] = None,
    dtype: Optional[Union[str, mindspore.dtype.TensorType]] = None,
    special_dtypes: Optional[Dict[str, str]] = None,
    verbose: bool = False,
    clean_result: bool = True,
    offload_buffers: bool = False,
):
    """
    Compute a device map for a given model giving priority to GPUs, then offload on CPU and finally offload to disk,
    such that:
    - we don't exceed the memory available of any of the GPU.
    - if offload to the CPU is needed, there is always room left on GPU 0 to put back the layer offloaded on CPU that
      has the largest size.
    - if offload to the CPU is needed,we don't exceed the RAM available on the CPU.
    - if offload to the disk is needed, there is always room left on the CPU to put back the layer offloaded on disk
      that has the largest size.

    <Tip>

    All computation is done analyzing sizes and dtypes of the model parameters. As a result, the model can be on the
    meta device (as it would if initialized within the `init_empty_weights` context manager).

    </Tip>

    Args:
        model (`nn.Module`):
            The model to analyze.
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available if unset.
            Example: `max_memory={0: "1GB"}`.
        no_split_module_classes (`List[str]`, *optional*):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        dtype (`str` or `mindspore.dtype.TensorType`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        special_dtypes (`Dict[str, str]`, *optional*):
            If provided, special dtypes to consider for some specific weights (will override dtype used as default for
            all weights).
        verbose (`bool`, *optional*, defaults to `False`):
            Whether or not to provide debugging statements as the function builds the device_map.
        clean_result (`bool`, *optional*, defaults to `True`):
            Clean the resulting device_map by grouping all submodules that go on the same device together.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            In the layers that are offloaded on the CPU or the hard drive, whether or not to offload the buffers as
            well as the parameters.
    """
    # Get default / clean up max_memory
    max_memory = get_max_memory(max_memory)
    if no_split_module_classes is None:
        no_split_module_classes = []
    elif not isinstance(no_split_module_classes, (list, tuple)):
        no_split_module_classes = [no_split_module_classes]

    devices = list(max_memory.keys())
    if "disk" not in devices:
        devices.append("disk")
    gpus = [device for device in devices if device not in ["cpu", "disk"]]

    # Devices that need to keep space for a potential offloaded layer.
    if "mps" in gpus:
        main_devices = ["mps"]
    elif len(gpus) > 0:
        main_devices = [gpus[0], "cpu"]
    else:
        main_devices = ["cpu"]

    module_sizes = compute_module_sizes(model, dtype=dtype, special_dtypes=special_dtypes)
    tied_parameters = find_tied_parameters(model)

    if check_tied_parameters_in_config(model) and len(tied_parameters) == 0:
        logger.warn(
            "The model weights are not tied. Please use the `tie_weights` method before using the `infer_auto_device` function."
        )

    device_map = OrderedDict()
    current_device = 0
    current_memory_used = 0
    device_memory_used = {}
    device_buffer_sizes = {}

    # Direct submodules and parameters
    modules_to_treat = (
        list(model.named_parameters(recurse=False))
        + list(model.named_children())
        + list(model.named_buffers(recurse=False))
    )
    # Initialize maximum largest layer, to know which space to keep in memory
    max_layer_size, max_layer_names = get_max_layer_size(modules_to_treat, module_sizes, no_split_module_classes)

    # Ready ? This is going to be a bit messy.
    while len(modules_to_treat) > 0:
        name, module = modules_to_treat.pop(0)
        if verbose:
            print(f"\nTreating module {name}.")
        # Max size in the remaining layers may have changed since we took one, so we maybe update it.
        max_layer_names = [n for n in max_layer_names if n != name and not n.startswith(name + ".")]
        if len(max_layer_names) == 0:
            max_layer_size, max_layer_names = get_max_layer_size(
                [(n, m) for n, m in modules_to_treat if isinstance(m, nn.Module)],
                module_sizes,
                no_split_module_classes,
            )
        # Assess size needed
        module_size = module_sizes[name]

        # We keep relevant tied parameters only: one of the tied parameters in the group is inside the current module
        # and the other is not.
        # Note: If we are currently processing the name `compute.weight`, an other parameter named e.g. `compute.weight_submodule.parameter`
        # needs to be considered outside the current module, hence the check with additional dots.
        tied_param_goups = [
            tied_group
            for tied_group in tied_parameters
            if any(name + "." in k + "." for k in tied_group) and not all(name + "." in k + "." for k in tied_group)
        ]

        if verbose and len(tied_param_goups) > 0:
            print(f"  Found the relevant tied param groups {tied_param_goups}")

        # Then we keep track of all the parameters that are tied to the current module, but not in the current module
        tied_params = sum(
            [[p for p in tied_group if name + "." not in p + "."] for tied_group in tied_param_goups], []
        )

        if verbose and len(tied_params) > 0:
            print(f"  So those parameters need to be taken into account {tied_params}")

        device = devices[current_device]
        current_max_size = max_memory[device] if device != "disk" else None
        current_memory_reserved = 0
        # Reduce max size available by the largest layer.
        if devices[current_device] in main_devices:
            current_max_size = current_max_size - max_layer_size
            current_memory_reserved = max_layer_size
        # Case 1 -> We're too big!
        if current_max_size is not None and current_memory_used + module_size > current_max_size:
            # Split or not split?
            modules_children = (
                []
                if isinstance(module, (mindspore.Tensor, nn.Parameter))
                else list(module.named_children())
            )
            if verbose:
                print(
                    f"Not enough space on {devices[current_device]} to put {name} (space available "
                    f"{current_max_size - current_memory_used}, module size {module_size})."
                )
            if len(modules_children) == 0 or module.__class__.__name__ in no_split_module_classes:
                # -> no split, we go to the next device
                if verbose:
                    print("This module cannot be split, going to the next device.")

                device_memory_used[device] = current_memory_used + current_memory_reserved
                current_device += 1
                modules_to_treat = [(name, module)] + modules_to_treat
                current_memory_used = 0
            else:
                # -> split, we replace the module studied by its children + parameters
                if verbose:
                    print(f"Splitting {name}.")
                modules_children = list(module.named_parameters(recurse=False)) + modules_children
                modules_to_treat = [(f"{name}.{n}", v) for n, v in modules_children] + modules_to_treat
                # Update the max layer size.
                max_layer_size, max_layer_names = get_max_layer_size(
                    [(n, m) for n, m in modules_to_treat if isinstance(m, nn.Module)],
                    module_sizes,
                    no_split_module_classes,
                )

        # Case 2, it fits! We're not entirely out of the wood though, because we may have some tied parameters.
        elif len(tied_params) > 0:
            # First locate all tied modules
            tied_module_names = []
            tied_modules = []
            for tied_param in tied_params:
                tied_module_index = [i for i, (n, _) in enumerate(modules_to_treat) if n in tied_param][0]
                tied_module_names.append(modules_to_treat[tied_module_index][0])
                tied_modules.append(modules_to_treat[tied_module_index][1])
            if verbose:
                print(
                    f"  It looks like {name} is going to fit on {devices[current_device]} but we have tied "
                    f"parameters to account for.\n  - Names {tied_params}\n  - Module names {tied_module_names}"
                )

            # Let's see if it all fits first
            module_size_with_ties = module_size
            for tied_param, tied_module_name in zip(tied_params, tied_module_names):
                module_size_with_ties += module_sizes[tied_module_name] - module_sizes[tied_param]

            if current_max_size is None or current_memory_used + module_size_with_ties <= current_max_size:
                # We really really fit!
                if verbose:
                    print(f"Putting {name} and {tied_module_names} on {devices[current_device]}.")
                current_memory_used += module_size_with_ties
                device_map[name] = devices[current_device]
                for tied_module_name in tied_module_names:
                    if tied_module_name in [m[0] for m in modules_to_treat]:
                        # The module may have been removed by a previous iteration of this loop.
                        tied_module_index = [i for i, (n, _) in enumerate(modules_to_treat) if n == tied_module_name][
                            0
                        ]
                        modules_to_treat.pop(tied_module_index)
                    device_map[tied_module_name] = devices[current_device]

                if not offload_buffers and isinstance(module, nn.Module):
                    current_buffer_size = compute_module_total_buffer_size(
                        module, dtype=dtype, special_dtypes=special_dtypes
                    )
                    device_buffer_sizes[device] = device_buffer_sizes.get(device, 0) + current_buffer_size

            else:
                # We don't fit with the tied modules. Next question is: can we split one of the tied modules to make it
                # smaller or do we need to go on the next device?
                if verbose:
                    print(
                        f"Not enough space on {devices[current_device]} to put {name} and {tied_module_names} (space "
                        f"available {current_max_size - current_memory_used}, needed size {module_size_with_ties})."
                    )
                split_happened = False
                for tied_module_name, tied_module in zip(tied_module_names, tied_modules):
                    tied_module_children = list(tied_module.named_children())
                    if len(tied_module_children) == 0 or tied_module.__class__.__name__ in no_split_module_classes:
                        # can't break this one.
                        continue

                    if verbose:
                        print(f"Splitting {tied_module_name}.")
                    tied_module_children = list(tied_module.named_parameters(recurse=False)) + tied_module_children
                    tied_module_children = [(f"{tied_module_name}.{n}", v) for n, v in tied_module_children]
                    tied_module_index = [i for i, (n, _) in enumerate(modules_to_treat) if n == tied_module_name][0]

                    modules_to_treat = (
                        [(name, module)]
                        + modules_to_treat[:tied_module_index]
                        + tied_module_children
                        + modules_to_treat[tied_module_index + 1 :]
                    )
                    # Update the max layer size.
                    max_layer_size, max_layer_names = get_max_layer_size(
                        [(n, m) for n, m in modules_to_treat if isinstance(m, nn.Module)],
                        module_sizes,
                        no_split_module_classes,
                    )
                    split_happened = True
                    break

                if not split_happened:
                    # If the tied module is not split, we go to the next device
                    if verbose:
                        print("None of the tied module can be split, going to the next device.")

                    device_memory_used[device] = current_memory_used + current_memory_reserved
                    current_device += 1
                    modules_to_treat = [(name, module)] + modules_to_treat
                    current_memory_used = 0

        else:
            if verbose:
                if current_max_size is None:
                    print(f"Putting {name} (size={module_size}) on {devices[current_device]}.")
                else:
                    print(
                        f"Putting {name} (size={module_size}) on {devices[current_device]} "
                        f"(available={current_max_size - current_memory_used})."
                    )
            current_memory_used += module_size
            device_memory_used[device] = current_memory_used + current_memory_reserved
            device_map[name] = devices[current_device]

            if not offload_buffers and isinstance(module, nn.Module):
                current_buffer_size = compute_module_total_buffer_size(
                    module, dtype=dtype, special_dtypes=special_dtypes
                )
                device_buffer_sizes[device] = device_buffer_sizes.get(device, 0) + current_buffer_size

    if clean_result:
        device_map = clean_device_map(device_map)

    non_gpu_buffer_size = device_buffer_sizes.get("cpu", 0) + device_buffer_sizes.get("disk", 0)
    if non_gpu_buffer_size > 0 and not offload_buffers:
        is_buffer_fit_any_gpu = False
        for gpu_device, gpu_max_memory in max_memory.items():
            if gpu_device in ('cpu', 'disk'):
                continue

            if not is_buffer_fit_any_gpu:
                gpu_memory_used = device_memory_used.get(gpu_device, 0)

                if gpu_max_memory >= non_gpu_buffer_size + gpu_memory_used:
                    is_buffer_fit_any_gpu = True

        if len(gpus) > 0 and not is_buffer_fit_any_gpu:
            warnings.warn(
                f"Current model requires {non_gpu_buffer_size} bytes of buffer for offloaded layers, which seems does "
                f"not fit any GPU's remaining memory. If you are experiencing a OOM later, please consider using "
                f"offload_buffers=True."
            )

    return device_map


def compute_module_sizes(
    model: nn.Module,
    dtype: Optional[str] = None,
    special_dtypes: Optional[Dict[str, str]] = None,
    buffers_only: bool = False,
):
    """
    Compute the size of each submodule of a given model.
    """
    if dtype is not None:
        dtype = _get_proper_dtype(dtype)
        dtype_size = dtype_byte_size(dtype)
    if special_dtypes is not None:
        special_dtypes = {key: _get_proper_dtype(dtyp) for key, dtyp in special_dtypes.items()}
        special_dtypes_size = {key: dtype_byte_size(dtyp) for key, dtyp in special_dtypes.items()}
    module_sizes = defaultdict(int)

    module_list = []

    if not buffers_only:
        module_list = named_module_tensors(model, recurse=True)
    else:
        module_list = model.named_buffers(recurse=True)

    for name, tensor in module_list:
        if special_dtypes is not None and name in special_dtypes:
            size = tensor.numel() * special_dtypes_size[name]
        elif dtype is None:
            size = tensor.numel() * dtype_byte_size(tensor.dtype)
        elif str(tensor.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            # According to the code in set_module_tensor_to_device, these types won't be converted
            # so use their original size here
            size = tensor.numel() * dtype_byte_size(tensor.dtype)
        else:
            size = tensor.numel() * min(dtype_size, dtype_byte_size(tensor.dtype))
        name_parts = name.split(".")
        for idx in range(len(name_parts) + 1):
            module_sizes[".".join(name_parts[:idx])] += size

    return module_sizes

def compute_module_total_buffer_size(
    model: nn.Module,
    dtype: Optional[str] = None,
    special_dtypes: Optional[Dict[str, str]] = None,
):
    """
    Compute the total size of buffers in each submodule of a given model.
    """
    module_sizes = compute_module_sizes(model, dtype=dtype, special_dtypes=special_dtypes, buffers_only=True)
    return module_sizes.get("", 0)


def get_max_layer_size(
    modules: List[Tuple[str, nn.Module]], module_sizes: Dict[str, int], no_split_module_classes: List[str]
):
    """
    Utility function that will scan a list of named modules and return the maximum size used by one full layer. The
    definition of a layer being:
    - a module with no direct children (just parameters and buffers)
    - a module whose class name is in the list `no_split_module_classes`

    Args:
        modules (`List[Tuple[str, nn.Module]]`):
            The list of named modules where we want to determine the maximum layer size.
        module_sizes (`Dict[str, int]`):
            A dictionary mapping each layer name to its size (as generated by `compute_module_sizes`).
        no_split_module_classes (`List[str]`):
            A list of class names for layers we don't want to be split.

    Returns:
        `Tuple[int, List[str]]`: The maximum size of a layer with the list of layer names realizing that maximum size.
    """
    max_size = 0
    layer_names = []
    modules_to_treat = modules.copy()
    while len(modules_to_treat) > 0:
        module_name, module = modules_to_treat.pop(0)
        modules_children = list(module.named_children()) if isinstance(module, nn.Module) else []
        if len(modules_children) == 0 or module.__class__.__name__ in no_split_module_classes:
            # No splitting this one so we compare to the max_size
            size = module_sizes[module_name]
            if size > max_size:
                max_size = size
                layer_names = [module_name]
            elif size == max_size:
                layer_names.append(module_name)
        else:
            modules_to_treat = [(f"{module_name}.{n}", v) for n, v in modules_children] + modules_to_treat
    return max_size, layer_names


def get_max_memory(max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None):
    """
    Get the maximum memory available if nothing is passed, converts string to int otherwise.
    """
    if max_memory is None:
        max_memory = {}
        try:
            group_size = get_group_size()
        except:
            group_size = 1
        device_target = mindspore.get_context('device_target')
        if device_target == 'GPU':
            devices_free_memory = get_gpus_free_memory()
            visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            export_command = "export CUDA_VISIBLE_DEVICES=0,1,2,3"
        elif device_target == 'Ascend':
            devices_free_memory = get_npus_free_memory()
            visible_devices = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', None)
            export_command = "export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3"

        if visible_devices is not None:
            visible_devices = visible_devices.split(',')
            if len(visible_devices) < group_size:
                raise RuntimeError(f'There are {group_size} process with only {len(visible_devices)} visible devices, '\
                                   f'please use `{export_command}` to set enough devices')
            if len(visible_devices) > group_size:
                visible_devices = visible_devices[:group_size]
        else:
            visible_devices = list(range(group_size))

        if group_size != len(visible_devices):
            raise ValueError('The number of process must be equal to visible devices, but got '
                             f'{group_size} process and {len(visible_devices)} devices.')
        for i in visible_devices:
            i = int(i)
            max_memory[i] = convert_file_size_to_int(f'{devices_free_memory[i][1]}MIB')

        return max_memory

    for key in max_memory:
        if isinstance(max_memory[key], str):
            max_memory[key] = convert_file_size_to_int(max_memory[key])

    # Need to sort the device by type to make sure that we allocate the gpu first.
    # As gpu/npu/xpu are represented by int, we need to sort them first.
    gpu_devices = [k for k in max_memory.keys() if isinstance(k, int)]
    gpu_devices.sort()
    # check if gpu/npu/xpu devices are available and if not, throw a warning
    num_devices = mindspore.hal.device_count()
    for device in gpu_devices:
        if device >= num_devices or device < 0:
            logger.warning(f"Device {device} is not available, available devices are {list(range(num_devices))}")
    # Add the other devices in the preset order if they are available
    all_devices = gpu_devices + [k for k in ["mps", "cpu", "disk"] if k in max_memory.keys()]
    # Raise an error if a device is not recognized
    for k in max_memory.keys():
        if k not in all_devices:
            raise ValueError(
                f"Device {k} is not recognized, available devices are integers(for GPU/XPU), 'mps', 'cpu' and 'disk'"
            )
    max_memory = {k: max_memory[k] for k in all_devices}

    return max_memory


def _get_param_device(param, device_map):
    if param in device_map:
        return device_map[param]
    parent_param = ".".join(param.split(".")[:-1])
    if parent_param == param:
        raise ValueError(f"The `device_map` does not contain the module {param}.")
    else:
        return _get_param_device(parent_param, device_map)


def check_tied_parameters_on_same_device(tied_params, device_map):
    """
    Check if tied parameters are on the same device

    Args:
        tied_params (`List[List[str]]`):
            A list of lists of parameter names being all tied together.

        device_map (`Dict[str, Union[int, str, torch.device]]`):
            A map that specifies where each submodule should go.

    """
    for tie_param in tied_params:
        tie_param_devices = {}
        for param in tie_param:
            tie_param_devices[param] = _get_param_device(param, device_map)
        if len(set(tie_param_devices.values())) > 1:
            logger.warn(
                f"Tied parameters are on different devices: {tie_param_devices}. "
                "Please modify your custom device map or set `device_map='auto'`. "
            )



def _get_named_modules(
    module: nn.Module,
    memo: Optional[Set[nn.Module]] = None,
    prefix: str = "",
    remove_duplicate: bool = True,
):
    """
    Return an iterator over all modules in the network, yielding both the name of the module as well as the module
    itself. Copied from PyTorch `nn.Module.named_modules` for compatability with torch < 2.0 versions with
    `remove_duplicate` option added.

    Args:
        memo (set of `nn.Module`, *optional*):
            A memo to store the set of modules already added to the result
        prefix (`str`, *optional*):
            A prefix that will be added to the name of the module
        remove_duplicate (`bool`, *optional*):
            Whether to remove the duplicated module instances in the result or not

    Yields:
        (str, Module): Tuple of name and module

    Note:
        Duplicate modules are returned only once. In the following example, ``l`` will be returned only once.
    """
    if memo is None:
        memo = set()
    if module not in memo:
        if remove_duplicate:
            memo.add(module)
        yield prefix, module
        for name, sub_module in module._modules.items():
            if module is None:
                continue
            submodule_prefix = prefix + ("." if prefix else "") + name
            yield from _get_named_modules(sub_module, memo, submodule_prefix, remove_duplicate)


def _get_named_parameters(module: nn.Module, prefix="", recurse=True, remove_duplicate: bool = True):
    """
    Help yield various names + members of modules. Copied from PyTorch `nn.Module.named_modules` for
    compatability with torch < 2.0 versions with `remove_duplicate` option added.
    """
    memo = set()
    modules = (
        _get_named_modules(module, prefix=prefix, remove_duplicate=remove_duplicate) if recurse else [(prefix, module)]
    )
    for module_prefix, module in modules:
        members = module._parameters.items()
        for k, v in members:
            if v is None or v in memo:
                continue
            if remove_duplicate:
                memo.add(v)
            name = module_prefix + ("." if module_prefix else "") + k
            yield name, v

def find_tied_parameters(model: nn.Module, **kwargs):
    """
    Find the tied parameters in a given model.

    <Tip warning={true}>

    The signature accepts keyword arguments, but they are for the recursive part of this function and you should ignore
    them.

    </Tip>

    Args:
        model (`nn.Module`): The model to inspect.

    Returns:
        List[List[str]]: A list of lists of parameter names being all tied together.

    Example:

    ```py
    >>> from collections import OrderedDict
    >>> import nn as nn

    >>> model = nn.Sequential(OrderedDict([("linear1", nn.Linear(4, 4)), ("linear2", nn.Linear(4, 4))]))
    >>> model.linear2.weight = model.linear1.weight
    >>> find_tied_parameters(model)
    [['linear1.weight', 'linear2.weight']]
    ```
    """

    # get ALL model parameters and thier names
    all_named_parameters = {name: param for name, param in _get_named_parameters(model, remove_duplicate=False)}

    # get ONLY unique named parameters,
    # if parameter is tied and have multiple names, it will be included only once
    no_duplicate_named_parameters = {
        name: param for name, param in _get_named_parameters(model, remove_duplicate=True)
    }

    # the difference of the two sets will give us the tied parameters
    tied_param_names = set(all_named_parameters.keys()) - set(no_duplicate_named_parameters.keys())

    # 'tied_param_names' contains the names of parameters that are tied in the model, but we do not know
    # which names refer to the same parameter. To identify this, we need to group them together.
    tied_param_groups = {}
    for tied_param_name in tied_param_names:
        tied_param = all_named_parameters[tied_param_name]
        for param_name, param in no_duplicate_named_parameters.items():
            # compare if parameters are the same, if so, group thier names together
            if param is tied_param:
                if param_name not in tied_param_groups:
                    tied_param_groups[param_name] = []
                tied_param_groups[param_name].append(tied_param_name)

    return FindTiedParametersResult([sorted([weight] + list(set(tied))) for weight, tied in tied_param_groups.items()])


class FindTiedParametersResult(list):
    """
    This is a subclass of a list to handle backward compatibility for Transformers. Do not rely on the fact this is not
    a list or on the `values` method as in the future this will be removed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def values(self):
        # TODO: at the next Transformers release (4.28.0) issue a deprecation warning here.
        return sum([x[1:] for x in self], [])


def check_tied_parameters_in_config(model: nn.Module):
    """
    Check if there is any indication in the given model that some weights should be tied.

    Args:
        model (`torch.nn.Module`): The model to inspect

    Returns:
        bool: True if the model needs to have tied weights
    """

    # based on model.tie_weights() method
    has_tied_word_embedding = False
    has_tied_encoder_decoder = False
    has_tied_module = False

    if "PreTrainedModel" in [c.__name__ for c in inspect.getmro(model.__class__)]:
        has_tied_word_embedding = (
            hasattr(model, "config")
            and getattr(model.config, "tie_word_embeddings", False)
            and model.get_output_embeddings()
        )
        has_tied_encoder_decoder = (
            hasattr(model, "config")
            and getattr(model.config, "is_encoder_decoder", False)
            and getattr(model.config, "tie_encoder_decoder", False)
        )
        has_tied_module = any(hasattr(module, "_tie_weights") for module in model.modules())

    return any([has_tied_word_embedding, has_tied_encoder_decoder, has_tied_module])


def retie_parameters(model, tied_params):
    """
    Reties tied parameters in a given model if the link was broken (for instance when adding hooks).

    Args:
        model (`nn.Module`):
            The model in which to retie parameters.
        tied_params (`List[List[str]]`):
            A mapping parameter name to tied parameter name as obtained by `find_tied_parameters`.
    """
    for tied_group in tied_params:
        param_to_tie = None
        # two loops : the first one to set param_to_tie , the second one to change the values of tied_group
        for param_name in tied_group:
            module = model
            splits = param_name.split(".")
            for split in splits[:-1]:
                module = getattr(module, split)
            param = getattr(module, splits[-1])
            if param_to_tie is None:
                param_to_tie = param
                break
        if param_to_tie is not None:
            for param_name in tied_group:
                module = model
                splits = param_name.split(".")
                for split in splits[:-1]:
                    module = getattr(module, split)
                setattr(module, splits[-1], param_to_tie)

def _get_proper_dtype(dtype):
    """
    Just does mindspore.dtype.TensorType(dtype) if necessary.
    """
    if isinstance(dtype, str):
        # We accept "mindspore.float16" or just "float16"
        dtype = dtype.replace("mindspore.", "")
        dtype = getattr(mindspore, dtype)
    return dtype


def named_module_tensors(
    module: nn.Module, include_buffers: bool = True, recurse: bool = False, remove_non_persistent: bool = False
):
    """
    A helper function that gathers all the tensors (parameters + buffers) of a given module. If `include_buffers=True`
    it's the same as doing `module.named_parameters(recurse=recurse) + module.named_buffers(recurse=recurse)`.

    Args:
        module (`nn.Module`):
            The module we want the tensors on.
        include_buffer (`bool`, *optional*, defaults to `True`):
            Whether or not to include the buffers in the result.
        recurse (`bool`, *optional`, defaults to `False`):
            Whether or not to go look in every submodule or just return the direct parameters and buffers.
        remove_non_persistent (`bool`, *optional*, defaults to `False`):
            Whether or not to remove the non persistent buffer from the buffers. Useful only when include_buffers =
            True
    """
    yield from module.named_parameters(recurse=recurse)

    if include_buffers:
        non_persistent_buffers = set()
        if remove_non_persistent:
            non_persistent_buffers = get_non_persistent_buffers(module, recurse=recurse)
        for named_buffer in module.named_buffers(recurse=recurse):
            name, _ = named_buffer
            if name not in non_persistent_buffers:
                yield named_buffer


def get_non_persistent_buffers(module: nn.Module, recurse: bool = False):
    """
    Gather all non persistent buffers of a given modules into a set

    Args:
        module (`nn.Module`):
            The module we want the non persistent buffers on.
        recurse (`bool`, *optional*, defaults to `False`):
            Whether or not to go look in every submodule or just return the direct non persistent buffers.
    """

    non_persistent_buffers_set = module._non_persistent_buffers_set
    if recurse:
        for _, m in module.named_modules():
            non_persistent_buffers_set |= m._non_persistent_buffers_set

    return non_persistent_buffers_set

def clean_device_map(device_map: Dict[str, Union[int, str]], module_name: str = ""):
    """
    Cleans a device_map by grouping all submodules that go on the same device together.
    """
    # Get the value of the current module and if there is only one split across several keys, regroup it.
    prefix = "" if module_name == "" else f"{module_name}."
    values = [v for k, v in device_map.items() if k.startswith(prefix)]
    if len(set(values)) == 1 and len(values) > 1:
        for k in [k for k in device_map if k.startswith(prefix)]:
            del device_map[k]
        device_map[module_name] = values[0]

    # Recurse over the children
    children_modules = [k for k in device_map.keys() if k.startswith(prefix) and len(k) > len(module_name)]
    idx = len(module_name.split(".")) + 1 if len(module_name) > 0 else 1
    children_modules = set(".".join(k.split(".")[:idx]) for k in children_modules)
    for child in children_modules:
        clean_device_map(device_map, module_name=child)

    return device_map

def modify_model_for_pp_infer(model: nn.Module, device_map, no_split_module_classes):
    current_device = get_rank()
    last_device = get_group_size() - 1
    reversed_device_map = {}
    for scope_name, device in device_map.items():
        if device not in reversed_device_map:
            reversed_device_map[device] = [scope_name]
        else:
            reversed_device_map[device].append(scope_name)

        if device != current_device:
            submodule = model.get_submodule(scope_name)
            if isinstance(submodule, nn.Embedding):
                new_embedding = EmbeddingIndentity(submodule.num_embeddings, submodule.embedding_dim, model.dtype)
                replace_submodule(model, scope_name, new_embedding)
            elif isinstance(submodule, nn.Linear):
                new_linear = LinearIndetity(submodule.in_features, submodule.out_features, model.dtype)
                replace_submodule(model, scope_name, new_linear)
            elif submodule.__class__.__name__ in no_split_module_classes:
                new_layer = DecoderLayerIdentity(submodule.self_attn.layer_idx, submodule.self_attn.config)
                replace_submodule(model, scope_name, new_layer)
            else:
                new_layer = nn.Identity()
                replace_submodule(model, scope_name, new_layer)

    if current_device < last_device:
        current_last_layer = model.get_submodule(reversed_device_map[current_device][-1])
        current_last_layer._forward = current_last_layer.forward
        current_last_layer.forward = types.MethodType(send_forward, current_last_layer)
        current_last_layer.dist = current_device + 1

    if current_device > 0:
        current_first_layer = model.get_submodule(reversed_device_map[current_device][0])
        current_first_layer._forward = current_first_layer.forward
        current_first_layer.forward = types.MethodType(receive_forward, current_first_layer)
        current_first_layer.src = current_device - 1

    model_last_layer = model.get_submodule(next(reversed(device_map)))
    model_last_layer._forward = model_last_layer.forward
    model_last_layer.forward = types.MethodType(broadcast_forward, model_last_layer)
    model_last_layer.src = last_device

    return model


def replace_submodule(model, submodule_path, new_module):
    parent_path, _, child_name = submodule_path.rpartition('.')

    parent_module = model.get_submodule(parent_path) if parent_path else model

    setattr(parent_module, child_name, new_module)

def send_forward(self, *args, **kwargs):
    output = self._forward(*args, **kwargs)
    isend(output[0], self.dist)
    return output

def receive_forward(self, *args, **kwargs):
    hidden_states = args[0]
    hidden_states = irecv(hidden_states, src=self.src)
    output = self._forward(*((hidden_states,) + args[1:]), **kwargs)
    return output

def broadcast_forward(self, *args, **kwargs):
    output = self._forward(*args, **kwargs)
    output = broadcast(output, src=self.src)
    return output


class DecoderLayerIdentity(nn.Module):
    def __init__(self, layer_idx, config):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_key_value_heads = config.num_key_value_heads

    def forward(self, *args, **kwargs):
        output_attentions = kwargs.get('output_attentions', False)
        use_cache = kwargs.get('use_cache', False)
        past_key_value = kwargs.get('past_key_value', None)
        hidden_states = args[0]
        bs, seq_len, _ = hidden_states.shape
        if output_attentions:
            logger.warning('`output_attentions` should set to `False` during multi-process inference.')

        if past_key_value is not None:
            past_key_value.update(
                ops.empty(bs, self.num_key_value_heads, seq_len, 0),
                ops.empty(bs, self.num_key_value_heads, seq_len, 0),
                self.layer_idx)

        output = (hidden_states,)

        if output_attentions:
            output = output + (None,)

        if use_cache:
            output = output + (past_key_value,)

        output = output + (None,)
        return output

class EmbeddingIndentity(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = dtype

    def forward(self, input):
        return ops.empty(input.shape + (self.embedding_dim,), dtype=self.dtype)

class LinearIndetity(nn.Module):
    def __init__(self, in_features, out_features, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

    def forward(self, input):
        return ops.empty(input.shape[:-1] + (self.out_features,), dtype=self.dtype)
