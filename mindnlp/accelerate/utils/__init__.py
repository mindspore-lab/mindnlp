"""accelerate utils"""
from .dataclasses import (
    DistributedType,
    MindFormersPlugin
)
from .environment import (
    str_to_bool
)
from .imports import (
    is_mindformers_available
)
from .modeling import (
    # calculate_maximum_sizes,
    # check_device_map,
    check_tied_parameters_in_config,
    check_tied_parameters_on_same_device,
    compute_module_sizes,
    convert_file_size_to_int,
    dtype_byte_size,
    find_tied_parameters,
    get_balanced_memory,
    get_max_layer_size,
    get_max_memory,
    # get_mixed_precision_context_manager,
    # id_tensor_storage,
    infer_auto_device_map,
    # is_peft_model,
    # load_checkpoint_in_model,
    # load_offloaded_weights,
    # load_state_dict,
    named_module_tensors,
    modify_model_for_pp_infer,
    find_usefull_files,
    # retie_parameters,
    # set_module_tensor_to_device,
    # shard_checkpoint,
)

from .other import (
    wait_for_everyone
)

from .mindformers import (
    MindFormersDummyDataLoader,
    MindFormersDummyScheduler
)

if is_mindformers_available():
    from .mindformers import (
        MindFormersEngine,
        MindFormersOptimizerWrapper,
        MindFormersSchedulerWrapper,
        initialize as mindformers_initialize,
        prepare_data_loader as mindformers_prepare_data_loader,
        prepare_model_optimizer_scheduler as mindformers_prepare_model_optimizer_scheduler
    )
