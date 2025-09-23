import mindtorch
from ._apis import npu, cpu, gpu, meta, numpy
from .configs import DEVICE_TARGET, cpu_use_numpy
from ._bind import is_autocast_enabled

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


api_map = {
    'cpu': numpy if cpu_use_numpy() else cpu,
    'npu': npu,
    'meta': meta,
    'cuda': gpu
}

class Dispatcher(metaclass=SingletonMeta):
    def dispatch(self, func_name, *args, **kwargs):
        device = kwargs.pop("device", None)
        if isinstance(device, str):
            device = mindtorch.device(device)

        if device is None:
            tensors = (
                [arg for arg in args[0] if mindtorch.is_tensor(arg)]
                if isinstance(args[0], (tuple, list))
                else [arg for arg in args if mindtorch.is_tensor(arg)]
            )

            if len(tensors) == 1:
                device = tensors[0].device

            else:
                devices = {tensor.device for tensor in tensors}
                if len(devices) > 1:
                    raise ValueError("All tensor arguments must be on the same device.")

                device = next(iter(devices), mindtorch.device("cpu"))

        if DEVICE_TARGET == "Ascend" and device.type == "cuda":
            device.type = "npu"

        device_type = device.type

        # func = self._registry[device_type].get(func_name, None)
        func = getattr(api_map[device_type], func_name, None)
        if func is None:
            raise RuntimeError(
                f"No implementation for function: {func_name} on {device_type}."
            )
        return func(*args, **kwargs), device


dispatcher = Dispatcher()

