from mindnlp import core
from .types import device as device_
from ._prims import ascend, cpu, numpy, meta, ascend_310b
from .configs import DEVICE_TARGET, CPU_USE_NUMPY_OP, SOC

device_map = {"cpu": "CPU", "npu": "Ascend", "cuda": "GPU"}


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Dispatcher(metaclass=SingletonMeta):
    def __init__(self):
        self._registry = {"cpu": {}, "npu": {}, "gpu": {}, 'numpy': {}, 'meta': {}}

    def register(self, func_name, device, func):
        self._registry[device][func_name] = func

    def dispatch(self, func_name, *args, **kwargs):
        device = kwargs.pop("device", None)
        if isinstance(device, str):
            device = device_(device)

        if device is None:
            tensors = (
                [arg for arg in args[0] if core.is_tensor(arg)]
                if isinstance(args[0], (tuple, list))
                else [arg for arg in args if core.is_tensor(arg)]
            )

            if len(tensors) == 1:
                device = tensors[0].device

            else:
                devices = {tensor.device for tensor in tensors}

                if len(devices) > 1:
                    raise ValueError("All tensor arguments must be on the same device.")

                device = next(iter(devices), device_("cpu"))

        if DEVICE_TARGET == 'Ascend' and device.type == 'cuda':
            device.type = 'npu'

        device_type = device.type

        if CPU_USE_NUMPY_OP and device_type == 'cpu':
            device_type = 'numpy'

        func = self._registry[device_type].get(func_name, None)
        if func is None:
            raise RuntimeError(
                f"No implementation for function: {func_name} on {device_type}."
            )
        return func(*args), device


dispatcher = Dispatcher()
if SOC == 'ascend310b':
    for func_name in ascend_310b.__all__:
        dispatcher.register(func_name, "npu", getattr(ascend_310b, func_name))
else:
    for func_name in ascend.__all__:
        dispatcher.register(func_name, "npu", getattr(ascend, func_name))

for func_name in cpu.__all__:
    dispatcher.register(func_name, "cpu", getattr(cpu, func_name))

for func_name in numpy.__all__:
    dispatcher.register(func_name, "numpy", getattr(numpy, func_name))

for func_name in meta.__all__:
    dispatcher.register(func_name, "meta", getattr(meta, func_name))

