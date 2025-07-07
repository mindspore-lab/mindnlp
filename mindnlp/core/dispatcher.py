from mindnlp import core
from mindnlp.core.types import device as device_
from mindnlp.core._prims import ascend, cpu

device_map = {
    'cpu': 'CPU',
    'npu': 'Ascend',
    'cuda': 'GPU'
}

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class Dispatcher(metaclass=SingletonMeta):
    def __init__(self):
        self._registry = {
            'cpu': {},
            'npu': {},
            'gpu': {}
        }

    def register(self, func_name, device, func):
        self._registry[device][func_name] = func

    def dispatch(self, func_name, *args, **kwargs):
        device = kwargs.pop('device', None)
        if isinstance(device, str):
            device = device_(device)

        if device is None:
            device = args[0].device

        func = self._registry[device.type].get(func_name, None)
        if func is None:
            raise RuntimeError(f"No implementation for function: {func_name} on {device.type}.")
        return func(*args), device

dispatcher = Dispatcher()
for func_name in ascend.__all__:
    dispatcher.register(func_name.replace('_npu', ''), 'npu', getattr(ascend, func_name))
for func_name in cpu.__all__:
    dispatcher.register(func_name.replace('_cpu', ''), 'cpu', getattr(cpu, func_name))
