from ._apis import cpu, gpu, meta, numpy, npu_910a, npu_910b, npu_310b, npu_310p
from .configs import CPU_USE_NUMPY_OP, SOC, ENABLE_DISPATCH, DEVICE_TARGET

if SOC == 'ascend910':
    npu = npu_910a
elif SOC in ['ascend910b', 'ascend910_93']:
    npu = npu_910b
elif SOC in ['ascend310b', 'ascend310b1', 'ascend310b4']:
    npu = npu_310b
elif SOC == 'ascend310p':
    npu = npu_310p
else:
    raise ValueError(f'Unsupported SOC: {SOC}')

api_map = {
    'CPU': numpy if CPU_USE_NUMPY_OP else cpu,
    'Ascend': npu,
    'Meta': meta,
    'GPU': gpu,
    'cpu': numpy if CPU_USE_NUMPY_OP else cpu,
    'npu': npu,
    'cuda': gpu,
    'meta': meta,
}

DISPATCH_WHITE_LIST = ['inplace_zero', 'inplace_fill_scalar']

if ENABLE_DISPATCH:
    def execute(func_name, *args, **kwargs):
        device_from_list = kwargs.pop('device_from_list', False)
        device_position = kwargs.pop('device_position', 0)
        device_type = kwargs.pop('device', None)

        if device_type is None:
            if device_from_list:
                device_type = args[0][0]._device
            else:
                device_type = args[device_position]._device
            device_type = device_type[:device_type.find(':')] if ':' in device_type else device_type

        func = getattr(api_map[device_type], func_name, None)
        if func is None:
            raise RuntimeError(
                f"No implementation for function: {func_name} on {device_type}."
            )
        return func(*args, **kwargs)
else:
    def execute(func_name, *args, **kwargs):
        device_from_list = kwargs.pop('device_from_list', False)
        device_position = kwargs.pop('device_position', 0)
        device_type = kwargs.pop('device', None)

        # for device = meta
        if device_type is None:
            if device_from_list:
                device_type = args[0][0].init
            else:
                device_type = args[device_position].init

        if device_type is None or device_type not in ('meta', 'Meta'):
            device_type = DEVICE_TARGET

        func = getattr(api_map[device_type], func_name, None)
        if func is None:
            raise RuntimeError(
                f"No implementation for function: {func_name} on {device_type}."
            )
        return func(*args, **kwargs)
