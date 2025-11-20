from ._apis import cpu, gpu, meta, numpy, npu_910a, npu_910b, npu_310b, npu_310p
from .configs import cpu_use_numpy, SOC, ENABLE_DISPATCH, DEVICE_TARGET

if SOC == 'ascend910':
    npu = npu_910a
elif SOC in ['ascend910b', 'ascend910_93']:
    npu = npu_910b
elif SOC in ['ascend310b1', 'ascend310b4']:
    npu = npu_310b
elif SOC == 'ascend310p':
    npu = npu_310p
else:
    raise ValueError(f'Unsupported SOC: {SOC}')

api_map = {
    'CPU': numpy if cpu_use_numpy() else cpu,
    'Ascend': npu,
    'Ascend:0': npu,
    'Meta': meta,
    'GPU': gpu,
    'cpu': numpy if cpu_use_numpy() else cpu,
    'npu': npu,
    'cuda': gpu,
    'meta': meta,
}


def execute(func_name, *args, **kwargs):
    device_from_list = kwargs.pop('device_from_list', False)
    device_position = kwargs.pop('device_position', 0)
    device_type = kwargs.pop('device', None)

    if device_type is None:
        if ENABLE_DISPATCH or 'inplace' in func_name:
            if device_from_list:
                device_type = args[0][0]._device
            else:
                device_type = args[device_position]._device
        else:
            device_type = DEVICE_TARGET

    # # func = self._registry[device_type].get(func_name, None)
    func = getattr(api_map[device_type], func_name, None)
    # func = getattr(api_map['Ascend'], func_name, None)
    if func is None:
        raise RuntimeError(
            f"No implementation for function: {func_name} on {device_type}."
        )
    return func(*args, **kwargs)
