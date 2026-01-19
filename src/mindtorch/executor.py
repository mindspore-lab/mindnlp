import mindtorch
from ._apis import cpu, gpu, meta, numpy, npu_910a, npu_910b, npu_310b, npu_310p
from .configs import CPU_USE_NUMPY_OP, SOC, ENABLE_DISPATCH, DEVICE_TARGET, CAPTURE_INF_NAN

if SOC == 'ascend910':
    npu = npu_910a
elif SOC in ['ascend910b', 'ascend910_93']:
    npu = npu_910b
elif SOC in ['ascend310b', 'ascend310b1', 'ascend310b4']:
    npu = npu_310b
elif SOC == 'ascend310p':
    npu = npu_310p
else:
    npu = npu_910a

api_map = {
    'cpu': numpy if CPU_USE_NUMPY_OP else cpu,
    'npu': npu,
    'cuda': gpu,
    'meta': meta,
    'Ascend': npu,
    'GPU': gpu,
    'CPU': cpu,
}

# Normalize device type to lowercase for consistent tensor.init values
DEVICE_NORMALIZE = {
    'CPU': 'cpu',
    'GPU': 'cuda',
    'Ascend': 'npu',
    'Meta': 'meta',
}

def normalize_device(device_type):
    """Normalize device type to lowercase for consistent tensor.init values."""
    return DEVICE_NORMALIZE.get(device_type, device_type)

DISPATCH_WHITE_LIST = ['inplace_zero', 'inplace_fill_scalar']
SKIP_NAN_CHECK = ['empty', 'empty_like']

# Debug flag to trace function calls
DEBUG_TRACE = False
DEBUG_ALL_OPS = False  # Set True to trace all operations

def ensure_output(outs, device_type):
    # Normalize device_type to lowercase for consistent tensor.init values
    normalized_device = normalize_device(device_type)
    if isinstance(outs, (tuple, list)):
        for out in outs:
            if isinstance(out, mindtorch.Tensor):
                out.init = normalized_device

    else:
        outs.init = normalized_device
    return outs

if ENABLE_DISPATCH:
    def execute(func_name, *args, **kwargs):
        device_from_list = kwargs.pop('device_from_list', False)
        device_position = kwargs.pop('device_position', 0)
        device_type = kwargs.pop('device', None)

        if device_type is None:
            if device_from_list:
                device_type = args[0][0].init
            else:
                device_type = args[device_position].init

        if device_type is None:
            device_type = DEVICE_TARGET



        func = getattr(api_map[device_type], func_name, None)
        if func is None:
            raise RuntimeError(
                f"No implementation for function: {func_name} on {device_type}."
            )
        outs = func(*args, **kwargs)
        outs = ensure_output(outs, device_type)

        # Debug trace for function calls
        if DEBUG_TRACE and ('inplace' in func_name or DEBUG_ALL_OPS):
            # arg_shapes = [getattr(a, 'shape', None) for a in args[:3]]
            # print(f"[TRACE] {func_name} on {device_type}, shapes: {arg_shapes}")
            def hook(grad):
                print(func_name)
            if isinstance(outs, tuple):
                for out in outs:
                    try:
                        out.register_hook(hook)
                    except:
                        pass
            else:
                try:
                    outs.register_hook(hook)
                except:
                    pass

        if CAPTURE_INF_NAN:
            if func_name in SKIP_NAN_CHECK:
                pass
            else:
                isfinite_op = getattr(api_map[device_type], 'isfinite')
                if isinstance(outs, tuple):
                    for out in outs:
                        assert isfinite_op(out).asnumpy().all()
                else:
                    assert isfinite_op(outs).asnumpy().all()

        return outs
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

        # Debug trace for function calls
        if DEBUG_TRACE and ('inplace' in func_name or DEBUG_ALL_OPS):
            arg_shapes = [getattr(a, 'shape', None) for a in args[:3]]
            print(f"[TRACE] {func_name} on {device_type}, shapes: {arg_shapes}")

        func = getattr(api_map[device_type], func_name, None)
        if func is None:
            raise RuntimeError(
                f"No implementation for function: {func_name} on {device_type}."
            )
        if CAPTURE_INF_NAN:
            outs = func(*args, **kwargs)
            if func_name in SKIP_NAN_CHECK:
                return outs

            isfinite_op = getattr(api_map[device_type], 'isfinite')
            if isinstance(outs, tuple):
                for out in outs:
                    assert isfinite_op(out).asnumpy().all()
            else:
                assert isfinite_op(outs).asnumpy().all()
            return outs

        return func(*args, **kwargs)
