from ..._dispatch import dispatch


def parameters_to_vector(parameters):
    params = list(parameters)
    if not params:
        raise ValueError("parameters is empty")
    flat = []
    for p in params:
        numel = 1
        for s in p.data.shape:
            numel *= s
        flat.append(p.data.reshape((numel,)))
    return dispatch("cat", flat[0].device.type, flat, 0)


def vector_to_parameters(vec, parameters):
    params = list(parameters)
    offset = 0
    for p in params:
        numel = 1
        for s in p.shape:
            numel *= s
        p.data = vec[offset:offset + numel].reshape(p.shape)
        offset += numel
