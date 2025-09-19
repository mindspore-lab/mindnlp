from .. import _dtype

def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.

    Args:
        x (ndarray): Input array.
        shape:

    Returns:
        ndarray: Output array of the shape.
    """
    if x is None:
        return None
    ndim = len(shape)
    lead = x.dim() - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdim=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y

py2dtype = {
    bool: _dtype.bool,
    int: _dtype.int64,
    float: _dtype.float32,
}
