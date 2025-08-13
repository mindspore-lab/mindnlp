def apply_chat_template_wrapper(fn):
    def wrapper(*args, **kwargs):
        return_tensors = kwargs.get('return_tensors', None)
        if return_tensors is not None and return_tensors == 'ms':
            kwargs['return_tensors'] = 'pt'
        return fn(*args, **kwargs)
    return wrapper
