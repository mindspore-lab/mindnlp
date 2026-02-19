class SchemaParam:
    def __init__(self, name, *, kw_only=False, default=None, has_default=False, mutates=False):
        self.name = name
        self.kw_only = kw_only
        self.default = default
        self.has_default = has_default
        self.mutates = mutates


def _torch_param_name(name):
    if name == "self":
        return "input"
    return name


class OpSchema:
    def __init__(self, schema):
        self.schema = schema
        self.name, self.params = _parse_schema(schema)

    def bind(self, args, kwargs, *, op_name=None):
        name = op_name or self.name
        if kwargs is None:
            kwargs = {}
        params = self.params
        positional_params = [p for p in params if not p.kw_only]
        if len(args) > len(positional_params):
            expected = len(positional_params)
            plural = "s" if expected != 1 else ""
            raise TypeError(
                f"{name}() takes {expected} positional argument{plural} but {len(args)} were given"
            )
        for key in kwargs:
            if key not in {p.name for p in params}:
                raise TypeError(f"{name}() got an unexpected keyword argument '{key}'")
        for idx, value in enumerate(args):
            param = positional_params[idx]
            if param.name in kwargs:
                arg_name = _torch_param_name(param.name)
                raise TypeError(f"{name}() got multiple values for argument '{arg_name}'")
        provided = {p.name for p in positional_params[: len(args)]} | set(kwargs.keys())
        missing = [p.name for p in params if p.name not in provided and not p.has_default]
        if missing:
            missing_fmt = ", ".join(f'"{_torch_param_name(m)}"' for m in missing)
            raise TypeError(
                f"{name}() missing {len(missing)} required positional arguments: {missing_fmt}"
            )
        return True


def _parse_schema(schema):
    sig = schema.split("->", 1)[0].strip()
    if "(" not in sig:
        return schema, []
    name, params = sig.split("(", 1)
    params = params.rsplit(")", 1)[0].strip()
    if not params:
        return name.strip(), []
    tokens = [t.strip() for t in params.split(",")]
    kw_only = False
    parsed = []
    for token in tokens:
        if token == "*":
            kw_only = True
            continue
        default = None
        has_default = False
        if "=" in token:
            left, default = token.split("=", 1)
            left = left.strip()
            default = default.strip()
            has_default = True
        else:
            left = token
        parts = left.split()
        name = parts[-1]
        type_part = " ".join(parts[:-1])
        mutates = "!" in type_part
        parsed.append(
            SchemaParam(name, kw_only=kw_only, default=default, has_default=has_default, mutates=mutates)
        )
    return name.strip(), parsed
