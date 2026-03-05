class SchemaParam:
    def __init__(self, name, *, type_name=None, kw_only=False, default=None, has_default=False, mutates=False, alias_set=None):
        self.name = name
        self.type_name = type_name
        self.kw_only = kw_only
        self.default = default
        self.has_default = has_default
        self.mutates = mutates
        self.alias_set = alias_set


class SchemaReturn:
    def __init__(self, alias_set=None):
        self.alias_set = alias_set


def _torch_param_name(name):
    if name == "self":
        return "input"
    return name


def _parse_type_name(type_part):
    text = type_part.strip()
    if "(" in text:
        text = text.split("(", 1)[0].strip()
    text = text.replace("!", "").strip()
    if text.endswith("?"):
        text = text[:-1]
    if text.endswith("[]"):
        text = text[:-2]
    return text


class OpSchema:
    def __init__(self, schema):
        self.schema = schema
        self.name, self.params, self.returns = _parse_schema(schema)

    def bind(self, args, kwargs, *, op_name=None, error_overrides=None):
        name = op_name or self.name
        if kwargs is None:
            kwargs = {}
        params = self.params
        positional_params = [p for p in params if not p.kw_only]

        def _format_type(value):
            if hasattr(value, "device") and hasattr(value, "dtype"):
                return "Tensor"
            if isinstance(value, bool):
                return "bool"
            if isinstance(value, int):
                return "int"
            if isinstance(value, float):
                return "float"
            if isinstance(value, str):
                return "str"
            return type(value).__name__

        def _format_got():
            parts = []
            for value in args:
                parts.append(_format_type(value))
            for key, value in kwargs.items():
                parts.append(f"{key}={_format_type(value)}")
            if not parts:
                return "()"
            return f"({', '.join(parts)})"

        def _maybe_override(kind):
            if not error_overrides:
                return False
            template = error_overrides.get(kind)
            if template is None:
                return False
            message = template.format(name=name, got=_format_got())
            raise TypeError(message)
        if len(args) > len(positional_params):
            expected = len(positional_params)
            plural = "s" if expected != 1 else ""
            _maybe_override("too_many")
            raise TypeError(
                f"{name}() takes {expected} positional argument{plural} but {len(args)} were given"
            )
        for key in kwargs:
            if key == "device" and key not in {p.name for p in params}:
                _maybe_override("unexpected")
                raise TypeError(f"{name}() got an unexpected keyword argument '{key}'")
            if key not in {p.name for p in params}:
                _maybe_override("unexpected")
                raise TypeError(f"{name}() got an unexpected keyword argument '{key}'")
        for idx, value in enumerate(args):
            param = positional_params[idx]
            if param.name in kwargs:
                arg_name = _torch_param_name(param.name)
                _maybe_override("duplicate")
                raise TypeError(f"{name}() got multiple values for argument '{arg_name}'")
        provided = {p.name for p in positional_params[: len(args)]} | {k for k in kwargs.keys() if k != "device" or k in {p.name for p in params}}
        missing = [p.name for p in params if p.name not in provided and not p.has_default]
        if missing:
            _maybe_override("missing")
            missing_fmt = ", ".join(f'"{_torch_param_name(m)}"' for m in missing)
            raise TypeError(
                f"{name}() missing {len(missing)} required positional arguments: {missing_fmt}"
            )

        # Minimal type checks for high-frequency schema mismatches that must
        # match torch call-site validation paths.
        self._validate_types(args, kwargs, name=name, error_overrides=error_overrides, got=_format_got())
        return True

    def _validate_types(self, args, kwargs, *, name, error_overrides, got):
        if kwargs is None:
            kwargs = {}

        def _raise_invalid_combo():
            if error_overrides and error_overrides.get("unexpected") is not None:
                raise TypeError(error_overrides["unexpected"].format(name=name, got=got))
            raise TypeError(f"{name}() received an invalid combination of arguments - got {got}")

        params = self.params
        positional = [p for p in params if not p.kw_only]
        bound = {}
        for idx, value in enumerate(args):
            if idx < len(positional):
                bound[positional[idx].name] = value
        bound.update(kwargs)

        for param in params:
            if param.name not in bound:
                continue
            value = bound[param.name]
            ptype = getattr(param, "type_name", None)
            if ptype == "bool" and not isinstance(value, bool):
                _raise_invalid_combo()


def _parse_schema(schema):
    parts = schema.split("->", 1)
    sig = parts[0].strip()
    return_sig = parts[1].strip() if len(parts) > 1 else ""
    if "(" not in sig:
        return schema, [], _parse_returns(return_sig)
    name, params = sig.split("(", 1)
    params = params.rsplit(")", 1)[0].strip()
    if not params:
        return name.strip(), [], _parse_returns(return_sig)
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
        parsed_type_name = _parse_type_name(type_part)
        mutates = "!" in type_part
        alias_set = None
        start = type_part.find("(")
        end = type_part.find(")", start + 1) if start != -1 else -1
        if start != -1 and end != -1:
            alias_part = type_part[start + 1:end].replace("!", "").strip()
            if alias_part:
                alias_set = alias_part
        parsed.append(
            SchemaParam(
                name,
                type_name=parsed_type_name,
                kw_only=kw_only,
                default=default,
                has_default=has_default,
                mutates=mutates,
                alias_set=alias_set,
            )
        )
    return name.strip(), parsed, _parse_returns(return_sig)


def _parse_returns(return_sig):
    if not return_sig:
        return []
    text = return_sig.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    if not text:
        return []
    tokens = [t.strip() for t in text.split(",")]
    returns = []
    for token in tokens:
        alias_set = None
        start = token.find("(")
        end = token.find(")", start + 1) if start != -1 else -1
        if start != -1 and end != -1:
            alias_part = token[start + 1:end].replace("!", "").strip()
            if alias_part:
                alias_set = alias_part
        returns.append(SchemaReturn(alias_set=alias_set))
    return returns
