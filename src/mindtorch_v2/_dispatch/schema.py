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

        for idx, _ in enumerate(args):
            param = positional_params[idx]
            if param.name in kwargs:
                arg_name = _torch_param_name(param.name)
                _maybe_override("duplicate")
                raise TypeError(f"{name}() got multiple values for argument '{arg_name}'")

        provided = {p.name for p in positional_params[: len(args)]} | {
            k for k in kwargs.keys() if k != "device" or k in {p.name for p in params}
        }
        missing = [p.name for p in params if p.name not in provided and not p.has_default]
        if missing:
            _maybe_override("missing")
            missing_fmt = ", ".join(f'"{_torch_param_name(m)}"' for m in missing)
            raise TypeError(
                f"{name}() missing {len(missing)} required positional arguments: {missing_fmt}"
            )

        # Minimal type checks for high-frequency schema mismatches that must
        # match torch call-site validation paths.
        self._validate_types(
            args,
            kwargs,
            name=name,
            error_overrides=error_overrides,
            got=_format_got(),
        )
        return True

    def _validate_types(self, args, kwargs, *, name, error_overrides, got):
        if kwargs is None:
            kwargs = {}

        op_short_name = name.split("::", 1)[-1]

        def _raise_invalid_combo(extra=None):
            if error_overrides and error_overrides.get("unexpected") is not None:
                payload = {"name": name, "got": got}
                if extra:
                    payload.update(extra)
                raise TypeError(error_overrides["unexpected"].format(**payload))
            raise TypeError(f"{name}() received an invalid combination of arguments - got {got}")

        params = self.params
        positional = [p for p in params if not p.kw_only]
        bound = {}
        for idx, value in enumerate(args):
            if idx < len(positional):
                bound[positional[idx].name] = value
        bound.update(kwargs)

        def _raise_invalid_combo_with_got(custom_got, extra=None):
            if error_overrides and error_overrides.get("unexpected") is not None:
                payload = {"name": name, "got": custom_got}
                if extra:
                    payload.update(extra)
                raise TypeError(error_overrides["unexpected"].format(**payload))
            raise TypeError(f"{name}() received an invalid combination of arguments - got {custom_got}")

        def _dimname_not_found(name_str, input_tensor):
            rank = len(getattr(input_tensor, "shape", ()) or ())
            dimnames = ", ".join(["None"] * rank)
            return RuntimeError(f"Name '{name_str}' not found in Tensor[{dimnames}].")

        def _normalize_dim_index(dim, rank):
            if dim < 0:
                dim += rank
            return dim

        def _transpose_sigs(dim0, dim1):
            t0 = _type_label(dim0)
            t1 = _type_label(dim1)
            int_sig = f"int, !{t1}!" if isinstance(dim0, int) and not isinstance(dim0, bool) else f"!{t0}!, int"
            name_sig = f"!int!, !{t1}!" if isinstance(dim0, int) and not isinstance(dim0, bool) else f"!{t0}!, !int!"
            return int_sig, name_sig

        def _validate_sum_mean_dim(value, input_tensor):
            # Match torch call-site validation for sum/mean(dim=...).
            if value is None:
                return
            if isinstance(value, bool):
                _raise_invalid_combo()
            if isinstance(value, int):
                return
            if isinstance(value, str):
                if value.isidentifier():
                    raise _dimname_not_found(value, input_tensor)
                raise RuntimeError(
                    "Invalid name: a valid identifier contains only digits, alphabetical characters, "
                    f"and/or underscore and starts with a non-digit. got: '{value}'."
                )
            if isinstance(value, (list, tuple)):
                if not value:
                    return
                if len(value) == 1 and isinstance(value[0], str):
                    name_value = value[0]
                    if name_value.isidentifier():
                        raise _dimname_not_found(name_value, input_tensor)
                    raise RuntimeError(
                        "Invalid name: a valid identifier contains only digits, alphabetical characters, "
                        f"and/or underscore and starts with a non-digit. got: '{name_value}'."
                    )
                if isinstance(value[0], str):
                    for item in value[1:]:
                        if not (item is None or isinstance(item, str)):
                            item_type = type(item).__name__
                            raise TypeError(f"expected None or string for Dimname but got {item_type}")
                    # Let backend handle real dimname semantics.
                    return

                first = value[0]
                if isinstance(first, bool):
                    _raise_invalid_combo()

                rank = len(getattr(input_tensor, "shape", ()) or ())
                seen = set()
                normalized = []

                for item in value:
                    if not isinstance(item, int):
                        item_type = type(item).__name__
                        raise TypeError(
                            f"{name}(): argument 'dim' failed to unpack the object at pos 2 "
                            f"with error \"type must be tuple of ints,but got {item_type}\""
                        )
                    if isinstance(item, bool):
                        # bool inside dim sequence is treated as an integer dim value.
                        item = int(item)
                    dim_idx = _normalize_dim_index(item, rank)
                    if dim_idx in seen:
                        raise RuntimeError(f"dim {dim_idx} appears multiple times in the list of dims")
                    seen.add(dim_idx)
                    normalized.append(item)

                # Mutate kwargs so backend kernels receive integer dims for bool entries.
                if isinstance(value, list):
                    value[:] = normalized
                elif "dim" in kwargs:
                    kwargs["dim"] = tuple(normalized)
                    bound["dim"] = kwargs["dim"]
                return
            _raise_invalid_combo()

        def _validate_prod_dim(value, input_tensor):
            # Match torch call-site validation for prod(dim=...).
            if value is None:
                return
            if isinstance(value, bool):
                _raise_invalid_combo()
            if isinstance(value, int):
                return
            if isinstance(value, str):
                if value.isidentifier():
                    raise _dimname_not_found(value, input_tensor)
                raise RuntimeError(
                    "Invalid name: a valid identifier contains only digits, alphabetical characters, "
                    f"and/or underscore and starts with a non-digit. got: '{value}'."
                )
            _raise_invalid_combo()

        def _validate_norm_dim(value):
            # Match torch call-site type validation for linalg_vector_norm dim.
            if value is None:
                return
            if isinstance(value, int) and not isinstance(value, bool):
                return
            if isinstance(value, bool):
                raise TypeError(
                    "linalg_vector_norm(): argument 'dim' (position 3) must be tuple of ints, "
                    "but found element of type bool at pos 0"
                )
            if isinstance(value, (list, tuple)):
                if not value:
                    return
                for idx, item in enumerate(value):
                    if not isinstance(item, int) or isinstance(item, bool):
                        item_type = type(item).__name__
                        if idx == 0:
                            raise TypeError(
                                "linalg_vector_norm(): argument 'dim' (position 3) must be tuple of ints, "
                                f"but found element of type {item_type} at pos 0"
                            )
                        raise TypeError(
                            "linalg_vector_norm(): argument 'dim' failed to unpack the object at pos 2 "
                            f"with error \"type must be tuple of ints,but got {item_type}\""
                        )
                return
            raise TypeError(
                "linalg_vector_norm(): argument 'dim' (position 3) must be tuple of ints, "
                f"not {type(value).__name__}"
            )

        def _validate_arg_reduce_dim(value):
            if value is None:
                return
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{op_short_name}(): argument 'dim' must be int, not {type(value).__name__}"
                )

        def _invalid_dimname(value):
            raise RuntimeError(
                "Invalid name: a valid identifier contains only digits, alphabetical characters, "
                f"and/or underscore and starts with a non-digit. got: '{value}'."
            )

        def _validate_all_any_dim(value):
            if value is None:
                return
            if isinstance(value, bool):
                _raise_invalid_combo_with_got("(Tensor, dim=bool)", {"dim_detail": "bool"})
                return
            if isinstance(value, int):
                return
            if isinstance(value, str):
                if value.isidentifier():
                    raise RuntimeError(
                        f"{op_short_name}: You passed a dimname (string) to this op in place of a dimension "
                        "index but it does not yet support this behavior. Please pass a dimension index to "
                        "work around this."
                    )
                _invalid_dimname(value)
                return
            if isinstance(value, (list, tuple)):
                if not value:
                    return
                first = value[0]
                if isinstance(first, (bool, str)):
                    _raise_invalid_combo_with_got("(Tensor, dim=list)", {"dim_detail": "list"})
                    return
                for item in value:
                    if not isinstance(item, int):
                        _raise_invalid_combo_with_got("(Tensor, dim=list)", {"dim_detail": "list"})
                        return
                return
            _raise_invalid_combo()

        def _validate_count_nonzero_dim(value):
            if value is None:
                return
            if isinstance(value, bool):
                _raise_invalid_combo_with_got("(Tensor, dim=bool)", {"dim_detail": "bool"})
                return
            if isinstance(value, str):
                _raise_invalid_combo_with_got("(Tensor, dim=str)", {"dim_detail": "str"})
                return
            if isinstance(value, int):
                return
            if isinstance(value, (list, tuple)):
                if not value:
                    return
                for item in value:
                    if not isinstance(item, int) or isinstance(item, bool):
                        _raise_invalid_combo_with_got("(Tensor, dim=list)", {"dim_detail": "list"})
                        return
                return
            _raise_invalid_combo()

        def _validate_std_var_dim(value):
            if value is None:
                return
            if isinstance(value, bool):
                _raise_invalid_combo_with_got("(Tensor, dim=bool)")
                return
            if isinstance(value, str):
                if value.isidentifier():
                    raise _dimname_not_found(value, bound.get("input"))
                raise RuntimeError(
                    "Invalid name: a valid identifier contains only digits, alphabetical characters, "
                    f"and/or underscore and starts with a non-digit. got: '{value}'."
                )

        def _validate_nan_reduction_dim(value):
            if value is None:
                return
            if isinstance(value, int) and not isinstance(value, bool):
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    if not isinstance(item, int) or isinstance(item, bool):
                        raise TypeError(
                            f"{op_short_name}(): argument 'dim' must be tuple of ints, not {type(item).__name__}"
                        )
                return
            raise TypeError(
                f"{op_short_name}(): argument 'dim' must be tuple of ints, not {type(value).__name__}"
            )

        def _type_label(value):
            if isinstance(value, bool):
                return "bool"
            if isinstance(value, int):
                return "int"
            if isinstance(value, float):
                return "float"
            if isinstance(value, str):
                return "str"
            if value is None:
                return "NoneType"
            if isinstance(value, list):
                if value:
                    inner = ", ".join(type(v).__name__ for v in value)
                    return f"list of [{inner}]"
                return "list"
            if isinstance(value, tuple):
                if value:
                    inner = ", ".join(type(v).__name__ for v in value)
                    return f"tuple of ({inner},)"
                return "tuple"
            return type(value).__name__

        def _validate_view_shape(value):
            if isinstance(value, bool):
                _raise_invalid_combo_with_got("(bool)", {"detail": "bool"})
                return
            if isinstance(value, int):
                return
            if isinstance(value, list):
                if all(isinstance(v, int) and not isinstance(v, bool) for v in value):
                    return
                if any(isinstance(v, str) for v in value):
                    raise TypeError(
                        f"{name}(): argument 'size' failed to unpack the object at pos 2 "
                        f"with error \"type must be tuple of ints,but got str\""
                    )
                _raise_invalid_combo_with_got("(list)", {"detail": _type_label(value)})
                return
            if isinstance(value, tuple):
                if all(isinstance(v, int) and not isinstance(v, bool) for v in value):
                    return
                if any(isinstance(v, str) for v in value):
                    raise TypeError(
                        f"{name}(): argument 'size' failed to unpack the object at pos 2 "
                        f"with error \"type must be tuple of ints,but got str\""
                    )
                _raise_invalid_combo_with_got("(tuple)", {"detail": _type_label(value)})
                return
            if isinstance(value, str):
                _raise_invalid_combo_with_got("(str)", {"detail": "str"})
                return
            if isinstance(value, float):
                _raise_invalid_combo_with_got("(float)", {"detail": "float"})
                return
            _raise_invalid_combo_with_got(f"({_type_label(value)})", {"detail": _type_label(value)})

        def _rank_of_input(input_tensor):
            return len(getattr(input_tensor, "shape", ()) or ())

        def _squeeze_dim_out_of_range(dim_value, rank):
            return IndexError(
                f"Dimension out of range (expected to be in range of [{-rank}, {rank - 1}], but got {dim_value})"
            )

        def _unsqueeze_dim_out_of_range(dim_value, rank):
            return IndexError(
                f"Dimension out of range (expected to be in range of [{-rank - 1}, {rank}], but got {dim_value})"
            )

        def _squeeze_invalid_combo(got_text, detail_text):
            raise TypeError(
                f"{name}() received an invalid combination of arguments - got {got_text}, but expected one of:\n"
                " * (Tensor input)\n"
                " * (Tensor input, int dim)\n"
                f"      didn't match because some of the arguments have invalid types: (Tensor, !{detail_text}!)\n"
                " * (Tensor input, tuple of ints dim)\n"
                f"      didn't match because some of the arguments have invalid types: (Tensor, !{detail_text}!)\n"
                " * (Tensor input, name dim)\n"
                f"      didn't match because some of the arguments have invalid types: (Tensor, !{detail_text}!)\n"
            )

        def _validate_squeeze_dim(value, input_tensor):
            rank = _rank_of_input(input_tensor)
            if isinstance(value, bool):
                _squeeze_invalid_combo("(Tensor, bool)", "bool")
            if isinstance(value, int):
                if value < -rank or value > rank - 1:
                    raise _squeeze_dim_out_of_range(value, rank)
                return
            if isinstance(value, str):
                if value.isidentifier():
                    raise _dimname_not_found(value, input_tensor)
                raise RuntimeError(
                    "Invalid name: a valid identifier contains only digits, alphabetical characters, "
                    f"and/or underscore and starts with a non-digit. got: '{value}'."
                )
            if value is None:
                raise RuntimeError("Please look up dimensions by name, got: name = None.")
            if isinstance(value, float):
                _squeeze_invalid_combo("(Tensor, float)", "float")
            if isinstance(value, (list, tuple)):
                if not value:
                    return
                seq_kind = "list" if isinstance(value, list) else "tuple"
                seq_types = ", ".join(type(v).__name__ for v in value)
                if isinstance(value[0], bool):
                    _squeeze_invalid_combo(f"(Tensor, {seq_kind})", f"{seq_kind} of [{seq_types}]")
                seen = set()
                for item in value:
                    if not isinstance(item, int) or isinstance(item, bool):
                        item_type = type(item).__name__
                        if seq_kind == "list":
                            _squeeze_invalid_combo(f"(Tensor, list)", f"list of [{seq_types}]")
                        raise TypeError(
                            f"{name}(): argument 'dim' failed to unpack the object at pos 2 "
                            f"with error \"type must be tuple of ints,but got {item_type}\""
                        )
                    norm = _normalize_dim_index(item, rank)
                    if norm < 0 or norm >= rank:
                        raise _squeeze_dim_out_of_range(item, rank)
                    if norm in seen:
                        raise RuntimeError(f"dim {norm} appears multiple times in the list of dims")
                    seen.add(norm)
                return
            _squeeze_invalid_combo(f"(Tensor, {_type_label(value)})", _type_label(value))

        def _validate_unsqueeze_dim(value, input_tensor):
            rank = _rank_of_input(input_tensor)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{name}(): argument 'dim' (position 2) must be int, not {type(value).__name__}"
                )
            if value < -rank - 1 or value > rank:
                raise _unsqueeze_dim_out_of_range(value, rank)

        def _validate_topk_k(value):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{name}(): argument 'k' (position 2) must be int, not {type(value).__name__}"
                )

        def _validate_topk_dim(value):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{name}(): argument 'dim' (position 3) must be int, not {type(value).__name__}"
                )

        def _validate_cum_dim(value, input_tensor):
            if isinstance(value, bool):
                _raise_invalid_combo_with_got("(Tensor, bool)")
            if isinstance(value, int):
                return
            if isinstance(value, str):
                if value.isidentifier():
                    raise _dimname_not_found(value, input_tensor)
                raise RuntimeError(
                    "Invalid name: a valid identifier contains only digits, alphabetical characters, "
                    f"and/or underscore and starts with a non-digit. got: '{value}'."
                )
            _raise_invalid_combo_with_got(f"(Tensor, {_type_label(value)})")

        def _validate_sort_dim(value, input_tensor):
            if isinstance(value, bool):
                _raise_invalid_combo_with_got("(Tensor, bool)")
            if isinstance(value, int):
                return
            if isinstance(value, str):
                if value.isidentifier():
                    raise _dimname_not_found(value, input_tensor)
                raise RuntimeError(
                    "Invalid name: a valid identifier contains only digits, alphabetical characters, "
                    f"and/or underscore and starts with a non-digit. got: '{value}'."
                )
            _raise_invalid_combo_with_got(f"(Tensor, {_type_label(value)})")

        def _normalize_permute_dims(value):
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, list):
                return value
            raise TypeError(
                f"{name}(): argument 'dims' (position 2) must be tuple of ints, not {type(value).__name__}"
            )

        def _validate_permute_dims(value, input_tensor):
            rank = _rank_of_input(input_tensor)
            dims = _normalize_permute_dims(value)
            if dims and isinstance(dims[0], bool):
                raise TypeError(
                    "permute(): argument 'dims' (position 2) must be tuple of ints, "
                    "but found element of type bool at pos 0"
                )
            if not dims:
                if rank != 0:
                    raise RuntimeError(
                        "permute(sparse_coo): number of dimensions in the tensor input does not match "
                        "the length of the desired ordering of dimensions i.e. "
                        f"input.dim() = {rank} is not equal to len(dims) = 0"
                    )
                return
            if len(dims) != rank:
                raise RuntimeError(
                    "permute(sparse_coo): number of dimensions in the tensor input does not match "
                    "the length of the desired ordering of dimensions i.e. "
                    f"input.dim() = {rank} is not equal to len(dims) = {len(dims)}"
                )
            seen = set()
            for idx, item in enumerate(dims):
                if not isinstance(item, int) or isinstance(item, bool):
                    item_type = type(item).__name__
                    if idx == 0:
                        raise TypeError(
                            "permute(): argument 'dims' (position 2) must be tuple of ints, "
                            f"but found element of type {item_type} at pos 0"
                        )
                    raise TypeError(
                        f"permute(): argument 'dims' failed to unpack the object at pos 2 "
                        f"with error \"type must be tuple of ints,but got {item_type}\""
                    )
                if item < -rank or item > rank - 1:
                    raise IndexError(
                        f"Dimension out of range (expected to be in range of [{-rank}, {rank - 1}], but got {item})"
                    )
                norm = _normalize_dim_index(item, rank)
                if norm in seen:
                    raise RuntimeError("permute(): duplicate dims are not allowed.")
                seen.add(norm)

        def _validate_transpose_dims(dim0, dim1):
            valid0 = isinstance(dim0, int) and not isinstance(dim0, bool)
            valid1 = isinstance(dim1, int) and not isinstance(dim1, bool)
            if valid0 and valid1:
                return
            got_text = f"({_type_label(dim0)}, {_type_label(dim1)})"
            int_sig, name_sig = _transpose_sigs(dim0, dim1)
            _raise_invalid_combo_with_got(
                got_text,
                {
                    "transpose_int_sig": int_sig,
                    "transpose_name_sig": name_sig,
                },
            )

        for param in params:
            if param.name not in bound:
                continue
            value = bound[param.name]
            ptype = getattr(param, "type_name", None)
            if op_short_name == "sum" and param.name == "dim":
                input_tensor = bound.get("input")
                _validate_sum_mean_dim(value, input_tensor)
                continue
            if op_short_name == "mean" and param.name == "dim":
                input_tensor = bound.get("input")
                _validate_sum_mean_dim(value, input_tensor)
                continue
            if op_short_name == "prod" and param.name == "dim":
                input_tensor = bound.get("input")
                _validate_prod_dim(value, input_tensor)
                continue
            if op_short_name == "norm" and param.name == "dim":
                _validate_norm_dim(value)
                continue
            if op_short_name in {"argmax", "argmin"} and param.name == "dim":
                _validate_arg_reduce_dim(value)
                continue
            if op_short_name in {"all", "any"} and param.name == "dim":
                _validate_all_any_dim(value)
                continue
            if op_short_name == "count_nonzero" and param.name == "dim":
                _validate_count_nonzero_dim(value)
                continue
            if op_short_name in {"std", "var"} and param.name == "dim":
                _validate_std_var_dim(value)
                continue
            if op_short_name in {"nansum", "nanmean"} and param.name == "dim":
                _validate_nan_reduction_dim(value)
                continue
            if op_short_name == "view" and param.name == "shape":
                _validate_view_shape(value)
                continue
            if op_short_name == "squeeze" and param.name == "dim":
                input_tensor = bound.get("input")
                _validate_squeeze_dim(value, input_tensor)
                continue
            if op_short_name == "unsqueeze" and param.name == "dim":
                input_tensor = bound.get("input")
                _validate_unsqueeze_dim(value, input_tensor)
                continue
            if op_short_name == "permute" and param.name == "dims":
                input_tensor = bound.get("input")
                _validate_permute_dims(value, input_tensor)
                continue
            if op_short_name == "topk" and param.name == "k":
                _validate_topk_k(value)
                continue
            if op_short_name == "topk" and param.name == "dim":
                _validate_topk_dim(value)
                continue
            if op_short_name in {"cumsum", "cumprod"} and param.name == "dim":
                input_tensor = bound.get("input")
                _validate_cum_dim(value, input_tensor)
                continue
            if op_short_name in {"argsort", "sort"} and param.name == "dim":
                input_tensor = bound.get("input")
                _validate_sort_dim(value, input_tensor)
                continue
            if ptype == "bool" and not isinstance(value, bool):
                _raise_invalid_combo()

        if op_short_name == "transpose" and "dim0" in bound and "dim1" in bound:
            _validate_transpose_dims(bound["dim0"], bound["dim1"])


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
