"""Minimal distributed checkpoint state_dict helpers for mindtorch_v2.

This MVP intentionally provides a narrow surface:
- returns model and optimizer state dictionaries
- restores model and optimizer state dictionaries

It is designed for single-node DDP training recovery flows.
"""

def _unwrap_model(model):
    # DDP/DataParallel expose the wrapped module at `.module`.
    return getattr(model, "module", model)


def get_state_dict(
    model,
    optimizer=None,
    *,
    rank0_only=False,
    rank=None,
    as_payload=False,
):
    """Return model and optimizer state dicts.

    Args:
        model: module or parallel wrapper exposing ``state_dict``.
        optimizer: optional optimizer exposing ``state_dict``.
        rank0_only: if True, return payload only for rank 0.
        rank: optional explicit rank used with ``rank0_only``.
        as_payload: if True, return a single checkpoint payload dict.

    Returns:
        Tuple[dict, dict|None]: ``(model_state_dict, optim_state_dict)``.
    """
    if rank0_only:
        current_rank = 0 if rank is None else int(rank)
        if current_rank != 0:
            return None if as_payload else (None, None)
    else:
        current_rank = 0 if rank is None else int(rank)

    base_model = _unwrap_model(model)
    model_state = base_model.state_dict()
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if as_payload:
        return {
            "model": model_state,
            "optim": optim_state,
            "meta": {
                "rank0_only": bool(rank0_only),
                "rank": int(current_rank),
            },
        }
    return model_state, optim_state


def set_state_dict(
    model,
    optimizer=None,
    *,
    payload=None,
    model_state_dict=None,
    optim_state_dict=None,
    strict=True,
    allow_partial_optimizer=False,
):
    """Restore model and optimizer state dicts.

    Args:
        model: module or parallel wrapper exposing ``load_state_dict``.
        optimizer: optional optimizer exposing ``load_state_dict``.
        model_state_dict: state dict for model.
        optim_state_dict: optional state dict for optimizer.
        strict: forwarded to ``model.load_state_dict``.
    """
    if payload is not None:
        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError("payload.meta must be a dict")
        if not isinstance(meta.get("rank"), int):
            raise ValueError("payload.meta.rank must be an int")
        model_state_dict = payload.get("model")
        if optim_state_dict is None:
            optim_state_dict = payload.get("optim")

    if model_state_dict is None:
        if payload is None and not strict:
            return {
                "missing_keys": [],
                "unexpected_keys": [],
                "loaded_keys_count": 0,
                "restored_optimizer_state_keys_count": 0,
            }
        raise ValueError("model_state_dict must not be None")

    base_model = _unwrap_model(model)
    expected_keys = set(base_model.state_dict().keys())
    provided_keys = set(model_state_dict.keys())
    incompatible = base_model.load_state_dict(model_state_dict, strict=strict)
    restored_optimizer_state_keys_count = 0
    if optimizer is not None and optim_state_dict is not None:
        # Safe optimizer restore: preserve runtime-owned parameter objects and
        # copy only serializable group options/state.
        loaded_groups = optim_state_dict.get("param_groups", [])
        if (not allow_partial_optimizer) and len(loaded_groups) != len(optimizer.param_groups):
            raise ValueError("optimizer param_groups length mismatch")
        if len(loaded_groups) == len(optimizer.param_groups):
            for group, loaded in zip(optimizer.param_groups, loaded_groups):
                for k, v in loaded.items():
                    if k not in ("params", "param_ids"):
                        group[k] = v
        loaded_state = optim_state_dict.get("state", {})
        optimizer.state = {}
        for k, v in loaded_state.items():
            try:
                key = int(k)
            except Exception:
                key = k
            optimizer.state[key] = v
        restored_optimizer_state_keys_count = len(loaded_state)

    missing_keys = list(getattr(incompatible, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
    # Current module.load_state_dict only populates these in strict mode;
    # for checkpoint reporting we expose them in both modes.
    if not strict:
        missing_keys = sorted(expected_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_keys)
    loaded_keys_count = len(expected_keys & provided_keys)
    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "loaded_keys_count": loaded_keys_count,
        "restored_optimizer_state_keys_count": restored_optimizer_state_keys_count,
    }
