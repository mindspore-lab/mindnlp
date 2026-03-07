"""Minimal distributed checkpoint state_dict helpers for mindtorch_v2.

This MVP intentionally provides a narrow surface:
- returns model and optimizer state dictionaries
- restores model and optimizer state dictionaries

It is designed for single-node DDP training recovery flows.
"""

def _unwrap_model(model):
    # DDP/DataParallel expose the wrapped module at `.module`.
    return getattr(model, "module", model)


def get_state_dict(model, optimizer=None):
    """Return model and optimizer state dicts.

    Args:
        model: module or parallel wrapper exposing ``state_dict``.
        optimizer: optional optimizer exposing ``state_dict``.

    Returns:
        Tuple[dict, dict|None]: ``(model_state_dict, optim_state_dict)``.
    """
    base_model = _unwrap_model(model)
    model_state = base_model.state_dict()
    optim_state = optimizer.state_dict() if optimizer is not None else None
    return model_state, optim_state


def set_state_dict(
    model,
    optimizer=None,
    *,
    model_state_dict,
    optim_state_dict=None,
    strict=True,
):
    """Restore model and optimizer state dicts.

    Args:
        model: module or parallel wrapper exposing ``load_state_dict``.
        optimizer: optional optimizer exposing ``load_state_dict``.
        model_state_dict: state dict for model.
        optim_state_dict: optional state dict for optimizer.
        strict: forwarded to ``model.load_state_dict``.
    """
    base_model = _unwrap_model(model)
    base_model.load_state_dict(model_state_dict, strict=strict)
    if optimizer is not None and optim_state_dict is not None:
        # Safe optimizer restore: preserve runtime-owned parameter objects and
        # copy only serializable group options/state.
        loaded_groups = optim_state_dict.get("param_groups", [])
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
