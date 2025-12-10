from typing import (  # noqa: UP035, F401  # (Dict, List, Tuple) imported by torch.jit.annotations
    Any,
    Callable,
    Dict,
    Final,
    ForwardRef,
    get_args,
    get_origin,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

class FunctionModifiers:
    """
    Used to denote the behavior of a function in TorchScript. See export() and
    ignore() for details.
    """

    UNUSED = "unused (ignored and replaced with raising of an exception)"
    IGNORE = "ignore (leave as a call to Python, cannot be torch.jit.save'd)"
    EXPORT = "export (compile this function even if nothing calls it)"
    DEFAULT = "default (compile if called from a exported function / forward)"
    COPY_TO_SCRIPT_WRAPPER = (
        "if this method is not scripted, copy the python method onto the scripted model"
    )
    _DROP = "_drop (function is fully ignored, declaration can be unscriptable)"

def unused(fn):
    """
    This decorator indicates to the compiler that a function or method should
    be ignored and replaced with the raising of an exception. This allows you
    to leave code in your model that is not yet TorchScript compatible and still
    export your model.

        Example (using ``@torch.jit.unused`` on a method)::

            import torch
            import torch.nn as nn


            class MyModule(nn.Module):
                def __init__(self, use_memory_efficient):
                    super().__init__()
                    self.use_memory_efficient = use_memory_efficient

                @torch.jit.unused
                def memory_efficient(self, x):
                    import pdb

                    pdb.set_trace()
                    return x + 10

                def forward(self, x):
                    # Use not-yet-scriptable memory efficient mode
                    if self.use_memory_efficient:
                        return self.memory_efficient(x)
                    else:
                        return x + 10


            m = torch.jit.script(MyModule(use_memory_efficient=False))
            m.save("m.pt")

            m = torch.jit.script(MyModule(use_memory_efficient=True))
            # exception raised
            m(torch.rand(100))
    """
    if isinstance(fn, property):
        prop = fn
        setattr(  # noqa: B010
            prop.fget, "_torchscript_modifier", FunctionModifiers.UNUSED
        )

        if prop.fset:
            setattr(  # noqa: B010
                prop.fset, "_torchscript_modifier", FunctionModifiers.UNUSED
            )

        return prop

    fn._torchscript_modifier = FunctionModifiers.UNUSED
    return fn

# allows BroadcastingList instance to be subscriptable
class BroadcastingListCls:
    def __getitem__(self, types):
        return


# mypy doesn't support parameters on types, so we have to explicitly type each
# list size
BroadcastingList1 = BroadcastingListCls()
for i in range(2, 7):
    globals()[f"BroadcastingList{i}"] = BroadcastingList1

def is_scripting():
    False
