import sys
import accelerate
from transformers.utils import _LazyModule

_import_structure = {
    "utils": [
        'DistributedType',

    ]
}

sys.modules[__name__] = _LazyModule(
    'accelerate',
    accelerate.__file__,
    _import_structure,
    module_spec=__spec__,
    extra_objects={"__version__": accelerate.__version__},
)
