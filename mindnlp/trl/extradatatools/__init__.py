'''
This is a empty page.
'''

from typing import TYPE_CHECKING

from ..import_utils import _LazyModule


_import_structure = {
    "hf_argparser": ["HfArgumentParser"],
    "data_collator": ["DataCollatorMixin",
                      "DefaultDataCollator",
                      "DataCollatorWithPadding",
                      "DataCollatorForTokenClassification",
                      "DataCollatorForSeq2Seq",
                      "DataCollatorForLanguageModeling",
                      ]
}

if TYPE_CHECKING:
    from .data_collator import *
    from .hf_argparser import *
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__)
