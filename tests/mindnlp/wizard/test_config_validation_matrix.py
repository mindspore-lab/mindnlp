import pytest

from mindnlp.wizard.merge.common import ModelReference
from mindnlp.wizard.merge.config import InputModelDefinition, MergeConfiguration


def _model(name: str) -> InputModelDefinition:
    return InputModelDefinition(model=ModelReference.parse(name))


@pytest.mark.parametrize(
    "cfg,err",
    [
        (
            {
                "merge_method": "linear",
            },
            "Exactly one of models, slices, or modules must be specified",
        ),
        (
            {
                "merge_method": "linear",
                "models": [_model("a"), _model("b")],
                "tokenizer_source": "base",
                "tokenizer": {"source": "union"},
            },
            "Cannot specify both tokenizer_source and tokenizer",
        ),
        (
            {
                "merge_method": "slerp",
                "models": [_model("a"), _model("b")],
            },
            "requires base_model",
        ),
        (
            {
                "merge_method": "nearswap",
                "models": [_model("a"), _model("b")],
            },
            "requires base_model",
        ),
    ],
)
def test_config_validation_matrix(cfg, err):
    with pytest.raises(RuntimeError, match=err):
        MergeConfiguration.model_validate(cfg)
