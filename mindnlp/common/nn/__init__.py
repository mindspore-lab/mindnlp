"""nn modules for legacy mindspore."""
from mindnlp.utils import less_min_pynative_first

if less_min_pynative_first:
    from .transformer import Transformer, TransformerDecoder, TransformerEncoder, \
        TransformerEncoderLayer, TransformerDecoderLayer, MultiheadAttention
else:
    from mindspore.nn import Transformer, TransformerDecoder, TransformerEncoder, \
        TransformerEncoderLayer, TransformerDecoderLayer
    from mindspore.nn import MultiHeadAttention as MultiheadAttention

__all__ = []
__all__.extend([
    'Transformer', 'TransformerEncoder', 'TransformerDecoder',
    'TransformerEncoderLayer', 'TransformerDecoderLayer',
    'MultiheadAttention'])
