"""nn modules for legacy mindspore."""
from mindnlp.utils import less_min_pynative_first

if less_min_pynative_first:
    from .transformer import Transformer, TransformerDecoder, TransformerEncoder, \
        TransformerEncoderLayer, TransformerDecoderLayer, MultiheadAttention
else:
    from mindspore.nn import Transformer, TransformerDecoder, TransformerEncoder, \
        TransformerEncoderLayer, TransformerDecoderLayer, MultiheadAttention

__all__ = [
    'Transformer', 'TransformerEncoder', 'TransformerDecoder',
    'TransformerEncoderLayer', 'TransformerDecoderLayer',
    'MultiheadAttention']
