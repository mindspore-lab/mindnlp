"""largely copy from llama and adapt for cogvlm"""
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, List, Union, Dict, Any
try:
    from typing import Literal
except:
    from typing_extensions import Literal
import math
import numpy as np
import mindspore
from mindspore import ops, nn, Tensor
from mindspore.common.initializer import initializer, Normal
from mindspore.dataset import transforms,vision
from mindnlp.modules.functional import finfo
from ...modeling_utils import PreTrainedModel
from ...tokenization_utils import PreTrainedTokenizer
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast,CausalLMOutputWithPast

from .configuration_cogvlm import CogVLMConfig
from .visual import EVA2CLIPModel
if TYPE_CHECKING:
    from mindnlp.utils import ModelOutput

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape, dtype: mindspore.dtype, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = ops.full((tgt_len, tgt_len),finfo(dtype=dtype,attr='min'),dtype=dtype)
    mask_cond = ops.arange(mask.shape[-1])
    mask = mask.masked_fill(mask_cond < (mask_cond + 1).view(mask.shape[-1], 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = ops.cat([ops.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: mindspore.Tensor, dtype: mindspore.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(mindspore.bool_), finfo(dtype=dtype,attr='min'))


class RMSNorm(nn.Cell):

    """
    This class represents a Root Mean Square Normalization (RMSNorm) layer that can be used in neural networks for feature normalization.
    
    RMSNorm is a technique used to normalize the hidden states of a neural network layer.
    It calculates the variance of the hidden states and applies normalization based on the root mean square of the variance.
    
    This class inherits from the nn.Cell class in the MindSpore library.
    
    Attributes:
        weight (mindspore.Parameter): The weight parameter used for the normalization.
        variance_epsilon (float): A small value added to the variance to avoid division by zero.
    
    Methods:
        __init__(self, hidden_size, eps=1e-06):
            Initializes a new instance of the RMSNorm class.

            Args:

            - hidden_size (int): The size of the hidden states.
            - eps (float, optional): A small value added to the variance to avoid division by zero. Default is 1e-06.

        construct(self, hidden_states):
            Applies RMSNorm normalization to the given hidden states.

            Args:

            - hidden_states (mindspore.Tensor): The input hidden states to be normalized.

             Returns:

            - mindspore.Tensor: The normalized hidden states after applying RMSNorm.

    Note:
        - The RMSNorm layer assumes that the input hidden states have a shape of (batch_size, hidden_size).
        - The RMSNorm layer expects the input hidden states to have a floating-point data type.
    """
    def __init__(self, hidden_size, eps=1e-6):
        """
        Initializes a new instance of the RMSNorm class.

        Args:
            hidden_size (int): The size of the hidden layer in the neural network.
            eps (float, optional): The epsilon value used for numerical stability in the calculation of the variance.
                Defaults to 1e-06.

        Returns:
            None.

        Raises:
            ValueError: If the hidden_size is not a positive integer.
            TypeError: If the eps is not a float.
        """
        super().__init__()
        self.weight = mindspore.Parameter(ops.ones(hidden_size))
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        """
        Constructs an RMSNorm object.

        This method applies the RMS normalization technique to the given hidden states.

        Args:
            self (RMSNorm): The RMSNorm object.
            hidden_states (mindspore.Tensor): The input hidden states to be normalized.
                It should have the shape (batch_size, sequence_length, hidden_size).
                The data type should be convertible to float32.

        Returns:
            None: This method modifies the hidden_states tensor in-place.

        Raises:
            TypeError: If the input hidden_states tensor is not of type mindspore.Tensor.
            ValueError: If the input hidden_states tensor does not have the correct shape.
            ValueError: If the input hidden_states tensor data type cannot be converted to float32.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(mindspore.float32)
        variance = hidden_states.pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class MLP(nn.Cell):

    """
    This class represents a Multi-Layer Perceptron (MLP) neural network model, which is used for various machine learning tasks.
    The MLP class inherits from the nn.Cell class, which is a fundamental building block for creating neural network models.

    Attributes:
        hidden_size (int): The size of the hidden layer in the MLP.
        intermediate_size (int): The size of the intermediate layer in the MLP.
        gate_proj (nn.Dense): The dense layer responsible for projecting the input to the intermediate size.
        up_proj (nn.Dense): The dense layer responsible for projecting the input to the intermediate size.
        down_proj (nn.Dense): The dense layer responsible for projecting the intermediate size back to the hidden size.
        act_fn (function): The activation function used in the hidden layer of the MLP.

    Methods:
        construct(x): Constructs the forward pass of the MLP given an input tensor.

    Example:
        ```python
        >>> config = MLPConfig(hidden_size=128, intermediate_size=64, hidden_act="relu")
        >>> mlp = MLP(config)
        >>> input_tensor = torch.randn(10, 128)
        >>> output = mlp.construct(input_tensor)
        ```

    Note:
        The MLP class assumes that the ACT2FN dictionary, containing activation functions, is defined in the global scope.
    """
    def __init__(self, config):
        """
        Initializes an instance of the MLP class.

        Args:
            self: The instance of the MLP class.
            config:
                An object containing configuration parameters for the MLP.

                - hidden_size (int): The size of the hidden layer.
                - intermediate_size (int): The size of the intermediate layer.
                - hidden_act (str): The activation function for the hidden layer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.up_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.down_proj = nn.Dense(self.intermediate_size, self.hidden_size, has_bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def construct(self, x):
        """
        Method to construct a down_proj output based on the given input x.

        Args:
            self (MLP): The instance of the MLP class.
            x: Input tensor of shape (batch_size, features) to be processed.

        Returns:
            None: This method does not return any value directly. The down_proj output is stored in the internal state.

        Raises:
            TypeError: If the input x is not of the expected type.
            ValueError: If the dimensions of the input x are not compatible with the operations.
        """
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def get_expert_mask(token_type_ids):
    """
    Args:
        token_type_ids (Tensor): A 2D tensor representing the token type ids.
            Each element indicates the type of token in the input sequence.
            Shape: (batch_size, sequence_length).

    Returns:
        vision_token_mask: A boolean tensor identifying vision tokens in the input sequence.
        language_token_mask: A boolean tensor identifying language tokens in the input sequence.

    Raises:
        None.
    """
    vision_token_mask = ops.zeros_like(token_type_ids, dtype=mindspore.bool_)
    vision_token_mask[:, :-1] = (token_type_ids[:, :-1] == VISION_TOKEN_TYPE) & (token_type_ids[:, 1:] == VISION_TOKEN_TYPE)
    language_token_mask = ~vision_token_mask
    return vision_token_mask, language_token_mask


class VisionExpertMLP(nn.Cell):

    """
    The VisionExpertMLP class represents a multi-layer perceptron (MLP) model designed for expert processing of
    vision-related and language-related inputs. This class inherits from the nn.Cell module.

    Attributes:
        language_mlp (MLP): An instance of the MLP class for processing language-related inputs.
        vision_mlp (MLP): An instance of the MLP class for processing vision-related inputs.

    Methods:
        construct(hidden_states, token_type_ids): Processes the input hidden states based on the token type IDs to produce the output.

        Detailed Description:

            - The VisionExpertMLP class initializes with two instances of the MLP class, language_mlp, and vision_mlp,
            to process language-related and vision-related inputs, respectively.
            - The construct method operates on the hidden states and token type IDs to calculate the output.
            - The construct method employs the vision_mlp and language_mlp instances to process the hidden states
            based on the vision and language token masks, and then aggregates the results to produce the final output.

        The construct method takes the following parameters:
            - hidden_states (mindspore.Tensor(B, L, D)): The input hidden states to be processed.
            - token_type_ids (mindspore.Tensor(B, L)): The token type IDs to determine the type of input.

        The construct method returns:
            - output (mindspore.Tensor(B, L, D)): The processed output based on the input hidden states and token type IDs.

    Note:
        The construct method leverages the get_expert_mask function to obtain vision
        and language token masks for processing the hidden states.

    """
    def __init__(self, config):
        """
        Initializes an instance of the VisionExpertMLP class.

        Args:
            self: The instance of the VisionExpertMLP class.
            config: A configuration object that contains the necessary parameters for the VisionExpertMLP.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.language_mlp = MLP(config)
        self.vision_mlp = MLP(config)

    def construct(self, hidden_states: "mindspore.Tensor(B, L, D)", token_type_ids: "mindspore.Tensor(B, L)"):
        """
        Constructs the expert output by applying vision and language MLPs on the given hidden states.

        Args:
            self: An instance of the VisionExpertMLP class.
            hidden_states (mindspore.Tensor): A tensor of shape (B, L, D) containing the hidden states.
                B represents the batch size, L represents the sequence length, and D represents the hidden size.
            token_type_ids (mindspore.Tensor): A tensor of shape (B, L) containing the token type ids.
                It identifies whether each token in the sequence belongs to the vision or language modality.

        Returns:
            mindspore.Tensor: A tensor of shape (B, L, D) representing the expert output.
                The output tensor is constructed by applying vision MLP on the hidden states of vision tokens
                and language MLP on the hidden states of language tokens.

        Raises:
            None
        """
        output = ops.zeros(hidden_states.shape, dtype=hidden_states.dtype)
        vision_token_mask, language_token_mask = get_expert_mask(token_type_ids)
        output[vision_token_mask] = self.vision_mlp(hidden_states[vision_token_mask])
        output[language_token_mask] = self.language_mlp(hidden_states[language_token_mask])
        return output


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> mindspore.Tensor:
    """
    Perform scaled dot product attention.

    Args:
        query (mindspore.Tensor):
            The query tensor of shape (..., L, H), where L is the length of the query sequence and H is the hidden size.
        key (mindspore.Tensor):
            The key tensor of shape (..., S, H), where S is the length of the key sequence and H is the hidden size.
        value (mindspore.Tensor): The value tensor of shape (..., S, V), where V is the value size.
        attn_mask (mindspore.Tensor, optional): The attention mask tensor of shape (L, S) or (..., L, S) with elements being 0 or 1.
            It masks the attention weights. Defaults to None.
        dropout_p (float, optional): The dropout probability. Defaults to 0.0.
        is_causal (bool, optional): Whether to apply causal masking.
            If True, attn_mask must be None. Defaults to False.
        scale (float, optional): The scale factor for the attention weights.
            If None, it will be calculated as 1 / sqrt(H). Defaults to None.

    Returns:
        mindspore.Tensor: The attention weights tensor of shape (..., L, S).

    Raises:
        AssertionError: If is_causal is True and attn_mask is not None.
    """
    # Efficient implementation equivalent to the following:
    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    attn_bias = ops.zeros((L, S), dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = ops.ones((L, S), dtype=mindspore.bool_).tril(diagonal=0)
        attn_bias = attn_bias.masked_fill(temp_mask.logical_not(), finfo(dtype=query.dtype,attr='min'))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == mindspore.bool_:
            attn_bias=  attn_bias.masked_fill_(attn_mask.logical_not(), finfo(dtype=query.dtype,attr='min'))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.swapaxes(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = ops.softmax(attn_weight, axis=-1)
    attn_weight = ops.dropout(attn_weight, dropout_p)
    return attn_weight @ value


def attention_fn(
        query_layer: "mindspore.Tensor(B, H, L, HD)",
        key_layer: "mindspore.Tensor(B, H, L, HD)",
        value_layer: "mindspore.Tensor(B, H, L, HD)",
        attention_mask: "mindspore.Tensor(B, H, L, HD)",
        *,
        scaling_attention_score: bool = True,
        attention_dropout: nn.Cell = None):
    """
    The attention_fn function calculates the attention scores between the given query, key, and value layers,
    using an attention mask if provided.
    It then applies softmax to normalize the attention scores and performs an optional attention dropout.
    Finally, it computes the context layer by multiplying the attention scores with the value layer.

    Args:
        query_layer (mindspore.Tensor(B, H, L, HD)):
            The query layer tensor, where B represents the batch size, H is the number of attention heads,
            L is the sequence length, and HD is the hidden dimension.
        key_layer (mindspore.Tensor(B, H, L, HD)):
            The key layer tensor, with the same shape as the query_layer.
        value_layer (mindspore.Tensor(B, H, L, HD)): The value layer tensor, with the same shape as the query_layer.
        attention_mask (mindspore.Tensor(B, H, L, HD)):
            The attention mask tensor, with the same shape as the query_layer.
            It is used to mask out certain positions in the attention scores.

    Keyword Args:
        scaling_attention_score (bool, optional):
            Determines whether to scale the attention scores by dividing the query_layer by the square root
            of its hidden dimension. Defaults to True.
        attention_dropout (nn.Cell, optional):
            An attention dropout cell that applies dropout to the attention scores. Defaults to None.

    Returns:
        context_layer: The context layer tensor obtained by multiplying the attention scores with the value layer.
            It has the same shape as the query_layer.

    Raises:
        None.
    """
    if scaling_attention_score:
        query_layer = query_layer / math.sqrt(query_layer.shape[-1])
    attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))
    attention_scores = attention_scores + attention_mask
    attention_scores = ops.softmax(attention_scores, axis=-1, dtype=mindspore.float32).to(query_layer.dtype)
    if attention_dropout is not None:
        attention_scores = attention_dropout(attention_scores)
    context_layer = ops.matmul(attention_scores, value_layer)
    return context_layer


class RotaryEmbedding(mindspore.nn.Cell):

    """
    The 'RotaryEmbedding' class represents a rotary positional embedding layer in the mindspore.nn framework.
    This class inherits from the mindspore.nn.Cell class.

    Attributes:
        dim (int): The dimensionality of the embedding.
        max_position_embeddings (int): The maximum number of positions in the input sequence.
        base (int): The base value used for computing the inverse frequency.
        inv_freq (Tensor): The tensor containing the inverse frequency values.
        max_seq_len_cached (int): The maximum sequence length for which the cos and sin values are cached.
        cos_cached (Tensor): The cached cosine values for the positions.
        sin_cached (Tensor): The cached sine values for the positions.

    Methods:
        __init__(self, dim, max_position_embeddings=2048, base=10000):
            Initializes a new instance of the 'RotaryEmbedding' class.

            Args:

            - dim (int): The dimensionality of the embedding.
            - max_position_embeddings (int): The maximum number of positions in the input sequence. Default is 2048.
            - base (int): The base value used for computing the inverse frequency. Default is 10000.

            Returns:

            - None

        _compute_inv_freq(self):
            Computes the inverse frequency values for the embedding.

            Returns:

            - Tensor: The tensor containing the inverse frequency values.

        _set_cos_sin_cache(self, seq_len, dtype):
            Sets the cosine and sine values cache for the given sequence length and data type.

            Args:

            - seq_len (int): The length of the sequence.
            - dtype (mindspore.dtype): The data type of the input.

            Returns:

            - None

        construct(self, x, seq_len):
            Constructs the rotary embeddings for the given input and sequence length.

            Args:

            - x (Tensor): The input tensor.
            - seq_len (int): The length of the sequence.

            Returns:

            - Tuple[Tensor, Tensor]: The cosine and sine embeddings for the given sequence length and input.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        """
        Initializes an instance of the RotaryEmbedding class.

        Args:
            self (RotaryEmbedding): The instance of the class.
            dim (int): The dimensionality of the embeddings.
            max_position_embeddings (int, optional): The maximum number of position embeddings. Default is 2048.
            base (int, optional): The base value used for computing the inverse frequency. Default is 10000.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = self._compute_inv_freq()
        self.max_seq_len_cached = 0

    def _compute_inv_freq(self):
        """
        Compute the inverse frequency values for RotaryEmbedding.

        Args:
            self: RotaryEmbedding class instance. Represents the current instance of RotaryEmbedding.

        Returns:
            None: The method does not return any value but internally updates the inverse frequency values for RotaryEmbedding.

        Raises:
            TypeError: If the operation involving data types is invalid.
            ValueError: If the dimensions are not compatible for the computation.
        """
        return 1.0 / (
                self.base
                ** (ops.arange(0, self.dim, 2).to(mindspore.float32) / self.dim)
        )

    def _set_cos_sin_cache(self, seq_len, dtype):
        '''
        Set the cosine and sine cache for rotary embedding.

        Args:
            self (RotaryEmbedding): The instance of the RotaryEmbedding class.
            seq_len (int): The length of the sequence.
            dtype (str): The data type for the cache.

        Returns:
            None: The method sets the cosine and sine cache for the rotary embedding and does not return any value.

        Raises:
            None.
        '''
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = ops.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos()[:, None, :].to(dtype)
        self.sin_cached = emb.sin()[:, None, :].to(dtype)

    def construct(self, x, seq_len):
        """
        This method constructs the rotary embedding for the input sequence.

        Args:
            self (RotaryEmbedding): The instance of the RotaryEmbedding class.
            x:
                The input tensor representing the sequence.

                - Type: tensor
                - Purpose: It is the input sequence for which the rotary embedding needs to be constructed.
            seq_len:
                The length of the input sequence.

                - Type: int
                - Purpose: It defines the length of the input sequence for which the rotary embedding needs to be constructed.
                - Restrictions: Must be a positive integer.

        Returns:
            None: This method does not return any value directly.
                Instead, it updates the internal state of the RotaryEmbedding instance.

        Raises:
            None.
        """
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """
    Rotates the given tensor by 180 degrees along the last dimension and swaps the two halves.

    Args:
        x (tensor): The input tensor to be rotated. It should have at least 1 dimension.

    Returns:
        tensor: The rotated tensor with the same shape as the input tensor. It has the same type and order as the input tensor.

    Raises:
        None.
    """
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return ops.cat((-x2, x1), axis=x1.ndim - 1)


def apply_rotary_pos_emb_index_bhs(q, k, cos, sin, position_id,unsqueeze_dim=1):
    """
    Apply rotary positional embedding index to inputs.

    Args:
        q (Tensor): The input tensor representing queries.
        k (Tensor): The input tensor representing keys.
        cos (Tensor): The tensor containing cosine values.
        sin (Tensor): The tensor containing sine values.
        position_id (int): The index indicating the position of the embedding to be applied.
        unsqueeze_dim (int, optional): The dimension to unsqueeze the cosine and sine tensors (default: 1).

    Returns:
        Tuple[Tensor, Tensor]: A tuple of modified query tensor (`q`) and key tensor (`k`).

    Raises:
        None.

    Note:
        The function applies a rotary positional embedding to the input tensors `q` and `k` using the cosine (`cos`) and sine (`sin`) values.
        The embedding is applied at the specified `position_id`, with the cosine and sine values unsqueezed along the `unsqueeze_dim` dimension.

    Example:
        ```python
        >>> q = torch.tensor([1, 2, 3])
        >>> k = torch.tensor([4, 5, 6])
        >>> cos = torch.tensor([0, 0.5, 1])
        >>> sin = torch.tensor([1, 0.5, 0])
        >>> position_id = 1
        >>> unsqueeze_dim = 2
        >>> apply_rotary_pos_emb_index_bhs(q, k, cos, sin, position_id, unsqueeze_dim)
        (tensor([1.0000, 3.5000, 3.0000]), tensor([4.0000, 6.5000, 6.0000]))
        ```
    """
    cos = cos.squeeze()
    sin = sin.squeeze()
    cos = cos[position_id].unsqueeze(unsqueeze_dim)
    sin = sin[position_id].unsqueeze(unsqueeze_dim)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k


class VisionExpertAttention(nn.Cell):

    """
    This class represents a vision expert attention mechanism used in a neural network model. It is a subclass of nn.Cell.

    Attributes:
        config (Config): The configuration object for the attention mechanism.
        hidden_size (int): The size of the hidden state.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        max_position_embeddings (int): The maximum number of position embeddings.
        rotary_emb (RotaryEmbedding): The rotary embedding layer used for positional encoding.
        vision_expert_query_key_value (nn.Dense): The dense layer for vision expert query-key-value computation.
        vision_expert_dense (nn.Dense): The dense layer for vision expert output computation.
        language_expert_query_key_value (nn.Dense): The dense layer for language expert query-key-value computation.
        language_expert_dense (nn.Dense): The dense layer for language expert output computation.

    Methods:
        __init__(self, config): Initializes the VisionExpertAttention object.
        _swapaxes_for_scores(self, tensor): Transposes a 3D tensor into a 4D tensor.
        construct(self, hidden_states, token_type_ids, position_ids, attention_mask, past_key_value, output_attentions, use_cache):
            Constructs the attention mechanism.

    """
    def __init__(self, config):
        """
        Initializes an instance of the VisionExpertAttention class.

        Args:
            self (VisionExpertAttention): The current instance of the class.
            config: The configuration object containing various settings for the attention mechanism.

        Returns:
            None

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.vision_expert_query_key_value = nn.Dense(self.hidden_size, self.hidden_size * 3, has_bias=False)
        self.vision_expert_dense = nn.Dense(self.hidden_size, self.hidden_size, has_bias=False)
        self.language_expert_query_key_value = nn.Dense(self.hidden_size, self.hidden_size * 3, has_bias=False)
        self.language_expert_dense = nn.Dense(self.hidden_size, self.hidden_size, has_bias=False)

    def _swapaxes_for_scores(self, tensor):
        """Transpose a 3D tensor [B, L, H*HD] into a 4D tensor with size [B H L HD]."""
        new_tensor_shape = tensor.shape[:-1] + (self.num_heads, self.head_dim)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def construct(
            self,
            hidden_states: mindspore.Tensor,
            token_type_ids: mindspore.Tensor,
            position_ids: mindspore.Tensor,
            attention_mask: Optional[mindspore.Tensor] = None,
            past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """
        Constructs the VisionExpertAttention.

        Args:
            self: The object itself.
            hidden_states (mindspore.Tensor): The input hidden states. Shape (batch_size, sequence_length, hidden_size).
            token_type_ids (mindspore.Tensor): The token type ids. Shape (batch_size, sequence_length).
            position_ids (mindspore.Tensor): The position ids. Shape (batch_size, sequence_length).
            attention_mask (Optional[mindspore.Tensor], optional):
                The attention mask tensor. Shape (batch_size, sequence_length). Defaults to None.
            past_key_value (Optional[Tuple[mindspore.Tensor]], optional): The past key and value tensors. Defaults to None.
            output_attentions (bool, optional): Whether to output attentions. Defaults to False.
            use_cache (bool, optional): Whether to use cache. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
                A tuple containing the attention output tensor,
            an optional tensor, and an optional tuple of past key and value tensors.

        Raises:
            ValueError: If the shape of the context layer is not (batch_size, num_heads, sequence_length, head_dim).

        """
        bsz, q_len, _ = hidden_states.shape
        vision_token_mask, language_token_mask = get_expert_mask(token_type_ids)

        shape = list(hidden_states.shape)
        shape[-1] = shape[-1] * 3
        shape = tuple(shape)

        mixed_raw_layer = ops.zeros(shape,dtype=hidden_states.dtype)
        mixed_raw_layer[vision_token_mask] = self.vision_expert_query_key_value(hidden_states[vision_token_mask])
        mixed_raw_layer[language_token_mask] = self.language_expert_query_key_value(hidden_states[language_token_mask])
        query_states, key_states, value_states = ops.split(mixed_raw_layer, self.hidden_size, axis=-1)

        query_states = self._swapaxes_for_scores(query_states)  # B, H, L, HD
        key_states = self._swapaxes_for_scores(key_states)  # B, H, L, HD
        value_states = self._swapaxes_for_scores(value_states)  # B, H, L, HD

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max() + 1)

        tmp = [i.asnumpy() for i in [mixed_raw_layer,query_states, key_states, value_states,cos, sin]]

        query_states, key_states = apply_rotary_pos_emb_index_bhs(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            key_states = ops.cat([past_key_value[0], key_states], axis=2)
            value_states = ops.cat([past_key_value[1], value_states], axis=2)
        past_key_value = (key_states, value_states) if use_cache else None
        context_layer = attention_fn(
            query_layer=query_states, key_layer=key_states, value_layer=value_states, attention_mask=attention_mask,
            scaling_attention_score=True, attention_dropout=None)

        if context_layer.shape != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {context_layer.shape}"
            )

        context_layer = context_layer.swapaxes(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = ops.zeros(context_layer.shape, dtype=hidden_states.dtype)
        attn_output[vision_token_mask] = self.vision_expert_dense(context_layer[vision_token_mask])
        attn_output[language_token_mask] = self.language_expert_dense(context_layer[language_token_mask])
        if output_attentions:
            warnings.warn("output_attentions is not implemented.")

        return attn_output, None, past_key_value


class CogVLMDecoderLayer(nn.Cell):

    """
    CogVLMDecoderLayer represents a single layer of the Vision-Language Multimodal Transformer decoder.
    The layer consists of a vision expert attention module, a vision expert MLP module, and layer normalization modules.

    Attributes:
        hidden_size (int): The size of the hidden layers in the configuration.
        self_attn (VisionExpertAttention): The vision expert attention module.
        mlp (VisionExpertMLP): The vision expert MLP module.
        input_layernorm (RMSNorm): The layer normalization module for the input.
        post_attention_layernorm (RMSNorm): The layer normalization module after the attention module.

    Methods:
        construct:
            Constructs the decoder layer.

    Returns:
        Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
            A tuple containing the hidden states of the layer and optionally attention weights and present key value.
    """
    def __init__(self, config):
        """
        Initialize CogVLMDecoderLayer with given configuration.

        Args:
            self (CogVLMDecoderLayer): The instance of CogVLMDecoderLayer.
            config: The configuration object containing the model parameters.

        Returns:
            None

        Raises:
            None
            """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = VisionExpertAttention(config=config)
        self.mlp = VisionExpertMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def construct(
            self,
            hidden_states: mindspore.Tensor,
            token_type_ids: mindspore.Tensor,
            position_ids: mindspore.Tensor,
            attention_mask: Optional[mindspore.Tensor] = None,
            past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
        """
        CogVLMDecoderLayer.construct method.

        Args:
            self: The instance of the CogVLMDecoderLayer class.
            hidden_states (mindspore.Tensor): The input hidden states tensor.
            token_type_ids (mindspore.Tensor): The token type ids tensor.
            position_ids (mindspore.Tensor): The position ids tensor.
            attention_mask (Optional[mindspore.Tensor], optional): An optional tensor for attention mask. Defaults to None.
            past_key_value (Optional[Tuple[mindspore.Tensor]], optional): An optional tuple of past key values. Defaults to None.
            output_attentions (Optional[bool], optional): A flag to output attentions. Defaults to False.
            use_cache (Optional[bool], optional): A flag to use cache. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
                A tuple containing the output hidden states tensor and optionally a tuple of present key values.

        Raises:
            None.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, token_type_ids=token_type_ids)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs  # type: ignore


class CogVLMPreTrainedModel(PreTrainedModel):

    """
    The `CogVLMPreTrainedModel` class is a subclass of `PreTrainedModel` and represents a pre-trained language model
    for cognitive vision and language tasks. This class provides methods for initializing the weights of the
    model's neural network cells.

    Methods:
        `_init_weights(self, cell)`:
            Initializes the weights of the specified neural network cell.

            - If the cell is a `nn.Dense` type, the weights are initialized using a normal distribution with a mean of 0
            and a standard deviation specified by `self.config.initializer_range`.
            - If the cell has a bias, the bias weights are initialized to zeros.
            - If the cell is an `nn.Embedding` type, the weights are initialized using a
            normal distribution with a mean of 0 and a standard deviation specified by `self.config.initializer_range`.
            - If the cell has a padding index, the corresponding weight value is set to 0.

    Note:
        The `CogVLMPreTrainedModel` class assumes that the `PreTrainedModel` class has been properly implemented and imported.
    """
    config_class = CogVLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["CogVLMDecoderLayer", "TransformerLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, cell):
        """
        Initialize the weights and biases for a given neural network cell.

        Args:
            self (CogVLMPreTrainedModel): The instance of the CogVLMPreTrainedModel class.
            cell: The neural network cell for which the weights and biases are to be initialized.
                It can be of type nn.Dense or nn.Embedding.

        Returns:
            None.

        Raises:
            ValueError: If the provided cell is not of type nn.Dense or nn.Embedding.
            TypeError: If the cell's weight or bias data cannot be set due to incompatible shapes or data types.
            IndexError: If the padding index for the cell's weight is out of range.
        """
        if isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(mean=0,sigma=self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.initializer_range, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))


def is_empty(images_list: Optional[List[List[mindspore.Tensor]]]):
    '''
    Args:
        images_list (Optional[List[List[mindspore.Tensor]]]): A list of lists of mindspore tensors representing images.
            Can be None or an empty list.

    Returns:
        None: Returns True if the input images_list is None or empty, otherwise False.

    Raises:
        None.
    '''
    if images_list is None or len(images_list) == 0:
        return True
    for image_list in images_list:
        if len(image_list):
            return False
    return True


def build_position_ids(x:"mindspore.Tensor(B, L)", attention_mask: Optional["mindspore.Tensor(B, L)"] = None) -> "mindspore.Tensor(B, L)":
    """
    Builds position IDs for each element in the input tensor.

    Args:
        x (mindspore.Tensor(B, L)): The input tensor of shape (B, L), where B represents the batch size and L represents the sequence length.
        attention_mask (Optional[mindspore.Tensor(B, L)]): An optional tensor of shape (B, L) representing the attention mask.
            If provided, the positions where the attention mask is False will be assigned -1 in the output tensor.
            Defaults to None.

    Returns:
        mindspore.Tensor(B, L): A tensor of shape (B, L) representing the position IDs.
            The values in the tensor indicate the position of each element in the input tensor.
        Elements that belong to a language token or a vision token are assigned a specific value
            called 'LANGUAGE_TOKEN_TYPE' and 'VISION_TOKEN_TYPE', respectively.

    Raises:
        None.
    """
    if attention_mask is not None:
        tmp = x.copy()
        bool_a = attention_mask.bool()
        bool_a = ~bool_a
        tmp[bool_a] = -1
    else:
        tmp = x.copy()
    is_boi_eoi = ops.zeros_like(x, dtype=mindspore.bool_)
    is_boi_eoi[:, 1:] |= (tmp[:, 1:] == VISION_TOKEN_TYPE) & (tmp[:, :-1] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, 0] |= (tmp[:, 0] == VISION_TOKEN_TYPE)
    is_boi_eoi[:, :-1] |= (tmp[:, :-1] == VISION_TOKEN_TYPE) & (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, -1] |= (tmp[:, -1] == VISION_TOKEN_TYPE)
    tmp[is_boi_eoi] = LANGUAGE_TOKEN_TYPE

    # final position ids
    y = ops.zeros_like(x, dtype=mindspore.int64)
    y[:, 1:] = (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE) | ((tmp[:, 1:] == VISION_TOKEN_TYPE) & (tmp[:, :-1] == LANGUAGE_TOKEN_TYPE))
    y = y.cumsum(axis=-1)
    return y

def my_index_put(index_tensor,update_data,original_data):
    """
    as mindspore in GPU does not support this operation: tensor.index_put, I simply implement this function instead.
    """
    left,right = -1,-1
    for j,i in enumerate(index_tensor[0]):
        if i:
            if left == -1:
                left = j
            else:
                right = j
    original_data[0][left:right+1] = update_data
    return original_data

class CogVLMModel(CogVLMPreTrainedModel):

    '''
    Represents a CogVLM (Cognitive Vision and Language Model) for multimodal learning, combining vision and
    language information for various NLP and computer vision tasks.

    This class inherits from CogVLMPreTrainedModel and implements methods for encoding images and constructing
    the model for language and vision processing. It also includes methods for forward pass, getting  and setting
    input embeddings, and preparing attention masks for the decoder.

    The CogVLMModel class includes the following methods:

    - __init__: Initializes the CogVLMModel with the provided configuration.
    - encode_images: Encodes the input images and returns the image features.
    - construct: Constructs the model for language and vision processing and returns the output.
    - llm_forward: Performs the forward pass for the CogVLMModel and returns the output.
    - get_input_embeddings: Returns the input embeddings for the model.
    - set_input_embeddings: Sets the input embeddings for the model.
    - _prepare_decoder_attention_mask: Prepares attention masks for the decoder based on the provided inputs.

    The CogVLMModel class also includes an __init__ method to initialize the model and handle configuration parameters.
    Additionally, it inherits methods from the CogVLMPreTrainedModel class.

    '''
    def __init__(self, config):
        """
        __init__(self, config)
        Initialize the CogVLMModel with the provided configuration.

        Args:
            self: The instance of the CogVLMModel class.
            config: An object containing the configuration parameters for the model, such as pad_token_id, vocab_size,
                hidden_size, num_hidden_layers, rms_norm_eps, and other relevant settings. It is of type
                'config' and is required for initializing the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx,dtype=mindspore.float32)
        self.layers = nn.CellList([CogVLMDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.vision = EVA2CLIPModel(config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def encode_images(self, images: List[List[mindspore.Tensor]]) -> mindspore.Tensor:
        """
        Encodes a batch of images into their corresponding image features using the CogVLMModel.

        Args:
            self (CogVLMModel): The instance of the CogVLMModel class.
            images (List[List[mindspore.Tensor]]): A list of lists of mindspore.Tensor objects representing the images.
                Each inner list contains a batch of images. Each image is represented as a mindspore.Tensor object.

        Returns:
            mindspore.Tensor: A tensor containing the image features.

        Raises:
            None.
        """
        images_list, images = images, []

        images = []
        for image_list in images_list:
            for image in image_list:
                images.append(image)

        images = ops.stack(images)
        images_features = self.vision(images)
        return images_features

    def construct(
            self,
            input_ids: mindspore.Tensor = None,
            images: List[List[mindspore.Tensor]] = None,
            token_type_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[List[mindspore.Tensor]] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """take care of image_encode, token_type_ids, position_ids and (attention_mask = None is fine)"""
        if past_key_values is not None:
            pass  # generate mode with past_key_values. the image features are already mapped
        else:
            # not allow for inputs_embeds, because we want to process image feature

            assert input_ids is not None and inputs_embeds is None, f"{input_ids} {inputs_embeds}"
            if not is_empty(images):  # multi-modality

                assert token_type_ids is not None, "multi-modality requires `token_type_ids`!"

                assert len(input_ids) == len(images), f"{len(input_ids)} {len(images)}"
                inputs_embeds = self.embed_tokens(input_ids)
                images_features = self.encode_images(images)
                images_features = mindspore.Tensor(images_features)
                images_features = images_features.squeeze(0)
                #inputs_embeds = inputs_embeds.index_put([token_type_ids == VISION_TOKEN_TYPE], images_features)
                inputs_embeds = my_index_put(token_type_ids == VISION_TOKEN_TYPE, images_features, inputs_embeds)

            else:  # single-modality
                if token_type_ids is None:
                    token_type_ids = ops.ones_like(input_ids, dtype=mindspore.int64) * LANGUAGE_TOKEN_TYPE
                assert not (token_type_ids == VISION_TOKEN_TYPE).any(), f"{(token_type_ids == VISION_TOKEN_TYPE).sum()}"
                inputs_embeds = self.embed_tokens(input_ids)
            if position_ids is None:
                position_ids = build_position_ids(token_type_ids, attention_mask)

            input_ids = None
        return self.llm_forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def llm_forward(
            self,
            input_ids: mindspore.Tensor = None,
            token_type_ids: mindspore.Tensor = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[List[mindspore.Tensor]] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """largely copy from llama forward and adapt for cogvlm with `token_type_ids`"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = ops.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=mindspore.int64
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = ops.ones(
                (batch_size, seq_length_with_past), dtype=mindspore.bool_
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        out = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )[0]
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def get_input_embeddings(self):
        """
        This method returns the input embeddings for the CogVLMModel.

        Args:
            self: A reference to the current instance of the class CogVLMModel.

        Returns:
            embed_tokens: The method returns the embed_tokens attribute of the CogVLMModel instance.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the CogVLMModel.

        Args:
            self (CogVLMModel): The instance of the CogVLMModel class.
            value: The input embeddings to be set.

        Returns:
            None.

        Raises:
            None.

        This method sets the 'embed_tokens' attribute of the CogVLMModel instance to the provided 'value'.
        The 'embed_tokens' attribute represents the input embeddings used for the model.
        By setting this attribute, the input embeddings can be customized or updated during runtime.

        Note:
            The 'value' parameter should be compatible with the expected format of the input embeddings.
            Ensure that the 'value' matches the required shape and data type for the model's input embeddings.

        Example:
            ```python
            >>> model = CogVLMModel()
            >>> embeddings = get_input_embeddings()
            >>> model.set_input_embeddings(embeddings)
            ```
        """
        self.embed_tokens = value

    # noinspection PyMethodMayBeStatic
    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """
        Prepare the decoder attention mask for the CogVLMModel.

        This method takes five parameters:

        - self: The instance of the CogVLMModel class.
        - attention_mask: The attention mask tensor of shape (batch_size, sequence_length) or None.
        It masks the attention scores by specifying which tokens should be attended to and which should not.
        - input_shape: The shape of the input tensor of shape (batch_size, sequence_length, hidden_size).
        - inputs_embeds: The embedded input tensor of shape (batch_size, sequence_length, hidden_size).
        - past_key_values_length: The length of past key values, used for causal mask generation.

        Returns:
            combined_attention_mask:
                The combined attention mask tensor of shape (batch_size, sequence_length, sequence_length) or None.
                It combines the input attention mask and the causal mask.

        Raises:
            None.

        Note:
            - The input_shape parameter is used to determine if the input tensor has a hidden dimension greater than 1.
            If so, a causal mask is generated using _make_causal_mask() function.
            - The attention_mask parameter is expanded to match the shape of the inputs_embeds tensor if not None,
            and then combined with the causal mask.
            - The combined_attention_mask tensor is returned as the result of this method.
        """
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask


def _history_to_prompt(signal_type, history, query):
    """
    Converts a given history of queries and responses into a formatted prompt.

    Args:
        signal_type (str):
            The type of signal, which can be one of the following:

            - 'base': indicates a base signal.
            - 'vqa': indicates a signal for Visual Question Answering.
            - 'chat': indicates a signal for chat conversation.
        history (list): A list of tuples representing the history of queries and responses.
            Each tuple contains two elements: the old query and its corresponding response.
        query (str): The current query.

    Returns:
        str: The formatted prompt string.

    Raises:
        AssertionError: If the provided signal_type is not one of the predefined types.
    """
    if signal_type == 'base':
        return query
    elif signal_type == 'vqa':
        answer_format = 'Short answer:'
    elif signal_type == 'chat':
        answer_format = 'Answer:'
    else:
        assert False, f"Unknown signal type {signal_type}"

    prompt = ''
    for i, (old_query, response) in enumerate(history):
        prompt += 'Question: ' + old_query + " {} ".format(answer_format) + response + "\n"
    prompt += 'Question: {} {}'.format(query, answer_format)
    return prompt


class CogVLMForCausalLM(CogVLMPreTrainedModel):

    """
    CogVLMForCausalLM is a class for generating language using a CogVLM (Cognitive Vision Language Model)
    for causal language modeling. This class inherits from the CogVLMPreTrainedModel and includes methods
    for constructing, preparing inputs for generation, updating model keyword arguments for generation,
    and reordering cache.

    Methods:
        __init__(self, config): Initializes the class with a given configuration.
        get_input_embeddings(self): Returns the model's input embeddings.
        set_input_embeddings(self, value): Sets the model's input embeddings to a given value.
        get_output_embeddings(self): Returns the model's output embeddings.
        set_output_embeddings(self, new_embeddings): Sets the model's output embeddings to a given value.
        set_decoder(self, decoder): Sets the model's decoder to a given value.
        get_decoder(self): Returns the model's decoder.
        construct(self, input_ids, images, token_type_ids, attention_mask, position_ids, past_key_values, inputs_embeds,
            use_cache, output_attentions, output_hidden_states, return_dict, labels): Constructs the model with given
            inputs and returns the output.
        _prepare_attention_mask_for_generation(self, inputs, pad_token_id, eos_token_id):
            Prepares the attention mask for generation.
        prepare_inputs_for_generation(self, input_ids, token_type_ids, images, past_key_values, attention_mask, inputs_embeds):
            Prepares inputs for generation.
        _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, standardize_cache_format):
            Updates model keyword arguments for generation.
        _reorder_cache(self, past_key_values, beam_idx): Reorders the cache.
        build_conversation_input_ids(self, tokenizer, query, history, images, template_version):
            Builds input IDs for a conversation with a given tokenizer, query, history, images, and template version.

    """
    _auto_class = "AutoModelForCausalLM"

    def __init__(self, config):
        """
        Initialize the CogVLMForCausalLM class.

        Args:
            self (CogVLMForCausalLM): The instance of the CogVLMForCausalLM class.
            config:
                A dictionary containing configuration parameters for the model.

                - Type: dict
                - Purpose: This parameter holds various configuration settings required to initialize the model.
                - Restrictions: Must be a valid dictionary containing necessary configuration keys.

        Returns:
            None.

        Raises:
            KeyError: If the 'config' dictionary does not contain required keys.
            ValueError: If the values in the 'config' dictionary are invalid or out of range.
            TypeError: If the input parameters are of incorrect types.
        """
        super().__init__(config)
        self.model = CogVLMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Returns the input embeddings of the CogVLMForCausalLM model.

        Args:
            self: An instance of the CogVLMForCausalLM class.

        Returns:
            None: The method retrieves and returns the input embeddings of the model. These embeddings represent the
                learned representations of the input tokens.

        Raises:
            None.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set input embeddings for the CogVLMForCausalLM model.

        Args:
            self (CogVLMForCausalLM): The current instance of the CogVLMForCausalLM class.
            value (Tensor): The input embeddings to be set for the model. It should be a tensor of shape (vocab_size, embedding_dim).

        Returns:
            None.

        Raises:
            None.

        This method sets the input embeddings for the CogVLMForCausalLM model. It assigns the input embeddings to the
        'embed_tokens' attribute of the model, which is responsible for handling the input embeddings during the
        model's forward pass.

        Note:
            The input embeddings should be a tensor of shape (vocab_size, embedding_dim), where 'vocab_size' is
            the size of the vocabulary and 'embedding_dim' is the dimension of the embedding space.
        """
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """Return the output embeddings from the CogVLMForCausalLM model.

        This method takes no additional arguments other than the instance itself.

        Args:
            self: An instance of the CogVLMForCausalLM class.

        Returns:
            lm_head: This method returns the output embeddings of the CogVLMForCausalLM model.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the CogVLMForCausalLM model.

        Args:
            self (CogVLMForCausalLM): The instance of the CogVLMForCausalLM class.
            new_embeddings: The new embeddings to be set. This parameter can be of any type.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        Sets the decoder for the CogVLMForCausalLM class.

        Args:
            self (CogVLMForCausalLM): The instance of the CogVLMForCausalLM class.
            decoder: The decoder to be set for the CogVLMForCausalLM instance.

        Returns:
            None.

        Raises:
            None.
        """
        self.model = decoder

    def get_decoder(self):
        """
        Returns the decoder model used for causal language modeling in CogVLMForCausalLM.

        Args:
            self (CogVLMForCausalLM): The instance of the CogVLMForCausalLM class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.model

    def construct(
            self,
            input_ids: mindspore.Tensor = None,
            images: List[List[mindspore.Tensor]] = None,
            token_type_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[List[mindspore.Tensor]] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Constructs the CogVLMForCausalLM model.

        Args:
            self: An instance of the CogVLMForCausalLM class.
            input_ids (mindspore.Tensor, optional): The input tensor containing sequence tokens. Default is None.
            images (List[List[mindspore.Tensor]], optional): The list of image tensors. Default is None.
            token_type_ids (mindspore.Tensor, optional): The tensor containing token type ids. Default is None.
            attention_mask (mindspore.Tensor, optional): The tensor containing attention mask. Default is None.
            position_ids (mindspore.Tensor, optional): The tensor containing position ids. Default is None.
            past_key_values (List[mindspore.Tensor], optional): The list of past key values. Default is None.
            inputs_embeds (mindspore.Tensor, optional): The tensor containing input embeddings. Default is None.
            use_cache (bool, optional): Whether to use cache. Default is None.
            output_attentions (bool, optional): Whether to output attentions. Default is None.
            output_hidden_states (bool, optional): Whether to output hidden states. Default is None.
            return_dict (bool, optional): Whether to return a dictionary. Default is None.
            labels (mindspore.Tensor, optional): The tensor containing labels. Default is None.

        Returns:
            Union[Tuple, CausalLMOutputWithPast]:
                A tuple or an instance of CausalLMOutputWithPast class containing the following:

                - loss (mindspore.Tensor, optional): The loss value. None if labels is None.
                - logits (mindspore.Tensor): The output logits.
                - past_key_values (List[mindspore.Tensor]): The list of past key values.
                - hidden_states (mindspore.Tensor): The hidden states.
                - attentions (mindspore.Tensor): The attentions.

        Raises:
            None.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            images=images,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = ops.cross_entropy#CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _prepare_attention_mask_for_generation(
            self,
            inputs: mindspore.Tensor,
            pad_token_id: Optional[int],
            eos_token_id: Optional[Union[int, List[int]]],
    ) -> mindspore.Tensor:
        """
        Prepare attention mask for generation.

        Args:
            self (CogVLMForCausalLM): An instance of the CogVLMForCausalLM class.
            inputs (mindspore.Tensor): The input tensor.
            pad_token_id (Optional[int]): The ID of the padding token. Defaults to None.
            eos_token_id (Optional[Union[int, List[int]]]): The ID(s) of the end-of-sentence token. Defaults to None.

        Returns:
            mindspore.Tensor: The attention mask tensor.

        Raises:
            None.
        """
        return ops.ones(inputs.shape[:2], dtype=mindspore.int64)  # type: ignore

    def prepare_inputs_for_generation(
            self, input_ids, token_type_ids, images=None, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        Prepare inputs for generation.

        This method prepares the inputs for generating text using the CogVLMForCausalLM model. It takes the following parameters:

        Args:
            self (CogVLMForCausalLM): The instance of the CogVLMForCausalLM class.
            input_ids (torch.Tensor): The input tensor containing the token IDs of the input sequence.
            token_type_ids (torch.Tensor): The token type IDs tensor.
            images (torch.Tensor, optional): The tensor containing the image features. Defaults to None.
            past_key_values (torch.Tensor, optional): The tensor containing the past key values for generation. Defaults to None.
            attention_mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.
            inputs_embeds (torch.Tensor, optional): The tensor containing the embedded inputs. Defaults to None.

        Returns:
            dict: A dictionary containing the model inputs.

        Raises:
            None: This method does not raise any exceptions.

        Note:
            The input_ids, token_type_ids, and attention_mask tensors should have the same shape and dimensionality.
            If position_ids are not provided, they are built using the token_type_ids and attention_mask tensors.
            If past_key_values are provided, the input_ids, token_type_ids, and position_ids tensors are sliced to keep
            only the last token. The model_inputs dictionary is then constructed with the relevant tensors.

        Example:
            ```python
            >>> model = CogVLMForCausalLM()
            >>> input_ids = torch.tensor([[1, 2, 3]])
            >>> token_type_ids = torch.tensor([[0, 0, 0]])
            >>> inputs = model.prepare_inputs_for_generation(input_ids, token_type_ids)
            >>> print(inputs)
            {'input_ids': tensor([[3]]), 'token_type_ids': tensor([[0]]), 'images': None, 'position_ids': tensor([[2]]), 'past_key_values': None, 'use_cache': None, 'attention_mask': None}
            ```
        """
        # build position_ids if needed
        position_ids = kwargs.get("position_ids", None)
        if position_ids is None:
            position_ids = build_position_ids(token_type_ids, attention_mask)

        if past_key_values:
            input_ids = input_ids[:, -1:]
            token_type_ids = token_type_ids[:, -1:]
            position_ids = position_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "token_type_ids": token_type_ids,
                "images": images,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
            self,
            outputs: "ModelOutput",
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        '''
        Method for updating model keyword arguments for generation.

        Args:
            self: CogVLMForCausalLM
                The instance of the CogVLMForCausalLM class.
            outputs: ModelOutput
                The output from the model.
            model_kwargs: Dict[str, Any]
                The keyword arguments for the model.
            is_encoder_decoder: bool, optional
                Indicates whether the model is an encoder-decoder, defaults to False.
            standardize_cache_format: bool, optional
                Indicates whether to standardize the cache format, defaults to False.

        Returns:
            Dict[str, Any]
                Returns the updated model keyword arguments.

        Raises:
            None
        '''
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            new_token_type_ids = ops.ones((token_type_ids.shape[0], 1), dtype=token_type_ids.dtype,
                                            ) * LANGUAGE_TOKEN_TYPE
            model_kwargs["token_type_ids"] = ops.cat([token_type_ids, new_token_type_ids], axis=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = ops.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], axis=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = ops.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    axis=-1,
                )

        return model_kwargs

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Reorders the cached past states based on the provided beam index.

        Args:
            self (CogVLMForCausalLM): The instance of the CogVLMForCausalLM class.
            past_key_values (tuple): A tuple containing the past key values for each layer.
            beam_idx (Tensor): A tensor representing the indices of the beams to reorder past states.

        Returns:
            None: This method does not return any value but updates the reordered_past variable.

        Raises:
            TypeError: If the input parameters are not of the expected types.
            IndexError: If the index provided in beam_idx is out of bounds.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past

    def build_conversation_input_ids(
            self,
            tokenizer: "PreTrainedTokenizer",
            *,
            query: str,
            history: Optional[List[Tuple[str, str]]] = None,
            images: Optional[List["PIL.Image"]] = None,
            template_version: Optional[Literal["base", "chat", "vqa"]] = None,
    ):
        """
        This method builds conversation input IDs for the CogVLMForCausalLM class.

        Args:
            self: The instance of the class.
            tokenizer (PreTrainedTokenizer):
                The tokenizer used for tokenizing the input. It is required for encoding the input text and images.
            query (str): The query text for the conversation.
            history (Optional[List[Tuple[str, str]]]):
                A list of tuples containing the conversation history, where each tuple represents (user, bot) dialogue turns.
                Defaults to None.
            images (Optional[List[PIL.Image]]):
                A list of PIL images representing the visual context of the conversation. Defaults to None.
            template_version (Optional[Literal['base', 'chat', 'vqa']]):
                The version of the conversation template to be used. Defaults to None.

        Returns:
            dict: A dictionary containing the input_ids, token_type_ids, attention_mask, and images.
                The input_ids are the tokenized input for the conversation, token_type_ids specify the type of each token
                (language or vision), attention_mask indicates the position of valid tokens, and images represent
                the processed visual input.
        
        Raises:
            AssertionError: If the number of images provided is more than one.
        """
        image_size: int = self.config.vision_config['image_size']
        patch_size: int = self.config.vision_config['patch_size']
        template_version = template_version or self.config.template_version
        assert images is None or len(images) <= 1, "not support multi images by now."
        history = history or []
        text = _history_to_prompt(template_version, history, query)
        input_ids = [tokenizer.bos_token_id]
        token_type_ids = [LANGUAGE_TOKEN_TYPE]
        if images is not None and len(images) == 1:
            # vision
            transform = transforms.Compose(
                [
                    vision.Resize(
                        (image_size, image_size), interpolation= vision.Inter.BICUBIC),
                    vision.ToTensor(),
                    vision.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711),is_hwc=False),
                ]
            )
            images = transform(images[0])
            # language
            vision_token_num = (image_size // patch_size) * (image_size // patch_size) + 2
            input_ids += [tokenizer.pad_token_id] * vision_token_num
            token_type_ids += [VISION_TOKEN_TYPE] * vision_token_num
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        input_ids += text_ids
        token_type_ids += [LANGUAGE_TOKEN_TYPE] * len(text_ids)
        attention_mask = [1] * len(input_ids)
        return {
            'input_ids': mindspore.tensor(input_ids, dtype=mindspore.int64),
            'token_type_ids': mindspore.tensor(token_type_ids, dtype=mindspore.int64),
            'attention_mask': mindspore.tensor(attention_mask, dtype=mindspore.int64),
            'images': images,
        }

__all__ = [
    "MLP",
    "RMSNorm",
    'VisionExpertMLP',
    'RotaryEmbedding',
    'VisionExpertAttention',
    'CogVLMDecoderLayer',
    'CogVLMPreTrainedModel',
    'CogVLMModel',
    'CogVLMForCausalLM',
]
