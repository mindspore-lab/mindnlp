# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
LayoutLM Models config
"""
from mindnlp.utils import logging
from ...configuration_utils import PretrainedConfig

logger = logging.get_logger(__name__)

LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/layoutlm-base-uncased": (
        "https://hf-mirror.com/microsoft/layoutlm-base-uncased/resolve/main/config.json"
    ),
    "microsoft/layoutlm-large-uncased": (
        "https://hf-mirror.com/microsoft/layoutlm-large-uncased/resolve/main/config.json"
    ),
}


class LayoutLMConfig(PretrainedConfig):
    """LayoutLMConfig"""
    model_type = "layoutlm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        max_2d_position_embeddings=1024,
        **kwargs,
    ):
        """
        Initializes a LayoutLMConfig object.
        
        Args:
            self: The instance of the class.
            vocab_size (int, optional): The size of the vocabulary. Defaults to 30522.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 768.
            num_hidden_layers (int, optional): The number of hidden layers. Defaults to 12.
            num_attention_heads (int, optional): The number of attention heads. Defaults to 12.
            intermediate_size (int, optional): The size of the intermediate layer in the transformer encoder. 
                Defaults to 3072.
            hidden_act (str, optional): The activation function for the hidden layers. Defaults to 'gelu'.
            hidden_dropout_prob (float, optional): The dropout probability for the hidden layers. Defaults to 0.1.
            attention_probs_dropout_prob (float, optional): The dropout probability for the attention probabilities. 
                Defaults to 0.1.
            max_position_embeddings (int, optional): The maximum sequence length that this model might ever be used with. 
                Defaults to 512.
            type_vocab_size (int, optional): The size of the token type vocabulary. Defaults to 2.
            initializer_range (float, optional): The standard deviation of the truncated_normal_initializer for 
                initializing all weight matrices. Defaults to 0.02.
            layer_norm_eps (float, optional): The epsilon value to use in LayerNorm layers. Defaults to 1e-12.
            pad_token_id (int, optional): The id of the padding token. Defaults to 0.
            position_embedding_type (str, optional): The type of position embedding. Defaults to 'absolute'.
            use_cache (bool, optional): Whether to use cache for the model. Defaults to True.
            max_2d_position_embeddings (int, optional): The maximum 2D sequence length that this model 
                might ever be used with. Defaults to 1024.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.max_2d_position_embeddings = max_2d_position_embeddings

__all__ = ['LayoutLMConfig']
