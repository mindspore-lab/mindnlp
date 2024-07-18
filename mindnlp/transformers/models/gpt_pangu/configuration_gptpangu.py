# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
PanGu_Alpha Models config
"""
from ...configuration_utils import PretrainedConfig


GPTPANGU_PRETRAINED_CONFIG_ARCHIVE_MAP = ['pangu-350M', 'pangu-2_6B', 'pangu-13B']


class GPTPanguConfig(PretrainedConfig):
    """GPTPanguConfig"""
    model_type = "gpt_pangu"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=40000,
        max_position_embeddings=1024,
        hidden_size=2560,
        intermediate_size=None,
        num_layers=32,
        num_heads=32,
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        scale_attn_weights=True,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        use_cache=True,
        bos_token_id=9,
        eos_token_id=9,
        **kwargs,
    ):
        """
        This method initializes an instance of the GPTPanguConfig class.
        
        Args:
            self: The instance of the class.
            vocab_size (int, optional): The size of the vocabulary. Defaults to 40000.
            max_position_embeddings (int, optional): The maximum position index. Defaults to 1024.
            hidden_size (int, optional): The hidden size of the model. Defaults to 2560.
            intermediate_size (int, optional): The size of the intermediate layer in the transformer encoder. Defaults to None.
            num_layers (int, optional): The number of layers in the transformer encoder. Defaults to 32.
            num_heads (int, optional): The number of attention heads in the transformer encoder. Defaults to 32.
            activation_function (str, optional): The activation function used in the transformer layers. Defaults to 'gelu'.
            resid_pdrop (float, optional): The dropout probability for the residual connections. Defaults to 0.1.
            embd_pdrop (float, optional): The dropout probability for the embedding layer. Defaults to 0.1.
            attn_pdrop (float, optional): The dropout probability for the attention layers. Defaults to 0.1.
            layer_norm_epsilon (float, optional): The epsilon value for layer normalization. Defaults to 1e-05.
            scale_attn_weights (bool, optional): Whether to scale the attention weights. Defaults to True.
            initializer_range (float, optional): The range of the initializer. Defaults to 0.02.
            summary_type (str, optional): The type of summary produced by the model. Defaults to 'cls_index'.
            summary_use_proj (bool, optional): Whether to use projection in the summary. Defaults to True.
            summary_activation (str, optional): The activation function used in the summary. Defaults to None.
            summary_proj_to_labels (bool, optional): Whether to project to labels in the summary. Defaults to True.
            summary_first_dropout (float, optional): The dropout probability for the first summary layer. Defaults to 0.1.
            use_cache (bool, optional): Whether to use cache in the model. Defaults to True.
            bos_token_id (int, optional): The beginning of sequence token id. Defaults to 9.
            eos_token_id (int, optional): The end of sequence token id. Defaults to 9.
        
        Returns:
            None.
        
        Raises:
            None
        """
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.scale_attn_weights = scale_attn_weights
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

__all__ = ['GPTPanguConfig']
