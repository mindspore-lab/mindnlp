# mindnlp/models/mimi/configuration_mimi.py

from mindnlp.configs import MINDNLP_CONFIG_URL_BASE
from ...configuration_utils import PretrainedConfig

MINDNLP_MODEL_CONFIG_URL_BASE = MINDNLP_CONFIG_URL_BASE + "mimi/"

MIMI_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mimi-base-uncased": MINDNLP_MODEL_CONFIG_URL_BASE + "mimi-base-uncased/config.json",
}


class MimiConfig(PretrainedConfig):


    def __init__(
        self,
        sampling_rate=24_000,
        frame_rate=12.5,
        audio_channels=1,
        hidden_size=512,
        num_filters=64,
        num_residual_layers=1,
        upsampling_ratios=None,
        kernel_size=7,
        last_kernel_size=3,
        residual_kernel_size=3,
        dilation_growth_rate=2,
        use_causal_conv=True,
        pad_mode="constant",
        compress=2,
        trim_right_ratio=1.0,
        codebook_size=2048,
        codebook_dim=256,
        num_quantizers=32,
        use_conv_shortcut=False,
        vector_quantization_hidden_dimension=256,
        num_semantic_quantizers=1,
        upsample_groups=512,
        num_hidden_layers=8,
        intermediate_size=2048,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=None,
        hidden_act="gelu",
        max_position_embeddings=8000,
        initializer_range=0.02,
        norm_eps=1e-5,
        use_cache=False,
        rope_theta=10000.0,
        sliding_window=250,
        attention_dropout=0.0,
        layer_scale_initial_scale=0.01,
        attention_bias=False,
        **kwargs,
    ):
        self.sampling_rate = sampling_rate
        self.frame_rate = frame_rate
        self.audio_channels = audio_channels
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.num_residual_layers = num_residual_layers
        self.upsampling_ratios = upsampling_ratios if upsampling_ratios else [8, 6, 5, 4]
        self.kernel_size = kernel_size
        self.last_kernel_size = last_kernel_size
        self.residual_kernel_size = residual_kernel_size
        self.dilation_growth_rate = dilation_growth_rate
        self.use_causal_conv = use_causal_conv
        self.pad_mode = pad_mode
        self.compress = compress
        self.trim_right_ratio = trim_right_ratio
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim if codebook_dim is not None else hidden_size
        self.num_quantizers = num_quantizers
        self.use_conv_shortcut = use_conv_shortcut
        self.vector_quantization_hidden_dimension = vector_quantization_hidden_dimension
        self.upsample_groups = upsample_groups
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.layer_scale_initial_scale = layer_scale_initial_scale
        self.attention_bias = attention_bias

        if num_semantic_quantizers >= self.num_quantizers:
            raise ValueError(
                f"The number of semantic quantizers should be lower than the total number of quantizers {self.num_quantizers}, but is currently {num_semantic_quantizers}."
            )
        self.num_semantic_quantizers = num_semantic_quantizers
        super().__init__(**kwargs)

    @property
    def encodec_frame_rate(self) -> int:
        hop_length = np.prod(self.upsampling_ratios)
        return math.ceil(self.sampling_rate / hop_length)

    @property
    def num_codebooks(self) -> int:
        # alias to num_quantizers
        return self.num_quantizers


__all__ = ["MimiConfig"]
