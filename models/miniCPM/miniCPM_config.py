class MiniCPMConfig:
    model_type = "minicpm"

    def __init__(
        self,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=8,
        vocab_size=128256,
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        pad_token_id=None,
        bos_token_id=128000,
        eos_token_id=128001,
        hidden_act="silu",
        rope_theta=500000.0,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=False,
        torch_dtype="float16",
        **kwargs,
    ):
        # ❌ 去掉 super().__init__()

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache
        self.torch_dtype = torch_dtype

        for k, v in kwargs.items():
            setattr(self, k, v)
