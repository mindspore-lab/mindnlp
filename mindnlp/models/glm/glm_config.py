

"""
GLM Model Config
"""

from mindnlp.abc.backbones.pretrained import PretrainedConfig

class GLMConfig(PretrainedConfig):
    """
    GLMConfig
    """

    def __init__(
        self,
        num_layers,
        vocab_size,
        hidden_size,
        num_attention_heads,
        embedding_dropout_prob,
        attention_dropout_prob,
        output_dropout_prob,
        max_sequence_length,
        max_memory_length,
        checkpoint_activations,
        checkpoint_num_layers=1,
        parallel_output=True,
        relative_encoding=False,
        block_position_encoding=False,
        output_predict=True,
        spell_length=None,
        spell_func='lstm',
        attention_scale=1.0,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.embedding_dropout_prob = embedding_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.output_dropout_prob = output_dropout_prob
        self.max_sequence_length = max_sequence_length
        self.max_memory_length = max_memory_length
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.parallel_output = parallel_output
        self.relative_encoding = relative_encoding
        self.block_position_encoding = block_position_encoding
        self.spell_length = spell_length
        self.spell_func = spell_func
        self.attention_scale = attention_scale
        self.output_predict = output_predict

        if spell_length is not None:
            self.prompt_spell = PromptSpell(spell_length, self.hidden_size, spell_func)


